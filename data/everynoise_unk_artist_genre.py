# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-02

import datasets
import json
import os
import random
import requests
import tqdm
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock
from urllib.parse import quote

file_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/").rstrip("/")
DATASET_DIR = f"{file_dir}/datasets/spotify_45m_tracks_metadata"


# Generate tasks
def get_unk_artist():
    artist_idx_name = {}
    dataset = datasets.load_from_disk(DATASET_DIR)
    # dataset = dataset.filter(lambda x: x['artist_genre_idx'] == 0 and x['artist_popularity'] > 0.0, num_proc=32)
    dataset = dataset.filter(lambda x: x['artist_genre_idx'] <= 0, num_proc=32)
    for item in tqdm.tqdm(dataset, total=len(dataset)):
        artist_idx_name[str(item['artist_idx'])] = str(item['artist_name'])
    print(f"Collected {len(artist_idx_name)} artists.")
    output_path = f"{file_dir}/everynoise/unk_artist_genre/unk_artist.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(artist_idx_name, ensure_ascii=False, indent=0))
    print(f"File saved: {output_path}")


# Get html response string
def fetch_html(url: str):
    headers = {
        "Origin": "https://everynoise.com",
        "Referer": "https://everynoise.com/",
        "Accept": "*/*",
        "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randrange(100, 140)}.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    html_text = resp.text.strip()
    return html_text


# Send request and purge result
def fetch_artist_genres(artist_idx: int, artist_name: str):
    url = f"https://everynoise.com/lookup.cgi?who={quote(artist_name)}&mode=map"
    genre_list = []
    artist_genres = "<UNK>"
    try:
        html = fetch_html(url=url)
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select('a[href^="engenremap-"]'):
            genre_name = a.get_text(" ", strip=True)
            if genre_name and genre_name not in genre_list:
                genre_list.append(genre_name)
    except Exception:
        return artist_idx, ""
    if genre_list:
        artist_genres = ", ".join(genre_list)
    return artist_idx, artist_genres


# Save final json
def save_final_result(artist_idx_genre_idx: dict, output_json: str):
    result = {str(k): int(v) for k, v in artist_idx_genre_idx.items() if int(v) > 0}
    with open(output_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=0))
    print(f"Save {len(result)} items to file: {output_json}")


# Process tasks (single thread)
def batch_run(save_every: int = 10000):
    output_txt = f"{file_dir}/everynoise/unk_artist_genre/unk_artist_result.txt"
    output_json = f"{file_dir}/everynoise/unk_artist_genre/artist_genre_idx_patch.json"
    with open(f"{file_dir}/everynoise/unk_artist_genre/unk_artist.json", "r", encoding="utf-8") as f:
        artist_idx_name = json.loads(f.read())
    with open(f"{file_dir}/everynoise/engenremap.json", "r", encoding="utf-8") as f:
        engenremap = json.loads(f.read())
    genre_idx_map = {item['genre']: item['index'] for item in engenremap}
    artist_idx_name = {int(k): str(v) for k, v in artist_idx_name.items()}
    artist_idx_genre_idx = {}
    if os.path.isfile(output_txt):
        with open(output_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                line = line.strip()
                parts = line.split(" | ")
                artist_idx = int(parts[0].strip())
                artist_genres = "".join(parts[1:])
                artist_genre = artist_genres.split(", ")[0].strip()
                artist_genre_idx = int(genre_idx_map.get(artist_genre, 0))
                artist_idx_genre_idx[artist_idx] = artist_genre_idx
            except:
                print(f"Failed to read line {i}: {line}")
    counter = 0
    for artist_idx, artist_name in tqdm.tqdm(artist_idx_name.items()):
        counter = counter + 1
        if artist_idx in artist_idx_genre_idx:
            continue
        artist_idx, artist_genres = fetch_artist_genres(artist_idx=artist_idx, artist_name=artist_name)
        if artist_genres:
            with open(output_txt, "a", encoding="utf-8") as f:
                text = f"{artist_idx} | {artist_genres}\n"
                f.write(text)
        else:
            print(f"Failed to fetch item and skip: artist_idx={artist_idx} artist_name={artist_name}")
        if counter > 1 and (counter - 1) % save_every == 0:
            save_final_result(artist_idx_genre_idx, output_json)
    save_final_result(artist_idx_genre_idx, output_json)
    return artist_idx_genre_idx


# Process tasks (multi thread)
def batch_run_multi_thread(max_workers: int = 4, save_every: int = 10000):
    output_txt = f"{file_dir}/everynoise/unk_artist_genre/unk_artist_result.txt"
    output_json = f"{file_dir}/everynoise/unk_artist_genre/artist_genre_idx_patch.json"
    output_txt_lock = FileLock(f"{output_txt}.lock")
    output_json_lock = FileLock(f"{output_json}.lock")
    with open(f"{file_dir}/everynoise/unk_artist_genre/unk_artist.json", "r", encoding="utf-8") as f:
        artist_idx_name = json.loads(f.read())
    with open(f"{file_dir}/everynoise/engenremap.json", "r", encoding="utf-8") as f:
        engenremap = json.loads(f.read())
    genre_idx_map = {str(item['genre']).strip(): int(item['index']) for item in engenremap}
    artist_idx_name = {int(k): str(v) for k, v in artist_idx_name.items()}
    artist_idx_genre_idx = {}
    if os.path.isfile(output_txt):
        with output_txt_lock:
            with open(output_txt, "r", encoding="utf-8") as f:
                lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" | ")
                artist_idx = int(parts[0].strip())
                artist_genres = "".join(parts[1:]).strip()
                artist_genre = artist_genres.split(", ")[0].strip()
                artist_genre_idx = int(genre_idx_map.get(artist_genre, 0))
                artist_idx_genre_idx[artist_idx] = artist_genre_idx
            except Exception:
                print(f"Failed to read line {i}: {line}")
        print(f"Skip {len(artist_idx_genre_idx)} items.")
    pending_items = [(k, v) for k, v in artist_idx_name.items() if k not in artist_idx_genre_idx]
    initial_count = max(0, len(artist_idx_name) - len(pending_items))
    counter = initial_count
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_artist_idx = {}
        for artist_idx, artist_name in pending_items:
            future = executor.submit(fetch_artist_genres, artist_idx=artist_idx, artist_name=artist_name)
            future_to_artist_idx[future] = artist_idx
        for future in tqdm.tqdm(as_completed(future_to_artist_idx), initial=initial_count, total=len(artist_idx_name)):
            counter = counter + 1
            artist_idx = future_to_artist_idx[future]
            artist_genres = ""
            try:
                artist_idx, artist_genres = future.result()
            except Exception as e:
                print(f"Failed to fetch artist_idx={artist_idx}: {e}")
            if artist_genres:
                with output_txt_lock:
                    with open(output_txt, "a", encoding="utf-8") as f:
                        text = f"{artist_idx} | {artist_genres}\n"
                        f.write(text)
                artist_genre = artist_genres.split(", ")[0].strip()
                artist_genre_idx = int(genre_idx_map.get(artist_genre, 0))
                artist_idx_genre_idx[artist_idx] = artist_genre_idx
            if counter > 1 and (counter - 1) % save_every == 0:
                with output_json_lock:
                    save_final_result(artist_idx_genre_idx, output_json)
    with output_json_lock:
        save_final_result(artist_idx_genre_idx, output_json)
    return artist_idx_genre_idx


if __name__ == "__main__":
    get_unk_artist()
    # batch_run()
    # batch_run_multi_thread()
