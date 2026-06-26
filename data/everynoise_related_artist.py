# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-05-26

import datasets
import json
import os
import random
import requests
import time
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock

file_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/").rstrip("/")
DATASET_DIR = f"{file_dir}/datasets/spotify_45m_tracks_metadata"
PROXY_INFO = {
    "http": "http://127.0.0.1:20171",
    "https": "http://127.0.0.1:20171"
}
IS_PROXY = True


# Generate tasks
def get_all_artist():
    artist_id_idx = {}
    dataset = datasets.load_from_disk(DATASET_DIR)
    for item in tqdm.tqdm(dataset, total=len(dataset)):
        artist_id_idx[str(item['artist_id'])] = int(item['artist_idx'])
    print(f"Collected {len(artist_id_idx)} artists.")
    output_path = f"{file_dir}/all_artist.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(artist_id_idx, ensure_ascii=False, indent=0))
    print(f"File saved: {output_path}")


# Send request and purge result
def fetch_related_artist(artist_id: str):
    html_text = "{" + f'"{artist_id}":null' + "}"
    url = f"https://everynoise.com/api/canon/{artist_id}"
    headers = {
        "Origin": "https://everynoise.com",
        "Referer": f"https://everynoise.com/artistprofile.cgi?id={artist_id}",
        "Accept": "*/*",
        "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randrange(100, 140)}.0.0.0 Safari/537.36"
    }
    proxies = PROXY_INFO if IS_PROXY else None
    resp = requests.get(url, headers=headers, timeout=60, proxies=proxies)
    resp.raise_for_status()
    html_raw = resp.text.strip()
    try:
        html_dict = json.loads(html_raw)
        if not html_dict:
            return html_text
        html_text = json.dumps(html_dict, separators=(",", ":"))
    except Exception:
        print(f"Failed to parse json result for item `{artist_id}`: {html_raw}")
    if not html_text:
        html_text = "{" + f'"{artist_id}":null' + "}"
    return html_text


# html_text to final item key-value
def parse_html_text(html_text: str, artist_id_idx: dict, top_k: int = 10):
    artist_idx = 0
    related_idxs = []
    html_text = html_text.strip()
    item_id_ids = json.loads(html_text)
    if not item_id_ids:
        return artist_idx, related_idxs
    artist_id = list(item_id_ids.keys())[0]
    artist_idx = artist_id_idx.get(artist_id) or 0
    if not item_id_ids[artist_id]:
        return artist_idx, related_idxs
    for item_id in item_id_ids[artist_id]:
        item_idx = artist_id_idx.get(item_id) or 0
        if item_idx > 0 and item_idx not in related_idxs:
            related_idxs.append(item_idx)
    related_idxs = related_idxs[:top_k]
    return artist_idx, related_idxs


# Save final json
def save_final_result(artist_idx_related_idxs: dict, output_json: str):
    original_items = []
    if os.path.isfile(output_json):
        try:
            with open(output_json) as f:
                original_items = json.loads(f.read())
        except Exception as e:
            print(f"Failed to read result json: {e}")
    items = []
    for k, v in artist_idx_related_idxs.items():
        artist_idx = str(k)
        related_idxs = list(v)
        if not related_idxs:
            continue
        item_str = f"{json.dumps(artist_idx)}:{json.dumps(related_idxs, separators=(',', ':'))}"
        items.append(item_str)
    json_str = "{\n" + ",\n".join(items) + "\n}"
    if len(items) > len(original_items):
        with open(output_json, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Save {len(items)} items to file: {output_json}")
    else:
        print(f"Skip saving {len(items)} items to file: original file has {len(original_items)} items.")
    return


# Process tasks (single thread)
def batch_run(save_every: int = 10000):
    output_txt = f"{file_dir}/everynoise/related_artist/related_artist_raw.txt"
    output_json = f"{file_dir}/everynoise/related_artist/related_artist_idx.json"
    with open(f"{file_dir}/everynoise/related_artist/all_artist.json", "r", encoding="utf-8") as f:
        artist_id_idx = json.loads(f.read())
    artist_idx_related_idxs = {}
    if os.path.isfile(output_txt):
        with open(output_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            html_text = line.strip()
            if not html_text:
                continue
            try:
                artist_idx, related_idxs = parse_html_text(html_text=html_text, artist_id_idx=artist_id_idx)
                artist_idx_related_idxs[artist_idx] = related_idxs
            except:
                print(f"Failed to read line {i}: {line}")
    counter = 0
    for artist_id, artist_idx in tqdm.tqdm(artist_id_idx.items()):
        counter = counter + 1
        if artist_idx in artist_idx_related_idxs:
            continue
        try:
            html_text = fetch_related_artist(artist_id=artist_id)
            with open(output_txt, "a", encoding="utf-8") as f:
                text = html_text + "\n"
                f.write(text)
            artist_idx, related_idxs = parse_html_text(html_text=html_text, artist_id_idx=artist_id_idx)
            artist_idx_related_idxs[artist_idx] = related_idxs
        except Exception as e:
            print(f"Failed to process item `{artist_id}`: {e}")
        if counter > 1 and (counter - 1) % save_every == 0:
            save_final_result(artist_idx_related_idxs, output_json)
    save_final_result(artist_idx_related_idxs, output_json)
    return artist_idx_related_idxs


# Process tasks (multi thread)
def batch_run_multi_thread(max_workers: int = 3, save_every: int = 10000):
    sort_by_artist_pop = True
    output_txt = f"{file_dir}/everynoise/related_artist/related_artist_raw.txt"
    output_json = f"{file_dir}/everynoise/related_artist/related_artist_idx.json"
    output_txt_lock = FileLock(f"{output_txt}.lock")
    output_json_lock = FileLock(f"{output_json}.lock")
    with open(f"{file_dir}/everynoise/related_artist/all_artist.json", "r", encoding="utf-8") as f:
        artist_id_idx = json.loads(f.read())
    if sort_by_artist_pop:
        artist_id_idx = dict(sorted(artist_id_idx.items(), key=lambda item: int(item[1])))
    artist_idx_related_idxs = {}
    if os.path.isfile(output_txt):
        with open(output_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            html_text = line.strip()
            if not html_text:
                continue
            try:
                artist_idx, related_idxs = parse_html_text(html_text=html_text, artist_id_idx=artist_id_idx)
                artist_idx_related_idxs[artist_idx] = related_idxs
            except:
                print(f"Failed to read line {i}: {line}")
        print(f"Skip {len(artist_idx_related_idxs)} items.")
    pending_items = [(k, v) for k, v in artist_id_idx.items() if v not in artist_idx_related_idxs]
    initial_count = max(0, len(artist_id_idx) - len(pending_items))
    counter = initial_count
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_artist_id = {}
        for artist_id, artist_idx in pending_items:
            future = executor.submit(fetch_related_artist, artist_id=artist_id)
            future_to_artist_id[future] = artist_id
        t1 = time.time()
        error_count = 0
        for future in tqdm.tqdm(as_completed(future_to_artist_id), initial=initial_count, total=len(artist_id_idx)):
            counter = counter + 1
            artist_id = future_to_artist_id[future]
            try:
                html_text = future.result()
                with output_txt_lock:
                    with open(output_txt, "a", encoding="utf-8") as f:
                        text = html_text + "\n"
                        f.write(text)
                artist_idx, related_idxs = parse_html_text(html_text=html_text, artist_id_idx=artist_id_idx)
                artist_idx_related_idxs[artist_idx] = related_idxs
                error_count = 0
            except Exception as e:
                error_count = error_count + 1
                print(f"Failed to fetch artist_id={artist_id}: {e}")
            if counter > 1 and (counter - 1) % save_every == 0:
                if sort_by_artist_pop:
                    artist_idx_related_idxs = dict(sorted(artist_idx_related_idxs.items(), key=lambda item: int(item[0])))
                with output_json_lock:
                    save_final_result(artist_idx_related_idxs, output_json)
                t2 = time.time()
                sec_per_item = round((t2 - t1) / (counter - initial_count), 3)
                eta_hour = int(sec_per_item * max(0, len(artist_id_idx) - counter - initial_count) / 60 / 60)
                print(f"ETA time: {eta_hour}h - {sec_per_item}s/it")
            if PROXY_INFO and error_count > 0 and error_count % 10 == 0:
                global IS_PROXY
                IS_PROXY = not IS_PROXY
                print(f"Too many error. Switch proxy to: {IS_PROXY}")
            if error_count > 20:
                print(f"Too many error. Sleep 10 min from: {time.ctime()}")
                time.sleep(360)
                error_count = 0
    if sort_by_artist_pop:
        artist_idx_related_idxs = dict(sorted(artist_idx_related_idxs.items(), key=lambda item: int(item[0])))
    with output_json_lock:
        save_final_result(artist_idx_related_idxs, output_json)
    return artist_idx_related_idxs


if __name__ == "__main__":
    # get_all_artist()
    # batch_run()
    batch_run_multi_thread()
