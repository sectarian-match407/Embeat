# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-03-11

import json
import random
import re
import requests
import time
import tqdm
import os
from bs4 import BeautifulSoup, Tag

file_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/").rstrip("/")
GENRE_LIST_URL = "https://everynoise.com/everynoise1d.html"
GENRE_DETAIL_URL = "https://everynoise.com/engenremap-###.html"
ENGENREMAP_URL = "https://everynoise.com/engenremap.html"
OUTPUT_DIR = f"{file_dir}/everynoise/genre_map"
SKIP_EXISTED = True


def fetch_html(url: str):
    headers = {
        "Origin": "https://everynoise.com",
        "Referer": "https://everynoise.com/",
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    html_text = resp.text.strip()
    return html_text


def get_genre_list():
    url = GENRE_LIST_URL
    html = fetch_html(url=url)
    soup = BeautifulSoup(html, "html.parser")
    selector = "body > table > tbody > tr > td:nth-child(3) > a"
    nodes = soup.select(selector)
    if not nodes:
        nodes = soup.select("body > table > tr > td:nth-child(3) > a")
    genre_list = []
    for node in nodes:
        genre_name = node.get_text(" ", strip=True).replace("»", "").strip()
        if genre_name:
            genre_list.append(genre_name)
    return genre_list


def get_genre_detail(url: str):
    html = fetch_html(url=url)
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select("div.genre")
    artists = []
    for node in nodes:
        artist_result = parse_artist_node(node)
        if artist_result.get("artist_id", ""):
            artists.append(artist_result)
    genres = []
    for node in nodes:
        genre_result = parse_genre_node(node)
        if genre_result.get("genre", ""):
            genres.append(genre_result)
    genre_name = soup.title.get_text(strip=True) if soup.title else None
    genre_name = genre_name.replace("Every Noise at Once", "").strip().strip("-").strip()
    result = {
        "genre_name": genre_name,
        "genre_artists": artists,
        "nearby_genres": genres,
    }
    return result


def parse_artist_node(node: Tag):
    result = {
        "artist_id": "",
        "artist_name": "",
        "preview_track_id": "",
        "preview_track_url": "",
        "weight": 1.0,
        "map_x": 0,
        "map_y": 0
    }
    selector = node.get("id", "")
    if not selector.startswith("item"):
        return result
    onclick = node.get("onclick", "")
    ONCLICK_PATTERN = re.compile(r'playx\("([^"]+)",\s*"([^"]+)"')
    match = ONCLICK_PATTERN.search(onclick)
    nav = node.select_one("a.navlink")
    style_text = node.get("style")
    STYLE_PATTERN = re.compile(r"([a-zA-Z-]+)\s*:\s*([^;]+)")
    style_dict = {k.strip(): v.strip() for k, v in STYLE_PATTERN.findall(style_text)}
    result['artist_id'] = str(nav['href']).split("?id=")[-1].strip()
    result['artist_name'] = str(match.group(2)).strip() if match else node.get_text(" ", strip=True).replace("»", "").strip()
    result['preview_track_id'] = str(match.group(1)).strip() if match else ""
    result['preview_track_url'] = node.get("preview_url")
    result['weight'] = round(float(style_dict.get("font-size").replace("%", "").strip()) / 100, 2)
    result['map_x'] = int(style_dict.get("left").replace("px", "").strip())
    result['map_y'] = int(style_dict.get("top").replace("px", "").strip())
    return result


def parse_genre_node(node: Tag):
    result = {
        "genre": "",
        "preview_track_id": "",
        "preview_track_url": "",
        "weight": 1.0,
        "map_x": 0,
        "map_y": 0
    }
    selector = node.get("id", "")
    if not selector.startswith("nearby"):
        return result
    onclick = node.get("onclick", "")
    ONCLICK_PATTERN = re.compile(r'playx\("([^"]+)",\s*"([^"]+)"')
    match = ONCLICK_PATTERN.search(onclick)
    style_text = node.get("style")
    STYLE_PATTERN = re.compile(r"([a-zA-Z-]+)\s*:\s*([^;]+)")
    style_dict = {k.strip(): v.strip() for k, v in STYLE_PATTERN.findall(style_text)}
    result['genre'] = str(match.group(2)).strip() if match else node.get_text(" ", strip=True).replace("»", "").strip()
    result['preview_track_id'] = str(match.group(1)).strip() if match else ""
    result['preview_track_url'] = node.get("preview_url")
    result['weight'] = round(float(style_dict.get("font-size").replace("%", "").strip()) / 100, 2)
    result['map_x'] = int(style_dict.get("left").replace("px", "").strip())
    result['map_y'] = int(style_dict.get("top").replace("px", "").strip())
    return result


def batch_run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Getting genre list...")
    genre_list = get_genre_list()
    output_path = f"{OUTPUT_DIR}/genre_rank.txt"
    genre_list_output = [genre + "\n" for genre in genre_list]
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(genre_list_output)
    print(f"Genre list length: {len(genre_list)}")
    print(f"Batch getting genre detail...")
    for genre in tqdm.tqdm(genre_list):
        genre_str = re.sub(r"[^a-zA-Z0-9]", "", genre)
        output_path = f"{OUTPUT_DIR}/{genre_str}.json"
        if SKIP_EXISTED and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            continue
        genre_url = GENRE_DETAIL_URL.replace("###", genre_str)
        try:
            genre_result = get_genre_detail(url=genre_url)
            if not genre_result.get("genre_artists") and not genre_result.get("nearby_genres"):
                raise ValueError("No result error.")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(genre_result, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Failed to process genre `{genre}`: {e}")
        time.sleep(random.randrange(5, 10))


def get_engenremap():
    is_add_unk = True
    output_path = f"{file_dir}/everynoise/engenremap.json"
    onclick_pattern = re.compile(r'playx\("([^"]+)",\s*"([^"]+)"')
    style_pattern = re.compile(r"([a-zA-Z-]+)\s*:\s*([^;]+)")
    html = fetch_html(url=ENGENREMAP_URL)
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select("div.genre")
    result = []
    for node in nodes:
        item = {
            "index": 0,
            "genre": "",
            "preview_track_id": "",
            "preview_track_url": "",
            "weight": 0.0,
            "map_x": 0,
            "map_y": 0,
            "norm_x": 0.0,
            "norm_y": 0.0
        }
        if is_add_unk and not result:
            result.append(item)
            continue
        onclick = node.get("onclick", "")
        match = onclick_pattern.search(onclick)
        style_text = node.get("style") or ""
        style_dict = {k.strip(): v.strip() for k, v in style_pattern.findall(style_text)}
        item['index'] = len(result) + 1
        item['genre'] = str(match.group(2)).strip() if match else node.get_text(" ", strip=True).replace("»", "").strip()
        item['preview_track_id'] = str(match.group(1)).strip() if match else ""
        item['preview_track_url'] = node.get("preview_url") or ""
        item['weight'] = round(float(style_dict.get("font-size", "100%").replace("%", "").strip()) / 100, 2)
        item['map_x'] = int(style_dict.get("left", "0px").replace("px", "").strip())
        item['map_y'] = int(style_dict.get("top", "0px").replace("px", "").strip())
        result.append(item)
    x_values = [item['map_x'] for item in result]
    y_values = [item['map_y'] for item in result]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    x_span = max(max_x - min_x, 1)
    y_span = max(max_y - min_y, 1)
    for item in result:
        item['norm_x'] = round((item['map_x'] - min_x) / x_span, 6)
        item['norm_y'] = round((item['map_y'] - min_y) / y_span, 6)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Written {len(result)} genre items to file: {output_path}")
    return result


if __name__ == "__main__":
    # batch_run()
    get_engenremap()
