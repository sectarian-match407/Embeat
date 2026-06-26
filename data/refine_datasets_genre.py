# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-22

import datasets
import hdbscan
import json
import math
import numpy as np
import os
import re
import tqdm
from scipy.spatial.distance import cdist

file_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/").rstrip("/")
EVERYNOISE_JSON_DIR = f"{file_dir}/everynoise/genre_map"
ENGENREMAP_FILE = f"{file_dir}/everynoise/engenremap.json"
ARTIST_GENRE_IDX_PATCH_FILE = f"{file_dir}/everynoise/unk_artist_genre/artist_genre_idx_patch.json"
ORIGINAL_DATASET_DIR = f"{file_dir}/datasets/spotify_45m_tracks_metadata_original"
OUTPUT_DATASET_DIR = f"{file_dir}/datasets/spotify_45m_tracks_metadata"
os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
with open(ENGENREMAP_FILE, "r", encoding="utf-8") as f:
    ENGENREMAP_DATA = json.load(f)
GLOBAL_GENRE_DICT = {item['index']: item for item in ENGENREMAP_DATA}
GENRE_INDEX_DICT = {str(item['genre']).strip(): int(item['index']) for item in ENGENREMAP_DATA if item['genre'].strip()}
print(f"Loaded {len(GENRE_INDEX_DICT)} genres.")
with open(ARTIST_GENRE_IDX_PATCH_FILE, "r", encoding="utf-8") as f:
    ARTIST_GENRE_IDX_PATCH = json.loads(f.read())
ARTIST_GENRE_IDX_PATCH = {int(k): int(v) for k, v in ARTIST_GENRE_IDX_PATCH.items()}
print(f"Loaded {len(ARTIST_GENRE_IDX_PATCH)} unk_artist_genre.")


def generate_artist_genre_map():
    use_engenremap = True
    if use_engenremap:
        lines = [str(item['genre']).strip() for item in ENGENREMAP_DATA]
        if lines[0] == "":
            lines[0] = "<UNK>"
        if lines[0] != "<UNK>":
            lines.insert(0, "<UNK>")
    else:
        input_path = f"{EVERYNOISE_JSON_DIR}/genre_rank.txt"
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        if lines[0] != "<UNK>":
            lines.insert(0, "<UNK>")
    result = {}
    for i, line in enumerate(lines):
        result[str(i)] = line.strip()
    output_path = f"{OUTPUT_DATASET_DIR}/artist_genre_map.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"File saved: {output_path}")


def read_json_files():
    result = {}
    artist_genres_dict, artist_genre_id_dict = {}, {}
    with open(f"{EVERYNOISE_JSON_DIR}/genre_rank.txt", "r", encoding="utf-8") as f:
        genres = f.readlines()
    genres = [genre.strip() for genre in genres if genre.strip()]
    genres.insert(0, "<UNK>")
    for genre in tqdm.tqdm(genres):
        if genre == "<UNK>":
            continue
        genre_str = re.sub(r"[^a-zA-Z0-9]", "", genre)
        json_path = f"{EVERYNOISE_JSON_DIR}/{genre_str}.json"
        if not os.path.isfile(json_path):
            print(f"File not exist: {json_path}")
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        genre_name = data['genre_name'].lower().strip()
        try:
            genre_index = GENRE_INDEX_DICT.get(genre_name, 0)
        except:
            print(f"genre_name `{genre_name}` is not in genre_rank.txt. Treat as <UNK>.")
            genre_index = 0
        for item in data['genre_artists']:
            artist_id = str(item['artist_id'])
            weight = float(item['weight']) if genre_index > 0 else 0.0
            if artist_id not in result:
                result[artist_id] = {
                    "genres_name": "",
                    "genre_indexs": [],
                    "local_weights": []
                }
            if genre_name not in result[artist_id]['genres_name']:
                result[artist_id]['genres_name'] = result[artist_id]['genres_name'] + f", {genre_name}"
                result[artist_id]['genres_name'] = result[artist_id]['genres_name'].strip(",").strip()
            if genre_index not in result[artist_id]['genre_indexs']:
                result[artist_id]['genre_indexs'].append(genre_index)
                result[artist_id]['local_weights'].append(weight)
    for artist_id, artist_genres_info in result.items():
        artist_genres_info = rank_genres(artist_genres_info)
        artist_genres_dict[artist_id] = artist_genres_info['genres_name']
        artist_genre_id_dict[artist_id] = artist_genres_info['genre_indexs'][0]
    return artist_genres_dict, artist_genre_id_dict


def rank_genres(artist_genres_info: dict):
    genre_indexs = artist_genres_info.get("genre_indexs", [])
    local_weights = artist_genres_info.get("local_weights", [])
    if len(genre_indexs) <= 1:
        return artist_genres_info
    if len(genre_indexs) != len(local_weights):
        print(f"Error in func `rank_genres`: {artist_genres_info}")
        return artist_genres_info
    final_genre_indexs = []
    for _ in range(len(genre_indexs)):
        best_genre = get_final_genre(genre_indexs=genre_indexs, local_weights=local_weights)
        if best_genre <= 0:
            continue
        final_genre_indexs.append(best_genre)
        best_genre_position = genre_indexs.index(best_genre)
        genre_indexs.pop(best_genre_position)
        local_weights.pop(best_genre_position)
    if final_genre_indexs:
        artist_genres_info['genre_indexs'] = final_genre_indexs
        artist_genres_info['genres_name'] = ""
        for genre_index in final_genre_indexs:
            genre_name = GLOBAL_GENRE_DICT.get(genre_index, {}).get("genre", "")
            if genre_name:
                artist_genres_info['genres_name'] = artist_genres_info['genres_name'] + f", {genre_name}"
                artist_genres_info['genres_name'] = artist_genres_info['genres_name'].strip(",").strip()
    return artist_genres_info


def get_final_genre(genre_indexs: list, local_weights: list):
    scale = 0.015  # safe zone scale (value from `get_best_scale_value`)
    alpha = 2.0  # punishment outside safe zone
    if not genre_indexs or len(genre_indexs) != len(local_weights):
        return 0
    if len(genre_indexs) == 1:
        return genre_indexs[0]
    artist_genre_infos = []
    for genre_index, local_weight in zip(genre_indexs, local_weights):
        if genre_index not in GLOBAL_GENRE_DICT:
            continue
        genre_info = GLOBAL_GENRE_DICT[genre_index]
        global_weight = genre_info['weight']
        hybrid_weight = global_weight * local_weight
        artist_genre_infos.append({
            "index": genre_index,
            "genre": genre_info['genre'],
            "norm_x": genre_info['norm_x'],
            "norm_y": genre_info['norm_y'],
            "hybrid_weight": hybrid_weight
        })
    if not artist_genre_infos:
        print(f"`artist_genre_infos` is empty. Defalut to <UNK>.")
        return 0
    total_mass = sum(genre_info['hybrid_weight'] for genre_info in artist_genre_infos)
    if total_mass == 0:
        return artist_genre_infos[0]
    # centroid_x: $$C_x = \frac{\sum_{i=1}^{n} (W_i \cdot X_i)}{\sum_{i=1}^{n} W_i}$$
    centroid_x = sum(genre_info['hybrid_weight'] * genre_info['norm_x'] for genre_info in artist_genre_infos) / total_mass
    # centroid_y: $$C_y = \frac{\sum_{i=1}^{n} (W_i \cdot Y_i)}{\sum_{i=1}^{n} W_i}$$
    centroid_y = sum(genre_info['hybrid_weight'] * genre_info['norm_y'] for genre_info in artist_genre_infos) / total_mass
    best_index = None
    max_score = -1.0
    for genre_info in artist_genre_infos:
        # dist: $$D = \sqrt{(X_i - C_x)^2 + (Y_i - C_y)^2}$$
        dist = math.sqrt((genre_info['norm_x'] - centroid_x)**2 + (genre_info['norm_y'] - centroid_y)**2)
        # final score: $$Score = \frac{W_{global} \times W_{local}}{1 + \left(\frac{D}{S}\right)^\alpha}$$
        score = genre_info['hybrid_weight'] / (1.0 + (dist / scale) ** alpha)
        if score > max_score:
            max_score = score
            best_index = int(genre_info['index'])
    return best_index


def get_best_scale_value(min_cluster_size: int = 5):
    default_value = 0.01
    coords = np.array([[genre['norm_x'], genre['norm_y']] for genre in ENGENREMAP_DATA])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(coords)
    unique_clusters = set(labels)
    unique_clusters.discard(-1)
    if len(unique_clusters) < 2:
        print("Number of clusters is less than 2. Please reduce `min_cluster_size`.")
        return default_value
    cluster_centroids = []
    for cluster_id in unique_clusters:
        points_in_cluster = coords[labels == cluster_id]
        centroid = points_in_cluster.mean(axis=0)
        cluster_centroids.append(centroid)
    cluster_centroids = np.array(cluster_centroids)
    dist_matrix = cdist(cluster_centroids, cluster_centroids, metric="euclidean")
    nearest_distances = []
    for i in range(len(cluster_centroids)):
        no_zero_matrix = np.delete(dist_matrix[i], i)
        nearest_distances.append(np.min(no_zero_matrix))
    median_nearest_dist = np.median(nearest_distances)
    optimal_scale = median_nearest_dist / 2.0
    print(f"Current min_cluster_size: {min_cluster_size}")
    print(f"Cluster number: {len(unique_clusters)}")
    print(f"Best scale value: {optimal_scale:.4f}")
    return optimal_scale


def map_fn(data, artist_genres_dict: dict, artist_genre_id_dict: dict):
    artist_id = data['artist_id']
    if not artist_id:
        print(f"Item {data['artist_name']} has no artist_id.")
        data['artist_genres'] = ""
        data['artist_genre_idx'] = 0
        return data
    old_artist_genre = data['artist_genres'].split(",")[0].lower().strip()
    new_artist_genres = artist_genres_dict.get(artist_id)
    data['artist_genres'] = new_artist_genres if new_artist_genres is not None else data['artist_genres']
    new_artist_genre_idx = artist_genre_id_dict.get(artist_id, 0)
    if new_artist_genre_idx == 0:
        artist_idx = data['artist_idx']
        new_artist_genre_idx = ARTIST_GENRE_IDX_PATCH.get(artist_idx, 0)
        data['artist_genres'] = ""
    if new_artist_genre_idx == 0 and old_artist_genre:
        new_artist_genre_idx = GENRE_INDEX_DICT.get(old_artist_genre, 0)
    data['artist_genre_idx'] = new_artist_genre_idx
    return data


def generate_refined_dataset():
    artist_genres_dict, artist_genre_id_dict = read_json_files()
    dataset = datasets.load_from_disk(ORIGINAL_DATASET_DIR)
    dataset = dataset.map(map_fn, num_proc=16, fn_kwargs={"artist_genres_dict": artist_genres_dict, "artist_genre_id_dict": artist_genre_id_dict})
    dataset.save_to_disk(dataset_path=OUTPUT_DATASET_DIR, max_shard_size="4GB", num_proc=1)
    print(dataset)
    print(dataset[0])
    print(f"Refined dataset saved: {OUTPUT_DATASET_DIR}")


if __name__ == "__main__":
    # get_best_scale_value()
    generate_refined_dataset()
    generate_artist_genre_map()
