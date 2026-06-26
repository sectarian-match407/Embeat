# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-21

import numpy as np
import torch
from dataclasses import dataclass
from datasets import load_from_disk
from typing import Any


# Dataset configurations
@dataclass
class DatasetConfig:
    # HuggingFace dataset path
    dataset_path: str
    # Artist index max
    artist_max_idx: int = 99999
    # year_idx = [year_min, year_max] - year_base + 1
    year_base: int = 1950
    year_min: int = 1950
    year_max: int = 2030


# Get int value safely
def safe_int(value: Any, default: int = 0):
    try:
        return int(value)
    except Exception:
        return int(default)


# Get float value safely
def safe_float(value: Any, default: float):
    try:
        return float(value)
    except Exception:
        return float(default)


# Index time signature: 3 ~ 7 -> 1 ~ 5, 0 = others
def time_signature_to_idx(time_signature: Any):
    time_signature = safe_int(time_signature, default=0)
    if time_signature in (3, 4, 5, 6, 7):
        return time_signature - 2
    return 0


# Index year: year_idx = [year_min, year_max] - year_base + 1, 0 = others
def year_to_idx(year: Any, *, base: int, year_min: int, year_max: int):
    year = safe_int(year, default=0)
    if year_min <= year <= year_max:
        return year - base + 1
    return 0


# Index musical key: -1 ~ 11 -> 0 ~ 12, 1 = C
def key_to_idx(key: Any):
    key = safe_int(key, default=-1)
    if key == -1:
        return 0
    if 0 <= key <= 11:
        return key + 1
    return 0


# Index musical mode: 0 ~ 1 -> 0 ~ 2, 1 = major, 2 = minor, 0 = unknown
def mode_to_idx(mode: Any):
    if mode is None:
        return 0
    mode = safe_int(mode, default=-1)
    if mode in (0, 1):
        return mode + 1
    return 0


# Index tempo: tempo / 10 -> 1 ~ 11, 0 = unknown
def tempo_to_idx(tempo: Any):
    tempo_value = safe_int(tempo, 0)
    if tempo_value <= 0:
        return 0
    if tempo_value >= 170:
        tempo_value = int(tempo_value / 2)
    return max(1, min(int(tempo_value / 10) - 5, 11))


# Build 7-dim dense vector
def build_dense_vector(row: dict):
    danceability = safe_float(row.get("danceability"), 0.5)
    energy = safe_float(row.get("energy"), 0.5)
    valence = safe_float(row.get("valence"), 0.5)
    acousticness = safe_float(row.get("acousticness"), 0.5)
    liveness = safe_float(row.get("liveness"), 0.5)
    speechiness = safe_float(row.get("speechiness"), 0.5)
    instrumentalness = safe_float(row.get("instrumentalness"), 0.5)
    vec = np.array(
        [
            danceability - 0.5,
            energy - 0.5,
            valence - 0.5,
            acousticness - 0.5,
            liveness - 0.5,
            speechiness - 0.5,
            instrumentalness - 0.5,
        ],
        dtype=np.float32
    )
    return vec


# Process items from HF dataset, return each item as a dict
class SpotifyTracksDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        if "@" in cfg.dataset_path:
            path, length = cfg.dataset_path.split("@")
            path = path.strip()
            length = int(length)
        else:
            path = cfg.dataset_path.strip()
            length = 0
        self.ds = load_from_disk(path)
        if length > 0:
            self.ds = self.ds.select(range(length))

    def get_dataset_item(self, idx: int):
        return self.ds[int(idx)]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.ds[int(idx)]
        artist_idx = safe_int(row.get("artist_idx"), default=0)
        if not (0 <= artist_idx <= self.cfg.artist_max_idx):
            artist_idx = 0
        genre_idx = safe_int(row.get("artist_genre_idx"), default=0)
        if genre_idx < 0:
            genre_idx = 0
        ts_idx = time_signature_to_idx(row.get("time_signature"))
        year_idx = year_to_idx(row.get("release_year"), base=self.cfg.year_base, year_min=self.cfg.year_min, year_max=self.cfg.year_max)
        key_idx = key_to_idx(row.get("key"))
        mode_idx = mode_to_idx(row.get("mode"))
        tempo_idx = tempo_to_idx(row.get("tempo"))
        dense = build_dense_vector(row)
        weight = safe_float(row.get("popularity"), 0.0)
        return {
            "artist_idx": torch.tensor(artist_idx, dtype=torch.long),
            "genre_idx": torch.tensor(genre_idx, dtype=torch.long),
            "ts_idx": torch.tensor(ts_idx, dtype=torch.long),
            "year_idx": torch.tensor(year_idx, dtype=torch.long),
            "key_idx": torch.tensor(key_idx, dtype=torch.long),
            "mode_idx": torch.tensor(mode_idx, dtype=torch.long),
            "tempo_idx": torch.tensor(tempo_idx, dtype=torch.long),
            "dense": torch.from_numpy(dense),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "track_id": row.get("track_id") or "",
            "album_id": row.get("album_id") or "",
            "isrc": row.get("isrc") or ""
        }
