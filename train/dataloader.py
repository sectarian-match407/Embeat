# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-17

import json
import numpy as np
import os
import sys
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from train.dataset import DatasetConfig, SpotifyTracksDataset
from train.sampler import Candidate, PairSampler, PairSamplerConfig


# Dataloader configurations
@dataclass
class DataLoaderConfig:
    # Batch size
    batch_size: int = 64
    # Workers to fetch data
    num_workers: int = 0
    # True if using GPU to train
    pin_memory: bool = True
    # True normally
    drop_last: bool = True
    # Reduce memory when num_workers > 1
    prefetch_factor: int = 1
    # Keep workers to speed up when epoch transition
    persistent_workers: bool = True
    # Is return track_id/album_id/isrc when training
    include_ids: bool = True


# Generate pair sample: {anchor, positive}, anchor[] = SpotifyTracksDataset[idx]
# Pack to batch format if batch_size > 1: anchor[B][] = SpotifyTracksDataset[idx]
class PairIterableDataset(IterableDataset):
    def __init__(self, dataset: SpotifyTracksDataset, sampler: PairSampler, batch_size: int, include_ids: bool):
        super().__init__()
        self.dataset = dataset
        self.sampler = sampler
        self.seed = int(self.sampler.config.random_seed)
        self.batch_size = int(batch_size)
        self.include_ids = bool(include_ids)
        self.random_generator = None
        self.dataset_length = 0
        self.get_artist_genre_patch()

    # Read additional artist genre patch as a dict
    def get_artist_genre_patch(self):
        self.artist_genre_idx_patch = {}
        patch_file = ""
        patch_file_list = [f"{project_root}/artist_genre_idx_patch.json", f"{project_root}/train/artist_genre_idx_patch.json", f"{project_root}/data/everynoise/unk_artist_genre/artist_genre_idx_patch.json"]
        for p in patch_file_list:
            if os.path.isfile(p):
                patch_file = p
                break
        if not patch_file:
            print("WARNING: Optional file `artist_genre_idx_patch.json` is missing. System might have a bad training result on some edge cases.")
            return
        try:
            with open(patch_file, "r", encoding="utf-8") as f:
                artist_genre_idx_patch = json.loads(f.read())
            self.artist_genre_idx_patch = {int(k): int(v) for k, v in artist_genre_idx_patch.items()}
        except Exception as e:
            print(f"WARNING: Failed to read `artist_genre_idx_patch.json` and skip: {e}")

    # Get a int random index
    def sample_random_index(self):
        random_generator = self.random_generator
        dataset_length = int(self.dataset_length)
        random_index = int(random_generator.integers(0, dataset_length))
        return random_index

    # Convert dataset item into candidate format: eg. -1 = Unknown
    def dataset_item_to_candidate(self, dataset_index: int, dataset_item: dict):
        artist_index = int(dataset_item['artist_idx'].item())
        genre_index = int(dataset_item['genre_idx'].item())
        if genre_index <= 0:
            genre_index = self.artist_genre_idx_patch.get(artist_index) or 0
        tempo_index = int(dataset_item['tempo_idx'].item())
        key_index = int(dataset_item['key_idx'].item())
        key_pitch_class = key_index - 1
        if not 0 <= key_pitch_class <= 11:
            key_pitch_class = -1
        mode_index = int(dataset_item['mode_idx'].item())
        mode_raw = mode_index - 1
        if mode_raw not in [0, 1]:
            mode_raw = -1
        time_signature_index = int(dataset_item['ts_idx'].item())
        if 1 <= time_signature_index <= 5:
            time_signature = time_signature_index + 2
        else:
            time_signature = 0
        dense_vector = dataset_item['dense'].detach().cpu().numpy()
        popularity = float(dataset_item.get("weight").item()) if "weight" in dataset_item else 0.0
        candidate = Candidate(
            idx=int(dataset_index),
            artist_idx=int(artist_index),
            album_id=str(dataset_item.get("album_id") or ""),
            isrc=str(dataset_item.get("isrc") or ""),
            genre_idx=int(genre_index),
            tempo_idx=int(tempo_index),
            mode=int(mode_raw),
            key=int(key_pitch_class),
            time_signature=int(time_signature),
            dense=dense_vector,
            popularity=float(popularity),
        )
        return candidate

    # Get one candidate randomly
    def get_candidate(self):
        candidate_index = self.sample_random_index()
        dataset_item = self.dataset[candidate_index]
        candidate = self.dataset_item_to_candidate(candidate_index, dataset_item)
        self.sampler.add_candidate_to_cache(candidate)
        return candidate

    # List[Dict['anchor']] -> Dict['anchor']['artist_idx'][0...B-1]
    def collate_pair_batch(self, batch: list, include_ids: bool, sampled_anchor_count: int, dropped_anchor_count: int):
        def stack_pair(batch: list, field_name: str):
            anchor_tensor = torch.stack([item['anchor'][field_name] for item in batch], dim=0)
            positive_tensor = torch.stack([item['positive'][field_name] for item in batch], dim=0)
            return anchor_tensor, positive_tensor

        if not batch:
            raise ValueError("Empty batch in collate_pair_batch")
        result = {
            "artist_idx": stack_pair(batch, "artist_idx"),
            "genre_idx": stack_pair(batch, "genre_idx"),
            "ts_idx": stack_pair(batch, "ts_idx"),
            "year_idx": stack_pair(batch, "year_idx"),
            "key_idx": stack_pair(batch, "key_idx"),
            "mode_idx": stack_pair(batch, "mode_idx"),
            "tempo_idx": stack_pair(batch, "tempo_idx"),
            "dense": stack_pair(batch, "dense"),
            "weight": stack_pair(batch, "weight"),
        }
        sampled_anchor_count_value = int(sampled_anchor_count)
        dropped_anchor_count_value = int(dropped_anchor_count)
        if sampled_anchor_count_value > 0:
            anchor_drop_frac_value = float(dropped_anchor_count_value) / float(sampled_anchor_count_value)
        else:
            anchor_drop_frac_value = 0.0
        result['sampled_anchor_count'] = torch.tensor(sampled_anchor_count_value, dtype=torch.float32)
        result['dropped_anchor_count'] = torch.tensor(dropped_anchor_count_value, dtype=torch.float32)
        result['anchor_drop_frac'] = torch.tensor(anchor_drop_frac_value, dtype=torch.float32)
        if bool(include_ids):
            result['track_id'] = (
                [item['anchor']['track_id'] for item in batch],
                [item['positive']['track_id'] for item in batch],
            )
            result['album_id'] = (
                [item['anchor']['album_id'] for item in batch],
                [item['positive']['album_id'] for item in batch],
            )
            result['isrc'] = (
                [item['anchor']['isrc'] for item in batch],
                [item['positive']['isrc'] for item in batch],
            )
        return result

    # Generator: return a batch (anchor-positive pairs) each time
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_seed = self.seed
        else:
            worker_seed = self.seed + int(worker_info.id) + 1
        self.random_generator = np.random.default_rng(worker_seed)
        self.dataset_length = int(len(self.dataset))
        batch_size = int(self.batch_size)
        if batch_size <= 0:
            batch_size = 1
        while True:
            pair_items: list[dict] = []
            sampled_anchor_count = 0
            dropped_anchor_count = 0
            while len(pair_items) < int(batch_size):
                anchor_index = self.sample_random_index()
                anchor_item = self.dataset[anchor_index]
                if bool(self.sampler.config.anchor_require_known_genre):
                    if int(anchor_item['genre_idx'].item()) <= 0:
                        continue
                anchor_speechiness_centered = float(anchor_item['dense'][5].item())
                speechiness_max_exclusive_raw = float(getattr(self.sampler.config, "speechiness_max_exclusive_raw", 0.30))
                speechiness_max_exclusive_centered = speechiness_max_exclusive_raw - 0.5
                if anchor_speechiness_centered >= speechiness_max_exclusive_centered:
                    continue
                anchor_candidate = self.dataset_item_to_candidate(anchor_index, anchor_item)
                self.sampler.add_candidate_to_cache(anchor_candidate)
                sampled_anchor_count = sampled_anchor_count + 1
                positive_candidate = self.sampler.sample_positive(anchor_candidate, self.get_candidate)
                if positive_candidate is None:
                    dropped_anchor_count = dropped_anchor_count + 1
                    continue
                positive_item = self.dataset[int(positive_candidate.idx)]
                pair_items.append(
                    {
                        "anchor": anchor_item,
                        "positive": positive_item,
                    }
                )
            batch_result = self.collate_pair_batch(
                pair_items,
                include_ids=bool(self.include_ids),
                sampled_anchor_count=sampled_anchor_count,
                dropped_anchor_count=dropped_anchor_count,
            )
            yield batch_result


# Entry: build pair dataloader
def build_pair_dataloader(dataset_config: DatasetConfig, sampler_config: PairSamplerConfig, dataloader_config: DataLoaderConfig):
    dataset = SpotifyTracksDataset(dataset_config)
    sampler = PairSampler(sampler_config)
    iterable_dataset = PairIterableDataset(
        dataset=dataset,
        sampler=sampler,
        batch_size=int(dataloader_config.batch_size),
        include_ids=bool(dataloader_config.include_ids),
    )
    loader_kwargs = dict(
        batch_size=1,
        num_workers=int(dataloader_config.num_workers),
        pin_memory=bool(dataloader_config.pin_memory),
        drop_last=bool(dataloader_config.drop_last),
        collate_fn=lambda batch: batch[0],
    )
    if int(dataloader_config.num_workers) > 0:
        loader_kwargs['prefetch_factor'] = int(dataloader_config.prefetch_factor)
        loader_kwargs['persistent_workers'] = bool(dataloader_config.persistent_workers)
    loader = DataLoader(iterable_dataset, **loader_kwargs)
    return loader
