# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-21

import numpy as np
import torch
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any

# Dense vector index: train/dataset.py -> build_dense_vector
DENSE_DANCEABILITY_INDEX = 0
DENSE_ENERGY_INDEX = 1
DENSE_VALENCE_INDEX = 2
DENSE_ACOUSTICNESS_INDEX = 3
DENSE_LIVENESS_INDEX = 4
DENSE_SPEECHINESS_INDEX = 5
DENSE_INSTRUMENTALNESS_INDEX = 6


# Sampler configurations
@dataclass
class PairSamplerConfig:
    # Positive bucket A rate: sample from same album, fallback to bucket B if miss
    pos_album_ratio: float = 0.99
    # Positive bucket B rate: sample by near acoustic similarity, fallback to bucket A if miss
    pos_genre_acoustic_ratio: float = 0.01
    # Positive dense threshold: valence difference of anchor and positive should not be too big
    pos_valence_diff_max: float = 0.20
    # Positive dense threshold: energy difference of anchor and positive should not be too big
    pos_energy_diff_max: float = 0.20
    # Positive dense threshold: danceability difference of anchor and positive should not be too big
    pos_danceability_diff_max: float = 0.30
    # Positive dense threshold: acousticness difference of anchor and positive should not be too big
    pos_acousticness_diff_max: float = 0.30
    # Positive dense threshold: liveness difference of anchor and positive should not be too big
    pos_liveness_diff_max: float = 0.20
    # Positive dense threshold: speechiness difference of anchor and positive should not be too big
    pos_speechiness_diff_max: float = 0.05
    # Positive dense threshold: L2 distance of anchor and positive should not be too far
    pos_dense_weighted_l2_max: float = 0.30
    # Positive dense weight: valence weight in L2 distance calculation
    pos_dense_weight_valence: float = 1.0
    # Positive dense weight: energy weight in L2 distance calculation
    pos_dense_weight_energy: float = 1.0
    # Positive dense weight: danceability weight in L2 distance calculation
    pos_dense_weight_danceability: float = 1.0
    # Positive dense weight: acousticness weight in L2 distance calculation
    pos_dense_weight_acousticness: float = 0.5
    # Positive dense weight: liveness weight in L2 distance calculation
    pos_dense_weight_liveness: float = 0.5
    # Positive dense weight: speechiness weight in L2 distance calculation
    pos_dense_weight_speechiness: float = 1.0
    # Discrete hard constraints: mode of anchor and positive should be the same (relative major/minor will be excluded)
    require_mode_match_for_positive: bool = True
    # Discrete hard constraints: key of anchor and positive should be close (circle-of-fifths distance, chromatic-circle distance or relative major/minor)
    require_key_close_for_positive: bool = True
    # Discrete hard constraints: tempo of anchor and positive should be close
    require_tempo_close_for_positive: bool = True
    # Discrete hard constraints: time signature of anchor and positive should be the same
    require_time_signature_match_for_positive: bool = True
    # Discrete hard constraints: artist genre of anchor and positive should be the same
    require_genre_match_for_positive: bool = False
    # Tempo threshold: abs(anchor_tempo_idx - positive_tempo_idx) <= pos_tempo_idx_diff_max
    pos_tempo_idx_diff_max: int = 1
    # Circle-of-fifths distance threshold
    pos_key_fifths_dist_max: int = 2
    # chromatic-circle distance threshold
    pos_key_chromatic_dist_max: int = 1
    # Positive filter: skip positive if positive_genre is "Unknown"
    pos_require_known_genre: bool = True
    # Anchor filter: skip anchor if anchor_genre is "Unknown" (set to True if require_genre_match_for_positive == True)
    anchor_require_known_genre: bool = True
    # Negative loss mask: anchor and negative should not be from the same album
    neg_exclude_same_album: bool = True
    # Negative loss mask: anchor and negative should not have the same ISRC
    neg_exclude_same_isrc: bool = True
    # Positive sampling bias: higher value means prefer higher popularity sample, applies to positive candidate score
    pos_popularity_power: float = 0.5
    # Positive sampling bias: minimum popularity weight
    pos_popularity_min_weight: float = 0.10
    # Positive sampling bias: maximum popularity weight
    pos_popularity_max_weight: float = 1.00
    # Positive candidates ranking bias: anchor and positive are from a same album
    score_bonus_same_album: float = 0.08
    # Positive candidates ranking bias: anchor and positive are from a same artist
    score_bonus_same_artist: float = 0.04
    # Positive candidates ranking bias: anchor and positive have a same genre
    score_bonus_same_genre: float = 0.02
    # Positive candidates ranking bias: anchor and positive have a same mode
    score_bonus_mode_match: float = 0.00
    # Positive candidates ranking bias: anchor and positive have a close key
    score_bonus_key_close: float = 0.00
    # Positive candidates ranking bias: anchor and positive have a close tempo_idx
    score_bonus_tempo_close: float = 0.00
    # Positive candidates ranking bias: anchor and positive have a same time signature
    score_bonus_time_signature_match: float = 0.00
    # Global filter: speechiness_raw >= threshold will be excluded, to avoid talk-only audio
    speechiness_max_exclusive_raw: float = 0.20
    # Cache items per genre
    cache_per_genre: int = 4096
    # Cache items per album
    cache_per_album: int = 128
    # Limit number of album ids in cache (avoid memory leak)
    max_album_cache_keys: int = 10000
    # Try time in one of positive buckets (higher value consumes more time but lower anchor_drop_frac)
    max_positive_tries: int = 128
    # Global random seed
    random_seed: int = 616


# Subset of HuggingFace dataset
@dataclass(frozen=True)
class Candidate:
    idx: int
    artist_idx: int
    album_id: str
    isrc: str
    genre_idx: int  # 1..6291, unknown -> 0
    tempo_idx: int  # 0..11, unknown -> 0
    mode: int  # raw: 0/1, unknown -> -1
    key: int  # raw pitch class: 0..11, unknown -> -1
    time_signature: int  # raw: 3/4/5/..., unknown -> 0
    dense: np.ndarray  # shape (7,)
    popularity: float  # raw: 0..1 or 0..100


# Get circle-of-fifths distance by two pitch classes
def circle_of_fifths_dist(pitch_class_a: Any, pitch_class_b: Any):
    if isinstance(pitch_class_a, torch.Tensor) or isinstance(pitch_class_b, torch.Tensor):
        fifths_position_a = (pitch_class_a * 7) % 12
        fifths_position_b = (pitch_class_b * 7) % 12
        absolute_diff = torch.abs(fifths_position_a - fifths_position_b)
        wrapped_diff = 12 - absolute_diff
        distance = torch.minimum(absolute_diff, wrapped_diff)
        return distance
    else:
        pitch_class_a_array = np.asarray(pitch_class_a)
        pitch_class_b_array = np.asarray(pitch_class_b)
        fifths_position_a = (pitch_class_a_array * 7) % 12
        fifths_position_b = (pitch_class_b_array * 7) % 12
        absolute_diff = np.abs(fifths_position_a - fifths_position_b)
        wrapped_diff = 12 - absolute_diff
        distance = np.minimum(absolute_diff, wrapped_diff)
        return distance


# Get chromatic circle distance by two pitch classes
def chromatic_circle_dist(pitch_class_a: Any, pitch_class_b: Any):
    if isinstance(pitch_class_a, torch.Tensor) or isinstance(pitch_class_b, torch.Tensor):
        absolute_diff = torch.abs(pitch_class_a - pitch_class_b)
        wrapped_diff = 12 - absolute_diff
        distance = torch.minimum(absolute_diff, wrapped_diff)
        return distance
    else:
        pitch_class_a_array = np.asarray(pitch_class_a)
        pitch_class_b_array = np.asarray(pitch_class_b)
        absolute_diff = np.abs(pitch_class_a_array - pitch_class_b_array)
        wrapped_diff = 12 - absolute_diff
        distance = np.minimum(absolute_diff, wrapped_diff)
        return distance


# Normalize time signature value
def normalize_time_signature(time_signature: Any):
    if isinstance(time_signature, torch.Tensor):
        normalized_tensor = torch.where(
            time_signature == 6,
            torch.full_like(time_signature, 3),
            time_signature,
        )
        return normalized_tensor
    else:
        normalized_value = int(time_signature)
        if normalized_value == 6:
            normalized_value = 3
        return normalized_value


# Get the weighted L2 distance for two dense vectors
def weighted_l2_distance_dense_vectors(anchor_dense_vector: np.ndarray, candidate_dense_vector: np.ndarray, config: Any):
    def build_dense_weight_vector_numpy(config: Any):
        weight_vector = np.asarray(
            [
                float(config.pos_dense_weight_danceability),
                float(config.pos_dense_weight_energy),
                float(config.pos_dense_weight_valence),
                float(config.pos_dense_weight_acousticness),
                float(config.pos_dense_weight_liveness),
                float(config.pos_dense_weight_speechiness)
            ],
            dtype=np.float32
        )
        return weight_vector

    anchor_dense_float = np.asarray(anchor_dense_vector, dtype=np.float32)
    candidate_dense_float = np.asarray(candidate_dense_vector, dtype=np.float32)
    anchor_core_vector = np.asarray(
        [
            anchor_dense_float[DENSE_DANCEABILITY_INDEX],
            anchor_dense_float[DENSE_ENERGY_INDEX],
            anchor_dense_float[DENSE_VALENCE_INDEX],
            anchor_dense_float[DENSE_ACOUSTICNESS_INDEX],
            anchor_dense_float[DENSE_LIVENESS_INDEX],
            anchor_dense_float[DENSE_SPEECHINESS_INDEX]
        ],
        dtype=np.float32
    )
    candidate_core_vector = np.asarray(
        [
            candidate_dense_float[DENSE_DANCEABILITY_INDEX],
            candidate_dense_float[DENSE_ENERGY_INDEX],
            candidate_dense_float[DENSE_VALENCE_INDEX],
            candidate_dense_float[DENSE_ACOUSTICNESS_INDEX],
            candidate_dense_float[DENSE_LIVENESS_INDEX],
            candidate_dense_float[DENSE_SPEECHINESS_INDEX]
        ],
        dtype=np.float32
    )
    diff_vector = anchor_core_vector - candidate_core_vector
    weight_vector = build_dense_weight_vector_numpy(config)
    weighted_square_sum = float(np.sum(weight_vector * (diff_vector * diff_vector)))
    weighted_square_sum = max(0.0, weighted_square_sum)
    distance = float(np.sqrt(weighted_square_sum))
    return distance


# Map popularity to a sampling weight (positive prefers higher popularity)
def popularity_to_sampling_weight(popularity: float, config: PairSamplerConfig):
    pop = float(popularity)
    if pop < 0.0:
        pop = 0.0
    if pop > 1.0:
        pop = pop / 100.0
    if pop > 1.0:
        pop = 1.0
    pop = float(np.power(pop, float(config.pos_popularity_power)))
    min_w = float(config.pos_popularity_min_weight)
    max_w = float(config.pos_popularity_max_weight)
    weight = min_w + (max_w - min_w) * pop
    return weight


# Optional: discrete features join the hard constraints
def check_discrete_constraints(anchor: Any, candidate: Any, config: Any):
    require_mode_match = bool(config.require_mode_match_for_positive)
    require_key_close = bool(config.require_key_close_for_positive)
    is_relative_key = False
    if require_mode_match or require_key_close:
        anchor_mode_value = int(anchor.mode)
        candidate_mode_value = int(candidate.mode)
        anchor_pitch_class = int(anchor.key)
        candidate_pitch_class = int(candidate.key)
        anchor_related_key = int((anchor_pitch_class + 3) % 12) if anchor_mode_value == 0 else int((anchor_pitch_class - 3) % 12)
        is_relative_key = (candidate_mode_value != anchor_mode_value) and (candidate_pitch_class == anchor_related_key)
    if require_mode_match:
        if anchor_mode_value not in [0, 1] or candidate_mode_value not in [0, 1]:
            return False
        if candidate_mode_value != anchor_mode_value and not is_relative_key:
            return False
    if require_key_close:
        if not (0 <= anchor_pitch_class <= 11 and 0 <= candidate_pitch_class <= 11):
            return False
        fifths_distance = int(circle_of_fifths_dist(anchor_pitch_class, candidate_pitch_class))
        chromatic_distance = int(chromatic_circle_dist(anchor_pitch_class, candidate_pitch_class))
        fifths_threshold = int(config.pos_key_fifths_dist_max)
        chromatic_threshold = int(config.pos_key_chromatic_dist_max)
        if fifths_distance > fifths_threshold and chromatic_distance > chromatic_threshold and not is_relative_key:
            return False
    require_tempo_close = bool(config.require_tempo_close_for_positive)
    if require_tempo_close:
        anchor_tempo_index = int(anchor.tempo_idx)
        candidate_tempo_index = int(candidate.tempo_idx)
        if anchor_tempo_index <= 0 or candidate_tempo_index <= 0:
            return False
        tempo_diff_max = int(config.pos_tempo_idx_diff_max)
        if abs(candidate_tempo_index - anchor_tempo_index) > tempo_diff_max:
            return False
    require_time_signature_match = bool(config.require_time_signature_match_for_positive)
    if require_time_signature_match:
        anchor_time_signature = int(anchor.time_signature)
        candidate_time_signature = int(candidate.time_signature)
        if not (3 <= anchor_time_signature <= 7 and 3 <= candidate_time_signature <= 7):
            return False
        anchor_time_signature = normalize_time_signature(anchor_time_signature)
        candidate_time_signature = normalize_time_signature(candidate_time_signature)
        if candidate_time_signature != anchor_time_signature:
            return False
    anchor_require_known_genre = bool(config.anchor_require_known_genre)
    pos_require_known_genre = bool(config.pos_require_known_genre)
    require_genre_match = bool(config.require_genre_match_for_positive)
    anchor_genre_index = int(anchor.genre_idx)
    candidate_genre_index = int(candidate.genre_idx)
    if anchor_require_known_genre and anchor_genre_index <= 0:
        return False
    if pos_require_known_genre and candidate_genre_index <= 0:
        return False
    if require_genre_match and candidate_genre_index != anchor_genre_index:
        return False
    return True


# Define if a bucket item can be one of positive candidates
def is_positive_candidate(anchor: Any, candidate: Any, config: Any):
    if int(candidate.idx) == int(anchor.idx):
        return False
    anchor_dense_vector = np.asarray(anchor.dense, dtype=np.float32)
    candidate_dense_vector = np.asarray(candidate.dense, dtype=np.float32)
    if anchor_dense_vector.shape[0] < 7 or candidate_dense_vector.shape[0] < 7:
        return False
    speechiness_max_exclusive_raw = float(config.speechiness_max_exclusive_raw)
    speechiness_max_exclusive_centered = speechiness_max_exclusive_raw - 0.5
    anchor_speechiness_centered = float(anchor_dense_vector[DENSE_SPEECHINESS_INDEX])
    candidate_speechiness_centered = float(candidate_dense_vector[DENSE_SPEECHINESS_INDEX])
    if anchor_speechiness_centered >= speechiness_max_exclusive_centered:
        return False
    if candidate_speechiness_centered >= speechiness_max_exclusive_centered:
        return False
    vocal_threshold_centered = -0.5
    anchor_instrumentalness_centered = float(anchor_dense_vector[DENSE_INSTRUMENTALNESS_INDEX])
    candidate_instrumentalness_centered = float(candidate_dense_vector[DENSE_INSTRUMENTALNESS_INDEX])
    anchor_is_vocal_song = anchor_instrumentalness_centered <= vocal_threshold_centered
    if anchor_is_vocal_song:
        candidate_is_vocal_song = candidate_instrumentalness_centered <= vocal_threshold_centered
        if not candidate_is_vocal_song:
            return False
    energy_diff = abs(
        float(anchor_dense_vector[DENSE_ENERGY_INDEX]) - float(candidate_dense_vector[DENSE_ENERGY_INDEX])
    )
    valence_diff = abs(
        float(anchor_dense_vector[DENSE_VALENCE_INDEX]) - float(candidate_dense_vector[DENSE_VALENCE_INDEX])
    )
    acousticness_diff = abs(
        float(anchor_dense_vector[DENSE_ACOUSTICNESS_INDEX]) - float(candidate_dense_vector[DENSE_ACOUSTICNESS_INDEX])
    )
    liveness_diff = abs(
        float(anchor_dense_vector[DENSE_LIVENESS_INDEX]) - float(candidate_dense_vector[DENSE_LIVENESS_INDEX])
    )
    speechiness_diff = abs(
        float(anchor_dense_vector[DENSE_SPEECHINESS_INDEX]) - float(candidate_dense_vector[DENSE_SPEECHINESS_INDEX])
    )
    if energy_diff >= float(config.pos_energy_diff_max):
        return False
    if valence_diff >= float(config.pos_valence_diff_max):
        return False
    if acousticness_diff >= float(config.pos_acousticness_diff_max):
        return False
    if liveness_diff >= float(config.pos_liveness_diff_max):
        return False
    if speechiness_diff >= float(config.pos_speechiness_diff_max):
        return False
    danceability_diff_max = float(config.pos_danceability_diff_max)
    danceability_diff = abs(
        float(anchor_dense_vector[DENSE_DANCEABILITY_INDEX]) - float(candidate_dense_vector[DENSE_DANCEABILITY_INDEX])
    )
    if danceability_diff >= danceability_diff_max:
        return False
    weighted_distance = weighted_l2_distance_dense_vectors(anchor_dense_vector, candidate_dense_vector, config)
    weighted_distance_max = float(config.pos_dense_weighted_l2_max)
    if weighted_distance > weighted_distance_max:
        return False
    if not check_discrete_constraints(anchor, candidate, config):
        return False
    return True


# Build pair samples (anchor, positive)
class PairSampler:
    def __init__(self, config: PairSamplerConfig):
        self.config = config
        self.rng = np.random.default_rng(self.config.random_seed)
        self.genre_cache: Dict[int, List[Candidate]] = {}
        self.album_cache = OrderedDict()

    # Generate genre and album cache
    def add_candidate_to_cache(self, candidate: Candidate):
        def has_same_idx(candidates: List[Candidate], target_idx: int):
            for item in candidates:
                if int(item.idx) == int(target_idx):
                    return True
            return False

        genre_index = int(candidate.genre_idx)
        if genre_index not in self.genre_cache.keys():
            self.genre_cache[genre_index] = []
        if not has_same_idx(self.genre_cache[genre_index], int(candidate.idx)):
            self.genre_cache[genre_index].append(candidate)
        if len(self.genre_cache[genre_index]) > int(self.config.cache_per_genre):
            del_length = len(self.genre_cache[genre_index]) - int(self.config.cache_per_genre)
            del self.genre_cache[genre_index][:del_length]
        if candidate.album_id:
            album_id = str(candidate.album_id)
            if album_id in self.album_cache:
                self.album_cache.move_to_end(album_id)
            else:
                max_keys = int(self.config.max_album_cache_keys)
                if max_keys > 0 and len(self.album_cache) >= max_keys:
                    self.album_cache.popitem(last=False)
                self.album_cache[album_id] = []
            if not has_same_idx(self.album_cache[album_id], int(candidate.idx)):
                self.album_cache[album_id].append(candidate)
            if len(self.album_cache[album_id]) > int(self.config.cache_per_album):
                del_length = len(self.album_cache[album_id]) - int(self.config.cache_per_album)
                del self.album_cache[album_id][:del_length]

    # Higher discrete_bonus makes positive candidate closer to anchor
    def calculate_discrete_bonus(self, anchor: Candidate, candidate: Candidate):
        bonus_value = 0.0
        if bool(anchor.album_id) and str(anchor.album_id) == str(candidate.album_id):
            bonus_value = bonus_value + float(self.config.score_bonus_same_album)
        if bool(anchor.artist_idx) and str(anchor.artist_idx) == str(candidate.artist_idx):
            bonus_value = bonus_value + float(self.config.score_bonus_same_artist)
        if int(anchor.genre_idx) > 0 and int(candidate.genre_idx) > 0:
            if int(anchor.genre_idx) == int(candidate.genre_idx):
                bonus_value = bonus_value + float(self.config.score_bonus_same_genre)
        if int(anchor.mode) in [0, 1] and int(candidate.mode) in [0, 1]:
            if int(anchor.mode) == int(candidate.mode):
                bonus_value = bonus_value + float(self.config.score_bonus_mode_match)
        tempo_diff_max = int(self.config.pos_tempo_idx_diff_max)
        if int(anchor.tempo_idx) > 0 and int(candidate.tempo_idx) > 0:
            tempo_diff = abs(int(anchor.tempo_idx) - int(candidate.tempo_idx))
            if tempo_diff <= tempo_diff_max:
                bonus_value = bonus_value + float(self.config.score_bonus_tempo_close)
        if 0 <= int(anchor.key) <= 11 and 0 <= int(candidate.key) <= 11:
            fifths_distance = int(circle_of_fifths_dist(int(anchor.key), int(candidate.key)))
            chromatic_distance = int(chromatic_circle_dist(int(anchor.key), int(candidate.key)))
            if fifths_distance <= int(self.config.pos_key_fifths_dist_max) or chromatic_distance <= int(self.config.pos_key_chromatic_dist_max):
                bonus_value = bonus_value + float(self.config.score_bonus_key_close)
        anchor_time_signature = int(anchor.time_signature)
        candidate_time_signature = int(candidate.time_signature)
        if 3 <= anchor_time_signature <= 7 and 3 <= candidate_time_signature <= 7:
            anchor_time_signature = normalize_time_signature(anchor_time_signature)
            candidate_time_signature = normalize_time_signature(candidate_time_signature)
            if anchor_time_signature == candidate_time_signature:
                bonus_value = bonus_value + float(self.config.score_bonus_time_signature_match)
        return bonus_value

    # candidate_score = (weighted_distance / popularity_weight) - discrete_bonus, lower is better
    def calculate_candidate_score(self, anchor: Candidate, candidate: Candidate):
        weighted_distance = weighted_l2_distance_dense_vectors(anchor.dense, candidate.dense, config=self.config)
        popularity_weight = popularity_to_sampling_weight(candidate.popularity, config=self.config)
        base_score = float(weighted_distance) / max(1e-6, float(popularity_weight))
        discrete_bonus = self.calculate_discrete_bonus(anchor=anchor, candidate=candidate)
        score_value = base_score - float(discrete_bonus)
        return score_value, weighted_distance

    # Build positive sampling buckets
    def build_cached_source_sequence(self, anchor: Candidate):
        candidate_sources: List[Tuple[str, List[Candidate]]] = []
        album_ratio = float(self.config.pos_album_ratio)
        genre_ratio = float(self.config.pos_genre_acoustic_ratio)
        has_album_bucket = album_ratio > 0 and bool(anchor.album_id)
        has_genre_bucket = genre_ratio > 0 and int(anchor.genre_idx) != 0
        if not has_album_bucket and not has_genre_bucket:
            return []
        album_candidates = []
        genre_candidates = []
        if has_album_bucket:
            album_candidates = self.album_cache.get(str(anchor.album_id), [])
        if has_genre_bucket:
            genre_candidates = self.genre_cache.get(int(anchor.genre_idx), [])
        if has_album_bucket and not has_genre_bucket:
            candidate_sources = [("album_cache", album_candidates)]
            return candidate_sources
        if has_genre_bucket and not has_album_bucket:
            candidate_sources = [("genre_cache", genre_candidates)]
            return candidate_sources
        total_ratio = album_ratio + genre_ratio
        random_target = float(self.rng.random()) * max(1e-9, total_ratio)
        if random_target < album_ratio:
            candidate_sources.append(("album_cache", album_candidates))
            candidate_sources.append(("genre_cache", genre_candidates))
        else:
            candidate_sources.append(("genre_cache", genre_candidates))
            candidate_sources.append(("album_cache", album_candidates))
        return candidate_sources

    # Find the best positive in positive sampling buckets
    def find_best_positive_in_buckets(self, anchor: Candidate, source_candidates: List[Candidate]):
        if not source_candidates:
            return None
        best_candidate = None
        best_score_value = float("inf")
        max_positive_tries = min(len(source_candidates), int(self.config.max_positive_tries))
        for _ in range(max_positive_tries):
            random_index = int(self.rng.integers(0, len(source_candidates)))
            candidate = source_candidates[random_index]
            if not is_positive_candidate(anchor=anchor, candidate=candidate, config=self.config):
                continue
            candidate_score_value, _ = self.calculate_candidate_score(anchor=anchor, candidate=candidate)
            if candidate_score_value < best_score_value:
                best_candidate = candidate
                best_score_value = candidate_score_value
        return best_candidate

    # Fallback: find the best positive randomly
    def find_best_positive_in_random(self, anchor: Candidate, get_candidate: Callable[[], Candidate]):
        best_candidate = None
        best_score_value = float("inf")
        max_positive_tries = int(self.config.max_positive_tries)
        for _ in range(max_positive_tries):
            candidate = get_candidate()
            if not is_positive_candidate(anchor=anchor, candidate=candidate, config=self.config):
                continue
            candidate_score_value, _ = self.calculate_candidate_score(anchor=anchor, candidate=candidate)
            if candidate_score_value < best_score_value:
                best_candidate = candidate
                best_score_value = candidate_score_value
        return best_candidate

    # Get the best positive for anchor
    def sample_positive(self, anchor: Candidate, get_candidate: Callable[[], Candidate]):
        cached_source_sequence = self.build_cached_source_sequence(anchor=anchor)
        for source_name, source_candidates in cached_source_sequence:
            if not source_candidates:
                continue
            best_candidate = self.find_best_positive_in_buckets(anchor=anchor, source_candidates=source_candidates)
            if best_candidate is not None:
                return best_candidate
        best_candidate = self.find_best_positive_in_random(anchor=anchor, get_candidate=get_candidate)
        return best_candidate
