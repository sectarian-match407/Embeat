# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-21

import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Sequence, Dict, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from train.sampler import PairSamplerConfig, circle_of_fifths_dist, chromatic_circle_dist, normalize_time_signature
from train.sampler import DENSE_DANCEABILITY_INDEX, DENSE_ENERGY_INDEX, DENSE_VALENCE_INDEX, DENSE_ACOUSTICNESS_INDEX, DENSE_LIVENESS_INDEX, DENSE_SPEECHINESS_INDEX, DENSE_INSTRUMENTALNESS_INDEX


# Loss configurations
@dataclass
class MaskedInfoNCEConfig:
    # InfoNCE temperature (tau)
    tau: float = 0.07
    # Numerical stability
    eps: float = 1e-12
    # Import configurations from sampler
    sampler_config: PairSamplerConfig = field(default_factory=PairSamplerConfig)


# Get the weighted L2 distance for two dense vectors (tensor version)
def weighted_l2_distance_dense_matrix(anchor_dense_matrix: torch.Tensor, positive_dense_matrix: torch.Tensor, config: Any):
    def build_dense_weight_vector_torch(config: Any, target_device: torch.device, target_dtype: torch.dtype):
        weight_vector = torch.tensor(
            [
                float(config.pos_dense_weight_danceability),
                float(config.pos_dense_weight_energy),
                float(config.pos_dense_weight_valence),
                float(config.pos_dense_weight_acousticness),
                float(config.pos_dense_weight_liveness),
                float(config.pos_dense_weight_speechiness)
            ],
            device=target_device,
            dtype=target_dtype
        )
        return weight_vector

    anchor_core_matrix = torch.stack(
        [
            anchor_dense_matrix[:, DENSE_DANCEABILITY_INDEX],
            anchor_dense_matrix[:, DENSE_ENERGY_INDEX],
            anchor_dense_matrix[:, DENSE_VALENCE_INDEX],
            anchor_dense_matrix[:, DENSE_ACOUSTICNESS_INDEX],
            anchor_dense_matrix[:, DENSE_LIVENESS_INDEX],
            anchor_dense_matrix[:, DENSE_SPEECHINESS_INDEX]
        ],
        dim=1
    )
    positive_core_matrix = torch.stack(
        [
            positive_dense_matrix[:, DENSE_DANCEABILITY_INDEX],
            positive_dense_matrix[:, DENSE_ENERGY_INDEX],
            positive_dense_matrix[:, DENSE_VALENCE_INDEX],
            positive_dense_matrix[:, DENSE_ACOUSTICNESS_INDEX],
            positive_dense_matrix[:, DENSE_LIVENESS_INDEX],
            positive_dense_matrix[:, DENSE_SPEECHINESS_INDEX]
        ],
        dim=1
    )
    dense_weight_vector = build_dense_weight_vector_torch(
        config=config,
        target_device=anchor_dense_matrix.device,
        target_dtype=anchor_dense_matrix.dtype
    )
    core_diff_matrix = anchor_core_matrix[:, None, :] - positive_core_matrix[None, :, :]
    weighted_square_sum_matrix = torch.sum(core_diff_matrix * core_diff_matrix * dense_weight_vector.view(1, 1, -1), dim=2)
    weighted_distance_matrix = torch.sqrt(torch.clamp(weighted_square_sum_matrix, min=0.0))
    return weighted_distance_matrix


# Compare two strings in [B, B]
def build_string_diff_mask(anchor_values: Sequence[str], positive_values: Sequence[str], device: torch.device):
    anchor_list = [str(item) for item in anchor_values]
    positive_list = [str(item) for item in positive_values]
    anchor_size = len(anchor_list)
    positive_size = len(positive_list)
    diff_mask = torch.ones((anchor_size, positive_size), dtype=torch.bool, device=device)
    for anchor_index in range(anchor_size):
        anchor_value = anchor_list[anchor_index]
        for positive_index in range(positive_size):
            if anchor_value == positive_list[positive_index]:
                diff_mask[anchor_index, positive_index] = False
    return diff_mask


# Optional: discrete features join the hard constraints (tensor version, returns torch.bool([B, B]))
def build_discrete_mask(anchor: Dict[str, torch.Tensor], positive: Dict[str, torch.Tensor], config: Any):
    batch_size = int(anchor['dense'].shape[0])
    target_device = anchor['dense'].device
    optional_discrete_mask = torch.ones((batch_size, batch_size), device=target_device, dtype=torch.bool)
    require_mode_match = bool(config.require_mode_match_for_positive)
    require_key_close = bool(config.require_key_close_for_positive)
    if require_mode_match or require_key_close:
        anchor_mode_value = anchor['mode_idx'].long() - 1
        positive_mode_value = positive['mode_idx'].long() - 1
        anchor_key_index = anchor['key_idx'].long()
        positive_key_index = positive['key_idx'].long()
        anchor_pitch_class = (anchor_key_index - 1).clamp(min=0, max=11)
        positive_pitch_class = (positive_key_index - 1).clamp(min=0, max=11)
        anchor_related_key = (anchor_pitch_class + (3 - 6 * anchor_mode_value)) % 12
        is_relative_key_mask = (positive_mode_value[None, :] != anchor_mode_value[:, None]) & (positive_pitch_class[None, :] == anchor_related_key[:, None])
    if require_mode_match:
        anchor_mode_known_mask = anchor_mode_value >= 0
        positive_mode_known_mask = positive_mode_value >= 0
        mode_known_mask = anchor_mode_known_mask[:, None] & positive_mode_known_mask[None, :]
        mode_equal_mask = (anchor_mode_value[:, None] == positive_mode_value[None, :]) | is_relative_key_mask
        optional_discrete_mask = optional_discrete_mask & mode_known_mask & mode_equal_mask
    if require_key_close:
        anchor_key_known_mask = anchor_key_index > 0
        positive_key_known_mask = positive_key_index > 0
        key_known_mask = anchor_key_known_mask[:, None] & positive_key_known_mask[None, :]
        anchor_pitch_class = (anchor_key_index - 1).clamp(min=0, max=11)
        positive_pitch_class = (positive_key_index - 1).clamp(min=0, max=11)
        fifths_distance_matrix = circle_of_fifths_dist(anchor_pitch_class[:, None], positive_pitch_class[None, :])
        chromatic_distance_matrix = chromatic_circle_dist(anchor_pitch_class[:, None], positive_pitch_class[None, :])
        fifths_threshold = int(config.pos_key_fifths_dist_max)
        chromatic_threshold = int(config.pos_key_chromatic_dist_max)
        key_close_mask = (fifths_distance_matrix <= fifths_threshold) | (chromatic_distance_matrix <= chromatic_threshold) | is_relative_key_mask
        optional_discrete_mask = optional_discrete_mask & key_known_mask & key_close_mask
    require_tempo_close = bool(config.require_tempo_close_for_positive)
    if require_tempo_close:
        anchor_tempo_index = anchor['tempo_idx'].long()
        positive_tempo_index = positive['tempo_idx'].long()
        anchor_tempo_known_mask = anchor_tempo_index > 0
        positive_tempo_known_mask = positive_tempo_index > 0
        tempo_known_mask = anchor_tempo_known_mask[:, None] & positive_tempo_known_mask[None, :]
        tempo_diff_matrix = torch.abs(anchor_tempo_index[:, None] - positive_tempo_index[None, :])
        tempo_diff_max = int(config.pos_tempo_idx_diff_max)
        tempo_close_mask = tempo_diff_matrix <= tempo_diff_max
        optional_discrete_mask = optional_discrete_mask & tempo_known_mask & tempo_close_mask
    require_time_signature_match = bool(config.require_time_signature_match_for_positive)
    if require_time_signature_match:
        anchor_time_signature_index = anchor['ts_idx'].long()
        positive_time_signature_index = positive['ts_idx'].long()
        anchor_time_signature_known_mask = anchor_time_signature_index > 0
        positive_time_signature_known_mask = positive_time_signature_index > 0
        time_signature_known_mask = anchor_time_signature_known_mask[:, None] & positive_time_signature_known_mask[None, :]
        anchor_time_signature_raw = anchor_time_signature_index + 2
        positive_time_signature_raw = positive_time_signature_index + 2
        anchor_time_signature_normalized = normalize_time_signature(anchor_time_signature_raw)
        positive_time_signature_normalized = normalize_time_signature(positive_time_signature_raw)
        time_signature_equal_mask = anchor_time_signature_normalized[:, None] == positive_time_signature_normalized[None, :]
        optional_discrete_mask = optional_discrete_mask & time_signature_known_mask & time_signature_equal_mask
    anchor_require_known_genre = bool(config.anchor_require_known_genre)
    pos_require_known_genre = bool(config.pos_require_known_genre)
    require_genre_match = bool(config.require_genre_match_for_positive)
    anchor_genre_index = anchor['genre_idx'].long()
    positive_genre_index = positive['genre_idx'].long()
    if anchor_require_known_genre:
        anchor_genre_known_mask = anchor_genre_index > 0
        anchor_genre_known_mask = anchor_genre_known_mask[:, None]
        optional_discrete_mask = optional_discrete_mask & anchor_genre_known_mask
    if pos_require_known_genre:
        positive_genre_known_mask = positive_genre_index > 0
        positive_genre_known_mask = positive_genre_known_mask[:, None]
        optional_discrete_mask = optional_discrete_mask & positive_genre_known_mask
    if require_genre_match:
        genre_equal_mask = anchor_genre_index[:, None] == positive_genre_index[None, :]
        optional_discrete_mask = optional_discrete_mask & genre_equal_mask
    return optional_discrete_mask


# Define if an in-batch item can be one of positive candidates (tensor version, returns torch.bool([B, B]))
def build_positive_mask(anchor: Dict[str, torch.Tensor], positive: Dict[str, torch.Tensor], config: Any):
    anchor_dense_matrix = anchor['dense'].float()
    positive_dense_matrix = positive['dense'].float()
    batch_size = int(anchor_dense_matrix.shape[0])
    target_device = anchor_dense_matrix.device

    speechiness_max_exclusive_raw = float(config.speechiness_max_exclusive_raw)
    speechiness_max_exclusive_centered = speechiness_max_exclusive_raw - 0.5
    anchor_speechiness_matrix = anchor_dense_matrix[:, DENSE_SPEECHINESS_INDEX : DENSE_SPEECHINESS_INDEX + 1]
    positive_speechiness_matrix = positive_dense_matrix[None, :, DENSE_SPEECHINESS_INDEX]
    anchor_speechiness_ok_mask = anchor_speechiness_matrix < speechiness_max_exclusive_centered
    positive_speechiness_ok_mask = positive_speechiness_matrix < speechiness_max_exclusive_centered
    speechiness_ok_mask = anchor_speechiness_ok_mask & positive_speechiness_ok_mask

    vocal_threshold_centered = -0.5
    anchor_instrumentalness_matrix = anchor_dense_matrix[:, DENSE_INSTRUMENTALNESS_INDEX : DENSE_INSTRUMENTALNESS_INDEX + 1]
    positive_instrumentalness_matrix = positive_dense_matrix[None, :, DENSE_INSTRUMENTALNESS_INDEX]
    anchor_is_vocal_song_mask = anchor_instrumentalness_matrix <= vocal_threshold_centered
    positive_is_vocal_song_mask = positive_instrumentalness_matrix <= vocal_threshold_centered
    instrumentalness_ok_mask = torch.logical_or(
        torch.logical_not(anchor_is_vocal_song_mask),
        positive_is_vocal_song_mask
    )

    anchor_energy_matrix = anchor_dense_matrix[:, DENSE_ENERGY_INDEX : DENSE_ENERGY_INDEX + 1]
    positive_energy_matrix = positive_dense_matrix[None, :, DENSE_ENERGY_INDEX]
    energy_diff_matrix = torch.abs(anchor_energy_matrix - positive_energy_matrix)
    energy_ok_mask = energy_diff_matrix < float(config.pos_energy_diff_max)

    anchor_valence_matrix = anchor_dense_matrix[:, DENSE_VALENCE_INDEX : DENSE_VALENCE_INDEX + 1]
    positive_valence_matrix = positive_dense_matrix[None, :, DENSE_VALENCE_INDEX]
    valence_diff_matrix = torch.abs(anchor_valence_matrix - positive_valence_matrix)
    valence_ok_mask = valence_diff_matrix < float(config.pos_valence_diff_max)

    anchor_acousticness_matrix = anchor_dense_matrix[:, DENSE_ACOUSTICNESS_INDEX : DENSE_ACOUSTICNESS_INDEX + 1]
    positive_acousticness_matrix = positive_dense_matrix[None, :, DENSE_ACOUSTICNESS_INDEX]
    acousticness_diff_matrix = torch.abs(anchor_acousticness_matrix - positive_acousticness_matrix)
    acousticness_ok_mask = acousticness_diff_matrix < float(config.pos_acousticness_diff_max)

    anchor_liveness_matrix = anchor_dense_matrix[:, DENSE_LIVENESS_INDEX : DENSE_LIVENESS_INDEX + 1]
    positive_liveness_matrix = positive_dense_matrix[None, :, DENSE_LIVENESS_INDEX]
    liveness_diff_matrix = torch.abs(anchor_liveness_matrix - positive_liveness_matrix)
    liveness_ok_mask = liveness_diff_matrix < float(config.pos_liveness_diff_max)

    anchor_speechiness_matrix = anchor_dense_matrix[:, DENSE_SPEECHINESS_INDEX : DENSE_SPEECHINESS_INDEX + 1]
    positive_speechiness_matrix = positive_dense_matrix[None, :, DENSE_SPEECHINESS_INDEX]
    speechiness_diff_matrix = torch.abs(anchor_speechiness_matrix - positive_speechiness_matrix)
    speechiness_ok_mask = speechiness_ok_mask & (speechiness_diff_matrix < float(config.pos_speechiness_diff_max))

    danceability_diff_max = float(config.pos_danceability_diff_max)
    danceability_ok_mask = torch.ones((batch_size, batch_size), device=target_device, dtype=torch.bool)
    anchor_danceability_matrix = anchor_dense_matrix[:, DENSE_DANCEABILITY_INDEX : DENSE_DANCEABILITY_INDEX + 1]
    positive_danceability_matrix = positive_dense_matrix[None, :, DENSE_DANCEABILITY_INDEX]
    danceability_diff_matrix = torch.abs(anchor_danceability_matrix - positive_danceability_matrix)
    danceability_ok_mask = danceability_diff_matrix < danceability_diff_max

    weighted_distance_matrix = weighted_l2_distance_dense_matrix(anchor_dense_matrix, positive_dense_matrix, config)
    weighted_distance_max = float(config.pos_dense_weighted_l2_max)
    weighted_distance_ok_mask = weighted_distance_matrix <= weighted_distance_max

    dense_similarity_mask = (
        speechiness_ok_mask
        & instrumentalness_ok_mask
        & energy_ok_mask
        & valence_ok_mask
        & acousticness_ok_mask
        & liveness_ok_mask
        & danceability_ok_mask
        & weighted_distance_ok_mask
    )

    optional_discrete_mask = build_discrete_mask(anchor=anchor, positive=positive, config=config)

    positive_mask = dense_similarity_mask & optional_discrete_mask
    return positive_mask


# Multi-positive masked InfoNCE (Supervised Contrastive), in-batch negatives after masking
class MaskedInfoNCELoss(nn.Module):
    def __init__(self, config: MaskedInfoNCEConfig):
        super().__init__()
        self.config = config

    # batch: Dict[str, Tensor[B][anchor, positive]]
    def forward(self, anchor_emb: torch.Tensor, positive_emb: torch.Tensor, batch: dict):
        tau = float(self.config.tau)
        sim_scores = (anchor_emb @ positive_emb.t()) / max(1e-6, tau)
        batch_size = int(sim_scores.shape[0])
        device = sim_scores.device
        anchor_feature = {
            "key_idx": batch['key_idx'][0].to(device),
            "mode_idx": batch['mode_idx'][0].to(device),
            "ts_idx": batch['ts_idx'][0].to(device),
            "tempo_idx": batch['tempo_idx'][0].to(device),
            "genre_idx": batch['genre_idx'][0].to(device),
            "dense": batch['dense'][0].to(device)
        }
        positive_feature = {
            "key_idx": batch['key_idx'][1].to(device),
            "mode_idx": batch['mode_idx'][1].to(device),
            "ts_idx": batch['ts_idx'][1].to(device),
            "tempo_idx": batch['tempo_idx'][1].to(device),
            "genre_idx": batch['genre_idx'][1].to(device),
            "dense": batch['dense'][1].to(device)
        }

        multi_positive_mask = build_positive_mask(anchor_feature, positive_feature, config=self.config.sampler_config)
        diagonal_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        positive_mask = multi_positive_mask | diagonal_mask

        anchor_artist_index = batch['artist_idx'][0].to(device)
        positive_artist_index = batch['artist_idx'][1].to(device)
        artist_diff_mask = anchor_artist_index[:, None] != positive_artist_index[None, :]

        negative_mask = torch.logical_not(positive_mask)
        negative_mask = negative_mask & torch.logical_not(diagonal_mask)
        negative_mask = negative_mask & artist_diff_mask
        if bool(getattr(self.config.sampler_config, "neg_exclude_same_album", True)) and "album_id" in batch:
            anchor_album_ids, positive_album_ids = batch['album_id']
            album_diff_mask = build_string_diff_mask(anchor_album_ids, positive_album_ids, device=device)
            negative_mask = negative_mask & album_diff_mask
        if bool(getattr(self.config.sampler_config, "neg_exclude_same_isrc", True)) and "isrc" in batch:
            anchor_isrc_ids, positive_isrc_ids = batch['isrc']
            isrc_diff_mask = build_string_diff_mask(anchor_isrc_ids, positive_isrc_ids, device=device)
            negative_mask = negative_mask & isrc_diff_mask

        allowed_mask = positive_mask | negative_mask
        ignored_mask = torch.logical_not(allowed_mask)

        sim_scores = sim_scores.masked_fill(ignored_mask, float("-inf"))
        log_denom = torch.logsumexp(sim_scores, dim=1)
        log_sim_scores = sim_scores - log_denom[:, None]

        positive_count = positive_mask.sum(dim=1).clamp(min=1)
        pos_per_anchor = float(positive_count.float().mean().detach().cpu())

        loss_per_anchor = -(log_sim_scores.masked_fill(torch.logical_not(positive_mask), 0.0).sum(dim=1) / positive_count)
        loss = loss_per_anchor.mean()

        batch_allowed_frac = allowed_mask.float().mean(dim=1)
        batch_masked_frac = float((1.0 - batch_allowed_frac).mean().detach().cpu())

        anchor_genre_index = batch['genre_idx'][0].to(device)
        positive_genre_index = batch['genre_idx'][1].to(device)
        same_genre_mask = anchor_genre_index[:, None] == positive_genre_index[None, :]
        same_genre_negative_count = (negative_mask & same_genre_mask).sum().float()
        negative_count = negative_mask.sum().float()
        if float(negative_count.detach().cpu()) > 0.0:
            neg_same_genre_frac = float((same_genre_negative_count / negative_count).detach().cpu())
        else:
            neg_same_genre_frac = 0.0

        anchor_drop_frac = batch.get("anchor_drop_frac", 0.0)
        if isinstance(anchor_drop_frac, torch.Tensor):
            anchor_drop_frac = float(anchor_drop_frac.detach().cpu().item())

        stats = {
            "loss": float(loss.detach().cpu()),
            "anchor_drop_frac": float(anchor_drop_frac),
            "pos_per_anchor": float(pos_per_anchor),
            "neg_same_genre_frac": float(neg_same_genre_frac),
            "batch_masked_frac": float(batch_masked_frac)
        }
        return loss, stats
