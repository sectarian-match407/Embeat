# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-21

import torch
import torch.nn as nn
import torch.nn.functional as functional
from dataclasses import dataclass


# Model configurations
@dataclass
class EmbeatMLPConfig:
    # Data dimensions
    key_vocab_size: int = 13
    mode_vocab_size: int = 3
    time_signature_vocab_size: int = 6
    tempo_vocab_size: int = 5
    key_emb_dim: int = 8
    mode_emb_dim: int = 4
    time_signature_emb_dim: int = 4
    tempo_emb_dim: int = 4
    # Layer dimensions
    acoustic_input_dim: int = 7
    acoustic_output_dim: int = 64
    discrete_output_dim: int = 64
    # Backbone dimensions
    backbone_hidden_dims = (256, 256)
    embedding_dim: int = 64
    # Other settings
    backbone_dropout_p: float = 0.0
    acoustic_dropout_p: float = 0.0


# Return index (in range) or 0
def safe_index(indices: torch.Tensor, vocab_size: int):
    if indices.dtype != torch.long:
        indices = indices.long()
    if vocab_size <= 0:
        index_tensor = torch.zeros_like(indices, dtype=torch.long)
        return index_tensor
    in_range = (indices >= 0) & (indices < vocab_size)
    index_tensor = torch.where(in_range, indices, torch.zeros_like(indices, dtype=torch.long))
    return index_tensor


# Main model
class EmbeatMLP(nn.Module):
    def __init__(self, config: EmbeatMLPConfig):
        super().__init__()
        self.config = config
        # Data embeddings: 13 + 3 + 6 + 5 = 27 -> 8 + 4 + 4 + 4 = 20
        self.key_embedding = nn.Embedding(int(config.key_vocab_size), int(config.key_emb_dim))
        self.mode_embedding = nn.Embedding(int(config.mode_vocab_size), int(config.mode_emb_dim))
        self.time_signature_embedding = nn.Embedding(int(config.time_signature_vocab_size), int(config.time_signature_emb_dim))
        self.tempo_embedding = nn.Embedding(int(config.tempo_vocab_size), int(config.tempo_emb_dim))
        # Discrete tower: 20 -> 64 -> 64
        discrete_input_dim = int(self.config.key_emb_dim + self.config.mode_emb_dim + self.config.time_signature_emb_dim + self.config.tempo_emb_dim)
        self.discrete_tower = nn.Sequential(
            nn.Linear(int(discrete_input_dim), int(config.discrete_output_dim)),
            nn.BatchNorm1d(int(config.discrete_output_dim)),
            nn.PReLU(),
            nn.Linear(int(config.discrete_output_dim), int(config.discrete_output_dim)),
            nn.BatchNorm1d(int(config.discrete_output_dim)),
            nn.PReLU()
        )
        # Acoustic tower: 7 -> 64 -> 64
        self.acoustic_tower = nn.Sequential(
            nn.Linear(int(config.acoustic_input_dim), int(config.acoustic_output_dim)),
            nn.BatchNorm1d(int(config.acoustic_output_dim)),
            nn.PReLU(),
            nn.Linear(int(config.acoustic_output_dim), int(config.acoustic_output_dim)),
            nn.BatchNorm1d(int(config.acoustic_output_dim)),
            nn.PReLU()
        )
        # fusion_dim: 64 + 64 = 128
        # Backbone: 128 -> 256 -> 256 -> 64
        fusion_dim = int(self.config.acoustic_output_dim + self.config.discrete_output_dim)
        hidden_0, hidden_1 = config.backbone_hidden_dims
        self.backbone = nn.Sequential(
            nn.Linear(int(fusion_dim), int(hidden_0)),
            nn.BatchNorm1d(int(hidden_0)),
            nn.PReLU(),
            nn.Dropout(p=float(config.backbone_dropout_p)),
            nn.Linear(int(hidden_0), int(hidden_1)),
            nn.BatchNorm1d(int(hidden_1)),
            nn.PReLU(),
            nn.Dropout(p=float(config.backbone_dropout_p)),
            nn.Linear(int(hidden_1), int(config.embedding_dim))
        )

    # Apply full-zero tensor to some items in acoustic tower batch randomly
    def apply_acoustic_dropout(self, acoustic_vec: torch.Tensor):
        dropout_p = float(self.config.acoustic_dropout_p)
        if not self.training or dropout_p <= 0.0:
            return acoustic_vec
        batch_size = int(acoustic_vec.shape[0])
        keep_mask = (torch.rand(batch_size, device=acoustic_vec.device) >= dropout_p).to(acoustic_vec.dtype)
        keep_mask = keep_mask.unsqueeze(1)
        acoustic_vec_dropout = acoustic_vec * keep_mask
        return acoustic_vec_dropout

    # Embed discrete vectors and concat
    def build_discrete_vec(self, key_idx: torch.Tensor, mode_idx: torch.Tensor, time_signature_idx: torch.Tensor, tempo_idx: torch.Tensor):
        key_idx = safe_index(key_idx, self.config.key_vocab_size)
        mode_idx = safe_index(mode_idx, self.config.mode_vocab_size)
        time_signature_idx = safe_index(time_signature_idx, self.config.time_signature_vocab_size)
        tempo_idx = safe_index(tempo_idx, self.config.tempo_vocab_size)
        key_vec = self.key_embedding(key_idx)
        mode_vec = self.mode_embedding(mode_idx)
        time_signature_vec = self.time_signature_embedding(time_signature_idx)
        tempo_vec = self.tempo_embedding(tempo_idx)
        discrete_concat = torch.cat([key_vec, mode_vec, time_signature_vec, tempo_vec], dim=1)
        discrete_vec = self.discrete_tower(discrete_concat)
        return discrete_vec

    # Get final embedding in full mode: discrete + dense input
    def encode(self, key_idx: torch.Tensor, mode_idx: torch.Tensor, time_signature_idx: torch.Tensor, tempo_idx: torch.Tensor, dense: torch.Tensor):
        discrete_vec = self.build_discrete_vec(key_idx=key_idx, mode_idx=mode_idx, time_signature_idx=time_signature_idx, tempo_idx=tempo_idx)
        if dense.dtype != torch.float32:
            dense = dense.float()
        acoustic_vec = self.acoustic_tower(dense)
        acoustic_vec = self.apply_acoustic_dropout(acoustic_vec)
        fused_vec = torch.cat([acoustic_vec, discrete_vec], dim=1)
        embedding = self.backbone(fused_vec)
        embedding = functional.normalize(embedding, p=2.0, dim=1, eps=1e-12)
        return embedding

    # Forward function
    def forward(self, features: dict):
        key_idx = features['key_idx']
        mode_idx = features['mode_idx']
        time_signature_idx = features['ts_idx']
        tempo_idx = features['tempo_idx']
        dense_feature = features.get("dense", None)
        if dense_feature is None:
            if key_idx.dim() == 0:
                batch_size = 1
            else:
                batch_size = int(key_idx.shape[0])
            dense_feature = torch.zeros((batch_size, int(self.config.acoustic_input_dim)), dtype=torch.float32, device=key_idx.device)
        embedding = self.encode(key_idx=key_idx, mode_idx=mode_idx, time_signature_idx=time_signature_idx, tempo_idx=tempo_idx, dense=dense_feature)
        return embedding
