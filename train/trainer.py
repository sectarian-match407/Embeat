# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-02-13

import numpy as np
import os
import random
import sys
import time
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from train.loss import MaskedInfoNCELoss
from train.model import EmbeatMLP


# Trainer configurations
@dataclass
class TrainerConfig:
    # Max training step (batch unit)
    max_steps: int = 200
    # Log every N steps
    log_every: int = 10
    # Save checkpoint every N steps
    save_every: int = 100
    # Save directory
    ckpt_dir: str = "checkpoints"
    # Gradient clipping
    grad_clip_norm: float = 1.0
    # Use automatic mixed precision (faster and save more GRAM)
    use_amp: bool = True


# Set random seed everywhere
def seed_everything(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Get anchor_tensor, positive_tensor or negative_tensor from one batch
def build_features(batch: dict, device: torch.device, pair_key: str):
    pair_map = {"anchor": 0, "positive": 1}
    if pair_key not in pair_map:
        raise ValueError("pair_key must be one of: anchor / positive")
    idx = int(pair_map[pair_key])
    key_idx = batch["key_idx"][idx].to(device)
    mode_idx = batch["mode_idx"][idx].to(device)
    ts_idx = batch["ts_idx"][idx].to(device)
    tempo_idx = batch["tempo_idx"][idx].to(device)
    features = {
        "key_idx": key_idx,
        "mode_idx": mode_idx,
        "ts_idx": ts_idx,
        "tempo_idx": tempo_idx,
    }
    dense = batch["dense"][idx].to(device)
    features["dense"] = dense
    return features


# Save *.pt checkpoint
def save_checkpoint(path: str, model: EmbeatMLP, optimizer: torch.optim.Optimizer, lr_scheduler: object, step: int, config: dict):
    path = os.path.abspath(path.replace("\\", "/").strip())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "config": dict(config)
    }
    torch.save(payload, path)
    print("Saved checkpoint:", path)


# Main trainer
class EmbeatTrainer:
    def __init__(self, model: EmbeatMLP, loss_fn: MaskedInfoNCELoss, optimizer: torch.optim.Optimizer, lr_scheduler: object, loader: DataLoader, device: torch.device, config: TrainerConfig, run_config: Optional[dict] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loader = loader
        self.device = device
        self.config = config
        self.run_config = dict(run_config or {})
        self.use_amp = bool(config.use_amp) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.step = 0
        self.t0 = time.time()

    # Train function, call every step
    def train_step(self, batch: dict):
        anchor_features = build_features(batch, self.device, "anchor")
        positive_features = build_features(batch, self.device, "positive")
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            anchor_emb = self.model(anchor_features)
            positive_emb = self.model(positive_features)
            loss, stats = self.loss_fn(anchor_emb, positive_emb, batch=batch)
        grad_norm_value = 0.0
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if float(self.config.grad_clip_norm) > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip_norm))
                grad_norm_value = float(grad_norm.detach().cpu())
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if float(self.config.grad_clip_norm) > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip_norm))
                grad_norm_value = float(grad_norm.detach().cpu())
            self.optimizer.step()
        stats['grad_norm'] = float(grad_norm_value)
        self.lr_scheduler.step()
        current_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        stats["lr"] = current_learning_rate
        return loss, stats

    # Print log to screen, call every step, but log by conditions
    def log_step(self, stats: dict):
        if int(self.config.log_every) <= 0:
            return
        if self.step % int(self.config.log_every) != 0:
            return
        now = time.time()
        elapsed = now - self.t0
        step_per_sec = self.step / max(1e-9, elapsed)
        print(
            "step:", self.step,
            "loss:", round(stats.get("loss", 0.0), 6),
            "lr:", round(stats.get("lr", 0.0), 8),
            "grad_norm:", round(stats.get("grad_norm", 0.0), 4),
            "anchor_drop_frac:", round(stats.get("anchor_drop_frac", 0.0), 3),
            "pos_per_anchor:", round(stats.get("pos_per_anchor", 0.0), 3),
            "neg_same_genre_frac:", round(stats.get("neg_same_genre_frac", 0.0), 3),
            "batch_masked_frac:", round(stats.get("batch_masked_frac", 0.0), 3),
            "step_per_sec:", round(step_per_sec, 2)
        )

    # Save checkpoint, call every step, but save by conditions
    def save_step(self):
        if int(self.config.save_every) <= 0:
            return
        if self.step % int(self.config.save_every) != 0:
            return
        output_path = os.path.join(str(self.config.ckpt_dir), f"step_{self.step}.pt")
        save_checkpoint(output_path, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler, step=self.step, config=dict(self.run_config))

    # Start training
    def train(self):
        print("Starting training...")
        print("device:", self.device)
        max_steps = int(self.config.max_steps)
        try:
            for batch in self.loader:
                if self.step >= max_steps:
                    break
                loss, stats = self.train_step(batch)
                self.step = self.step + 1
                self.log_step(stats=stats)
                self.save_step()
        except KeyboardInterrupt:
            print("Interrupted by user. Saving last checkpoint...")
            output_path = os.path.join(str(self.config.ckpt_dir), f"step_{self.step}_interrupt.pt")
            save_checkpoint(output_path, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler, step=self.step, config=dict(self.run_config))
        dt = time.time() - self.t0
        print("Training done")
        print(f"Used time: {int(dt)}s")
