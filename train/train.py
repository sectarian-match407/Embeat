# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-23

import argparse
import os
import sys
import torch
from dataclasses import asdict
from typing import Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from train.dataloader import DataLoaderConfig, build_pair_dataloader
from train.dataset import DatasetConfig
from train.loss import MaskedInfoNCEConfig, MaskedInfoNCELoss
from train.model import EmbeatMLP, EmbeatMLPConfig
from train.sampler import PairSamplerConfig
from train.trainer import TrainerConfig, EmbeatTrainer, seed_everything


# Get CUDA device ID (-1=CPU)
def get_device(cuda_device: Optional[int] = None):
    if cuda_device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        if cuda_device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{cuda_device}")
    return device


# Entry function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=f"{project_root}/data/datasets/spotify_45m_tracks_metadata@10000000", help="HuggingFace dataset path. Supports 'path@N' for quick subset select.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of DataLoader workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--drop-last", action="store_true", help="Drop last incomplete batch.")
    parser.add_argument("--seed", type=int, default=616, help="Random seed.")
    parser.add_argument("--cuda-device", default=0, help="-1=CPU, 0=cuda:0, 1=cuda:1...")
    parser.add_argument("--max-steps", type=int, default=200, help="Max training steps (batch level).")
    parser.add_argument("--log-every", type=int, default=10, help="Print logs every N steps.")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N steps.")
    parser.add_argument("--ckpt-dir", default="checkpoints", help="Checkpoint output directory.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay for AdamW.")
    parser.add_argument("--lr-scheduler", type=str, default="constant", choices=["constant", "cosine"], help="Learning rate scheduler mode. Choose between: cosine, constant.")
    parser.add_argument("--lr-min", type=float, default=1e-9, help="Minimum learning rate for cosine annealing.")
    parser.add_argument("--cosine-t-max", type=int, default=0, help="CosineAnnealingLR T_max in steps. 0 means use max_steps.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Max grad norm for clipping (0.0 to disable).")
    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed precision training.")
    parser.add_argument("--cache-per-genre", type=int, default=4096, help="Sampler cache size per genre. Decrease this value to reduce memory usage.")
    parser.add_argument("--cache-per-album", type=int, default=128, help="Sampler cache size per album. Decrease this value to reduce memory usage.")
    parser.add_argument("--max-album-cache-keys", type=int, default=10000, help="Max number of album ids kept in sampler cache. Decrease this value to reduce memory usage.")
    parser.add_argument("--max-positive-tries", type=int, default=128, help="Try time in one of positive buckets. Higher value consumes more time but higher sample efficiency.")
    parser.add_argument("--tau", type=float, default=0.05, help="InfoNCE temperature (tau). Lower value makes bigger similarity gap.")
    args = parser.parse_args()
    seed_everything(int(args.seed))
    device = get_device(int(args.cuda_device))
    sampler_config = PairSamplerConfig(
        random_seed=int(args.seed),
        cache_per_genre=int(args.cache_per_genre),
        cache_per_album=int(args.cache_per_album),
        max_album_cache_keys=int(args.max_album_cache_keys),
        max_positive_tries=int(args.max_positive_tries)
    )
    loader = build_pair_dataloader(
        dataset_config=DatasetConfig(dataset_path=str(args.dataset)),
        sampler_config=sampler_config,
        dataloader_config=DataLoaderConfig(
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory),
            drop_last=bool(args.drop_last)
        )
    )
    model_config = EmbeatMLPConfig()
    loss_config = MaskedInfoNCEConfig(
        tau=float(args.tau),
        sampler_config=sampler_config,
    )
    trainer_config = TrainerConfig(
        max_steps=int(args.max_steps),
        log_every=int(args.log_every),
        save_every=int(args.save_every),
        ckpt_dir=str(args.ckpt_dir),
        grad_clip_norm=float(args.grad_clip_norm),
        use_amp=bool(args.amp)
    )
    model = EmbeatMLP(model_config).to(device)
    model.train()
    loss_fn = MaskedInfoNCELoss(loss_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    lr_scheduler = None
    lr_scheduler_mode = str(args.lr_scheduler).strip().lower()
    resolved_cosine_t_max = 0
    if lr_scheduler_mode == "cosine":
        resolved_cosine_t_max = int(args.cosine_t_max)
        if resolved_cosine_t_max <= 0:
            resolved_cosine_t_max = int(args.max_steps)
        resolved_cosine_t_max = max(1, int(resolved_cosine_t_max))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(resolved_cosine_t_max), eta_min=float(args.lr_min))
    if lr_scheduler_mode == "constant":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda current_step: 1.0)
    run_config = {
        "args": vars(args),
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "trainer_config": asdict(trainer_config),
        "lr_scheduler_config": {
            "mode": lr_scheduler_mode,
            "lr_min": float(args.lr_min),
            "cosine_t_max": int(resolved_cosine_t_max)
        }
    }
    trainer = EmbeatTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler, loader=loader, device=device, config=trainer_config, run_config=run_config)
    trainer.train()


if __name__ == "__main__":
    main()
