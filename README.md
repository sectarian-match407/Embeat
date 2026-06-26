<p align="center">
  <img src="assets/banner.png" alt="Embeat Banner" width="100%">
</p>

<p align="center">
  <b>Embeat - Acoustic Feature-Based Music Recommendation System</b>
</p>

<p align="center">
  <b>English</b> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <a href="https://github.com/gdstudio-org/Embeat"><img src="https://img.shields.io/github/stars/gdstudio-org/Embeat?style=social" alt="Stars"></a>
  <a href="https://github.com/gdstudio-org/Embeat/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-CC--BY--NC%204.0-blue" alt="License"></a>
</p>

---

## Introduction

Embeat is a music recommendation system built on Spotify acoustic feature data. It encodes audio features into vectors via a **contrastive learning model** and combines them with a **multi-channel recall** strategy to deliver high-quality music recommendations.

Key Features:

- **Acoustic Similarity**: The EmbeatMLP model, trained on Spotify Audio Features (key, tempo, energy, valence, etc.), encodes acoustic features into 64-dimensional vectors
- **Genre Awareness**: Leverages 6,000+ micro-genre tags to precisely assign genres to 2M+ artists, preventing "acoustically similar but stylistically different" recommendations
- **Multi-Channel Recall**: 5 parallel recall channels (Acoustic Similarity / Same-Genre Popular / Same Artist / Similar Artists / Playlist Collaborative Filtering), merged and scored for final output
- **Playlist Collaborative Filtering**: Track2Vec (Word2Vec-inspired) learns track co-occurrence patterns from 1.88M Spotify playlists
- **Millisecond-Level Response**: Powered by the Qdrant vector database, retrieval across 45M tracks completes in 30–100ms

## Roadmap

- [x] 2026-06-26: Open-source initial codebase + [EmbeatMLP model weights](checkpoints/EmbeatMLP/)
- [x] 2026-06-26: Open-source [45M tracks dataset](https://huggingface.co/datasets/GD-Studio/embeat_45m_spotify_tracks) + [technical documentation](https://www.bilibili.com/opus/1218087093501165591)
- [ ] 100 Stars: Open-source Qdrant database
- [ ] 1K Stars: Open-source Track2Vec model weights + 6.6M playlist dataset

## Demo

Below are example recommendation results from Embeat (seed track -> Top 5 recommendations):

<details>
<summary><b>晴天 - Jay Chou [mandopop, taiwan pop, c-pop]</b></summary>

| # | Track | Artist | Source |
|---|-------|--------|--------|
| 1 | 告白氣球 | Jay Chou | Same-Genre Popular, Same Artist, Playlist CF |
| 2 | 突然好想你 | Mayday | Same-Genre Popular, Similar Artists |
| 3 | 怎麼了 | Eric Chou | Same-Genre Popular, Playlist CF |
| 4 | 飞鸟和蝉 | Ren Ran | Playlist CF |
| 5 | 你的背包 | Eason Chan | Similar Artists |

</details>

<details>
<summary><b>Uptown Funk (feat. Bruno Mars) - Bruno Mars [dance pop, pop]</b></summary>

| # | Track | Artist | Source |
|---|-------|--------|--------|
| 1 | That's What I Like | Bruno Mars | Same-Genre Popular, Same Artist, Playlist CF |
| 2 | Timber | Pitbull | Same-Genre Popular, Playlist CF |
| 3 | CAN'T STOP THE FEELING! | Justin Timberlake | Playlist CF |
| 4 | Happy | Pharrell Williams | Playlist CF |
| 5 | Sugar | Maroon 5 | Playlist CF |

</details>

<details>
<summary><b>Sis puella magica! - 梶浦由記 [anime score, japanese vgm]</b></summary>

| # | Track | Artist | Source |
|---|-------|--------|--------|
| 1 | Conturbatio | 梶浦 由記 | Same Artist, Playlist CF |
| 2 | 輝く空の静寂には | Kalafina | Similar Artists |
| 3 | Forbidden Love | Cécile Corbel | Acoustic Similarity |
| 4 | ARIA | Kalafina | Similar Artists |
| 5 | Zoltraak | Evan Call | Same-Genre Popular |

</details>

### LLM Blind Evaluation

<b>For detailed comparison, please refer to the [technical documentation](https://www.bilibili.com/opus/1218087093501165591)</b>

Using the LLM-as-a-Judge method (GPT-5.5 / Gemini Flash 3.5 / Claude Sonnet 4.6), Embeat was blindly evaluated against NetEase Cloud Music in AB tests:

| Judge Model | Embeat Wins | NetEase Wins | Tie |
|-------------|:-----------:|:------------:|:---:|
| Claude Sonnet 4.6 | **8** | 2 | 0 |
| Gemini Flash 3.5 | **9** | 1 | 0 |
| GPT 5.5 | **6** | 4 | 0 |

## System Architecture

```
Input: track_id / track_name + artist_name
  │
  ├─ Channel 1: Acoustic Similarity Recall (EmbeatMLP vectors + genre filtering)
  ├─ Channel 2: Same-Genre Popular Recall (genre tags + popularity ranking)
  ├─ Channel 3: Same Artist Recall (artist_idx + popularity ranking)
  ├─ Channel 4: Similar Artists Recall (Spotify Related Artists + vector ranking)
  ├─ Channel 5: Playlist Collaborative Filtering (Track2Vec)
  │
  ├─ ISRC Deduplication / Re-ranking / Same-Artist Ratio Control
  │
  └─ Output: Top-K Recommendation List
```

### Model Details

**EmbeatMLP** - Acoustic Feature Encoding Model

- Input: Discrete features (key, mode, tempo, time_signature) + continuous features (energy, valence, danceability, etc., 7 dimensions)
- Architecture: Dual-tower MLP (Discrete Tower + Acoustic Tower -> Backbone)
- Output: 64-dimensional L2-normalized vectors
- Training: Masked InfoNCE Loss, batch_size=4096, converges in ~70 steps
- Extremely small parameter count, supports real-time CPU-only inference

**Track2Vec** - Playlist Collaborative Filtering Model

- Based on Word2Vec Skip-Gram, treating playlists as "sentences" and tracks as "words"
- Training data: 1.88M Spotify playlists
- Vocabulary: 1.09M tracks, 64-dimensional vectors
- Single query latency < 0.1ms

## Getting Started

### Requirements

- Python >= 3.10
- PyTorch >= 2.6, < 2.7 (required for training)
- CUDA >= 12.0 (required for training)
- [Qdrant](https://github.com/qdrant/qdrant/releases) (required for inference)

### Installation

```bash
conda create -n embeat python=3.10
conda activate embeat

# Install PyTorch (CUDA 12.x), see https://pytorch.org/get-started/locally/
pip install "torch>=2.6,<2.7" --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

### Train EmbeatMLP

```bash
# Prepare training data in HuggingFace Dataset format under data/datasets/
python -m train.train \
    --dataset data/datasets/spotify_45m_tracks_metadata@10000000 \
    --batch-size 4096 \
    --max-steps 200 \
    --lr 1e-4 \
    --tau 0.05 \
    --ckpt-dir checkpoints
```

### Train Track2Vec

```bash
# Prepare playlist training data (txt format, one playlist per line, space-separated track_ids)
cd train
python train_track2vec.py
```

### Inference: Compute Acoustic Similarity Between Two Tracks

```python
from infer.infer import infer

song_a = {"key": 7, "mode": 1, "tempo": 137, "time_signature": 4,
          "danceability": 0.54, "energy": 0.56, "speechiness": 0.02,
          "instrumentalness": 0.0, "valence": 0.41, "acousticness": 0.23,
          "liveness": 0.1}

song_b = {"key": 5, "mode": 0, "tempo": 87, "time_signature": 4,
          "danceability": 0.67, "energy": 0.65, "speechiness": 0.05,
          "instrumentalness": 0.03, "valence": 0.57, "acousticness": 0.27,
          "liveness": 0.19}

similarity = infer(sample_a=song_a, sample_b=song_b,
                   checkpoint_path="checkpoints/EmbeatMLP/model.pt")
print(f"Similarity: {similarity:.4f}")
```

### Inference: Qdrant-Based Music Recommendation

```bash
# 1. Start the Qdrant service and import the database
# 2. Query recommendations via command line
cd infer
python Embeat.py -t 5pIcwtJYNJx93l420oR2Vm   # Query by Spotify Track ID
python Embeat.py -s "晴天 - Jay Chou"   # Query by track name and artist
python Embeat.py -a "Jay Chou"   # Query by artist name
```

## Project Structure

```
Embeat/
├── assets/                 # Static assets
├── checkpoints/
│   ├── EmbeatMLP/          # EmbeatMLP model weights
│   └── Track2Vec/          # Track2Vec model weights (requires separate download)
├── data/                   # Data processing scripts (not fully organized)
├── infer/                  # Inference code
│   ├── Embeat.py           # Recommendation system core (multi-channel recall + re-ranking)
│   ├── infer.py            # EmbeatMLP inference entry point
│   ├── hf_to_qdrant.py     # HuggingFace Dataset -> Qdrant database
│   └── eval_infer.py       # Model evaluation utilities
├── train/                  # Training code
│   ├── model.py            # EmbeatMLP model definition
│   ├── dataset.py          # Dataset processing
│   ├── sampler.py          # Positive/negative sample sampler
│   ├── loss.py             # Masked InfoNCE Loss
│   ├── trainer.py          # Trainer
│   ├── train.py            # EmbeatMLP training entry point
│   └── train_track2vec.py  # Track2Vec training entry point
├── requirements.txt
└── LICENSE
```

## Links

<p align="center">
  <img src="assets/gdmusic_embeat.png" alt="Embeat Banner" width="100%">
</p>

- GD Music (Live Demo): [https://music.gdstudio.xyz](https://music.gdstudio.xyz)
- Bilibili: [https://space.bilibili.com/13715770](https://space.bilibili.com/13715770)
- Telegram: [https://t.me/gdstudio_music](https://t.me/gdstudio_music)

## Acknowledgements

- [Anna's Archive](https://annas-archive.org)
- [Every Noise at Once](https://everynoise.com)

## License

| Scope | License |
|-------|---------|
| Code, Model Weights | [MIT](LICENSE-MIT) |
| Datasets, Database | [CC-BY-NC 4.0](LICENSE) |
