# MRMP: Multi-Round Meta-Learning for Label Purification Without Trusted Labels

Official PyTorch implementation of the paper:

> **Multi-Round Meta-Learning for Label Purification Without Trusted Labels**  
> *(Under review at TPAMI, 2026)*

---

## Branch: `mini-Clothing1M`

This branch contains experiments on **miniClothing-1M**, a curated subset of the Clothing-1M dataset featuring real-world label noise.  
For general benchmark experiments (CIFAR) and remote sensing datasets (UC Merced, AID), see the [`main`](../../tree/main) branch.

---

## Overview

MRMP is a PyTorch implementation of a clean-label-free meta-learning framework for label purification under noisy supervision. The framework iteratively purifies label quality over multiple rounds by alternating between **base-model training** and **meta-purification** — without requiring any clean reference labels.

- 🔖 **Problem** — Label noise is pervasive in large-scale datasets, and especially severe in expert-annotated domains (remote sensing, medical imaging) where inconsistencies persist even among trained annotators.
- ❌ **Limitation of prior work** — Existing methods either assume fully noisy data (co-training) or require a small clean subset (meta-learning) — neither paradigm supports clean-label-free purification.
- ✅ **Our approach** — A multi-round meta-purification strategy that decouples base-model training from label refinement, preserving training stability without clean labels.
- 📐 **Theory** — We analyze the purification process under a stochastic contraction perspective and introduce a KL-divergence–based stopping criterion to adaptively select purification depth.
- 📊 **Results** — Consistently outperforms state-of-the-art on CIFAR and miniClothing-1M; achieves up to **62.61% label accuracy improvement** on expert-annotated RS datasets (UC Merced and AID).

---

## Key Features

- **Clean-label-free** — no clean reference subset required at any stage.
- **Decoupled purification** — base-model training and meta-purification alternate independently, preserving stability.
- **NoiseDetector** — attention-fusion network refining soft labels using EMA teacher pseudo-labels.
- **Finite-difference meta-gradient** — trains the NoiseDetector without second-order differentiation.
- **KL-divergence stopping** — exponential curve fitting on KL history to adaptively select optimal purification depth.
- **Accumulated label distributions** across rounds for reliable mislabeled instance detection and correction.

---

## Method Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                          Round  r                                │
│                                                                  │
│  base_split   ──►  Pre-train  (CNN + EMA teacher)                │
│                          │                                       │
│  meta_split   ──►  Meta-train  NoiseDetector                     │
│  (KFold)                 │   finite-difference grad (v, ε)       │
│                          ▼                                       │
│               Purify Dataset                                     │
│          (replace noisy labels with NoiseDetector output)        │
│                          │                                       │
│               KL divergence ──► stochastic contraction check     │
│               exponential curve fit ──► predict optimal round    │
│               ──►  save  .pt  +  .json                           │
└──────────────────────────────────────────────────────────────────┘
```

| Component | Description |
|---|---|
| `CNN` (ConvNeXt) | Student base model; cross-entropy with prior entropy regularisation |
| `EMA teacher` | Exponential moving average of the student; stable pseudo-labels |
| `NoiseDetector` | Attention-fusion network refining noisy labels using EMA pseudo-labels |
| `meta_update` | Finite-difference meta-gradient: direction `v` and step size `ε` |
| `KFold splits` | Each round uses a different fold as the held-out meta validation set |
| `KL stopping` | Exponential curve fit on KL history to predict optimal purification depth |

---

## Requirements

- Python ≥ 3.8, PyTorch ≥ 2.0, CUDA GPU (recommended)
- `torch` `torchvision` `timm` `tqdm` `scikit-learn` `ema_pytorch` `scipy`

---

## Installation

```bash
git clone https://github.com/mitsuyoshiohtsu/MRMP.git
cd MRMP
pip install torch torchvision timm tqdm scikit-learn ema_pytorch scipy
```

---

## Repository Structure

```
MRMP/
├── MRMP.py                  # Main pipeline: pre-train → meta-train → purify → repeat
├── MRMP_clothing1M.py       # miniClothing-1M variant of the main pipeline
├── meta_modules.py          # Meta-update: computes v and ε via finite-difference
├── model.py                 # NoiseDetector (AttentionFusion network)
├── data/
│   ├── miniclothing1m.py    # MiniClothing1M dataset class
│   ├── make_combined.py     # Generate paired clean/noisy label files
│   ├── data_selection.py    # Copy matching images from clothing1M source
│   └── utils.py             # Noise generation utilities
├── data_preparation.sh      # End-to-end data preparation for miniClothing-1M
├── setup.sh                 # Environment setup
├── main.sh                  # Training launcher
└── results/                 # Logs and metrics (auto-created)
```

---

## Datasets

### miniClothing-1M

miniClothing-1M is a curated subset of the full [Clothing-1M](https://github.com/Cysu/noisy_label) dataset,
containing only samples with both clean and noisy label annotations.
It is used for real-world label noise experiments.

#### 1. Obtain Clothing-1M

Contact `tong.xiao.work[at]gmail[dot]com` to obtain the download link.
Place the dataset at `../clothing1M/` relative to the repository root.
The expected source structure is:

```
../clothing1M/
├── clean_label_kv.txt
├── noisy_label_kv.txt
├── clean_val_key_list.txt
├── clean_test_key_list.txt
└── images/
    ├── 0/
    │   ├── 00/
    │   └── ...
    ├── ...
    └── 9/
```

#### 2. Run data preparation

```bash
bash data_preparation.sh
```

This runs two steps in sequence:

| Step | Script | Description |
|---|---|---|
| 1 | `data/make_combined.py` | Generates paired `clean_noisy_labels_{split}.txt` files under `../mini_clothing1M/` |
| 2 | `data/data_selection.py` | Copies matching images from `../clothing1M/` into `../mini_clothing1M/` |

The resulting miniClothing-1M directory structure:

```
../mini_clothing1M/
├── clean_noisy_labels_train.txt
├── clean_noisy_labels_val.txt
├── clean_noisy_labels_test.txt
├── clean_noisy_labels_total.txt
└── images/
    ├── 0/
    │   ├── 00/
    │   │   └── *.jpg
    │   └── ...
    ├── ...
    └── 9/
```

---

## Training

```bash
# miniClothing-1M (real-world noise)
python MRMP_clothing1M.py
```

---

## Arguments

### miniClothing-1M (MRMP_clothing1M.py)

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `miniclothing1M` | `miniclothing1M` only |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `32` | Batch size |
| `--n_epoch` | `40` | Max pre-training epochs per round |
| `--n_round` | `10` | Max purification rounds |
| `--start_round` | `1` | Starting round (for resuming) |
| `--meta_iter` | `1000` | Meta-training iterations per round |
| `--beta` | `0.9` | EMA decay factor |
| `--T` | `0.7` | NoiseDetector attention temperature |
| `--ES_step` | `5` | Early stopping patience (epochs) |
| `--load` | `None` | Pre-saved purified dataset path |
| `--gpuid` | `0` | GPU device ID |
| `--seed` | `123` | Random seed |

---

## Output

```
data/purified/
  └── {model_str}_at_round{r}.pt      # Purified dataset checkpoint

results/{dataset}/MRML/
  └── {model_str}.txt                 # Per-epoch accuracy / loss log
  └── {model_str}.json                # Round-level metrics
```

| Metric | Description |
|---|---|
| `data_acc` | Label accuracy after purification |
| `n2c_flip` | Noisy → correct flips (beneficial) |
| `n2n_flip` | Noisy → noisy flips (harmful) |
| `c2n_flip` | Correct → noisy flips (harmful) |
| `mean_kl` | Mean KL divergence between label distributions; convergence signal |
| `pred_opt_r` | Predicted optimal purification round |

---

## Citation

*(To be added after publication.)*

---

## License

MIT License.
