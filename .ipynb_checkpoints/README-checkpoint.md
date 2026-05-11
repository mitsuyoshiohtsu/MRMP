# MRMP: Multi-Round Meta-Learning for Label Purification Without Trusted Labels

Official PyTorch implementation of the paper:

> **Multi-Round Meta-Learning for Label Purification Without Trusted Labels**  
> *(Under review at TPAMI, 2026)*

---

## Branch: RS Datasets

This branch contains experiments on **remote sensing (RS) image classification** datasets — specifically **UC Merced** and **AID** — which feature expert-annotated labels prone to inter-annotator inconsistency. These datasets serve as realistic testbeds for clean-label-free label purification under human-level noise.

For the general benchmark experiments (CIFAR, miniClothing-1M), see the `main` branch.

---

## Overview

MRMP is a clean-label-free meta-learning framework for label purification under noisy supervision. It iteratively improves label quality over multiple rounds by alternating between **base-model training** and **meta-purification** — without requiring any trusted reference labels.

| | |
|---|---|
| 🔖 **Problem** | Label noise is pervasive in large-scale datasets, and especially severe in expert-annotated domains such as remote sensing and medical imaging, where inconsistencies persist even among trained annotators. |
| ❌ **Prior work** | Existing methods either assume a fully noisy setting (co-training) or require a small clean subset (meta-learning). Neither paradigm supports clean-label-free purification. |
| ✅ **Our approach** | A multi-round meta-purification strategy that decouples base-model training from label refinement, preserving training stability without any clean labels. |
| 📐 **Theory** | We analyze the purification process under a stochastic contraction perspective and introduce a KL-divergence-based stopping criterion to adaptively select purification depth. |
| 📊 **Results** | Consistently outperforms state-of-the-art methods on CIFAR and miniClothing-1M; achieves up to **62.61% label accuracy improvement** on UC Merced and AID. |

---

## Key Features

- **Clean-label-free** — no clean reference subset required at any stage.
- **Decoupled purification** — base-model training and meta-purification alternate independently, preserving stability.
- **NoiseDetector** — attention-fusion network that refines soft labels using EMA teacher pseudo-labels.
- **Finite-difference meta-gradient** — trains the NoiseDetector without second-order differentiation.
- **KL-divergence stopping** — exponential curve fitting on KL history to adaptively select the optimal purification depth.
- **Accumulated label distributions** across rounds for reliable mislabeled-instance detection and correction.

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
| `EMA teacher` | Exponential moving average of the student; provides stable pseudo-labels |
| `NoiseDetector` | Attention-fusion network that refines noisy labels using EMA pseudo-labels |
| `meta_update` | Finite-difference meta-gradient: direction `v` and step size `ε` |
| `KFold splits` | Each round uses a different fold as the held-out meta validation set |
| `KL stopping` | Exponential curve fit on KL history to predict the optimal purification depth |

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA GPU (recommended)
- `torch`, `torchvision`, `timm`, `tqdm`, `scikit-learn`, `ema_pytorch`, `scipy`

---

## Installation

```bash
git clone -b RS_datasets https://github.com/mitsuyoshiohtsu/MRMP.git
cd MRMP
pip install torch torchvision timm tqdm scikit-learn ema_pytorch scipy
```

---

## Repository Structure

```
MRMP/
├── MRMP.py              # Main pipeline: pre-train → meta-train → purify → repeat
├── meta_modules.py      # Meta-update: computes v and ε via finite-difference
├── model.py             # NoiseDetector (AttentionFusion network)
├── data/
│   ├── aid.py           # AID dataset loader with noise injection
│   ├── ucmerced.py      # UC Merced dataset loader with noise injection
│   └── utils.py         # Noise generation utilities
├── setup.sh             # Environment setup
├── main.sh              # Training launcher
└── results/             # Logs and metrics (auto-created)
```

---

## Datasets

### UC Merced Land Use Dataset
A 21-class aerial image dataset with 100 images per class (2100 total), manually annotated from USGS National Map imagery at 0.3 m/pixel resolution.

### AID (Aerial Image Dataset)
A 30-class aerial scene classification dataset with 10,000 images collected from Google Earth, varying in spatial resolution and imaging conditions.

Both datasets support **symmetric**, **pairflip**, and **human** noise injection via the loaders in `data/`.

---

## Training

```bash
# UC Merced — symmetric noise at 20%
python MRMP.py --dataset ucmerced --noise_type symmetric --noise_rate 0.2

# AID — pairflip noise at 40%
python MRMP.py --dataset aid --noise_type pairflip --noise_rate 0.4
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `ucmerced` | Dataset to use: `ucmerced` or `aid` |
| `--noise_type` | `symmetric` | Noise type: `symmetric`, `pairflip`, or `human` |
| `--noise_rate` | `0.0` | Fraction of corrupted labels (0.0–1.0) |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `32` | Batch size |
| `--n_epoch` | `40` | Max pre-training epochs per round |
| `--n_round` | `10` | Max purification rounds |
| `--start_round` | `1` | Starting round (for resuming) |
| `--meta_iter` | `1000` | Meta-training iterations per round |
| `--beta` | `0.9` | EMA decay factor |
| `--T` | `0.7` | NoiseDetector attention temperature |
| `--ES_step` | `5` | Early stopping patience (epochs) |
| `--load` | `None` | Path to a pre-saved purified dataset checkpoint |
| `--gpuid` | `0` | GPU device ID |
| `--seed` | `123` | Random seed |

---

## Output

```
data/purified/
  └── {model_str}_at_round{r}.pt      # Purified dataset checkpoint

results/{dataset}/MRML/
  ├── {model_str}.txt                 # Per-epoch accuracy and loss log
  └── {model_str}.json                # Round-level metrics
```

| Metric | Description |
|---|---|
| `data_acc` | Label accuracy after purification |
| `n2c_flip` | Noisy → correct label flips (beneficial) |
| `n2n_flip` | Noisy → noisy label flips (harmful) |
| `c2n_flip` | Correct → noisy label flips (harmful) |
| `mean_kl` | Mean KL divergence between label distributions; used as convergence signal |
| `pred_opt_r` | Predicted optimal purification round |

---

## Citation

*(To be added after publication.)*

---

## License

MIT License.
