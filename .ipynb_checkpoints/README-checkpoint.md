# MRMP: Multi-Round Meta-Learning for Label Purification Without Trusted Labels

Official PyTorch implementation of the paper:

> **Multi-Round Meta-Learning for Label Purification Without Trusted Labels**  
> *(Under review at TPAMI, 2026)*

MRMP is a PyTorch implementation of a clean-label-free meta-learning framework for label purification under noisy supervision. The framework iteratively purifies label quality over multiple rounds by alternating between **base-model training** and **meta-purification** — without requiring any clean reference labels.

This repository provides the core training and purification code used in our experiments, together with dataset utilities and reproducible evaluation protocols.

---

## Overview

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
├── MRMP.py              # Main pipeline: pre-train → meta-train → purify → repeat
├── meta_modules.py      # Meta-update: computes v and ε via finite-difference
├── model.py             # NoiseDetector (AttentionFusion network)
├── data/
│   ├── cifar.py         # CIFAR-10/100 with noise injection
│   └── utils.py         # Noise generation utilities
├── setup.sh             # Environment setup
├── main.sh              # Training launcher
└── results/             # Logs and metrics (auto-created)
```

---

## Training

```bash
# CIFAR-10, symmetric noise 20%
python MRMP.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2

# CIFAR-100, pairflip noise 40%
python MRMP.py --dataset cifar100 --noise_type pairflip --noise_rate 0.4

# Resume from a saved purified dataset
python MRMP.py --dataset cifar100 --noise_type symmetric --noise_rate 0.4 \
  --load ./data/purified/cifar100_MRML_symmetric_0.4_beta0.9_tau0.7_at_round3.pt \
  --start_round 4
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar100` | `cifar10` or `cifar100` |
| `--noise_type` | `symmetric` | `symmetric`, `pairflip`, or `human` |
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
