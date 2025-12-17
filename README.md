# MRMP: Multi-Round Meta-Purification (Sample Implementation)

This repository provides a minimal and clean implementation of Multi-Round Meta-Purification (MRMP).
The code is simplified for reproducibility while preserving the core algorithmic structure.

## 1. Overview
MRMP is a multi-round meta-learning framework for label purification under noisy labels.  
It consists of:
- Base model  
- EMA teacher model  
- NoiseDetector network  
- Meta-update module  
- Multi-round purification loop  

## Project Structure

The repository is organized as follows:

```text
MRMP/
├── MRMP.py                # Main training and multi-round purification pipeline
├── data/
│   ├── cifar.py           # CIFAR-10/100 dataset handling
│   └── utils.py           # Noise generation and utility functions
├── meta_modules.py        # Meta-learning update rules (v, ε)
├── model.py               # NoiseDetector network architecture
└── results/               # Experimental logs and purified label outputs

## 3. Key Components

### Dataset (CIFAR10/100)
- Loads CIFAR dataset  
- Injects noise (symmetric / pairflip / human)  
- Converts labels to one-hot  
- Returns: image, noisy_label, source_label, index  

### NoiseDetector
Neural network that predicts purified soft labels using:
- pseudo-labels from EMA  
- noisy labels  
- original source labels  

### Meta-Update
Computes:
- v : meta-gradient direction  
- epsilon : step size  
Using finite-difference approximation.

### Purification
Combines:
- student model  
- EMA teacher  
- NoiseDetector  
To update:
- train_noisy_labels  
- flip statistics  
- KL divergence  

### Multi-Round Loop
pretrain → meta_train → purify_dataset → save → check convergence  

## 4. Installation
pip install torch torchvision timm tqdm scikit-learn ema_pytorch

## 5. Training Example
python MRMP.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2

## 6. Output
- purified/*.pt : purified datasets  
- results/*.json : KL, flips, accuracies  
- terminal logs  

## 7. Citation
(To be added after publication)

## 8. License
MIT License.
