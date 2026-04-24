# Conditional Synthetic Face Generation GAN

A PyTorch implementation of a Conditional GAN (CGAN) that generates realistic 128×128 face images conditioned on facial attributes, trained on the CelebA dataset.

---

## Overview

This project trains a deep convolutional CGAN to synthesize photorealistic faces conditioned on three binary attributes:

- **Male** — gender
- **Young** — age group
- **Smiling** — expression

The model supports all 8 attribute combinations, generating distinct and coherent faces for each.

---

## Architecture

### Generator
- Input: latent vector `z` (dim=256) + attribute embeddings (dim=64 per attribute)
- 5× `UpsampleBlock` with **Conditional Batch Normalization (CBN)** — gamma/beta predicted from label embeddings
- Output: 128×128 RGB image via `tanh`
- EMA (Exponential Moving Average) copy maintained for stable inference (`β=0.999`)

### Discriminator
- 5× strided conv blocks with **Spectral Normalization** (128 → 4×4 feature map)
- Label projection via spectral-normalized linear layer
- **Hinge loss** with one-sided label smoothing
- **R1 gradient penalty** for training stability

### Training Details

| Hyperparameter | Value |
|---|---|
| Image size | 128×128 |
| Batch size | 64 |
| Epochs | 150 |
| Latent dim (Z) | 256 |
| Embedding dim | 64 per attribute |
| LR Generator | 1e-4 |
| LR Discriminator | 4e-4 (TTUR) |
| R1 gamma | 1.0 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Dataset | CelebA (~5,000 images/combo, balanced) |

---

## Project Structure

```
.
├── train.py                          # Main training script
├── resize.py                         # Image preprocessing/resizing
├── data_cleaning.ipynb               # Dataset cleaning notebook
├── logs_cgan__128_cp/
│   ├── losses.png                    # Generator & Discriminator loss curves
│   └── train_logs.txt                # Full training log
├── samples_cgan__128_cp/
│   └── epoch_0150.png                # Sample output at epoch 150
└── checkpoints_cgan__128_cp/
    └── checkpoint_150.pth            # Final model checkpoint (via Git LFS)
```

---

## Requirements

```bash
pip install torch torchvision tqdm numpy pandas matplotlib pillow
```

Tested with:
- Python 3.10+
- PyTorch 2.x
- CUDA (optional but recommended)

---

## Usage

### 1. Prepare the dataset

Place your CelebA images in `celeba_cgan/` and the labels CSV at `celeba_cgan_labels.csv`.

The CSV should have columns: `image_id`, `Male`, `Young`, `Smiling` (binary 0/1).

Use `data_cleaning.ipynb` and `resize.py` to preprocess the images.

### 2. Train

```bash
python train.py
```

Checkpoints are saved every 10 epochs to `checkpoints_cgan__128_cp/`.
Sample images are saved every 2 epochs (early) and every 5 epochs (later) to `samples_cgan__128_cp/`.

### 3. Load a checkpoint for inference

```python
import torch
from train import Generator

G = Generator()
ckpt = torch.load("checkpoints_cgan__128_cp/checkpoint_150.pth", map_location="cpu")
G.load_state_dict(ckpt["g_ema"])
G.eval()

z = torch.randn(1, 256)
labels = torch.tensor([[1, 1, 1]], dtype=torch.float32)  # Male, Young, Smiling
with torch.no_grad():
    img = G(z, labels)  # shape: (1, 3, 128, 128)
```

---

## Results

Loss curves and sample outputs are tracked across 150 epochs. The model uses an EMA generator for final inference, producing more stable and sharper results than the raw generator.

Sample grid at epoch 150 covers all 8 attribute combinations (Male/Female × Young/Old × Smiling/Not).

---

## Notes

- Checkpoint files (`.pth`) are stored via **Git LFS** due to their size (~207MB each)
- The CelebA dataset is not included — download it from [the official source](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Training was done on a PC with a CUDA-capable GPU
