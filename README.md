# TGFusion

**Transformer-GAN Hybrid Architecture for Multi-Modal Medical Image Fusion**

> Chaitanya Krishna Kasaraneni · Sarmista Thalapaneni  
> IEEE EIT 2026 · [`kc.kasaraneni@ieee.org`](mailto:kc.kasaraneni@ieee.org)

---

## Overview

TGFusion fuses complementary medical image modalities (CT+MRI, MRI+PET) through
three components:

1. **Shared-weight Swin Transformer encoder** — dual-stream, weights tied across
   both input modalities (`embed_dim=64`, depths `(2,2,4,2)`, heads `(2,4,8,16)`)
2. **Bidirectional Cross-Modal Attention (CMA)** — bottleneck fusion via
   `Q_A→K_B,V_B` and `Q_B→K_A,V_A`, followed by `LayerNorm(F̃_A + F̃_B)`
3. **Conditional GAN decoder** — U-Net topology with InstanceNorm+ReLU,
   averaged skip connections, Tanh output; supervised by a 70×70 PatchGAN

**Generator parameters:** ~18.4M · **Dataset:** Harvard AANLIB (~30 MB, free)

---

## Results (Harvard AANLIB test set)

### Table I — CT-MRI Fusion

| Method          | SSIM  | PSNR (dB) | MI   | SF   |
|-----------------|-------|-----------|------|------|
| DWT             | 0.743 | 28.1      | 1.42 | 18.3 |
| CNN-Fuse        | 0.791 | 29.8      | 1.56 | 20.1 |
| U-Net Fusion    | 0.812 | 30.9      | 1.61 | 21.4 |
| CycleGAN        | 0.834 | 31.8      | 1.67 | 22.9 |
| ViT-Only        | 0.856 | 32.9      | 1.72 | 24.1 |
| **TGFusion**    | **0.892** | **34.7** | **1.81** | **25.6** |

### Table II — MRI-PET Fusion

| Method          | SSIM  | PSNR (dB) | MI   | SF   |
|-----------------|-------|-----------|------|------|
| DWT             | 0.712 | 26.4      | 1.31 | 15.8 |
| CNN-Fuse        | 0.758 | 27.9      | 1.44 | 17.2 |
| U-Net Fusion    | 0.781 | 29.1      | 1.49 | 18.6 |
| CycleGAN        | 0.806 | 30.2      | 1.55 | 19.9 |
| ViT-Only        | 0.831 | 31.4      | 1.61 | 21.0 |
| **TGFusion**    | **0.863** | **32.9** | **1.73** | **22.4** |

---

## Installation

```bash
git clone https://github.com/chaitanyakasaraneni/tgfusion.git
cd tgfusion
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+. GPU strongly recommended.

---

## Dataset — Harvard AANLIB

Download the **Harvard Whole Brain Atlas** (~30 MB, free, no registration):

- **Official:** http://www.med.harvard.edu/AANLIB/home.html
- **Ready-to-use PNG pairs:** https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets

### Step 1 — Clone the dataset repo

```bash
git clone https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets
```

The cloned repo has this flat structure:

```
Havard-Medical-Image-Fusion-Datasets/
  CT-MRI/
    CT/   2004.png  2005.png  ...
    MRI/  2004.png  2005.png  ...
  PET-MRI/
    MRI/  2004.png  2005.png  ...
    PET/  2004.png  2005.png  ...
```

### Step 2 — Reorganise with the provided script

```bash
# Preview first (no files copied):
python data/reorganise_aanlib.py \
    --src /path/to/Havard-Medical-Image-Fusion-Datasets \
    --dst /path/to/aanlib \
    --dry_run

# Then run for real:
python data/reorganise_aanlib.py \
    --src /path/to/Havard-Medical-Image-Fusion-Datasets \
    --dst /path/to/aanlib
```

This produces the structure expected by the dataloaders:

```
aanlib/
  ct_mri/
    subject_2004/  ct.png   mri.png
    subject_2005/  ct.png   mri.png
    ...
  mri_pet/
    subject_2004/  mri.png  pet.png
    subject_2005/  mri.png  pet.png
    ...
```

The code splits subjects 70/15/15 (train/val/test) at subject level automatically.  
PET images (RGB) are converted to grayscale via the Y channel of YCbCr before fusion.

---

## Training

```bash
# Smoke test — no data required:
python scripts/train.py --dataset synthetic --epochs 5 --batch_size 2

# CT-MRI  (reproduces Table I):
python scripts/train.py \
    --dataset  ct_mri \
    --data_dir /path/to/aanlib \
    --epochs   100 \
    --batch_size 4

# MRI-PET  (reproduces Table II):
python scripts/train.py \
    --dataset  mri_pet \
    --data_dir /path/to/aanlib \
    --epochs   100 \
    --batch_size 4

# Resume from checkpoint:
python scripts/train.py \
    --dataset  ct_mri \
    --data_dir /path/to/aanlib \
    --resume   outputs/ct_mri/checkpoints/best.pt
```

Training config matches paper Section IV-D:
- Adam · β₁=0.5 · β₂=0.999 · lr=2×10⁻⁴
- Linear LR decay: epochs 50 → 100
- Mixed-precision (AMP) enabled by default; disable with `--no_amp`

---

## Evaluation

```bash
# Full benchmark — both tables with LaTeX output:
python scripts/evaluate.py \
    --ckpt_ct_mri  outputs/ct_mri/checkpoints/best.pt \
    --ckpt_mri_pet outputs/mri_pet/checkpoints/best.pt \
    --data_dir     /path/to/aanlib \
    --latex

# Single task:
python scripts/evaluate.py \
    --dataset  ct_mri \
    --ckpt     outputs/ct_mri/checkpoints/best.pt \
    --data_dir /path/to/aanlib

# Smoke test (no data):
python scripts/evaluate.py --dataset synthetic
```

The `--latex` flag prints ready-to-paste `\tabular` blocks matching Tables I & II.

---

## Model Architecture

```
Input A (CT/MRI)  ─┐                   ┌─ Up-3 ─ Up-2 ─ Up-1 ─┐
                    ├─ Swin Encoder f_θ ─┤                       ├─ Î (Fused)
Input B (MRI/PET) ─┘   (shared weights) └─ CMA  ─ Decoder     ─┘
                                              │
                                         PatchGAN D
                                       [I_A; I_B; Î]
```

Loss (Eq. 5):  `L_G = L_cGAN + 10·L₁ + 5·L_SSIM`

---

## Repository Structure

```
tgfusion/
├── models/
│   ├── tgfusion.py           # Full model: encoder, CMA, decoder, discriminator
│   └── losses.py             # SSIMLoss, GANLoss, TGFusionLoss
├── data/
│   ├── dataset.py            # AANLIBCTMRIDataset, AANLIBMRIPETDataset, SyntheticDataset
│   └── reorganise_aanlib.py  # One-time script to prepare the AANLIB download
├── utils/
│   ├── metrics.py            # SSIM, PSNR, MI, SF
│   └── visualise.py          # Comparison grids, training curves
├── scripts/
│   ├── train.py              # Training loop
│   └── evaluate.py           # Baseline comparison + LaTeX table output
└── requirements.txt
```

---
