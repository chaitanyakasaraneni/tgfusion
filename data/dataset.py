"""
TGFusion Dataset Loaders — Harvard AANLIB
==========================================
Paper Section IV-A: "All experiments use the Harvard Whole Brain Atlas
(AANLIB). We use 160 CT-MRI pairs and 245 MRI-PET pairs, split 70/15/15
at the subject level."

Dataset URL: http://www.med.harvard.edu/AANLIB/home.html
             (~30 MB, free, no registration required)
GitHub mirrors (ready-to-use PNG/JPEG pairs):
  https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets
  https://github.com/wenzhezhai/Medical-image-fusion-dataset

Expected directory layout
--------------------------
  aanlib/
    ct_mri/
      subject_001/
        ct.png   mri.png
      subject_002/
        ct.png   mri.png
      ...
    mri_pet/
      subject_001/
        mri.png   pet.png
      ...

For PET images (RGB): the Y channel of YCbCr is extracted before fusion
(paper Section IV-A).

Also includes a SyntheticDataset for smoke-testing without any data.
"""

import os
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_tensor_gray(pil_img, img_size=256):
    """Resize → grayscale → tensor in [-1, 1]."""
    pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
    pil_img = pil_img.convert('L')
    t = TF.to_tensor(pil_img)          # [0, 1]
    return t * 2.0 - 1.0               # [-1, 1]


def _pet_to_gray(pil_img, img_size=256):
    """
    PET images are RGB in AANLIB.
    Extract Y channel of YCbCr as per paper Section IV-A.
    """
    pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
    pil_img = pil_img.convert('YCbCr')
    y, _, _ = pil_img.split()
    t = TF.to_tensor(y)
    return t * 2.0 - 1.0


def _avg_fused_target(t_a, t_b):
    """Per-pixel average as proxy ground-truth fused image."""
    return (t_a + t_b) * 0.5


def _augment(img_a, img_b, target):
    """Consistent random augmentation across all three images."""
    if random.random() > 0.5:
        img_a  = TF.hflip(img_a)
        img_b  = TF.hflip(img_b)
        target = TF.hflip(target)
    if random.random() > 0.5:
        img_a  = TF.vflip(img_a)
        img_b  = TF.vflip(img_b)
        target = TF.vflip(target)
    angle = random.uniform(-10, 10)
    img_a  = TF.rotate(img_a,  angle)
    img_b  = TF.rotate(img_b,  angle)
    target = TF.rotate(target, angle)
    return img_a, img_b, target


def _subject_split(subjects, train_frac=0.70, val_frac=0.15, seed=42):
    """
    Split subject list into train/val/test at subject level (paper Section IV-A)
    to prevent data leakage.
    """
    rng = random.Random(seed)
    s   = subjects[:]
    rng.shuffle(s)
    n_train = int(len(s) * train_frac)
    n_val   = int(len(s) * val_frac)
    return {
        'train': s[:n_train],
        'val'  : s[n_train:n_train + n_val],
        'test' : s[n_train + n_val:],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Harvard AANLIB — CT-MRI  (Table I)
# ─────────────────────────────────────────────────────────────────────────────

class AANLIBCTMRIDataset(Dataset):
    """
    CT-MRI fusion dataset from Harvard AANLIB.
    Expects pairs of CT and MRI images as PNG/JPEG files per subject folder.

    Paper: 160 CT-MRI pairs, 70/15/15 subject-level split.
    """

    def __init__(self, root, split='train', img_size=256, augment=False):
        self.img_size = img_size
        self.augment  = augment and (split == 'train')

        root = Path(root) / 'ct_mri'
        subjects = sorted([d for d in root.iterdir() if d.is_dir()])
        splits   = _subject_split(subjects)
        folders  = splits[split]

        self.pairs = []
        for folder in folders:
            ct_files  = sorted(glob.glob(str(folder / 'ct*')))
            mri_files = sorted(glob.glob(str(folder / 'mri*')))
            for ct_path, mri_path in zip(ct_files, mri_files):
                self.pairs.append((ct_path, mri_path))

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No CT-MRI pairs found under {root}. "
                "Expected subfolders with ct.png and mri.png files.\n"
                "Download: https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ct_path, mri_path = self.pairs[idx]
        ct  = _to_tensor_gray(Image.open(ct_path),  self.img_size)
        mri = _to_tensor_gray(Image.open(mri_path), self.img_size)
        target = _avg_fused_target(ct, mri)
        if self.augment:
            ct, mri, target = _augment(ct, mri, target)
        return ct, mri, target


# ─────────────────────────────────────────────────────────────────────────────
# Harvard AANLIB — MRI-PET  (Table II)
# ─────────────────────────────────────────────────────────────────────────────

class AANLIBMRIPETDataset(Dataset):
    """
    MRI-PET fusion dataset from Harvard AANLIB.
    PET images are RGB; Y channel extracted (paper Section IV-A).

    Paper: 245 MRI-PET pairs, 70/15/15 subject-level split.
    """

    def __init__(self, root, split='train', img_size=256, augment=False):
        self.img_size = img_size
        self.augment  = augment and (split == 'train')

        root = Path(root) / 'mri_pet'
        subjects = sorted([d for d in root.iterdir() if d.is_dir()])
        splits   = _subject_split(subjects)
        folders  = splits[split]

        self.pairs = []
        for folder in folders:
            mri_files = sorted(glob.glob(str(folder / 'mri*')))
            pet_files = sorted(glob.glob(str(folder / 'pet*')))
            for mri_path, pet_path in zip(mri_files, pet_files):
                self.pairs.append((mri_path, pet_path))

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No MRI-PET pairs found under {root}. "
                "Expected subfolders with mri.png and pet.png files.\n"
                "Download: https://github.com/xianming-gu/Havard-Medical-Image-Fusion-Datasets"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mri_path, pet_path = self.pairs[idx]
        mri    = _to_tensor_gray(Image.open(mri_path), self.img_size)
        pet    = _pet_to_gray(Image.open(pet_path),    self.img_size)
        target = _avg_fused_target(mri, pet)
        if self.augment:
            mri, pet, target = _augment(mri, pet, target)
        return mri, pet, target


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset — for smoke testing (no data required)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    """
    Generates random CT-like and MRI-like image pairs on the fly.
    Use for smoke testing the pipeline without downloading AANLIB.

    Usage:
        python scripts/train.py --dataset synthetic --epochs 5 --batch_size 2
    """

    def __init__(self, size=200, img_size=256, split='train'):
        sizes = {'train': int(size * 0.7),
                 'val'  : int(size * 0.15),
                 'test' : size - int(size * 0.7) - int(size * 0.15)}
        self.n        = sizes[split]
        self.img_size = img_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng   = torch.Generator().manual_seed(idx)
        img_a = torch.randn(1, self.img_size, self.img_size, generator=rng)
        img_b = torch.randn(1, self.img_size, self.img_size, generator=rng)
        img_a = torch.tanh(img_a)
        img_b = torch.tanh(img_b)
        return img_a, img_b, (img_a + img_b) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Dataset factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(dataset_name, data_dir=None, split='train',
                     img_size=256, batch_size=4, num_workers=4):
    """
    Factory function used by train.py and evaluate.py.

    Args:
        dataset_name : 'ct_mri' | 'mri_pet' | 'synthetic'
        data_dir     : path to root of AANLIB download
        split        : 'train' | 'val' | 'test'
    """
    augment = (split == 'train')

    if dataset_name == 'ct_mri':
        ds = AANLIBCTMRIDataset(data_dir, split=split,
                                 img_size=img_size, augment=augment)
    elif dataset_name == 'mri_pet':
        ds = AANLIBMRIPETDataset(data_dir, split=split,
                                  img_size=img_size, augment=augment)
    elif dataset_name == 'synthetic':
        ds = SyntheticDataset(img_size=img_size, split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         "Choose from: ct_mri, mri_pet, synthetic")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
