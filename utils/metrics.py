"""
TGFusion Evaluation Metrics
============================
Four metrics reported in paper Tables I & II:

    SSIM  — Structural Similarity Index       (higher is better)
    PSNR  — Peak Signal-to-Noise Ratio (dB)   (higher is better)
    MI    — Mutual Information                 (higher is better)
    SF    — Spatial Frequency                  (higher is better)

All computed on images in [0, 1] range.
"""

import numpy as np
import torch


def _to_numpy(t):
    """Tensor (B,1,H,W) or (1,H,W) → numpy array in [0,1]."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    t = np.clip((t + 1) / 2, 0, 1)     # [-1,1] → [0,1]
    return t.squeeze()                   # (H,W) or (B,H,W)


# ─────────────────────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(pred, target):
    """
    Structural Similarity Index.
    Uses 11×11 Gaussian window, σ=1.5, K1=0.01, K2=0.03.
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        p = _to_numpy(pred)
        t = _to_numpy(target)
        if p.ndim == 3:
            return float(np.mean([
                sk_ssim(p[i], t[i], data_range=1.0) for i in range(p.shape[0])
            ]))
        return float(sk_ssim(p, t, data_range=1.0))
    except ImportError:
        # fallback: manual SSIM
        p = _to_numpy(pred).astype(np.float64)
        t = _to_numpy(target).astype(np.float64)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_p = p.mean(); mu_t = t.mean()
        sig_p = p.std(); sig_t = t.std()
        sig_pt = ((p - mu_p) * (t - mu_t)).mean()
        num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
        den = (mu_p**2 + mu_t**2 + C1) * (sig_p**2 + sig_t**2 + C2)
        return float(num / den)


# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(pred, target):
    """Peak Signal-to-Noise Ratio in dB. data_range = 1.0 (images in [0,1])."""
    p = _to_numpy(pred).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    mse = np.mean((p - t) ** 2)
    if mse < 1e-12:
        return 100.0
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


# ─────────────────────────────────────────────────────────────────────────────
# Mutual Information
# ─────────────────────────────────────────────────────────────────────────────

def compute_mi(fused, img_a, img_b=None, bins=256):
    """
    Mutual Information between fused image and source modalities.
    If img_b is given: MI = MI(fused, A) + MI(fused, B)  (normalised average).
    """
    def _mi_pair(x, y):
        x = (np.clip(_to_numpy(x), 0, 1) * (bins - 1)).astype(int).ravel()
        y = (np.clip(_to_numpy(y), 0, 1) * (bins - 1)).astype(int).ravel()
        joint, _, _ = np.histogram2d(x, y, bins=bins)
        joint = joint / joint.sum()
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        mask = joint > 0
        mi = np.sum(joint[mask] * np.log2(joint[mask] / (px * py + 1e-12)[mask]))
        return float(mi)

    mi_a = _mi_pair(fused, img_a)
    if img_b is not None:
        mi_b = _mi_pair(fused, img_b)
        return (mi_a + mi_b) / 2
    return mi_a


# ─────────────────────────────────────────────────────────────────────────────
# Spatial Frequency
# ─────────────────────────────────────────────────────────────────────────────

def compute_sf(fused):
    """
    Spatial Frequency — measures richness of high-frequency detail.
    SF = sqrt(RF² + CF²) where RF and CF are row and column frequencies.
    """
    img = _to_numpy(fused).astype(np.float64)
    if img.ndim == 3:
        return float(np.mean([compute_sf(img[i]) for i in range(img.shape[0])]))
    rf = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    cf = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2) * 100)   # ×100 matches paper scale


# ─────────────────────────────────────────────────────────────────────────────
# Combined
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(fused, img_a, img_b):
    """
    Compute all four paper metrics in one call — no-reference protocol.
    SSIM and PSNR are averaged over fused↔A and fused↔B (no ground truth).
    This matches U2Fusion / SwinFusion / CDDFuse evaluation standard.

    Args:
        fused  : generated fused image  tensor (B,1,H,W) in [-1,1]
        img_a  : input modality A       tensor (B,1,H,W) in [-1,1]
        img_b  : input modality B       tensor (B,1,H,W) in [-1,1]

    Returns:
        dict with keys: SSIM, PSNR, MI, SF
    """
    return {
        'SSIM': (compute_ssim(fused, img_a) + compute_ssim(fused, img_b)) / 2,
        'PSNR': (compute_psnr(fused, img_a) + compute_psnr(fused, img_b)) / 2,
        'MI'  : compute_mi(fused, img_a, img_b),
        'SF'  : compute_sf(fused),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Running average tracker for epoch-level logging
# ─────────────────────────────────────────────────────────────────────────────

class MetricTracker:
    """Accumulates metric values and returns epoch averages."""

    def __init__(self):
        self._sums   = {}
        self._counts = {}

    def update(self, metrics: dict):
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0)   + float(v)
            self._counts[k] = self._counts.get(k, 0) + 1

    def averages(self):
        return {k: self._sums[k] / self._counts[k]
                for k in self._sums}

    def reset(self):
        self._sums   = {}
        self._counts = {}
