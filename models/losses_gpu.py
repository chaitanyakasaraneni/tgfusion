"""
TGFusion Loss Functions  (v2 — fixed for sharpness)
====================================================
Composite generator loss:

    L_G = L_cGAN + λ₁·L_intensity + λ₂·L_gradient + λ₃·L_SSIM + λ₄·L_MSE

Changes from v1:
  - λ_grad raised from 10 → 50  (SF was stuck at ~6, needs >18)
  - λ_MSE = 5 added  (pixel-MSE against max(A,B) target; helps PSNR)
  - λ_intensity kept at 10, λ_SSIM kept at 5
  - Discriminator 'real' sample = max(A,B) per pixel (not pixel average)

No ground-truth target needed — all losses are no-reference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sobel gradient helper
# ─────────────────────────────────────────────────────────────────────────────

def _sobel_magnitude(x):
    """Per-pixel gradient magnitude of x (B,1,H,W). Output in [0, ~1]."""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x.float(), sobel_x, padding=1)
    gy = F.conv2d(x.float(), sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss terms
# ─────────────────────────────────────────────────────────────────────────────

class IntensityLoss(nn.Module):
    """L1 between fused pixels and per-pixel maximum of sources."""
    def forward(self, fused, img_a, img_b):
        target = torch.max(img_a, img_b)
        return F.l1_loss(fused, target)


class MSELoss(nn.Module):
    """
    MSE between fused and per-pixel maximum of sources.
    Directly penalises PSNR-relevant squared error.
    """
    def forward(self, fused, img_a, img_b):
        target = torch.max(img_a, img_b)
        return F.mse_loss(fused, target)


class GradientLoss(nn.Module):
    """
    L1 between fused gradient magnitude and the stronger source gradient.
    Drives SF upward — critical for sharpness.
    """
    def forward(self, fused, img_a, img_b):
        g_fused = _sobel_magnitude(fused)
        g_a     = _sobel_magnitude(img_a)
        g_b     = _sobel_magnitude(img_b)
        target  = torch.max(g_a, g_b)
        return F.l1_loss(g_fused, target)


class SSIMLoss(nn.Module):
    """
    1 - mean(SSIM(fused, A), SSIM(fused, B)).
    11×11 Gaussian kernel, σ=1.5.
    """
    def __init__(self, window_size=11, sigma=1.5,
                 C1=0.01 ** 2, C2=0.03 ** 2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        self.ws = window_size
        kernel  = self._gauss(window_size, sigma)
        self.register_buffer('kernel', kernel)

    @staticmethod
    def _gauss(size, sigma):
        c = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        k = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        return k

    def _ssim_pair(self, pred, target):
        pred   = (pred   + 1) / 2
        target = (target + 1) / 2
        pred   = pred.float()
        target = target.float()
        C   = pred.shape[1]
        pad = self.ws // 2
        k   = self.kernel.float().to(pred.device).expand(C, 1, -1, -1)

        mu1  = F.conv2d(pred,   k, padding=pad, groups=C)
        mu2  = F.conv2d(target, k, padding=pad, groups=C)
        mu1_sq = mu1 ** 2;  mu2_sq = mu2 ** 2;  mu12 = mu1 * mu2

        s1  = F.conv2d(pred   * pred,   k, padding=pad, groups=C) - mu1_sq
        s2  = F.conv2d(target * target, k, padding=pad, groups=C) - mu2_sq
        s12 = F.conv2d(pred   * target, k, padding=pad, groups=C) - mu12

        num = (2 * mu12 + self.C1) * (2 * s12  + self.C2)
        den = (mu1_sq + mu2_sq + self.C1) * (s1 + s2 + self.C2)
        return (num / den).mean()

    def forward(self, fused, img_a, img_b):
        ssim_a = self._ssim_pair(fused, img_a)
        ssim_b = self._ssim_pair(fused, img_b)
        return 1 - (ssim_a + ssim_b) / 2


class GANLoss(nn.Module):
    """cGAN BCE loss with label smoothing (real=0.9, fake=0.0)."""
    def __init__(self, real_label=0.9, fake_label=0.0):
        super().__init__()
        self.real = real_label
        self.fake = fake_label
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred, is_real: bool):
        val    = self.real if is_real else self.fake
        labels = torch.full_like(pred, val)
        return self.bce(pred, labels)


# ─────────────────────────────────────────────────────────────────────────────
# TGFusion composite loss  (v2)
# ─────────────────────────────────────────────────────────────────────────────

class TGFusionLoss(nn.Module):
    """
    Full composite no-reference loss for TGFusion:

        L_G = L_cGAN
              + λ_intensity · L_intensity   (default 10)
              + λ_grad      · L_gradient    (default 50  ← raised ×5)
              + λ_ssim      · L_SSIM        (default  5)
              + λ_mse       · L_MSE         (default  5  ← new)

    Discriminator 'real' sample = max(A, B) per pixel (no-reference).
    """

    def __init__(self,
                 lambda_intensity=10.0,
                 lambda_grad=50.0,       # ×5 vs v1
                 lambda_ssim=5.0,
                 lambda_mse=5.0,         # new
                 # legacy kwargs kept for backward compat
                 lambda_l1=None,
                 **_kwargs):
        super().__init__()
        self.lambda_intensity = lambda_l1 if lambda_l1 is not None else lambda_intensity
        self.lambda_grad      = lambda_grad
        self.lambda_ssim      = lambda_ssim
        self.lambda_mse       = lambda_mse

        self.gan       = GANLoss()
        self.intensity = IntensityLoss()
        self.mse_loss  = MSELoss()
        self.gradient  = GradientLoss()
        self.ssim      = SSIMLoss(window_size=11, sigma=1.5)

    def generator_loss(self, fake_pred, fused, img_a, img_b, target=None):
        """
        Args:
            fake_pred : discriminator output on fused image  (B,1,H',W')
            fused     : generated fused image                (B,1,H,W)
            img_a     : source modality A                    (B,1,H,W)
            img_b     : source modality B                    (B,1,H,W)
            target    : ignored (API compat)
        """
        l_gan       = self.gan(fake_pred, is_real=True)
        l_intensity = self.intensity(fused, img_a, img_b)
        l_mse       = self.mse_loss(fused,  img_a, img_b)
        l_grad      = self.gradient(fused,  img_a, img_b)
        l_ssim      = self.ssim(fused,      img_a, img_b)

        total = (l_gan
                 + self.lambda_intensity * l_intensity
                 + self.lambda_mse       * l_mse
                 + self.lambda_grad      * l_grad
                 + self.lambda_ssim      * l_ssim)

        return total, {
            "G_cGAN"     : l_gan.item(),
            "G_intensity": l_intensity.item(),
            "G_MSE"      : l_mse.item(),
            "G_grad"     : l_grad.item(),
            "G_SSIM"     : l_ssim.item(),
            "G_total"    : total.item(),
        }

    def discriminator_loss(self, real_pred, fake_pred):
        """Discriminator trained: real = max(A,B), fake = fused."""
        l_real = self.gan(real_pred,          is_real=True)
        l_fake = self.gan(fake_pred.detach(), is_real=False)
        total  = (l_real + l_fake) * 0.5
        return total, {
            "D_real" : l_real.item(),
            "D_fake" : l_fake.item(),
            "D_total": total.item(),
        }
