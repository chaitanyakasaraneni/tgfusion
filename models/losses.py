"""
TGFusion Loss Functions
=======================
Composite generator loss (paper Eq. 5):

    L_G = L_cGAN(G, D) + λ₁·L₁(G) + λ₂·L_SSIM(G)

    λ₁ = 10  (pixel fidelity)
    λ₂ = 5   (structural similarity)

Discriminator uses standard cGAN loss with label smoothing
    p_real = 0.9,  p_fake = 0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss.
    Uses 11×11 Gaussian kernel, σ=1.5 (paper Section III-E).
    Returns 1 − SSIM so it can be minimised.
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
        c  = torch.arange(size, dtype=torch.float32) - size // 2
        g  = torch.exp(-(c ** 2) / (2 * sigma ** 2))
        g  = g / g.sum()
        k  = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        return k                                    # (1,1,ws,ws)

    def forward(self, pred, target):
        # normalise [-1,1] → [0,1]
        pred   = (pred   + 1) / 2
        target = (target + 1) / 2
        C = pred.shape[1]
        pad = self.ws // 2
        k   = self.kernel.expand(C, 1, -1, -1)

        mu1     = F.conv2d(pred,    k, padding=pad, groups=C)
        mu2     = F.conv2d(target,  k, padding=pad, groups=C)
        mu1_sq  = mu1 ** 2
        mu2_sq  = mu2 ** 2
        mu12    = mu1 * mu2

        s1  = F.conv2d(pred   * pred,   k, padding=pad, groups=C) - mu1_sq
        s2  = F.conv2d(target * target, k, padding=pad, groups=C) - mu2_sq
        s12 = F.conv2d(pred   * target, k, padding=pad, groups=C) - mu12

        num = (2 * mu12 + self.C1) * (2 * s12  + self.C2)
        den = (mu1_sq + mu2_sq + self.C1) * (s1 + s2 + self.C2)
        return 1 - (num / den).mean()


class GANLoss(nn.Module):
    """
    cGAN loss with label smoothing.
        real_label = 0.9  (smoothed)
        fake_label = 0.0
    """

    def __init__(self, real_label=0.9, fake_label=0.0):
        super().__init__()
        self.real = real_label
        self.fake = fake_label
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred, is_real: bool):
        val    = self.real if is_real else self.fake
        labels = torch.full_like(pred, val)
        return self.bce(pred, labels)


class TGFusionLoss(nn.Module):
    """
    Full composite loss for TGFusion (paper Eq. 5).

        L_G = L_cGAN + λ₁·L₁ + λ₂·L_SSIM
    """

    def __init__(self, lambda_l1=10.0, lambda_ssim=5.0):
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.gan  = GANLoss()
        self.l1   = nn.L1Loss()
        self.ssim = SSIMLoss(window_size=11, sigma=1.5)

    def generator_loss(self, fake_pred, fused, target):
        """
        Args:
            fake_pred : discriminator output on generated image  (B, 1, H', W')
            fused     : generated fused image                    (B, 1, H, W)
            target    : ground-truth fused image                 (B, 1, H, W)
        Returns:
            total  : scalar loss tensor
            losses : dict of individual loss values
        """
        l_gan  = self.gan(fake_pred, is_real=True)      # fool discriminator
        l_l1   = self.l1(fused, target)
        l_ssim = self.ssim(fused, target)
        total  = l_gan + self.lambda_l1 * l_l1 + self.lambda_ssim * l_ssim
        return total, {
            "G_cGAN" : l_gan.item(),
            "G_L1"   : l_l1.item(),
            "G_SSIM" : l_ssim.item(),
            "G_total": total.item(),
        }

    def discriminator_loss(self, real_pred, fake_pred):
        """
        Args:
            real_pred : D output on real  (ground-truth) fused
            fake_pred : D output on generated fused (detached)
        """
        l_real = self.gan(real_pred,        is_real=True)
        l_fake = self.gan(fake_pred.detach(), is_real=False)
        total  = (l_real + l_fake) * 0.5
        return total, {
            "D_real" : l_real.item(),
            "D_fake" : l_fake.item(),
            "D_total": total.item(),
        }
