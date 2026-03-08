"""
TGFusion Visualisation Utilities
=================================
Saves comparison grids: [Input A | Input B | Fused (ours) | Target]
"""

import os
import torch
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def tensor_to_img(t):
    """(1, H, W) tensor in [-1,1] → (H, W) numpy in [0,1]."""
    img = t.squeeze(0).detach().cpu().float()
    img = (img + 1) / 2
    return img.clamp(0, 1).numpy()


def save_comparison_grid(img_a, img_b, fused, target, save_path, n=4):
    """
    Save a grid comparing input A, input B, fused output, and target.

    Args:
        img_a, img_b, fused, target : (B, 1, H, W) tensors
        save_path : output .png file path
        n         : number of samples to show
    """
    if not HAS_MPL:
        return

    n = min(n, img_a.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ['Input A (CT/MRI)', 'Input B (MRI/US)', 'TGFusion (Ours)', 'Target']
    cols = [img_a, img_b, fused, target]

    for row in range(n):
        for col_idx, (imgs, title) in enumerate(zip(cols, titles)):
            ax = axes[row, col_idx]
            ax.imshow(tensor_to_img(imgs[row]), cmap='gray', vmin=0, vmax=1)
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_single(tensor, save_path):
    """Save a single (1, H, W) or (H, W) tensor as a grayscale PNG."""
    if not HAS_MPL:
        return
    img = tensor_to_img(tensor) if tensor.dim() == 3 else tensor.numpy()
    plt.imsave(save_path, img, cmap='gray', vmin=0, vmax=1)


def plot_training_curves(log_path, out_path):
    """
    Parse train.log and plot loss + metric curves.
    Expects lines containing: SSIM=, PSNR=, G=, D=
    """
    if not HAS_MPL:
        return

    epochs, ssim, psnr, g_loss, d_loss = [], [], [], [], []

    with open(log_path) as f:
        for line in f:
            if 'Epoch' not in line:
                continue
            try:
                def _val(key):
                    start = line.index(f'{key}=') + len(key) + 1
                    end = line.index(' ', start) if ' ' in line[start:] else len(line)
                    # strip trailing non-numeric chars
                    val_str = line[start:end].rstrip('dBs\n')
                    return float(val_str)

                ep = int(line.split('[')[1].split('/')[0])
                epochs.append(ep)
                ssim.append(_val('SSIM'))
                psnr.append(_val('PSNR'))
                g_loss.append(_val('G'))
                d_loss.append(_val('D'))
            except (ValueError, IndexError):
                continue

    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, ssim, 'b-o', ms=3)
    axes[0].set_title('Validation SSIM')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('SSIM')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, psnr, 'g-o', ms=3)
    axes[1].set_title('Validation PSNR (dB)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, g_loss, 'r-o', ms=3, label='Generator')
    axes[2].plot(epochs, d_loss, 'k-o', ms=3, label='Discriminator')
    axes[2].set_title('Training Losses')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('TGFusion Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training curves saved → {out_path}")
