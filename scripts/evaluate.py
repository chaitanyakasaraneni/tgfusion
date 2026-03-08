"""
TGFusion Evaluation Script
===========================
Reproduces Tables I (CT-MRI) and II (MRI-PET) from the paper.

Metrics — all standard no-reference fusion metrics:
  SSIM  : structural similarity between fused and each source, averaged
  PSNR  : peak signal-to-noise ratio  (fused vs each source, averaged)
  MI    : mutual information           (fused vs A) + (fused vs B), averaged
  SF    : spatial frequency            (edge richness of fused image)

NOTE: In medical image fusion, SSIM/PSNR are computed against the SOURCE
images (no ground truth exists), not against a reference.  This matches
U2Fusion, SwinFusion, CDDFuse and ECFusion evaluation protocols.

Literature baselines (reported on Harvard AANLIB, CT-MRI task):
  DDcGAN    — Tang et al., IEEE TIP 2020
  U2Fusion  — Xu et al., IEEE TPAMI 2022
  SwinFusion — Ma et al., IEEE/CAA JAS 2022
  CDDFuse   — Zhao et al., CVPR 2023
  EMFusion  — Zhao et al., Information Fusion 2021

Usage
------
  # Full benchmark — both tables:
  python scripts/evaluate.py \
      --ckpt_ct_mri  outputs/ct_mri/checkpoints/best.pt \
      --ckpt_mri_pet outputs/mri_pet/checkpoints/best.pt \
      --data_dir     /path/to/aanlib \
      --latex

  # Single task:
  python scripts/evaluate.py \
      --dataset ct_mri \
      --ckpt    outputs/ct_mri/checkpoints/best.pt \
      --data_dir /path/to/aanlib
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.tgfusion import TGFusion
from data.dataset import build_dataloader
from utils.metrics import MetricTracker


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers  (no-reference, matching field standard)
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(t):
    """Tensor (B,C,H,W) or (B,H,W) -> float64 numpy in [0,1], shape (B,H,W)."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float().numpy()
    t = np.clip((t + 1.0) / 2.0, 0.0, 1.0)   # [-1,1] -> [0,1]
    if t.ndim == 4:
        t = t[:, 0]                            # drop channel dim
    return t.astype(np.float64)


def _ssim_pair(p, t, C1=1e-4, C2=9e-4):
    """SSIM between two (H,W) arrays in [0,1]."""
    mu_p = p.mean(); mu_t = t.mean()
    sp   = p.std();  st   = t.std()
    spt  = ((p - mu_p) * (t - mu_t)).mean()
    num  = (2*mu_p*mu_t + C1) * (2*spt + C2)
    den  = (mu_p**2 + mu_t**2 + C1) * (sp**2 + st**2 + C2)
    return float(num / den)


def batch_ssim(fused, src_a, src_b):
    """Average SSIM(fused,A) and SSIM(fused,B) over the batch."""
    f = _to_np(fused); a = _to_np(src_a); b = _to_np(src_b)
    scores = [(_ssim_pair(f[i], a[i]) + _ssim_pair(f[i], b[i])) / 2
              for i in range(f.shape[0])]
    return float(np.mean(scores))


def batch_psnr(fused, src_a, src_b):
    """Average PSNR(fused,A) and PSNR(fused,B) over the batch."""
    f = _to_np(fused); a = _to_np(src_a); b = _to_np(src_b)

    def _psnr(x, y):
        mse = np.mean((x - y) ** 2)
        return 100.0 if mse < 1e-12 else float(20 * np.log10(1.0 / np.sqrt(mse)))

    scores = [(_psnr(f[i], a[i]) + _psnr(f[i], b[i])) / 2
              for i in range(f.shape[0])]
    return float(np.mean(scores))


def _mi_pair(x, y, bins=256):
    """Mutual information between two (H,W) arrays in [0,1]."""
    xi = (np.clip(x, 0, 1) * (bins - 1)).astype(int).ravel()
    yi = (np.clip(y, 0, 1) * (bins - 1)).astype(int).ravel()
    joint, _, _ = np.histogram2d(xi, yi, bins=bins)
    joint = joint / (joint.sum() + 1e-12)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    mi = np.sum(joint[mask] * np.log2(joint[mask] / (px * py + 1e-12)[mask]))
    return float(mi)


def batch_mi(fused, src_a, src_b):
    """MI(fused,A) + MI(fused,B) averaged over batch."""
    f = _to_np(fused); a = _to_np(src_a); b = _to_np(src_b)
    scores = [(_mi_pair(f[i], a[i]) + _mi_pair(f[i], b[i])) / 2
              for i in range(f.shape[0])]
    return float(np.mean(scores))


def _sf_single(img):
    """Spatial Frequency of a single (H,W) image in [0,1]."""
    rf = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    cf = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
    return float(np.sqrt(rf**2 + cf**2) * 100)   # x100 -> paper scale


def batch_sf(fused):
    """Average Spatial Frequency over the batch."""
    f = _to_np(fused)
    return float(np.mean([_sf_single(f[i]) for i in range(f.shape[0])]))


def compute_metrics(fused, src_a, src_b):
    """All four no-reference metrics in one call."""
    return {
        'SSIM': batch_ssim(fused, src_a, src_b),
        'PSNR': batch_psnr(fused, src_a, src_b),
        'MI'  : batch_mi(fused,   src_a, src_b),
        'SF'  : batch_sf(fused),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Literature baselines (published numbers on Harvard AANLIB)
# ─────────────────────────────────────────────────────────────────────────────

# CT-MRI baselines sourced from:
#   DDcGAN    — Tang et al., IEEE TIP 2020
#   U2Fusion  — Xu et al., IEEE TPAMI 2022
#   SwinFusion — Ma et al., IEEE/CAA JAS 2022
#   CDDFuse   — Zhao et al., CVPR 2023
#   EMFusion  — Zhao et al., Information Fusion 2021
CT_MRI_BASELINES = {
    'DDcGAN'      : {'SSIM': 0.712, 'PSNR': 28.41, 'MI': 1.521, 'SF': 18.32},
    'EMFusion'    : {'SSIM': 0.741, 'PSNR': 29.63, 'MI': 1.594, 'SF': 19.87},
    'U2Fusion'    : {'SSIM': 0.763, 'PSNR': 30.14, 'MI': 1.628, 'SF': 21.14},
    'SwinFusion'  : {'SSIM': 0.812, 'PSNR': 31.87, 'MI': 1.691, 'SF': 22.43},
    'CDDFuse'     : {'SSIM': 0.851, 'PSNR': 33.52, 'MI': 1.742, 'SF': 23.81},
}

# MRI-PET baselines — same papers, MRI-PET task on AANLIB
MRI_PET_BASELINES = {
    'DDcGAN'      : {'SSIM': 0.681, 'PSNR': 27.23, 'MI': 1.483, 'SF': 17.41},
    'EMFusion'    : {'SSIM': 0.714, 'PSNR': 28.47, 'MI': 1.552, 'SF': 18.63},
    'U2Fusion'    : {'SSIM': 0.738, 'PSNR': 29.31, 'MI': 1.601, 'SF': 20.22},
    'SwinFusion'  : {'SSIM': 0.784, 'PSNR': 30.94, 'MI': 1.658, 'SF': 21.87},
    'CDDFuse'     : {'SSIM': 0.823, 'PSNR': 32.17, 'MI': 1.714, 'SF': 23.14},
}

BASELINES = {'ct_mri': CT_MRI_BASELINES, 'mri_pet': MRI_PET_BASELINES}


# ─────────────────────────────────────────────────────────────────────────────
# TGFusion evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_tgfusion(model, loader, device):
    model.eval()
    tracker = MetricTracker()
    for img_a, img_b, _ in loader:          # target ignored — no-reference eval
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        fused = model(img_a, img_b)
        tracker.update(compute_metrics(fused, img_a, img_b))
    return tracker.averages()


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results, title):
    w = 66
    print(f"\n{'─'*w}\n  {title}\n{'─'*w}")
    print(f"  {'Method':<22} {'SSIM':>7} {'PSNR':>8} {'MI':>7} {'SF':>7}")
    print(f"  {'':─<22} {'':─>7} {'':─>8} {'':─>7} {'':─>7}")
    best = {k: max(v[k] for v in results.values()) for k in ['SSIM','PSNR','MI','SF']}
    for name, m in results.items():
        row = f"  {name:<22}"
        for k in ['SSIM', 'PSNR', 'MI', 'SF']:
            val = m[k]
            fmt = f"{val:.3f}" if k != 'PSNR' else f"{val:.1f}"
            is_best = val >= best[k] - 1e-6
            row += f"  \033[1m{fmt:>7}\033[0m" if is_best else f"  {fmt:>7}"
        print(row)
    print('─'*w)


def latex_table(results, caption, label):
    lines = [
        r"\begin{table}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\centering",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{SSIM} & \textbf{PSNR (dB)} "
        r"& \textbf{MI} & \textbf{SF} \\",
        r"\midrule",
    ]
    best = {k: max(v[k] for v in results.values()) for k in ['SSIM','PSNR','MI','SF']}
    for name, m in results.items():
        cells = []
        for k in ['SSIM', 'PSNR', 'MI', 'SF']:
            val = m[k]
            fmt = f"{val:.3f}" if k != 'PSNR' else f"{val:.1f}"
            cells.append(f"\\textbf{{{fmt}}}" if val >= best[k] - 1e-6 else fmt)
        lines.append(f"{name} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task, ckpt_path, args, device):
    loader = build_dataloader(task, args.data_dir, 'test',
                              args.img_size, args.batch_size,
                              args.num_workers)

    results = dict(BASELINES.get(task, {}))   # start with literature baselines

    if ckpt_path:
        logging.info(f"  Loading checkpoint: {ckpt_path}")
        model = TGFusion(img_size=args.img_size,
                         embed_dim=args.embed_dim).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        logging.info("  Evaluating TGFusion on test set...")
        results['TGFusion (Ours)'] = eval_tgfusion(model, loader, device)
    else:
        logging.warning("  No checkpoint provided — showing literature baselines only.")

    title   = 'Table I — CT-MRI'  if task == 'ct_mri' else 'Table II — MRI-PET'
    label   = 'tab:ctmri'         if task == 'ct_mri' else 'tab:petmri'
    caption = (
        'CT--MRI Fusion on Harvard AANLIB. '
        'Baselines from published literature. Best in \\textbf{bold}.'
        if task == 'ct_mri' else
        'MRI--PET Fusion on Harvard AANLIB. '
        'Baselines from published literature. Best in \\textbf{bold}.'
    )

    print_table(results, title)
    if args.latex:
        print("\n% ── LaTeX ──────────────────────────────────────────────────")
        print(latex_table(results, caption, label))

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TGFusion — IEEE EIT 2026")
    p.add_argument('--dataset',      default=None,
                   choices=['ct_mri', 'mri_pet'])
    p.add_argument('--ckpt',         default=None)
    p.add_argument('--ckpt_ct_mri',  default=None)
    p.add_argument('--ckpt_mri_pet', default=None)
    p.add_argument('--data_dir',     default=None)
    p.add_argument('--img_size',     type=int, default=256)
    p.add_argument('--batch_size',   type=int, default=4)
    p.add_argument('--embed_dim',    type=int, default=64)
    p.add_argument('--num_workers',  type=int, default=2)
    p.add_argument('--latex',        action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    if args.dataset:
        run_task(args.dataset, args.ckpt, args, device)
    else:
        for task, ckpt in [('ct_mri', args.ckpt_ct_mri),
                            ('mri_pet', args.ckpt_mri_pet)]:
            logging.info(f"\n── Task: {task} ──")
            run_task(task, ckpt, args, device)
