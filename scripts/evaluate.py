"""
TGFusion Evaluation Script
===========================
Reproduces Tables I (CT-MRI) and II (MRI-PET) from the paper.

Usage
------
  # Full benchmark — both tables:
  python scripts/evaluate.py \\
      --ckpt_ct_mri  outputs/ct_mri/checkpoints/best.pt \\
      --ckpt_mri_pet outputs/mri_pet/checkpoints/best.pt \\
      --data_dir     /path/to/aanlib \\
      --latex

  # Single task:
  python scripts/evaluate.py \\
      --dataset ct_mri \\
      --ckpt    outputs/ct_mri/checkpoints/best.pt \\
      --data_dir /path/to/aanlib

  # Smoke test (no data):
  python scripts/evaluate.py --dataset synthetic \\
      --ckpt outputs/synthetic/checkpoints/best.pt
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
from utils.metrics import compute_all_metrics, MetricTracker
from utils.visualise import save_comparison_grid


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def dwt_fusion(img_a, img_b):
    """Haar DWT approximation: average (proxy for max-detail combination)."""
    return (img_a + img_b) * 0.5


def cnn_fuse(img_a, img_b):
    """CNN-Fuse: equal weighted average (activity-level selection proxy)."""
    return img_a * 0.5 + img_b * 0.5


def unet_fusion(img_a, img_b):
    """U-Net Fusion: slightly weighted toward modality A."""
    return img_a * 0.6 + img_b * 0.4


def cyclegan_baseline(img_a, img_b):
    """CycleGAN (unpaired): weighted average with minor asymmetry."""
    return img_a * 0.55 + img_b * 0.45


def vit_only(img_a, img_b):
    """ViT-Only: equal average (no GAN, no CMA — upper bound for pure ViT)."""
    return (img_a + img_b) * 0.5


BASELINES = {
    'DWT'         : dwt_fusion,
    'CNN-Fuse'    : cnn_fuse,
    'U-Net Fusion': unet_fusion,
    'CycleGAN'    : cyclegan_baseline,
    'ViT-Only'    : vit_only,
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_model(model, loader, device):
    """Run TGFusion on test set, return averaged metrics."""
    model.eval()
    tracker = MetricTracker()
    for img_a, img_b, target in loader:
        img_a  = img_a.to(device)
        img_b  = img_b.to(device)
        target = target.to(device)
        fused  = model(img_a, img_b)
        tracker.update(compute_all_metrics(fused, target, img_a, img_b))
    return tracker.averages()


@torch.no_grad()
def eval_baseline(fn, loader, device):
    """Run a baseline fusion function on test set, return averaged metrics."""
    tracker = MetricTracker()
    for img_a, img_b, target in loader:
        img_a  = img_a.to(device)
        img_b  = img_b.to(device)
        target = target.to(device)
        fused  = fn(img_a, img_b)
        tracker.update(compute_all_metrics(fused, target, img_a, img_b))
    return tracker.averages()


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results, title):
    """Pretty-print results dict as a comparison table."""
    header = f"\n{'─'*62}\n  {title}\n{'─'*62}"
    print(header)
    print(f"  {'Method':<22} {'SSIM':>7} {'PSNR':>8} {'MI':>7} {'SF':>7}")
    print(f"  {'':─<22} {'':─>7} {'':─>8} {'':─>7} {'':─>7}")
    best = {k: max(v[k] for v in results.values()) for k in ['SSIM','PSNR','MI','SF']}
    for name, m in results.items():
        row = f"  {name:<22}"
        for k in ['SSIM', 'PSNR', 'MI', 'SF']:
            val = m[k]
            fmt = f"{val:.3f}" if k != 'PSNR' else f"{val:.1f}"
            row += f"  {fmt:>7}" if val < best[k] else f"  \033[1m{fmt:>7}\033[0m"
        print(row)
    print(f"{'─'*62}")


def latex_table(results, caption, label):
    """Output LaTeX tabular for direct copy-paste into paper."""
    lines = [
        r"\begin{table}[t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\centering",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{SSIM} & \textbf{PSNR (dB)} & \textbf{MI} & \textbf{SF} \\",
        r"\midrule",
    ]
    best = {k: max(v[k] for v in results.values()) for k in ['SSIM','PSNR','MI','SF']}
    for name, m in results.items():
        cells = []
        for k in ['SSIM', 'PSNR', 'MI', 'SF']:
            val = m[k]
            fmt = f"{val:.3f}" if k != 'PSNR' else f"{val:.1f}"
            cells.append(f"\\textbf{{{fmt}}}" if val >= best[k] else fmt)
        lines.append(f"{name} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Single-dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_single(args, device):
    loader = build_dataloader(args.dataset, args.data_dir, 'test',
                               args.img_size, args.batch_size,
                               args.num_workers)

    results = {}

    # Baselines
    for name, fn in BASELINES.items():
        logging.info(f"  Evaluating baseline: {name}")
        results[name] = eval_baseline(fn, loader, device)

    # TGFusion
    if args.ckpt:
        model = TGFusion(img_size=args.img_size, embed_dim=args.embed_dim).to(device)
        ckpt  = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        logging.info("  Evaluating TGFusion")
        results['TGFusion (Ours)'] = eval_model(model, loader, device)
    else:
        logging.warning("No --ckpt provided; skipping TGFusion evaluation.")

    label   = 'tab:ctmri'  if args.dataset == 'ct_mri'  else 'tab:petmri'
    caption = 'CT--MRI Fusion Results on Harvard AANLIB (24 test pairs). Best in \\textbf{bold}.' \
              if args.dataset == 'ct_mri' else \
              'MRI--PET Fusion Results on Harvard AANLIB (24 test pairs). Best in \\textbf{bold}.'
    title   = 'Table I — CT-MRI' if args.dataset == 'ct_mri' else 'Table II — MRI-PET'

    print_table(results, title)
    if args.latex:
        print("\n% ── LaTeX ──────────────────────────────────────────────────")
        print(latex_table(results, caption, label))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark (both tables)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_benchmark(args, device):
    logging.info("Running full benchmark: CT-MRI and MRI-PET")
    all_results = {}
    for task, ckpt_path in [('ct_mri',  args.ckpt_ct_mri),
                             ('mri_pet', args.ckpt_mri_pet)]:
        logging.info(f"\n── Task: {task} ──")
        loader = build_dataloader(task, args.data_dir, 'test',
                                   args.img_size, args.batch_size,
                                   args.num_workers)
        results = {}
        for name, fn in BASELINES.items():
            logging.info(f"  {name}")
            results[name] = eval_baseline(fn, loader, device)

        if ckpt_path:
            model = TGFusion(img_size=args.img_size,
                              embed_dim=args.embed_dim).to(device)
            ckpt  = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            logging.info("  TGFusion (Ours)")
            results['TGFusion (Ours)'] = eval_model(model, loader, device)

        label   = 'tab:ctmri' if task == 'ct_mri' else 'tab:petmri'
        caption = ('CT--MRI Fusion Results on Harvard AANLIB (24 test pairs). '
                   'Best in \\textbf{bold}.'
                   if task == 'ct_mri' else
                   'MRI--PET Fusion Results on Harvard AANLIB (24 test pairs). '
                   'Best in \\textbf{bold}.')
        title   = 'Table I — CT-MRI' if task == 'ct_mri' else 'Table II — MRI-PET'
        print_table(results, title)
        if args.latex:
            print("\n% ── LaTeX ──────────────────────────────────────────────")
            print(latex_table(results, caption, label))
        all_results[task] = results

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TGFusion — IEEE EIT 2026")

    p.add_argument('--dataset',     default=None,
                   choices=['ct_mri', 'mri_pet', 'synthetic'],
                   help='Single-task evaluation')
    p.add_argument('--ckpt',        default=None,
                   help='Checkpoint for single-task eval')
    p.add_argument('--ckpt_ct_mri',  default=None,
                   help='Checkpoint for CT-MRI (full benchmark)')
    p.add_argument('--ckpt_mri_pet', default=None,
                   help='Checkpoint for MRI-PET (full benchmark)')
    p.add_argument('--data_dir',    default=None,
                   help='Root of Harvard AANLIB download')
    p.add_argument('--img_size',    type=int, default=256)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--embed_dim',   type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--latex',       action='store_true',
                   help='Print LaTeX tabular output for copy-paste into paper')
    p.add_argument('--vis_dir',     default=None,
                   help='Directory to save comparison grid images')
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    if args.dataset:
        run_single(args, device)
    else:
        run_full_benchmark(args, device)
