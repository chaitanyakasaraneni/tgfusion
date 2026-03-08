"""
TGFusion Training Script
=========================
Trains the Transformer-GAN hybrid for multi-modal medical image fusion
on the Harvard AANLIB dataset (CT-MRI and MRI-PET tasks).

Paper training config (Section IV-D):
  - Optimiser : Adam, β₁=0.5, β₂=0.999, lr=2×10⁻⁴
  - LR decay  : linear from epoch 50 → 100 to 0
  - Epochs    : 100
  - Batch size: 4
  - Input     : 256x256

Usage
------
  # Smoke test — no data needed:
  python scripts/train.py --dataset synthetic --epochs 5 --batch_size 2

  # Train CT-MRI (Table I):
  python scripts/train.py --dataset ct_mri --data_dir /path/to/aanlib --epochs 100

  # Train MRI-PET (Table II):
  python scripts/train.py --dataset mri_pet --data_dir /path/to/aanlib --epochs 100

  # Resume from checkpoint:
  python scripts/train.py --dataset ct_mri --data_dir /path/to/aanlib \\
                           --resume outputs/checkpoints/ct_mri/best.pt
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.tgfusion import TGFusion
from models.losses import TGFusionLoss
from data.dataset import build_dataloader
from utils.metrics import compute_all_metrics, MetricTracker
from utils.visualise import save_comparison_grid


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train TGFusion — IEEE EIT 2026")

    # Data
    p.add_argument('--dataset',    default='synthetic',
                   choices=['ct_mri', 'mri_pet', 'synthetic'],
                   help="'ct_mri' → Table I, 'mri_pet' → Table II")
    p.add_argument('--data_dir',   default=None,
                   help='Root of Harvard AANLIB download')
    p.add_argument('--img_size',   type=int, default=256)

    # Model
    p.add_argument('--embed_dim',  type=int, default=64,
                   help='Base channel width (paper: 64 → 18.4M params)')

    # Training (paper Section IV-D)
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr',         type=float, default=2e-4)
    p.add_argument('--beta1',      type=float, default=0.5)
    p.add_argument('--beta2',      type=float, default=0.999)
    p.add_argument('--decay_start',type=int, default=50,
                   help='Epoch at which LR linear decay begins')
    p.add_argument('--lambda_l1',  type=float, default=10.0)
    p.add_argument('--lambda_ssim',type=float, default=5.0)
    p.add_argument('--n_disc_upd', type=int, default=1,
                   help='Discriminator updates per generator update')

    # Misc
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--resume',     default=None, help='Path to checkpoint .pt')
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--vis_every',  type=int, default=5)
    p.add_argument('--output_dir', default='outputs')
    p.add_argument('--no_amp',     action='store_true',
                   help='Disable automatic mixed precision')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler — linear decay after decay_start (paper Section IV-D)
# ─────────────────────────────────────────────────────────────────────────────

def make_lr_lambda(args):
    def lr_lambda(epoch):
        if epoch < args.decay_start:
            return 1.0
        decay_epochs = max(args.epochs - args.decay_start, 1)
        return max(0.0, 1.0 - (epoch - args.decay_start) / decay_epochs)
    return lr_lambda


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):

    # ── Step 1: Output directories & logging ─────────────────────────────────
    out_dir  = Path(args.output_dir) / args.dataset
    ckpt_dir = out_dir / 'checkpoints'
    vis_dir  = out_dir / 'visuals'
    log_path = out_dir / 'train.log'
    for d in [ckpt_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger()
    log.info("=" * 60)
    log.info("  TGFusion Training — IEEE EIT 2026")
    log.info("=" * 60)
    log.info(f"[Step 1/6]  Output dir → {out_dir}")
    log.info(f"            Log file   → {log_path}")

    # ── Step 2: Device ───────────────────────────────────────────────────────
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available() and not args.no_amp
    log.info(f"[Step 2/6]  Device: {device}  |  "
             f"AMP (mixed precision): {'enabled' if use_amp else 'disabled'}")
    if device.type == 'cpu':
        log.info("            WARNING: training on CPU will be very slow. "
                 "GPU strongly recommended.")

    # ── Step 3: Data loaders ─────────────────────────────────────────────────
    log.info(f"[Step 3/6]  Loading dataset: '{args.dataset}'  "
             f"(img_size={args.img_size}, batch={args.batch_size})")
    train_loader = build_dataloader(args.dataset, args.data_dir, 'train',
                                    args.img_size, args.batch_size,
                                    args.num_workers)
    val_loader   = build_dataloader(args.dataset, args.data_dir, 'val',
                                    args.img_size, args.batch_size,
                                    args.num_workers)
    log.info(f"            Train: {len(train_loader.dataset)} pairs  "
             f"({len(train_loader)} batches)  |  "
             f"Val: {len(val_loader.dataset)} pairs  "
             f"({len(val_loader)} batches)")

    # ── Step 4: Model ────────────────────────────────────────────────────────
    log.info(f"[Step 4/6]  Building model  (embed_dim={args.embed_dim})")
    model  = TGFusion(img_size=args.img_size, embed_dim=args.embed_dim).to(device)
    n_gen  = sum(p.numel() for p in model.generator.parameters()) / 1e6
    n_disc = sum(p.numel() for p in model.discriminator.parameters()) / 1e6
    log.info(f"            Generator:     {n_gen:.2f}M params  "
             f"(paper target: 18.4M)")
    log.info(f"            Discriminator: {n_disc:.2f}M params")

    criterion = TGFusionLoss(lambda_l1=args.lambda_l1,
                              lambda_ssim=args.lambda_ssim)
    log.info(f"            Loss weights: λ_L1={args.lambda_l1}  "
             f"λ_SSIM={args.lambda_ssim}")

    # ── Step 5: Optimisers & schedulers ──────────────────────────────────────
    log.info(f"[Step 5/6]  Optimisers  Adam  "
             f"lr={args.lr}  β=({args.beta1}, {args.beta2})")
    log.info(f"            LR schedule: constant until epoch {args.decay_start}, "
             f"then linear decay → 0 by epoch {args.epochs}")

    opt_G   = optim.Adam(model.generator.parameters(),
                         lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D   = optim.Adam(model.discriminator.parameters(),
                         lr=args.lr, betas=(args.beta1, args.beta2))
    sched_G = optim.lr_scheduler.LambdaLR(opt_G, make_lr_lambda(args))
    sched_D = optim.lr_scheduler.LambdaLR(opt_D, make_lr_lambda(args))
    scaler  = GradScaler(enabled=use_amp)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_ssim   = 0.0
    if args.resume:
        log.info(f"            Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt['epoch'] + 1
        best_ssim   = ckpt.get('best_ssim', 0.0)
        log.info(f"            Resumed at epoch {ckpt['epoch']}  "
                 f"best_ssim={best_ssim:.4f}")

    # ── Step 6: Training loop ─────────────────────────────────────────────────
    log.info(f"[Step 6/6]  Starting training loop  "
             f"(epochs {start_epoch} → {args.epochs})")
    log.info("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0           = time.time()
        loss_tracker = MetricTracker()
        n_batches    = len(train_loader)

        log.info(f"  Epoch {epoch:03d}/{args.epochs}  ── train "
                 f"({n_batches} batches) ──────────────")

        for batch_idx, (img_a, img_b, target) in enumerate(train_loader, 1):
            img_a  = img_a.to(device)
            img_b  = img_b.to(device)
            target = target.to(device)

            # ── Discriminator update ─────────────────────────────────────────
            for _ in range(args.n_disc_upd):
                with autocast(enabled=use_amp):
                    fused     = model(img_a, img_b).detach()
                    real_pred = model.discriminate(img_a, img_b, target)
                    fake_pred = model.discriminate(img_a, img_b, fused)
                    d_loss, d_log = criterion.discriminator_loss(real_pred,
                                                                  fake_pred)
                opt_D.zero_grad()
                scaler.scale(d_loss).backward()
                scaler.step(opt_D)

            # ── Generator update ─────────────────────────────────────────────
            with autocast(enabled=use_amp):
                fused     = model(img_a, img_b)
                fake_pred = model.discriminate(img_a, img_b, fused)
                g_loss, g_log = criterion.generator_loss(fake_pred, fused,
                                                          target)
            opt_G.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(opt_G)
            scaler.update()

            loss_tracker.update({**d_log, **g_log})

            # Per-batch progress — log every 10% of epoch
            log_every = max(1, n_batches // 10)
            if batch_idx % log_every == 0 or batch_idx == n_batches:
                log.info(
                    f"    batch {batch_idx:04d}/{n_batches}  "
                    f"G={g_log['G_total']:.4f}  "
                    f"(cGAN={g_log['G_cGAN']:.4f}  "
                    f"L1={g_log['G_L1']:.4f}  "
                    f"SSIM={g_log['G_SSIM']:.4f})  "
                    f"D={d_log['D_total']:.4f}  "
                    f"lr={opt_G.param_groups[0]['lr']:.2e}"
                )

        sched_G.step()
        sched_D.step()

        # ── Validation ───────────────────────────────────────────────────────
        log.info(f"  Epoch {epoch:03d}/{args.epochs}  ── val "
                 f"({len(val_loader)} batches) ────────────────")
        model.eval()
        val_tracker = MetricTracker()
        with torch.no_grad():
            for i, (img_a, img_b, target) in enumerate(val_loader):
                img_a  = img_a.to(device)
                img_b  = img_b.to(device)
                target = target.to(device)
                fused  = model(img_a, img_b)
                val_tracker.update(
                    compute_all_metrics(fused, target, img_a, img_b)
                )
                if i == 0 and epoch % args.vis_every == 0:
                    grid_path = vis_dir / f'epoch_{epoch:03d}.png'
                    save_comparison_grid(img_a, img_b, fused, target, grid_path)
                    log.info(f"    visual grid saved → {grid_path}")

        val_avg  = val_tracker.averages()
        loss_avg = loss_tracker.averages()
        elapsed  = time.time() - t0

        # ── Epoch summary ────────────────────────────────────────────────────
        log.info(
            f"  Epoch {epoch:03d}/{args.epochs}  ── summary [{elapsed:.1f}s] "
            f"──────────────────"
        )
        log.info(
            f"    Train  G_total={loss_avg.get('G_total',0):.4f}  "
            f"D_total={loss_avg.get('D_total',0):.4f}"
        )
        log.info(
            f"    Val    SSIM={val_avg.get('SSIM',0):.4f}  "
            f"PSNR={val_avg.get('PSNR',0):.2f} dB  "
            f"MI={val_avg.get('MI',0):.4f}  "
            f"SF={val_avg.get('SF',0):.4f}"
        )

        # ── Checkpoints ──────────────────────────────────────────────────────
        ckpt_data = {
            'epoch'      : epoch,
            'model'      : model.state_dict(),
            'opt_G'      : opt_G.state_dict(),
            'opt_D'      : opt_D.state_dict(),
            'best_ssim'  : best_ssim,
            'val_metrics': val_avg,
            'args'       : vars(args),
        }

        if val_avg.get('SSIM', 0) > best_ssim:
            best_ssim = val_avg['SSIM']
            ckpt_data['best_ssim'] = best_ssim
            torch.save(ckpt_data, ckpt_dir / 'best.pt')
            log.info(f"    ✓ New best SSIM={best_ssim:.4f}  → saved best.pt")

        if epoch % args.save_every == 0:
            ckpt_file = ckpt_dir / f'epoch_{epoch:03d}.pt'
            torch.save(ckpt_data, ckpt_file)
            log.info(f"    ✓ Periodic checkpoint → {ckpt_file.name}")

        if epoch == args.decay_start:
            log.info(f"    ↓ LR decay started (epoch {epoch}  "
                     f"lr={opt_G.param_groups[0]['lr']:.2e})")

        log.info("-" * 60)

    # ── Done ─────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Training complete.")
    log.info(f"  Best val SSIM : {best_ssim:.4f}")
    log.info(f"  Checkpoints   : {ckpt_dir}")
    log.info(f"  Log file      : {log_path}")
    log.info("=" * 60)


if __name__ == '__main__':
    train(parse_args())
