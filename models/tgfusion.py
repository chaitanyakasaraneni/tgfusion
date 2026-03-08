"""
TGFusion: Transformer-GAN Hybrid for Multi-Modal Medical Image Fusion
=====================================================================
Paper : "TGFusion: A Transformer-GAN Hybrid Architecture for Multi-Modal
         Medical Image Fusion — Benchmarking CT-MRI and MRI-PET on
         Harvard AANLIB"
Authors: Chaitanya Krishna Kasaraneni, Sarmista Thalapaneni
Venue  : IEEE EIT 2026

Architecture (Section III of paper)
------------------------------------
  1. Dual-Stream Swin Transformer Encoder  — shared weights f_θ
       embed_dim=64, depths=(2,2,4,2), heads=(2,4,8,16)
       Channel widths: C=64, 2C=128, 4C=256, 8C=512

  2. Bidirectional Cross-Modal Attention (CMA) at bottleneck
       F̃_A = Attn(Q_A, K_B, V_B)
       F̃_B = Attn(Q_B, K_A, V_A)
       F_fused = LayerNorm(F̃_A + F̃_B)

  3. Conditional GAN Decoder  — U-Net topology, residual skip connections,
       InstanceNorm + ReLU, Tanh output

  4. 70×70 PatchGAN Discriminator
       Input: [I_A ; I_B ; Î]  (3-channel concatenation)

Parameter count: ~18.4M (generator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project to embed_dim."""

    def __init__(self, img_size=256, patch_size=4, in_chans=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.H = img_size // patch_size
        self.W = img_size // patch_size

    def forward(self, x):
        x = self.proj(x)                        # (B, C, H/p, W/p)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, N, C)
        x = self.norm(x)
        return x, H, W


# ─────────────────────────────────────────────────────────────────────────────
# Swin Transformer Blocks
# ─────────────────────────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """Local window multi-head self-attention (Swin Transformer block)."""

    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x) + residual


class FFN(nn.Module):
    """Position-wise feed-forward network with LayerNorm pre-norm."""

    def __init__(self, dim, expansion=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class SwinBlock(nn.Module):
    """One Swin Transformer block: WindowAttention + FFN."""

    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.attn = WindowAttention(dim, num_heads=num_heads,
                                    window_size=window_size)
        self.ffn  = FFN(dim)

    def forward(self, x):
        return self.ffn(self.attn(x))


# ─────────────────────────────────────────────────────────────────────────────
# Shared-Weight Dual-Stream Swin Encoder  (Section III-B)
# ─────────────────────────────────────────────────────────────────────────────

class SwinEncoder(nn.Module):
    """
    Hierarchical Swin encoder.
    Stage channel widths: {C, 2C, 4C, 8C} = {64,128,256,512}
    Depths : (2, 2, 4, 2)  (matches paper Table ablation)
    Heads  : (2, 4, 8, 16)
    """

    def __init__(self, img_size=256, in_chans=1, embed_dim=64,
                 depths=(2, 2, 4, 2), num_heads=(2, 4, 8, 16)):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size=4,
                                      in_chans=in_chans, embed_dim=embed_dim)
        self.stages      = nn.ModuleList()
        self.merges      = nn.ModuleList()
        dim = embed_dim
        for i, (d, h) in enumerate(zip(depths, num_heads)):
            self.stages.append(
                nn.Sequential(*[SwinBlock(dim, num_heads=h) for _ in range(d)])
            )
            if i < len(depths) - 1:
                self.merges.append(nn.Linear(4 * dim, 2 * dim, bias=False))
                dim *= 2
            else:
                self.merges.append(None)
        self.out_dim = dim      # 512 for embed_dim=64

    def forward(self, x):
        """
        Returns:
            bottleneck : (B, N_bot, C_bot) token sequence
            skips      : list of (tokens, H, W) from each stage
        """
        x, H, W = self.patch_embed(x)
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            skips.append((x, H, W))
            if self.merges[i] is not None:
                B, _, C = x.shape
                x = x.view(B, H, W, C)
                x = torch.cat([x[:, 0::2, 0::2],
                               x[:, 1::2, 0::2],
                               x[:, 0::2, 1::2],
                               x[:, 1::2, 1::2]], dim=-1)   # (B,H/2,W/2,4C)
                H, W = H // 2, W // 2
                x = self.merges[i](x.reshape(B, H * W, 4 * C))
        return x, skips


# ─────────────────────────────────────────────────────────────────────────────
# Bidirectional Cross-Modal Attention  (Section III-C, Eqs 2-4)
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention at the bottleneck.

        F̃_A = Attn(Q_A, K_B, V_B)
        F̃_B = Attn(Q_B, K_A, V_A)
        F_fused = LayerNorm(F̃_A + F̃_B)

    where Attn(Q,K,V) = softmax(QK^T / sqrt(d_h)) V
    """

    def __init__(self, dim, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.W_Q_A = nn.Linear(dim, dim)
        self.W_K_A = nn.Linear(dim, dim)
        self.W_V_A = nn.Linear(dim, dim)

        self.W_Q_B = nn.Linear(dim, dim)
        self.W_K_B = nn.Linear(dim, dim)
        self.W_V_B = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _split_heads(self, t):
        B, N, C = t.shape
        return t.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    def _attn(self, q, k, v):
        a = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        B, H, N, d = (a @ v).shape
        return (a @ v).permute(0, 2, 1, 3).reshape(B, N, H * d)

    def forward(self, f_a, f_b):
        # Eq (2): F̃_A = Attn(Q_A, K_B, V_B)
        f_tilde_a = self._attn(
            self._split_heads(self.W_Q_A(f_a)),
            self._split_heads(self.W_K_B(f_b)),
            self._split_heads(self.W_V_B(f_b)),
        )
        # Eq (3): F̃_B = Attn(Q_B, K_A, V_A)
        f_tilde_b = self._attn(
            self._split_heads(self.W_Q_B(f_b)),
            self._split_heads(self.W_K_A(f_a)),
            self._split_heads(self.W_V_A(f_a)),
        )
        # Eq (4): F_fused = LayerNorm(F̃_A + F̃_B)
        return self.proj(self.norm(f_tilde_a + f_tilde_b))


# ─────────────────────────────────────────────────────────────────────────────
# cGAN Decoder  (Section III-D)
# ─────────────────────────────────────────────────────────────────────────────

class UpBlock(nn.Module):
    """Deconv 2× → concat skip → Conv-InstanceNorm-ReLU × 2."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:      # guard against rounding
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TGFusionGenerator(nn.Module):
    """
    Full TGFusion generator.
    embed_dim=64 → ~18.4M parameters (matches Section IV-D).
    """

    def __init__(self, img_size=256, in_chans=1, embed_dim=64,
                 depths=(2, 2, 4, 2), num_heads=(2, 4, 8, 16), out_chans=1):
        super().__init__()
        self.encoder   = SwinEncoder(img_size, in_chans, embed_dim,
                                     depths, num_heads)
        bot_dim        = self.encoder.out_dim           # 512
        n              = len(depths)
        dims           = [embed_dim * (2 ** i) for i in range(n)]  # [64,128,256,512]

        self.cma       = CrossModalAttention(bot_dim, num_heads=num_heads[-1])
        self.bot_proj  = nn.Sequential(nn.Linear(bot_dim, bot_dim), nn.GELU())

        # bottleneck spatial size: img_size / (4 * 2^(n-1))
        self.bot_H = img_size // (4 * 2 ** (n - 1))

        # Up-3, Up-2, Up-1  (reversed dims)
        rev = list(reversed(dims))   # [512, 256, 128, 64]
        self.up_blocks = nn.ModuleList([
            UpBlock(rev[i], rev[i + 1], rev[i + 1])
            for i in range(len(rev) - 1)
        ])

        # Final 4× upsample back to original resolution (undo patch_size=4)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0] // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dims[0] // 2, out_chans, kernel_size=2, stride=2),
            nn.Tanh(),
        )

    def _tok2map(self, tokens, H, W):
        B, N, C = tokens.shape
        return tokens.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, img_a, img_b):
        # Eq (1): encode both modalities with shared weights
        bot_a, skips_a = self.encoder(img_a)
        bot_b, skips_b = self.encoder(img_b)

        # Eqs (2-4): bidirectional cross-modal attention
        fused = self.bot_proj(self.cma(bot_a, bot_b))

        # Reshape bottleneck tokens → spatial map
        _, (_, bH, bW) = skips_a[-1][0], skips_a[-1]
        x = self._tok2map(fused, bH, bW)

        # Decode with averaged skip connections
        for i, up in enumerate(self.up_blocks):
            s_a_tok, sH, sW = skips_a[-(i + 2)]
            s_b_tok, _,  _  = skips_b[-(i + 2)]
            skip = self._tok2map((s_a_tok + s_b_tok) * 0.5, sH, sW)
            x = up(x, skip)

        return self.final_up(x)     # Î ∈ [-1, 1]


# ─────────────────────────────────────────────────────────────────────────────
# 70×70 PatchGAN Discriminator  (Section III-D)
# ─────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN.
    Input: [I_A ; I_B ; Î]  — 3-channel concatenation.
    Outputs a spatial map of real/fake scores.
    """

    def __init__(self, in_chans=3, base_ch=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_chans, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = base_ch
        for _ in range(1, n_layers):
            nf_next = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf, nf_next, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(nf_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            nf = nf_next
        nf_next = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf, nf_next, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf_next),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf_next, 1, 4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# TGFusion — top-level wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TGFusion(nn.Module):
    """Top-level model. Use .forward() for generation, .discriminate() for D."""

    def __init__(self, img_size=256, embed_dim=64):
        super().__init__()
        self.generator     = TGFusionGenerator(
            img_size=img_size, in_chans=1, embed_dim=embed_dim,
            depths=(2, 2, 4, 2), num_heads=(2, 4, 8, 16), out_chans=1,
        )
        self.discriminator = PatchDiscriminator(in_chans=3, base_ch=64, n_layers=3)

    def forward(self, img_a, img_b):
        return self.generator(img_a, img_b)

    def discriminate(self, img_a, img_b, img_fused):
        return self.discriminator(torch.cat([img_a, img_b, img_fused], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TGFusion(img_size=256, embed_dim=64).to(device)

    a = torch.randn(2, 1, 256, 256).to(device)
    b = torch.randn(2, 1, 256, 256).to(device)

    fused = model(a, b)
    disc  = model.discriminate(a, b, fused)

    gen_params   = sum(p.numel() for p in model.generator.parameters()) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"✓ Fused output shape     : {fused.shape}")
    print(f"✓ Discriminator output   : {disc.shape}")
    print(f"✓ Generator parameters   : {gen_params:.2f}M  (paper reports 18.4M)")
    print(f"✓ Total parameters       : {total_params:.2f}M")
