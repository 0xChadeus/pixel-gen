"""Building blocks for the EDM UNet: ResNet blocks, attention, FiLM conditioning."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FourierFeatures(nn.Module):
    """Fourier feature embedding for noise level sigma (EDM style)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (B,) -> (B, dim)
        log_sigma = torch.log(sigma.clamp(min=1e-8))
        freqs = torch.arange(self.dim // 2, device=sigma.device, dtype=sigma.dtype)
        freqs = freqs * (4.0 * math.pi / (self.dim // 2))
        args = log_sigma.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.cos(), args.sin()], dim=-1)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale and shift features given conditioning."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        # x: (B, C, H, W), scale/shift: (B, C)
        return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    """ResNet block with FiLM conditioning and optional dropout."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # q,k,v: (B, heads, head_dim, N) -> rearrange for sdpa
        q = q.permute(0, 1, 3, 2)  # (B, heads, N, head_dim)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.out(h)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: spatial features attend to text token sequence."""

    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.GroupNorm(32, dim)
        self.norm_ctx = nn.LayerNorm(context_dim)
        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Linear(context_dim, dim * 2)
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial features
            context: (B, T, context_dim) text token embeddings
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, self.num_heads, self.head_dim, H * W)
        q = q.permute(0, 1, 3, 2)  # (B, heads, N, head_dim)

        ctx = self.norm_ctx(context)
        kv = self.kv(ctx).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]  # (B, T, heads, head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, heads, T, head_dim)
        v = v.permute(0, 2, 1, 3)

        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.out(h)


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
