"""Conditioning modules: CLIP text, palette transformer, timestep, resolution."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import FourierFeatures


class TimestepConditioner(nn.Module):
    """EDM-style noise level conditioning via Fourier features + MLP."""

    def __init__(self, cond_dim: int, fourier_dim: int = 256):
        super().__init__()
        self.fourier = FourierFeatures(fourier_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma: (B,) -> (B, cond_dim)"""
        return self.mlp(self.fourier(sigma))


class PaletteConditioner(nn.Module):
    """Encode a palette of up to max_colors OKLab colors into a conditioning vector.

    Uses a small transformer over palette color tokens, then mean pools.
    """

    def __init__(self, max_colors: int = 32, color_dim: int = 3,
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_heads: int = 4, cond_dim: int = 512):
        super().__init__()
        self.max_colors = max_colors
        self.color_proj = nn.Linear(color_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_colors, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, cond_dim)

        # Learned null embedding for classifier-free guidance
        self.null_embedding = nn.Parameter(torch.randn(1, cond_dim) * 0.02)

    def forward(self, palette: torch.Tensor, mask: torch.Tensor | None = None,
                force_null: bool = False) -> torch.Tensor:
        """
        Args:
            palette: (B, N, 3) OKLab colors, N <= max_colors
            mask: (B, N) bool, True for valid colors, False for padding
            force_null: If True, return null embedding (for CFG unconditional)

        Returns:
            (B, cond_dim) conditioning vector
        """
        if force_null:
            return self.null_embedding.expand(palette.shape[0], -1)

        B, N, _ = palette.shape
        # Pad to max_colors if needed
        if N < self.max_colors:
            padding = torch.zeros(B, self.max_colors - N, 3, device=palette.device)
            palette = torch.cat([palette, padding], dim=1)
            if mask is not None:
                mask_pad = torch.zeros(B, self.max_colors - N, dtype=torch.bool, device=mask.device)
                mask = torch.cat([mask, mask_pad], dim=1)

        x = self.color_proj(palette) + self.pos_embed[:, :palette.shape[1]]

        # Transformer expects src_key_padding_mask: True = ignore
        pad_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Mean pool over valid tokens
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)  # (B, N, 1)
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.out_proj(x)


class ResolutionConditioner(nn.Module):
    """Learnable embedding per supported resolution."""

    def __init__(self, supported_sizes: list[int], cond_dim: int):
        super().__init__()
        self.size_to_idx = {s: i for i, s in enumerate(supported_sizes)}
        self.embed = nn.Embedding(len(supported_sizes), cond_dim)

    def forward(self, size: int, batch_size: int, device: torch.device) -> torch.Tensor:
        idx = self.size_to_idx[size]
        idx_t = torch.full((batch_size,), idx, dtype=torch.long, device=device)
        return self.embed(idx_t)


class CLIPTextProjector(nn.Module):
    """Project CLIP pooled text embedding to conditioning space.

    Also projects token-level embeddings for cross-attention.
    """

    def __init__(self, clip_dim: int = 768, cond_dim: int = 512,
                 cross_attn_dim: int = 512):
        super().__init__()
        # Pooled embedding -> global conditioning
        self.global_proj = nn.Sequential(
            nn.Linear(clip_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # Token embeddings -> cross-attention keys/values
        self.token_proj = nn.Linear(clip_dim, cross_attn_dim)

        # Learned null embeddings for CFG
        self.null_global = nn.Parameter(torch.randn(1, cond_dim) * 0.02)
        self.null_tokens = nn.Parameter(torch.randn(1, 77, cross_attn_dim) * 0.02)

    def forward(self, pooled: torch.Tensor, tokens: torch.Tensor,
                force_null: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pooled: (B, clip_dim) CLIP pooled text embedding
            tokens: (B, 77, clip_dim) CLIP token embeddings
            force_null: Return null embeddings for CFG unconditional

        Returns:
            global_cond: (B, cond_dim)
            token_cond: (B, 77, cross_attn_dim)
        """
        if force_null:
            B = pooled.shape[0]
            return (
                self.null_global.expand(B, -1),
                self.null_tokens.expand(B, -1, -1),
            )
        return self.global_proj(pooled), self.token_proj(tokens)


class ConditioningAssembler(nn.Module):
    """Assembles all conditioning signals into a single FiLM vector + cross-attn tokens."""

    def __init__(self, cond_dim: int = 512, clip_dim: int = 768,
                 cross_attn_dim: int = 512, supported_sizes: list[int] | None = None):
        super().__init__()
        if supported_sizes is None:
            supported_sizes = [32, 64, 128]

        self.timestep_cond = TimestepConditioner(cond_dim)
        self.text_proj = CLIPTextProjector(clip_dim, cond_dim, cross_attn_dim)
        self.palette_cond = PaletteConditioner(cond_dim=cond_dim)
        self.resolution_cond = ResolutionConditioner(supported_sizes, cond_dim)

        # Merge: timestep + text_global + palette + resolution -> final cond
        self.merge = nn.Sequential(
            nn.Linear(cond_dim * 4, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

    def forward(
        self,
        sigma: torch.Tensor,
        text_pooled: torch.Tensor,
        text_tokens: torch.Tensor,
        palette: torch.Tensor,
        palette_mask: torch.Tensor | None,
        resolution: int,
        drop_text: bool = False,
        drop_palette: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cond: (B, cond_dim) FiLM conditioning vector
            cross_tokens: (B, 77, cross_attn_dim) for cross-attention
        """
        B = sigma.shape[0]
        device = sigma.device

        t_cond = self.timestep_cond(sigma)
        text_global, cross_tokens = self.text_proj(text_pooled, text_tokens, force_null=drop_text)
        pal_cond = self.palette_cond(palette, palette_mask, force_null=drop_palette)
        res_cond = self.resolution_cond(resolution, B, device)

        combined = torch.cat([t_cond, text_global, pal_cond, res_cond], dim=-1)
        cond = self.merge(combined)

        return cond, cross_tokens
