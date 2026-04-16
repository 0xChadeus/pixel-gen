"""Top-level inference pipeline: text prompt -> pixel art image."""

import numpy as np
import torch

from server.config import ServerConfig
from server.inference.clip_encoder import CLIPEncoder
from server.utils.color import (
    oklab_to_srgb_torch, denormalize_oklab, normalize_oklab, srgb_to_oklab_torch,
)
from server.utils.image_io import palette_hex_to_array
from server.postprocess.pipeline import postprocess
from model.unet import EDMUNet
from model.conditioning import ConditioningAssembler
from model.diffusion import EDMPrecond, HeunSampler

from server.utils.color import srgb_to_oklab


class InferencePipeline:
    """Orchestrates the full generation pipeline."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load CLIP text encoder
        self.clip = CLIPEncoder(config.clip_model, config.device)

        # Build model
        self.model = EDMUNet(
            in_channels=config.in_channels,
            out_channels=config.in_channels,
            base_channels=config.base_channels,
            channel_mults=config.channel_mults,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            num_heads=config.num_heads,
            cond_dim=config.cond_dim,
            cross_attn_dim=config.cross_attn_dim,
            dropout=config.dropout,
            self_condition=config.self_condition,
        ).to(self.device)

        self.cond_assembler = ConditioningAssembler(
            cond_dim=config.cond_dim,
            clip_dim=768,
            cross_attn_dim=config.cross_attn_dim,
        ).to(self.device)

        self.precond = EDMPrecond(self.model, sigma_data=config.sigma_data)

        self.sampler = HeunSampler()

        # Load checkpoint if provided
        if config.checkpoint_path:
            self.load_checkpoint(config.checkpoint_path)

        self.model.eval()
        self.cond_assembler.eval()

    def load_checkpoint(self, path: str):
        """Load model weights from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        if "cond_assembler" in ckpt:
            self.cond_assembler.load_state_dict(ckpt["cond_assembler"])
        if "ema_model" in ckpt:
            self.model.load_state_dict(ckpt["ema_model"])

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        palette_hex: list[str] | None = None,
        guidance_scale: float = 5.0,
        num_steps: int = 35,
        seed: int = -1,
        dither_mode: str | None = None,
        outline_cleanup: bool = True,
        num_colors: int = 16,
        init_image: np.ndarray | None = None,
        init_strength: float = 0.6,
        progress_callback=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a 128x128 pixel art sprite.

        Args:
            prompt: Text description
            palette_hex: List of hex color strings, or None for auto
            guidance_scale: CFG scale
            num_steps: Sampling steps
            seed: RNG seed (-1 for random)
            dither_mode: None, "ordered", or "floyd_steinberg"
            outline_cleanup: Whether to clean up outlines
            num_colors: Palette size for auto extraction
            init_image: Optional init image (H, W, 4) RGBA uint8
            init_strength: Init image strength (0=no change, 1=full generation)
            progress_callback: Called with (step, total)

        Returns:
            image: (128, 128, 4) RGBA uint8
            palette: (N, 3) RGB uint8 palette used
        """
        if seed >= 0:
            torch.manual_seed(seed)

        # Encode text
        text_pooled, text_tokens = self.clip.encode(prompt)
        # Unconditional embeddings for CFG
        uncond_pooled, uncond_tokens = self.clip.encode("")

        # Prepare palette conditioning
        if palette_hex:
            pal_rgb = palette_hex_to_array(palette_hex)
            pal_lab = srgb_to_oklab(pal_rgb).astype(np.float32)
            palette_t = torch.from_numpy(pal_lab).unsqueeze(0).to(self.device)
            palette_mask = torch.ones(1, len(palette_hex), dtype=torch.bool, device=self.device)
        else:
            palette_t = torch.zeros(1, 1, 3, device=self.device)
            palette_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        # Assemble conditioning (conditional + unconditional)
        cond, cross = self.cond_assembler(
            sigma=torch.ones(1, device=self.device),  # placeholder, actual sigma set per step
            text_pooled=text_pooled,
            text_tokens=text_tokens,
            palette=palette_t,
            palette_mask=palette_mask,
        )
        uncond_cond, uncond_cross = self.cond_assembler(
            sigma=torch.ones(1, device=self.device),
            text_pooled=uncond_pooled,
            text_tokens=uncond_tokens,
            palette=palette_t,
            palette_mask=palette_mask,
            drop_text=True,
            drop_palette=True,
        )

        shape = (1, 4, 128, 128)  # B, C, H, W

        # Sample with CFG
        def cfg_denoise(x, sigma_batch, x_self_cond=None):
            pred_c = self.precond(x, sigma_batch, cond, cross, x_self_cond)
            pred_u = self.precond(x, sigma_batch, uncond_cond, uncond_cross, x_self_cond)
            return pred_u + guidance_scale * (pred_c - pred_u)

        # Use the sampler
        sigmas = self.sampler.get_sigmas(num_steps, self.device)
        x = torch.randn(shape, device=self.device) * sigmas[0]
        x_self_cond = None

        for i in range(num_steps):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            sigma_batch = torch.full((1,), sigma_cur, device=self.device)
            denoised = cfg_denoise(x, sigma_batch, x_self_cond)

            if self.config.self_condition:
                x_self_cond = denoised.detach()

            d = (x - denoised) / sigma_cur
            x_next = x + d * (sigma_next - sigma_cur)

            if sigma_next > 0:
                sigma_batch_next = torch.full((1,), sigma_next, device=self.device)
                denoised_2 = cfg_denoise(x_next, sigma_batch_next, x_self_cond)
                d_2 = (x_next - denoised_2) / sigma_next
                x_next = x + (d + d_2) / 2 * (sigma_next - sigma_cur)

            x = x_next
            if progress_callback:
                progress_callback(i + 1, num_steps)

        # Convert from normalized OKLab to sRGB
        # x is (1, 4, H, W) with channels [L_norm, a_norm, b_norm, alpha_norm]
        x = x.squeeze(0).permute(1, 2, 0).cpu()  # (H, W, 4)
        oklab = denormalize_oklab(x[:, :, :3])
        rgb = oklab_to_srgb_torch(oklab)
        alpha = (x[:, :, 3:4] + 1.0) / 2.0  # denorm alpha from [-1,1] to [0,1]
        rgba = torch.cat([rgb, alpha.clamp(0, 1)], dim=-1)
        image_uint8 = (rgba * 255).clamp(0, 255).byte().numpy()

        # Post-process
        palette_arg = palette_hex_to_array(palette_hex) if palette_hex else None
        image_final, palette_out = postprocess(
            image_uint8,
            palette=palette_arg,
            num_colors=num_colors,
            dither_mode=dither_mode,
            outline_cleanup=outline_cleanup,
        )

        return image_final, palette_out
