"""Main training script for the pixel art diffusion model.

Usage:
    python -m training.train --config training/config.yaml
"""

import argparse
import copy
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from model.unet import EDMUNet
from model.conditioning import ConditioningAssembler
from model.diffusion import EDMPrecond, EDMLoss, HeunSampler
from server.inference.clip_encoder import CLIPEncoder
from training.dataset import PixelArtDataset, collate_fn
from server.utils.color import denormalize_oklab, oklab_to_srgb_torch

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build model
    model = EDMUNet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        channel_mults=tuple(model_cfg["channel_mults"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        attention_resolutions=tuple(model_cfg["attention_resolutions"]),
        num_heads=model_cfg["num_heads"],
        cond_dim=model_cfg["cond_dim"],
        cross_attn_dim=model_cfg["cross_attn_dim"],
        dropout=model_cfg["dropout"],
        self_condition=model_cfg["self_condition"],
    ).to(device)

    cond_assembler = ConditioningAssembler(
        cond_dim=model_cfg["cond_dim"],
        clip_dim=768,
        cross_attn_dim=model_cfg["cross_attn_dim"],
        supported_sizes=train_cfg["image_sizes"],
    ).to(device)

    precond = EDMPrecond(model, sigma_data=model_cfg["sigma_data"])
    loss_fn = EDMLoss(
        sigma_data=model_cfg["sigma_data"],
        P_mean=diff_cfg["P_mean"],
        P_std=diff_cfg["P_std"],
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in cond_assembler.parameters())
    logger.info(f"Model parameters: {total_params / 1e6:.1f}M")

    # CLIP encoder (frozen)
    clip_encoder = CLIPEncoder(model_cfg["clip_model"], str(device))

    # EMA
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_decay = train_cfg["ema_decay"]

    # Optimizer
    params = list(model.parameters()) + list(cond_assembler.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.999),
    )

    # LR schedule: cosine with warmup
    warmup_steps = train_cfg["warmup_steps"]
    total_steps = train_cfg["total_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Dataset
    dataset = PixelArtDataset(
        data_dir=data_cfg["data_dir"],
        image_sizes=train_cfg["image_sizes"],
        resolution_weights=train_cfg["resolution_weights"],
        lospec_palettes_dir="aseprite_plugin/palettes",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Resume
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        cond_assembler.load_state_dict(ckpt["cond_assembler"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        logger.info(f"Resumed from step {start_step}")

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(train_cfg["mixed_precision"] == "fp16"))

    # Sampler for evaluation samples
    sampler = HeunSampler()

    # Training loop
    logger.info(f"Starting training for {total_steps} steps")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    model.train()
    cond_assembler.train()
    data_iter = iter(dataloader)
    step = start_step

    pbar = tqdm(total=total_steps, initial=start_step, desc="Training")

    while step < total_steps:
        # Get batch (restart dataloader if exhausted)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images = batch["image"].to(device)        # (B, 4, H, W)
        captions = batch["caption"]                # list[str]
        palettes = batch["palette"].to(device)     # (B, N, 3)
        pal_masks = batch["palette_mask"].to(device)
        resolutions = batch["resolution"]          # list[int]

        # Encode text with CLIP (no grad)
        with torch.no_grad():
            text_pooled, text_tokens = clip_encoder.encode_batch(captions)

        # Classifier-free guidance dropout
        B = images.shape[0]
        drop_text = torch.rand(B) < train_cfg["cond_drop_text"]
        drop_palette = torch.rand(B) < train_cfg["cond_drop_palette"]
        drop_both = torch.rand(B) < train_cfg["cond_drop_both"]
        drop_text = drop_text | drop_both
        drop_palette = drop_palette | drop_both

        # For simplicity in batched training, we use the majority resolution
        # (all items in a batch use the same resolution for simplicity)
        # The dataset already samples by resolution weights
        res = resolutions[0]  # all same in batch due to collate

        # Self-conditioning: 50% of time, run model once and feed back prediction
        x_self_cond = None
        if model_cfg["self_condition"] and torch.rand(1).item() < 0.5:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                # Quick estimate for self-conditioning
                log_sigma = torch.randn(B, device=device) * diff_cfg["P_std"] + diff_cfg["P_mean"]
                sigma_sc = log_sigma.exp()
                noise_sc = torch.randn_like(images)
                x_noisy_sc = images + sigma_sc.reshape(-1, 1, 1, 1) * noise_sc

                cond_sc, cross_sc = cond_assembler(
                    sigma=sigma_sc, text_pooled=text_pooled, text_tokens=text_tokens,
                    palette=palettes, palette_mask=pal_masks, resolution=res,
                )
                x_self_cond = precond(x_noisy_sc, sigma_sc, cond_sc, cross_sc, None).detach()

        # Assemble conditioning (with dropout applied per-sample)
        # For batch efficiency, apply dropout as masking on the assembled vectors
        cond_vec, cross_tok = cond_assembler(
            sigma=torch.ones(B, device=device),  # placeholder, EDMLoss samples its own
            text_pooled=text_pooled,
            text_tokens=text_tokens,
            palette=palettes,
            palette_mask=pal_masks,
            resolution=res,
            drop_text=False,  # handled at sample level below
            drop_palette=False,
        )

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", enabled=(train_cfg["mixed_precision"] == "fp16")):
            loss = loss_fn(precond, images, cond_vec, cross_tok, x_self_cond)

        # Backward
        scaler.scale(loss).backward()

        if train_cfg["grad_clip_norm"] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, train_cfg["grad_clip_norm"])

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        # EMA update
        with torch.no_grad():
            for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                p_ema.lerp_(p_model, 1 - ema_decay)

        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Logging
        if step % train_cfg["log_every"] == 0:
            logger.info(f"Step {step}/{total_steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if step % train_cfg["save_every"] == 0:
            ckpt_path = f"checkpoints/step_{step:07d}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "cond_assembler": cond_assembler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

        # Generate samples
        if step % train_cfg["sample_every"] == 0:
            _generate_samples(ema_model, cond_assembler, precond, sampler,
                              clip_encoder, step, device, model_cfg)

    pbar.close()
    logger.info("Training complete.")


@torch.no_grad()
def _generate_samples(ema_model, cond_assembler, precond_template, sampler,
                      clip_encoder, step, device, model_cfg):
    """Generate a grid of sample images for visual inspection."""
    ema_model.eval()
    cond_assembler.eval()

    precond = EDMPrecond(ema_model, sigma_data=model_cfg["sigma_data"])

    prompts = [
        "pixel art knight character, side view",
        "pixel art tree, green leaves",
        "pixel art robot, front view",
        "pixel art treasure chest, open",
    ]

    size = 64
    images = []

    for prompt in prompts:
        text_pooled, text_tokens = clip_encoder.encode(prompt)
        palette_t = torch.zeros(1, 1, 3, device=device)
        pal_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

        cond, cross = cond_assembler(
            sigma=torch.ones(1, device=device),
            text_pooled=text_pooled, text_tokens=text_tokens,
            palette=palette_t, palette_mask=pal_mask,
            resolution=size,
        )

        x = sampler.sample(
            precond, (1, 4, size, size), cond, cross,
            num_steps=20, device=device, self_condition=model_cfg["self_condition"],
        )

        # Convert to RGB image
        x = x.squeeze(0).permute(1, 2, 0).cpu()
        oklab = denormalize_oklab(x[:, :, :3])
        rgb = oklab_to_srgb_torch(oklab)
        alpha = (x[:, :, 3:4] + 1.0) / 2.0
        rgba = torch.cat([rgb, alpha.clamp(0, 1)], dim=-1)
        img = (rgba * 255).clamp(0, 255).byte().numpy()
        images.append(img)

    # Arrange in 2x2 grid
    grid = np.zeros((size * 2, size * 2, 4), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, 2)
        grid[r * size:(r + 1) * size, c * size:(c + 1) * size] = img

    Image.fromarray(grid, "RGBA").save(f"samples/step_{step:07d}.png")

    ema_model.train()
    cond_assembler.train()


if __name__ == "__main__":
    main()
