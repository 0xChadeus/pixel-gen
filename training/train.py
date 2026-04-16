"""Main training script for the pixel art diffusion model.

Run directly in your terminal:
    python -m training.train --config training/config.yaml

Ctrl+C saves a checkpoint and exits cleanly.
Auto-resumes from the latest checkpoint on next run.

CLIP embeddings must be precomputed:
    python -m data.cache_embeddings --data-dir data/processed
"""

import argparse
import copy
import logging
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import yaml

from model.unet import EDMUNet
from model.conditioning import ConditioningAssembler
from model.diffusion import EDMPrecond, EDMLoss, HeunSampler
from training.dataset import PixelArtDataset, collate_fn
from server.utils.color import denormalize_oklab, oklab_to_srgb_torch

logger = logging.getLogger(__name__)

_interrupted = False


def _sigint_handler(signum, frame):
    global _interrupted
    logger.warning("Ctrl+C received — will save checkpoint and exit after current step.")
    _interrupted = True


def _find_latest_checkpoint(checkpoint_dir: str) -> Path | None:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def _save_checkpoint(path: Path, step: int, model, ema_model, cond_assembler,
                     optimizer, scheduler, scaler):
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "cond_assembler": cond_assembler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }, tmp)
    tmp.rename(path)
    logger.info(f"Saved checkpoint: {path.name}")


def _rotate_checkpoints(checkpoint_dir: str, keep: int):
    if keep <= 0:
        return
    ckpt_dir = Path(checkpoint_dir)
    existing = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(existing) > keep:
        old = existing.pop(0)
        old.unlink()
        logger.info(f"Rotated out: {old.name}")


def main():
    global _interrupted

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

    # Cap VRAM so the GPU stays usable for other tasks
    max_vram_gb = train_cfg.get("max_vram_gb", 0)
    if max_vram_gb > 0 and device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(device).total_memory / 1e9
        fraction = min(max_vram_gb / total_vram, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        logger.info(f"VRAM cap: {max_vram_gb:.1f} GB ({fraction:.0%} of {total_vram:.1f} GB)")

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

    # Enable gradient checkpointing to reduce VRAM
    model.gradient_checkpointing = True

    cond_assembler = ConditioningAssembler(
        cond_dim=model_cfg["cond_dim"],
        clip_dim=768,
        cross_attn_dim=model_cfg["cross_attn_dim"],
    ).to(device)

    precond = EDMPrecond(model, sigma_data=model_cfg["sigma_data"])
    loss_fn = EDMLoss(
        sigma_data=model_cfg["sigma_data"],
        P_mean=diff_cfg["P_mean"],
        P_std=diff_cfg["P_std"],
    )

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in cond_assembler.parameters())
    logger.info(f"Model parameters: {total_params / 1e6:.1f}M")

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

    # LR schedule
    warmup_steps = train_cfg["warmup_steps"]
    total_steps = train_cfg["total_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Dataset (loads cached CLIP embeddings — no CLIP model needed on GPU)
    dataset = PixelArtDataset(
        data_dir=data_cfg["data_dir"],
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

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(train_cfg["mixed_precision"] == "fp16"))

    # Checkpointing config
    ckpt_dir = "checkpoints"
    save_every = train_cfg.get("save_every", 5000)
    keep_checkpoints = train_cfg.get("keep_checkpoints", 5)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # Resume
    start_step = 0
    resume_path = args.resume
    if not resume_path:
        latest = _find_latest_checkpoint(ckpt_dir)
        if latest:
            resume_path = str(latest)
            logger.info(f"Auto-resuming from: {resume_path}")

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        cond_assembler.load_state_dict(ckpt["cond_assembler"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        logger.info(f"Resumed from step {start_step}")

    # Signal handler for clean Ctrl+C
    signal.signal(signal.SIGINT, _sigint_handler)

    # Log VRAM after setup
    allocated = torch.cuda.memory_allocated() / 1e9
    logger.info(f"VRAM after setup: {allocated:.2f} GB")

    # Epoch tracking
    batch_size = train_cfg["batch_size"]
    steps_per_epoch = max(dataset.total // batch_size, 1)
    epoch_ckpt_dir = os.path.join(ckpt_dir, "epochs")
    os.makedirs(epoch_ckpt_dir, exist_ok=True)

    # Training loop
    logger.info(f"Starting training for {total_steps} steps ({dataset.total} images, ~{steps_per_epoch} steps/epoch)")
    model.train()
    cond_assembler.train()
    data_iter = iter(dataloader)
    step = start_step
    grad_accum = train_cfg.get("gradient_accumulation", 1)

    train_start = time.monotonic()

    while step < total_steps:
        if _interrupted:
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        text_pooled = batch["text_pooled"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        palettes = batch["palette"].to(device)
        pal_masks = batch["palette_mask"].to(device)

        B = images.shape[0]

        # Self-conditioning (50% of the time)
        x_self_cond = None
        if model_cfg["self_condition"] and torch.rand(1).item() < 0.5:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                log_sigma = torch.randn(B, device=device) * diff_cfg["P_std"] + diff_cfg["P_mean"]
                sigma_sc = log_sigma.exp()
                noise_sc = torch.randn_like(images)
                x_noisy_sc = images + sigma_sc.reshape(-1, 1, 1, 1) * noise_sc
                cond_sc, cross_sc = cond_assembler(
                    sigma=sigma_sc, text_pooled=text_pooled, text_tokens=text_tokens,
                    palette=palettes, palette_mask=pal_masks,
                )
                x_self_cond = precond(x_noisy_sc, sigma_sc, cond_sc, cross_sc, None).detach()

        # Conditioning dropout for classifier-free guidance
        drop_text = False
        drop_palette = False
        r = torch.rand(1).item()
        drop_both = train_cfg.get("cond_drop_both", 0.05)
        drop_t = train_cfg.get("cond_drop_text", 0.1)
        drop_p = train_cfg.get("cond_drop_palette", 0.1)
        if r < drop_both:
            drop_text = True
            drop_palette = True
        elif r < drop_both + drop_t:
            drop_text = True
        elif r < drop_both + drop_t + drop_p:
            drop_palette = True

        # Assemble conditioning
        cond_vec, cross_tok = cond_assembler(
            sigma=torch.ones(B, device=device),
            text_pooled=text_pooled, text_tokens=text_tokens,
            palette=palettes, palette_mask=pal_masks,
            drop_text=drop_text, drop_palette=drop_palette,
        )

        # Forward + backward
        with torch.amp.autocast("cuda", enabled=(train_cfg["mixed_precision"] == "fp16")):
            loss = loss_fn(precond, images, cond_vec, cross_tok, x_self_cond)
            raw_loss = loss.item()
            if grad_accum > 1:
                loss = loss / grad_accum

        scaler.scale(loss).backward()

        # Gradient accumulation — only step optimizer every N steps
        if (step + 1) % grad_accum == 0:
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

        # Logging
        if step % train_cfg["log_every"] == 0 or step == start_step + 1:
            vram = torch.cuda.memory_allocated() / 1e9
            elapsed = time.monotonic() - train_start
            steps_done = step - start_step
            if steps_done > 0:
                secs_per_step = elapsed / steps_done
                eta_secs = secs_per_step * (total_steps - step)
                eta_h, eta_rem = divmod(int(eta_secs), 3600)
                eta_m = eta_rem // 60
                eta_str = f"{eta_h}h{eta_m:02d}m"
            else:
                eta_str = "..."
            epoch = step // steps_per_epoch
            logger.info(f"Step {step}/{total_steps} (epoch {epoch}) | Loss: {raw_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | VRAM: {vram:.1f}GB | ETA: {eta_str}")

        # Save intra-epoch checkpoint (rotated — keeps last N)
        if step % save_every == 0:
            _save_checkpoint(
                Path(ckpt_dir) / f"step_{step:07d}.pt", step,
                model, ema_model, cond_assembler, optimizer, scheduler, scaler,
            )
            _rotate_checkpoints(ckpt_dir, keep_checkpoints)

        # Save epoch checkpoint (permanent — never rotated)
        if step % steps_per_epoch == 0 and step > start_step:
            epoch = step // steps_per_epoch
            epoch_path = Path(epoch_ckpt_dir) / f"epoch_{epoch:04d}.pt"
            if not epoch_path.exists():
                _save_checkpoint(
                    epoch_path, step,
                    model, ema_model, cond_assembler, optimizer, scheduler, scaler,
                )
                logger.info(f"Epoch {epoch} complete ({step} steps)")

        # Generate samples
        if step % train_cfg["sample_every"] == 0:
            _generate_samples(ema_model, cond_assembler,
                              step, device, model_cfg)

    # Save final checkpoint on exit
    _save_checkpoint(
        Path(ckpt_dir) / f"step_{step:07d}.pt", step,
        model, ema_model, cond_assembler, optimizer, scheduler, scaler,
    )
    _rotate_checkpoints(ckpt_dir, keep_checkpoints)
    logger.info(f"Training {'interrupted' if _interrupted else 'complete'} at step {step}.")


_EVAL_PROMPTS = [
    "pixel art warrior with sword, front view",
    "pixel art green tree",
    "pixel art treasure chest",
    "pixel art slime monster",
]

# PICO-8 palette (16 colors) in OKLab — hardcoded to avoid runtime dependencies
_PICO8_RGB = [
    [0, 0, 0], [29, 43, 83], [126, 37, 83], [0, 135, 81],
    [171, 82, 54], [95, 87, 79], [194, 195, 199], [255, 241, 232],
    [255, 0, 77], [255, 163, 0], [255, 236, 39], [0, 228, 54],
    [41, 173, 255], [131, 118, 156], [255, 119, 168], [255, 204, 170],
]


def _ensure_eval_embeddings(device: torch.device) -> dict:
    """Load or compute CLIP embeddings for fixed evaluation prompts.

    Computes once and caches to samples/eval_embeddings.pt so CLIP
    is never loaded during training after the first run.
    """
    cache_path = Path("samples/eval_embeddings.pt")
    if cache_path.exists():
        return torch.load(cache_path, map_location=device, weights_only=True)

    logger.info("Computing CLIP embeddings for evaluation prompts (one-time)...")
    from server.inference.clip_encoder import CLIPEncoder
    clip = CLIPEncoder(device=str(device))

    embeddings = {}
    for i, prompt in enumerate(_EVAL_PROMPTS):
        pooled, tokens = clip.encode(prompt)
        embeddings[f"pooled_{i}"] = pooled.cpu()
        embeddings[f"tokens_{i}"] = tokens.cpu()

    torch.save(embeddings, cache_path)
    # Free CLIP from GPU
    del clip
    torch.cuda.empty_cache()
    logger.info(f"Cached evaluation embeddings to {cache_path}")
    return {k: v.to(device) for k, v in embeddings.items()}


def _sample_cfg_heun(precond, cond_c, cross_c, cond_u, cross_u,
                     size, device, self_condition, seed, guidance_scale=5.0):
    """Generate one sample with CFG Heun sampling. Returns (H, W, 4) uint8 RGBA."""
    num_steps = 20
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, 4, size, size, generator=rng, device=device)
    sigmas = HeunSampler().get_sigmas(num_steps, device)
    x = x * sigmas[0]
    x_self_cond = None

    for i in range(num_steps):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_batch = torch.full((1,), sigma_cur, device=device)

        sc = x_self_cond if self_condition else None
        pred_c = precond(x, sigma_batch, cond_c, cross_c, sc)
        pred_u = precond(x, sigma_batch, cond_u, cross_u, sc)
        denoised = pred_u + guidance_scale * (pred_c - pred_u)

        if self_condition:
            x_self_cond = denoised.detach()

        d = (x - denoised) / sigma_cur
        x_next = x + d * (sigma_next - sigma_cur)

        if sigma_next > 0:
            sigma_batch_next = torch.full((1,), sigma_next, device=device)
            sc2 = x_self_cond if self_condition else None
            pred_c2 = precond(x_next, sigma_batch_next, cond_c, cross_c, sc2)
            pred_u2 = precond(x_next, sigma_batch_next, cond_u, cross_u, sc2)
            denoised2 = pred_u2 + guidance_scale * (pred_c2 - pred_u2)
            d2 = (x_next - denoised2) / sigma_next
            x_next = x + (d + d2) / 2 * (sigma_next - sigma_cur)

        x = x_next

    x_out = x.squeeze(0).permute(1, 2, 0).cpu()
    oklab = denormalize_oklab(x_out[:, :, :3])
    rgb = oklab_to_srgb_torch(oklab)
    alpha = (x_out[:, :, 3:4] + 1.0) / 2.0
    rgba = torch.cat([rgb, alpha.clamp(0, 1)], dim=-1)
    return (rgba * 255).clamp(0, 255).byte().numpy()


@torch.no_grad()
def _generate_samples(ema_model, cond_assembler, step, device, model_cfg):
    """Generate a 3x4 evaluation grid at 128x128.

    Row 1: unconditional, Row 2: text-conditioned (CFG), Row 3: palette-conditioned (CFG)
    """
    ema_model.eval()
    cond_assembler.eval()

    precond = EDMPrecond(ema_model, sigma_data=model_cfg["sigma_data"])
    eval_embs = _ensure_eval_embeddings(device)

    from server.utils.color import srgb_to_oklab_torch
    pico8 = torch.tensor(_PICO8_RGB, dtype=torch.float32, device=device) / 255.0
    pico8_lab = srgb_to_oklab_torch(pico8).unsqueeze(0)
    pico8_mask = torch.ones(1, 16, dtype=torch.bool, device=device)

    guidance_scale = 5.0
    n_cols = 4
    base_seed = step
    self_cond = model_cfg["self_condition"]

    null_pooled = torch.zeros(1, 768, device=device)
    null_tokens = torch.zeros(1, 77, 768, device=device)
    null_pal = torch.zeros(1, 1, 3, device=device)
    null_pal_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

    def _null_cond():
        return cond_assembler(
            sigma=torch.ones(1, device=device),
            text_pooled=null_pooled, text_tokens=null_tokens,
            palette=null_pal, palette_mask=null_pal_mask,
            drop_text=True, drop_palette=True,
        )

    rows = []

    # Row 1: Unconditional
    row_imgs = []
    cond_u, cross_u = _null_cond()
    for j in range(n_cols):
        img = _sample_cfg_heun(precond, cond_u, cross_u, cond_u, cross_u,
                               128, device, self_cond, seed=base_seed + j)
        row_imgs.append(img)
    rows.append(np.concatenate(row_imgs, axis=1))

    # Row 2: Text-conditioned (CFG)
    row_imgs = []
    for j in range(n_cols):
        cond_c, cross_c = cond_assembler(
            sigma=torch.ones(1, device=device),
            text_pooled=eval_embs[f"pooled_{j}"], text_tokens=eval_embs[f"tokens_{j}"],
            palette=null_pal, palette_mask=null_pal_mask,
        )
        cond_u, cross_u = _null_cond()
        img = _sample_cfg_heun(precond, cond_c, cross_c, cond_u, cross_u,
                               128, device, self_cond, seed=base_seed + n_cols + j,
                               guidance_scale=guidance_scale)
        row_imgs.append(img)
    rows.append(np.concatenate(row_imgs, axis=1))

    # Row 3: Palette-conditioned (PICO-8 + CFG)
    row_imgs = []
    cond_c, cross_c = cond_assembler(
        sigma=torch.ones(1, device=device),
        text_pooled=null_pooled, text_tokens=null_tokens,
        palette=pico8_lab, palette_mask=pico8_mask,
        drop_text=True,
    )
    cond_u, cross_u = _null_cond()
    for j in range(n_cols):
        img = _sample_cfg_heun(precond, cond_c, cross_c, cond_u, cross_u,
                               128, device, self_cond, seed=base_seed + 2 * n_cols + j,
                               guidance_scale=guidance_scale)
        row_imgs.append(img)
    rows.append(np.concatenate(row_imgs, axis=1))

    grid = np.concatenate(rows, axis=0)
    out_path = f"samples/step_{step:07d}.png"
    Image.fromarray(grid, "RGBA").save(out_path)
    logger.info(f"Saved sample grid: {out_path}")

    ema_model.train()
    cond_assembler.train()


if __name__ == "__main__":
    main()
