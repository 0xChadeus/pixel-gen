"""Evaluation metrics for pixel art diffusion model.

Computes FID per resolution, alpha purity, and palette divergence.

Usage:
    python -m training.evaluate --checkpoint checkpoints/step_0050000.pt \
        --config training/config.yaml --num-samples 2048
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_alpha_purity(images: list[np.ndarray], threshold: int = 20) -> float:
    """Fraction of pixels with alpha cleanly near 0 or 255.

    Args:
        images: list of (H, W, 4) RGBA uint8 arrays
        threshold: distance from 0 or 255 to count as "pure"

    Returns:
        Float in [0, 1] — higher is better for pixel art.
    """
    pure = 0
    total = 0
    for img in images:
        alpha = img[:, :, 3].astype(np.float32)
        near_zero = alpha < threshold
        near_full = alpha > (255 - threshold)
        pure += (near_zero | near_full).sum()
        total += alpha.size
    return pure / max(total, 1)


def compute_palette_divergence(
    generated: list[np.ndarray],
    reference_dir: Path,
    num_ref: int = 1000,
    num_bins: int = 64,
) -> float:
    """KL divergence between color histograms of generated vs training images.

    Args:
        generated: list of (H, W, 4) RGBA uint8
        reference_dir: path to processed training data directory
        num_ref: number of reference images to sample
        num_bins: bins per color channel for histogram

    Returns:
        KL divergence (lower = closer to training distribution).
    """
    def _color_hist(images, num_bins):
        all_pixels = []
        for img in images:
            alpha = img[:, :, 3]
            rgb = img[:, :, :3][alpha > 128]
            if len(rgb) > 0:
                all_pixels.append(rgb)
        if not all_pixels:
            return np.ones(num_bins ** 3) / (num_bins ** 3)
        pixels = np.concatenate(all_pixels, axis=0)
        # Quantize to bins
        binned = (pixels / 256.0 * num_bins).astype(int).clip(0, num_bins - 1)
        indices = binned[:, 0] * num_bins ** 2 + binned[:, 1] * num_bins + binned[:, 2]
        hist = np.bincount(indices, minlength=num_bins ** 3).astype(np.float64)
        hist = hist / hist.sum()
        return hist + 1e-10  # avoid log(0)

    gen_hist = _color_hist(generated, num_bins)

    # Load reference images
    ref_pngs = sorted(reference_dir.glob("*.png"))[:num_ref]
    ref_images = []
    for p in ref_pngs:
        try:
            ref_images.append(np.array(Image.open(p).convert("RGBA")))
        except Exception:
            continue
    ref_hist = _color_hist(ref_images, num_bins)

    # KL divergence: sum(p * log(p/q))
    kl = np.sum(gen_hist * np.log(gen_hist / ref_hist))
    return float(kl)


@torch.no_grad()
def generate_samples(
    checkpoint_path: str,
    config_path: str,
    num_samples: int = 2048,
    size: int = 64,
    guidance_scale: float = 5.0,
    num_steps: int = 20,
    batch_size: int = 4,
) -> list[np.ndarray]:
    """Generate samples using a checkpoint for evaluation."""
    import yaml
    from model.unet import EDMUNet
    from model.conditioning import ConditioningAssembler
    from model.diffusion import EDMPrecond, HeunSampler
    from server.utils.color import denormalize_oklab, oklab_to_srgb_torch

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        dropout=0.0,
        self_condition=model_cfg["self_condition"],
    ).to(device)

    cond_assembler = ConditioningAssembler(
        cond_dim=model_cfg["cond_dim"],
        clip_dim=768,
        cross_attn_dim=model_cfg["cross_attn_dim"],
        supported_sizes=train_cfg["image_sizes"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema_model"])
    cond_assembler.load_state_dict(ckpt["cond_assembler"])
    model.eval()
    cond_assembler.eval()

    precond = EDMPrecond(model, sigma_data=model_cfg["sigma_data"])
    sampler = HeunSampler()

    images = []
    for i in tqdm(range(0, num_samples, batch_size), desc=f"Generating {size}px"):
        B = min(batch_size, num_samples - i)

        # Unconditional generation (null conditioning)
        null_pooled = torch.zeros(B, 768, device=device)
        null_tokens = torch.zeros(B, 77, 768, device=device)
        pal_t = torch.zeros(B, 1, 3, device=device)
        pal_m = torch.ones(B, 1, dtype=torch.bool, device=device)

        cond, cross = cond_assembler(
            sigma=torch.ones(B, device=device),
            text_pooled=null_pooled, text_tokens=null_tokens,
            palette=pal_t, palette_mask=pal_m, resolution=size,
            drop_text=True, drop_palette=True,
        )

        x = sampler.sample(
            precond, (B, 4, size, size), cond, cross,
            num_steps=num_steps, device=device,
            self_condition=model_cfg["self_condition"],
        )

        for j in range(B):
            sample = x[j].permute(1, 2, 0).cpu()
            oklab = denormalize_oklab(sample[:, :, :3])
            rgb = oklab_to_srgb_torch(oklab)
            alpha = (sample[:, :, 3:4] + 1.0) / 2.0
            rgba = torch.cat([rgb, alpha.clamp(0, 1)], dim=-1)
            img = (rgba * 255).clamp(0, 255).byte().numpy()
            images.append(img)

    return images


def compute_fid(generated_dir: str, reference_dir: str) -> float:
    """Compute FID between generated and reference image directories.

    Uses pytorch-fid under the hood.
    """
    from pytorch_fid.fid_score import calculate_fid_given_paths

    fid = calculate_fid_given_paths(
        [generated_dir, reference_dir],
        batch_size=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dims=2048,
    )
    return fid


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    data_dir: str = "data/processed",
    num_samples: int = 2048,
    output_csv: str = "checkpoints/fid_log.csv",
):
    """Run full evaluation on a checkpoint."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sizes = cfg["training"]["image_sizes"]
    step = int(Path(checkpoint_path).stem.split("_")[-1])

    results = {"step": step}

    for size in sizes:
        ref_dir = Path(data_dir) / str(size)
        if not ref_dir.exists() or len(list(ref_dir.glob("*.png"))) == 0:
            logger.info(f"Skipping {size}px — no reference data")
            continue

        logger.info(f"Evaluating at {size}x{size}...")

        # Generate samples
        images = generate_samples(
            checkpoint_path, config_path,
            num_samples=min(num_samples, 512),  # reduce for larger sizes
            size=size, batch_size=4,
        )

        # Alpha purity
        alpha_purity = compute_alpha_purity(images)
        results[f"alpha_purity_{size}"] = f"{alpha_purity:.4f}"
        logger.info(f"  Alpha purity: {alpha_purity:.4f}")

        # Palette divergence
        kl = compute_palette_divergence(images, ref_dir)
        results[f"palette_kl_{size}"] = f"{kl:.4f}"
        logger.info(f"  Palette KL: {kl:.4f}")

        # FID — save generated images to temp dir, compute against reference
        gen_dir = Path(f"samples/eval_fid_{size}")
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Save as RGB (FID expects RGB)
        for i, img in enumerate(images):
            rgb = img[:, :, :3]
            Image.fromarray(rgb, "RGB").save(gen_dir / f"{i:05d}.png")

        try:
            fid = compute_fid(str(gen_dir), str(ref_dir))
            results[f"fid_{size}"] = f"{fid:.2f}"
            logger.info(f"  FID: {fid:.2f}")
        except Exception as e:
            logger.warning(f"  FID computation failed: {e}")
            results[f"fid_{size}"] = "error"

        # Cleanup temp dir
        for f in gen_dir.glob("*.png"):
            f.unlink()
        gen_dir.rmdir()

    # Append to CSV
    csv_path = Path(output_csv)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(results.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(results)

    logger.info(f"Results saved to {csv_path}")
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate pixel art diffusion model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--output-csv", default="checkpoints/fid_log.csv")
    args = parser.parse_args()

    evaluate_checkpoint(
        args.checkpoint, args.config, args.data_dir,
        args.num_samples, args.output_csv,
    )


if __name__ == "__main__":
    main()
