"""Preprocess raw pixel art images for training.

Pipeline: grid detection -> downsample -> quality filter -> resize to target
-> palette extraction -> save with metadata.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans

from data.filter import is_valid_pixel_art
from data.attributes import detect_all_attributes
from server.postprocess.grid_snap import detect_grid_size, downsample_to_grid
from server.utils.color import srgb_to_oklab, oklab_to_srgb


TARGET_SIZE = 128


def extract_palette(img: np.ndarray, max_colors: int = 32) -> list[list[int]]:
    """Extract palette via K-means in OKLab."""
    alpha = img[:, :, 3]
    rgb = img[:, :, :3]
    opaque = rgb[alpha > 0]

    if len(opaque) == 0:
        return []

    unique = np.unique(opaque, axis=0)
    k = min(max_colors, len(unique))
    if k <= 0:
        return []

    lab = srgb_to_oklab(unique)
    kmeans = KMeans(n_clusters=k, n_init=5, max_iter=20, random_state=42)
    kmeans.fit(lab)

    palette_rgb = oklab_to_srgb(kmeans.cluster_centers_.astype(np.float32))
    return palette_rgb.tolist()


def process_image(img_path: Path, output_base: Path) -> bool:
    """Process a single image through the full pipeline."""
    try:
        img = np.array(Image.open(img_path).convert("RGBA"))
    except Exception:
        return False

    H, W = img.shape[:2]

    # Detect and correct upscaling
    grid = detect_grid_size(img)
    if grid > 1:
        img = downsample_to_grid(img, grid)
        H, W = img.shape[:2]

    # Quality filter
    valid, reason = is_valid_pixel_art(img)
    if not valid:
        return False

    target = TARGET_SIZE

    # Resize to target (nearest neighbor, pad to square)
    size = max(W, H)
    if size != target:
        # Pad to square first
        square = np.zeros((size, size, 4), dtype=np.uint8)
        y_off = (size - H) // 2
        x_off = (size - W) // 2
        square[y_off:y_off + H, x_off:x_off + W] = img
        img = square

        # Resize
        pil = Image.fromarray(img, "RGBA")
        pil = pil.resize((target, target), Image.NEAREST)
        img = np.array(pil)
    elif W != H:
        # Pad to square at current size, then resize
        square = np.zeros((size, size, 4), dtype=np.uint8)
        y_off = (size - H) // 2
        x_off = (size - W) // 2
        square[y_off:y_off + H, x_off:x_off + W] = img
        img = square
        if size != target:
            pil = Image.fromarray(img, "RGBA")
            pil = pil.resize((target, target), Image.NEAREST)
            img = np.array(pil)

    # Snap alpha to binary
    img[:, :, 3] = np.where(img[:, :, 3] >= 128, 255, 0)

    # Extract palette
    palette = extract_palette(img)

    # Detect attributes
    attributes = detect_all_attributes(img, palette, source_path=str(img_path.name))

    # Save
    out_dir = output_base / "128"
    out_dir.mkdir(parents=True, exist_ok=True)

    name = img_path.stem
    out_img = out_dir / f"{name}.png"
    out_meta = out_dir / f"{name}.json"

    Image.fromarray(img, "RGBA").save(out_img)

    meta = {
        "caption": "",  # filled by caption.py
        "palette": palette,
        "original_size": [int(W), int(H)],
        "grid_detected": int(grid),
        "source_name": img_path.name,
        "attributes": attributes,
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f)

    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess pixel art for training")
    parser.add_argument("--input", required=True, help="Input directory with raw images")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    all_images = list(input_dir.rglob("*.png"))
    print(f"Found {len(all_images)} images")

    accepted = 0
    for img_path in tqdm(all_images, desc="Processing"):
        if process_image(img_path, output_dir):
            accepted += 1

    print(f"\nProcessed: {accepted}/{len(all_images)} images accepted")
    size_dir = output_dir / "128"
    if size_dir.exists():
        count = len(list(size_dir.glob("*.png")))
        print(f"  128x128: {count} images")


if __name__ == "__main__":
    main()
