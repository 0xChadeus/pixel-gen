"""Download pixel art datasets from HuggingFace and other sources."""

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


DATASETS = {
    "limbicnation": {
        "hf_id": "Limbicnation/pixel-art-character",
        "image_col": "image",
        "license": "CC0",
    },
    "lpc_4view": {
        "hf_id": "carlosuperb/lpc-4view-pixel-art-diffusion",
        "image_col": "image",
        "license": "CC-BY-SA 3.0",
    },
    "pico8_sprites": {
        "hf_id": "Fraser/pico-8-games",
        "image_col": "spritesheet",
        "license": "CC0",
    },
    "pixel_art_thliang": {
        "hf_id": "thliang01/Pixel_Art",
        "image_col": "image",
        "license": "CC0",
    },
    "diffusiondb_pixel": {
        "hf_id": "jainr3/diffusiondb-pixelart",
        "image_col": "image",
        "license": "CC0",
    },
    "opengameart_cc0": {
        "hf_id": "nyuuzyou/OpenGameArt-CC0",
        "image_col": "image",
        "license": "CC0",
        "subset": "2d-art",
    },
    "vatsadev_pixel": {
        "hf_id": "VatsaDev/pixel-art",
        "image_col": "image",
        "license": "CC0",
    },
}


def download_dataset(name: str, output_dir: str, max_items: int = 0):
    """Download a single dataset and save images as PNG."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
        return

    info = DATASETS[name]
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name} ({info['hf_id']})...")
    subset = info.get("subset")
    try:
        ds = load_dataset(info["hf_id"], name=subset, split="train")
    except Exception:
        # Some datasets use different split names or configs
        try:
            ds = load_dataset(info["hf_id"], split="train")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            return

    count = 0
    for i, item in enumerate(tqdm(ds, desc=name)):
        if max_items > 0 and count >= max_items:
            break

        img = item.get(info["image_col"])
        if img is None:
            continue

        if not isinstance(img, Image.Image):
            continue

        img = img.convert("RGBA")
        img.save(out / f"{name}_{i:06d}.png")
        count += 1

    print(f"Saved {count} images to {out}")


def main():
    parser = argparse.ArgumentParser(description="Download pixel art datasets")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        help="Which datasets to download")
    parser.add_argument("--max-items", type=int, default=0,
                        help="Max items per dataset (0=all)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for name in args.datasets:
        download_dataset(name, args.output, args.max_items)


if __name__ == "__main__":
    main()
