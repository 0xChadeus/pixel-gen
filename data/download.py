"""Download pixel art datasets from HuggingFace and Kaggle."""

import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm


DATASETS = {
    "kaggle_pixelart": {
        "source": "kaggle",
        "kaggle_id": "ebrahimelgazar/pixel-art",
        "license": "CC0",
    },
    "free_pixelart": {
        "source": "huggingface",
        "hf_id": "bghira/free-to-use-pixelart",
        "image_col": "image",
        "license": "permissive",
    },
    "limbicnation": {
        "source": "huggingface",
        "hf_id": "Limbicnation/pixel-art-character",
        "image_col": "image",
        "license": "CC0",
    },
    "kaggle_pixel_chars": {
        "source": "kaggle",
        "kaggle_id": "volodymyrpivoshenko/pixel-characters-dataset",
        "license": "CC0",
    },
    "diffusiondb_pixel": {
        "source": "huggingface",
        "hf_id": "jainr3/diffusiondb-pixelart",
        "image_col": "image",
        "license": "CC0",
    },
    "pixel_art_thliang": {
        "source": "huggingface",
        "hf_id": "thliang01/Pixel_Art",
        "image_col": "image",
        "license": "CC0",
    },
    "vatsadev_pixel": {
        "source": "huggingface",
        "hf_id": "VatsaDev/pixel-art",
        "image_col": "image",
        "license": "CC0",
    },
}


def download_hf_dataset(name: str, output_dir: str, max_items: int = 0):
    """Download a HuggingFace dataset and save images as PNG."""
    from datasets import load_dataset

    info = DATASETS[name]
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name} ({info['hf_id']})...")
    subset = info.get("subset")
    try:
        ds = load_dataset(info["hf_id"], name=subset, split="train")
    except Exception:
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
        if img is None or not isinstance(img, Image.Image):
            continue

        img = img.convert("RGBA")
        img.save(out / f"{name}_{i:06d}.png")
        count += 1

    print(f"Saved {count} images to {out}")


def download_kaggle_dataset(name: str, output_dir: str, max_items: int = 0):
    """Download a Kaggle dataset via kagglehub and save images as PNG."""
    import kagglehub

    info = DATASETS[name]
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name} ({info['kaggle_id']})...")
    path = kagglehub.dataset_download(info["kaggle_id"])

    count = 0
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for img_path in tqdm(sorted(Path(path).rglob("*")), desc=name):
        if img_path.suffix.lower() not in image_exts:
            continue
        if max_items > 0 and count >= max_items:
            break
        try:
            img = Image.open(img_path).convert("RGBA")
            img.save(out / f"{name}_{count:06d}.png")
            count += 1
        except Exception:
            continue

    print(f"Saved {count} images to {out}")


def download_dataset(name: str, output_dir: str, max_items: int = 0):
    """Download a dataset by name, dispatching to the appropriate backend."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
        return

    info = DATASETS[name]
    if info["source"] == "kaggle":
        download_kaggle_dataset(name, output_dir, max_items)
    else:
        download_hf_dataset(name, output_dir, max_items)


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
