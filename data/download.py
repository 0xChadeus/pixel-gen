"""Download pixel art datasets from HuggingFace and Kaggle."""

import argparse
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from tqdm import tqdm


DATASETS = {
    "kaggle_pixelart": {
        "source": "kaggle",
        "kaggle_id": "ebrahimelgazar/pixel-art",
        "license": "CC0",
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
    "pixel_art_thliang": {
        "source": "huggingface",
        "hf_id": "thliang01/Pixel_Art",
        "image_col": "image",
        "license": "CC0",
    },
    "free_pixelart": {
        "source": "huggingface_urls",
        "hf_id": "bghira/free-to-use-pixelart",
        "url_col": "full_image_url",
        "allowed_domains": {"pixilart.com", "www.pixilart.com", "art.pixilart.com"},
        "license": "permissive",
    },
    "opengameart": {
        "source": "opengameart_hf",
        "hf_ids": [
            "nyuuzyou/OpenGameArt-CC0",
            "nyuuzyou/OpenGameArt-CC-BY-3.0",
            "nyuuzyou/OpenGameArt-CC-BY-4.0",
            "nyuuzyou/OpenGameArt-CC-BY-SA-3.0",
            "nyuuzyou/OpenGameArt-CC-BY-SA-4.0",
        ],
        "split": "2d_art",
        "allowed_domains": {"opengameart.org"},
        "license": "CC",
    },
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
REQUEST_TIMEOUT = (30, 60)  # (connect, read)
USER_AGENT = "PixelGenDataCollector/1.0"


def _validate_url(url: str, allowed_domains: set[str]) -> bool:
    """Validate URL scheme and domain."""
    try:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.hostname in allowed_domains
    except Exception:
        return False


def _fetch_image(url: str, session: requests.Session) -> Image.Image | None:
    """Fetch and validate an image from a URL."""
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()
        size = int(resp.headers.get("Content-Length", 0))
        if size > MAX_FILE_SIZE:
            return None
        content = resp.content
        if len(content) > MAX_FILE_SIZE:
            return None
        from io import BytesIO
        return Image.open(BytesIO(content)).convert("RGBA")
    except Exception:
        return None


def download_hf_dataset(name: str, output_dir: str, max_items: int = 0):
    """Download a HuggingFace dataset with embedded images."""
    from datasets import load_dataset

    info = DATASETS[name]
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name} ({info['hf_id']})...")
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


def download_hf_url_dataset(name: str, output_dir: str, max_items: int = 0):
    """Download a HuggingFace dataset that contains image URLs (not embedded images)."""
    from datasets import load_dataset

    info = DATASETS[name]
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name} ({info['hf_id']}) via URLs...")
    try:
        ds = load_dataset(info["hf_id"], split="train")
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return

    allowed = info["allowed_domains"]
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    count = 0
    for i, item in enumerate(tqdm(ds, desc=name)):
        if max_items > 0 and count >= max_items:
            break
        url = item.get(info["url_col"], "")
        if not url or not _validate_url(url, allowed):
            continue
        img = _fetch_image(url, session)
        if img is None:
            continue
        img.save(out / f"{name}_{count:06d}.png")
        count += 1
        time.sleep(0.5)

    print(f"Saved {count} images to {out}")


def download_opengameart_hf(name: str, output_dir: str, max_items: int = 0):
    """Download OpenGameArt preview images from nyuuzyou HF metadata datasets."""
    from datasets import load_dataset

    info = DATASETS[name]
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    allowed = info["allowed_domains"]
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    count = 0
    for hf_id in info["hf_ids"]:
        if max_items > 0 and count >= max_items:
            break
        print(f"Loading {hf_id} ({info['split']})...")
        try:
            ds = load_dataset(hf_id, split=info["split"])
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        for item in tqdm(ds, desc=hf_id.split("/")[-1]):
            if max_items > 0 and count >= max_items:
                break
            previews = item.get("preview_images", [])
            if not previews:
                continue
            url = previews[0]
            if not _validate_url(url, allowed):
                # OGA URLs are http, not https — allow http for opengameart.org
                parsed = urlparse(url)
                if parsed.scheme == "http" and parsed.hostname in allowed:
                    pass  # allow http for OGA
                else:
                    continue
            img = _fetch_image(url, session)
            if img is None:
                # Try http -> https or vice versa
                alt_url = url.replace("http://", "https://") if url.startswith("http://") else url.replace("https://", "http://")
                img = _fetch_image(alt_url, session)
            if img is None:
                continue
            img.save(out / f"{name}_{count:06d}.png")
            count += 1
            time.sleep(1.0)

    print(f"Saved {count} images to {out}")


def download_kaggle_dataset(name: str, output_dir: str, max_items: int = 0):
    """Download a Kaggle dataset via kagglehub."""
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
    """Download a dataset by name."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
        return

    source = DATASETS[name]["source"]
    if source == "kaggle":
        download_kaggle_dataset(name, output_dir, max_items)
    elif source == "huggingface":
        download_hf_dataset(name, output_dir, max_items)
    elif source == "huggingface_urls":
        download_hf_url_dataset(name, output_dir, max_items)
    elif source == "opengameart_hf":
        download_opengameart_hf(name, output_dir, max_items)


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
