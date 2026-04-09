"""Per-image palette extraction using K-means in OKLab space."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

from server.utils.color import srgb_to_oklab, oklab_to_srgb


def extract_palette(image: np.ndarray, max_colors: int = 32) -> np.ndarray:
    """Extract optimal palette from image via K-means in OKLab.

    Args:
        image: (H, W, 4) RGBA uint8

    Returns:
        (K, 3) RGB uint8 palette
    """
    alpha = image[:, :, 3]
    rgb = image[:, :, :3]
    opaque = rgb[alpha > 0]

    if len(opaque) == 0:
        return np.zeros((1, 3), dtype=np.uint8)

    unique = np.unique(opaque, axis=0)
    k = min(max_colors, len(unique))

    if k <= 1:
        return unique[:1]

    lab = srgb_to_oklab(unique)
    kmeans = KMeans(n_clusters=k, n_init=5, max_iter=20, random_state=42)
    kmeans.fit(lab)

    return oklab_to_srgb(kmeans.cluster_centers_.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Extract palettes for processed sprites")
    parser.add_argument("--data-dir", required=True, help="Processed data directory")
    parser.add_argument("--max-colors", type=int, default=32)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for size_dir in sorted(data_dir.iterdir()):
        if not size_dir.is_dir():
            continue

        images = sorted(size_dir.glob("*.png"))
        print(f"Extracting palettes for {len(images)} images in {size_dir.name}/")

        for img_path in tqdm(images, desc=size_dir.name):
            meta_path = img_path.with_suffix(".json")

            img = np.array(Image.open(img_path).convert("RGBA"))
            palette = extract_palette(img, args.max_colors)

            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = {}

            meta["palette"] = palette.tolist()

            with open(meta_path, "w") as f:
                json.dump(meta, f)

    print("Done.")


if __name__ == "__main__":
    main()
