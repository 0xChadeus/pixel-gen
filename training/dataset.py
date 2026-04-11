"""Multi-resolution pixel art dataset for training.

Loads images, cached CLIP embeddings, and palettes. CLIP embeddings
must be precomputed via `python -m data.cache_embeddings`.
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from server.utils.color import srgb_to_oklab_torch, normalize_oklab, srgb_to_oklab
from training.augment import augment


class PixelArtDataset(Dataset):
    """Dataset for multi-resolution pixel art training.

    Expected directory structure:
        data_dir/
            32/
                sprite_0001.png
                sprite_0001.json     # {"caption": "...", "palette": [...]}
                sprite_0001.emb.pt   # {"pooled": (768,), "tokens": (77, 768)}
                ...
    """

    def __init__(
        self,
        data_dir: str,
        image_sizes: list[int] = None,
        resolution_weights: list[float] = None,
        lospec_palettes_dir: str | None = None,
    ):
        if image_sizes is None:
            image_sizes = [32, 64, 128]
        if resolution_weights is None:
            resolution_weights = [0.15, 0.25, 0.60]

        self.image_sizes = image_sizes
        self.resolution_weights = resolution_weights

        self.items_by_res: dict[int, list[dict]] = {}
        for size in image_sizes:
            res_dir = Path(data_dir) / str(size)
            items = []
            if res_dir.exists():
                for png in sorted(res_dir.glob("*.png")):
                    emb_path = png.with_suffix(".emb.pt")
                    meta_path = png.with_suffix(".json")

                    if not emb_path.exists():
                        continue  # skip images without cached embeddings

                    meta = {}
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta = json.load(f)

                    items.append({
                        "image_path": str(png),
                        "emb_path": str(emb_path),
                        "palette": np.array(meta.get("palette", []), dtype=np.uint8) if meta.get("palette") else None,
                    })
            self.items_by_res[size] = items

        self.total = sum(len(v) for v in self.items_by_res.values())

        # Load Lospec palettes for augmentation
        self.lospec_palettes = []
        if lospec_palettes_dir and os.path.isdir(lospec_palettes_dir):
            for gpl in Path(lospec_palettes_dir).glob("*.gpl"):
                pal = _parse_gpl(gpl)
                if pal is not None and len(pal) >= 4:
                    self.lospec_palettes.append(pal)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Map flat index to (resolution, item)
        offset = 0
        item = None
        size = self.image_sizes[0]
        for s in self.image_sizes:
            items = self.items_by_res.get(s, [])
            if idx < offset + len(items):
                item = items[idx - offset]
                size = s
                break
            offset += len(items)

        if item is None:
            return {
                "image": torch.zeros(4, 32, 32),
                "text_pooled": torch.zeros(768),
                "text_tokens": torch.zeros(77, 768),
                "palette": torch.zeros(1, 3),
                "palette_mask": torch.zeros(1, dtype=torch.bool),
                "resolution": 32,
            }

        # Load image
        img = Image.open(item["image_path"]).convert("RGBA")
        img = img.resize((size, size), Image.NEAREST)
        img_np = np.array(img)

        palette = item["palette"]

        # Augment
        img_np, palette = augment(img_np, palette,
                                  lospec_palettes=self.lospec_palettes if self.lospec_palettes else None)

        # Convert to OKLab + alpha tensor
        rgb = torch.from_numpy(img_np[:, :, :3].copy()).float() / 255.0
        alpha = torch.from_numpy(img_np[:, :, 3:4].copy()).float() / 255.0

        oklab = srgb_to_oklab_torch(rgb)
        normalized = normalize_oklab(oklab)
        alpha_norm = alpha * 2.0 - 1.0
        image_tensor = torch.cat([normalized, alpha_norm], dim=-1).permute(2, 0, 1)

        # Load cached CLIP embeddings
        emb = torch.load(item["emb_path"], weights_only=True)
        text_pooled = emb["pooled"]
        text_tokens = emb["tokens"]

        # Prepare palette tensor
        if palette is not None and len(palette) > 0:
            pal_lab = srgb_to_oklab(palette).astype(np.float32)
            pal_tensor = torch.from_numpy(pal_lab)
            pal_mask = torch.ones(len(palette), dtype=torch.bool)
        else:
            pal_tensor = torch.zeros(1, 3)
            pal_mask = torch.ones(1, dtype=torch.bool)

        return {
            "image": image_tensor,
            "text_pooled": text_pooled,
            "text_tokens": text_tokens,
            "palette": pal_tensor,
            "palette_mask": pal_mask,
            "resolution": size,
        }


def _parse_gpl(path: Path) -> np.ndarray | None:
    """Parse a GIMP .gpl palette file into (N, 3) RGB uint8 array."""
    colors = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("GIMP") or line.startswith("Name:") or line.startswith("Columns:"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                        if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                            colors.append([r, g, b])
                    except ValueError:
                        continue
    except Exception:
        return None
    if colors:
        return np.array(colors, dtype=np.uint8)
    return None


class ResolutionGroupedSampler(torch.utils.data.Sampler):
    """Yields batches where all items share the same resolution."""

    def __init__(self, dataset: PixelArtDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

        self.indices_by_res: dict[int, list[int]] = {}
        offset = 0
        for size in dataset.image_sizes:
            n = len(dataset.items_by_res.get(size, []))
            self.indices_by_res[size] = list(range(offset, offset + n))
            offset += n

        self.available_sizes = [s for s in dataset.image_sizes
                                if len(self.indices_by_res.get(s, [])) > 0]
        self.weights = [dataset.resolution_weights[dataset.image_sizes.index(s)]
                        for s in self.available_sizes]

    def __iter__(self):
        while True:
            size = random.choices(self.available_sizes, weights=self.weights, k=1)[0]
            indices = self.indices_by_res[size]
            batch = [random.choice(indices) for _ in range(self.batch_size)]
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-length palettes."""
    images = torch.stack([b["image"] for b in batch])
    text_pooled = torch.stack([b["text_pooled"] for b in batch])
    text_tokens = torch.stack([b["text_tokens"] for b in batch])
    resolutions = [b["resolution"] for b in batch]

    max_colors = max(b["palette"].shape[0] for b in batch)
    palettes = []
    masks = []
    for b in batch:
        pal = b["palette"]
        mask = b["palette_mask"]
        pad_n = max_colors - pal.shape[0]
        if pad_n > 0:
            pal = torch.cat([pal, torch.zeros(pad_n, 3)])
            mask = torch.cat([mask, torch.zeros(pad_n, dtype=torch.bool)])
        palettes.append(pal)
        masks.append(mask)

    return {
        "image": images,
        "text_pooled": text_pooled,
        "text_tokens": text_tokens,
        "palette": torch.stack(palettes),
        "palette_mask": torch.stack(masks),
        "resolution": resolutions,
    }
