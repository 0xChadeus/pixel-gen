"""Multi-resolution pixel art dataset for training."""

import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from server.utils.color import srgb_to_oklab_torch, normalize_oklab
from training.augment import augment


class PixelArtDataset(Dataset):
    """Dataset for multi-resolution pixel art training.

    Expected directory structure:
        data_dir/
            32/
                sprite_0001.png
                sprite_0001.json  # {"caption": "...", "palette": [[r,g,b], ...]}
                ...
            64/
                ...
            128/
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

        # Load all items per resolution
        self.items_by_res: dict[int, list[dict]] = {}
        for size in image_sizes:
            res_dir = Path(data_dir) / str(size)
            items = []
            if res_dir.exists():
                for png in sorted(res_dir.glob("*.png")):
                    meta_path = png.with_suffix(".json")
                    meta = {}
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta = json.load(f)
                    items.append({
                        "image_path": str(png),
                        "caption": meta.get("caption", "pixel art sprite"),
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
        # Sample a resolution according to weights
        size = random.choices(self.image_sizes, weights=self.resolution_weights, k=1)[0]
        items = self.items_by_res.get(size, [])

        if not items:
            # Fallback to any available resolution
            for s in self.image_sizes:
                if self.items_by_res[s]:
                    items = self.items_by_res[s]
                    size = s
                    break

        if not items:
            # Return zeros if no data (shouldn't happen in practice)
            return {
                "image": torch.zeros(4, 32, 32),
                "caption": "pixel art sprite",
                "palette": torch.zeros(1, 3),
                "palette_mask": torch.zeros(1, dtype=torch.bool),
                "resolution": 32,
            }

        item = random.choice(items)

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

        # Alpha: [0,1] -> [-1,1]
        alpha_norm = alpha * 2.0 - 1.0

        # Stack: (H, W, 4) -> (4, H, W)
        image_tensor = torch.cat([normalized, alpha_norm], dim=-1).permute(2, 0, 1)

        # Prepare palette tensor
        if palette is not None and len(palette) > 0:
            from server.utils.color import srgb_to_oklab
            pal_lab = srgb_to_oklab(palette).astype(np.float32)
            pal_tensor = torch.from_numpy(pal_lab)
            pal_mask = torch.ones(len(palette), dtype=torch.bool)
        else:
            pal_tensor = torch.zeros(1, 3)
            pal_mask = torch.ones(1, dtype=torch.bool)

        return {
            "image": image_tensor,
            "caption": item["caption"],
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


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-length palettes."""
    images = torch.stack([b["image"] for b in batch])
    captions = [b["caption"] for b in batch]
    resolutions = [b["resolution"] for b in batch]

    # Pad palettes to same length
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
        "caption": captions,
        "palette": torch.stack(palettes),
        "palette_mask": torch.stack(masks),
        "resolution": resolutions,
    }
