"""Pixel-art-safe data augmentation.

Only augmentations that preserve grid alignment are used:
- Horizontal/vertical flip (safe)
- Palette recoloring (safe, teaches palette flexibility)
- NO rotation, NO arbitrary scaling, NO shearing
"""

import random
import numpy as np

from server.utils.color import srgb_to_oklab, oklab_to_srgb


def augment(image: np.ndarray, palette: np.ndarray | None = None,
            lospec_palettes: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply pixel-art-safe augmentations.

    Args:
        image: (H, W, 4) RGBA uint8
        palette: (N, 3) RGB uint8 extracted palette for this image
        lospec_palettes: List of (M, 3) RGB uint8 palettes for recoloring

    Returns:
        augmented image, augmented palette (or None if unchanged)
    """
    # Horizontal flip (50%)
    if random.random() < 0.5:
        image = np.flip(image, axis=1).copy()

    # Vertical flip (10%, only useful for symmetric objects)
    if random.random() < 0.1:
        image = np.flip(image, axis=0).copy()

    # Palette recoloring (20%)
    if random.random() < 0.2 and palette is not None and lospec_palettes:
        image, palette = _recolor_palette(image, palette, lospec_palettes)

    return image, palette


def _recolor_palette(image: np.ndarray, old_palette: np.ndarray,
                     lospec_palettes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Remap sprite colors to a random Lospec palette of similar size.

    Matches each old palette color to the nearest color in the new palette
    by luminance ordering, creating a visually coherent recoloring.
    """
    # Pick a random palette with enough colors
    candidates = [p for p in lospec_palettes if len(p) >= len(old_palette)]
    if not candidates:
        return image, old_palette

    new_palette = random.choice(candidates)

    # Sort both palettes by luminance (L channel in OKLab)
    old_lab = srgb_to_oklab(old_palette)
    new_lab = srgb_to_oklab(new_palette)

    old_order = np.argsort(old_lab[:, 0])
    new_order = np.argsort(new_lab[:, 0])

    # Map old colors to new by luminance rank
    color_map = {}
    for i, old_idx in enumerate(old_order):
        # Map to corresponding luminance position in new palette
        new_idx_pos = int(i * len(new_order) / len(old_order))
        new_idx_pos = min(new_idx_pos, len(new_order) - 1)
        new_idx = new_order[new_idx_pos]
        old_color = tuple(old_palette[old_idx])
        color_map[old_color] = new_palette[new_idx]

    # Apply remapping
    result = image.copy()
    H, W, _ = image.shape
    for y in range(H):
        for x in range(W):
            if image[y, x, 3] == 0:
                continue
            key = tuple(image[y, x, :3])
            if key in color_map:
                result[y, x, :3] = color_map[key]

    new_pal = np.array(list(color_map.values()), dtype=np.uint8)
    return result, new_pal
