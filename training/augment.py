"""Pixel-art-safe data augmentation.

All augmentations preserve grid alignment and discrete palettes:
- Geometric: horizontal/vertical flip, 90° rotation, integer translation
- Color: hue shift, saturation jitter, brightness jitter, palette recolor
- Destructive: random cutout
"""

import math
import random

import numpy as np

from server.utils.color import srgb_to_oklab, oklab_to_srgb


def augment(image: np.ndarray, palette: np.ndarray | None = None,
            lospec_palettes: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply pixel-art-safe augmentations.

    Args:
        image: (H, W, 4) RGBA uint8
        palette: (N, 3) RGB uint8 extracted palette
        lospec_palettes: List of (M, 3) palettes for recoloring

    Returns:
        augmented image, augmented palette
    """
    # Geometric
    if random.random() < 0.5:
        image = np.flip(image, axis=1).copy()
    if random.random() < 0.1:
        image = np.flip(image, axis=0).copy()
    if random.random() < 0.25:
        k = random.choice([1, 2, 3])
        image = np.rot90(image, k).copy()
    if random.random() < 0.15:
        image = _translate(image, max_frac=0.25)

    # Color (via palette remapping)
    if random.random() < 0.3:
        image, palette = _hue_shift(image, palette)
    if random.random() < 0.2:
        image, palette = _saturation_jitter(image, palette, lo=0.5, hi=1.5)
    if random.random() < 0.2:
        image, palette = _brightness_jitter(image, palette, max_delta=0.1)
    if random.random() < 0.15 and palette is not None and lospec_palettes:
        image, palette = _palette_recolor(image, palette, lospec_palettes)

    # Destructive (last)
    if random.random() < 0.15:
        image = _cutout(image, min_frac=0.1, max_frac=0.4)

    return image, palette


# --- Geometric helpers ---

def _translate(image: np.ndarray, max_frac: float) -> np.ndarray:
    H, W = image.shape[:2]
    max_shift = int(max(H, W) * max_frac)
    dy = random.randint(-max_shift, max_shift)
    dx = random.randint(-max_shift, max_shift)
    return np.roll(np.roll(image, dy, axis=0), dx, axis=1).copy()


def _cutout(image: np.ndarray, min_frac: float, max_frac: float) -> np.ndarray:
    H, W = image.shape[:2]
    ch = random.randint(int(H * min_frac), int(H * max_frac))
    cw = random.randint(int(W * min_frac), int(W * max_frac))
    y = random.randint(0, H - ch)
    x = random.randint(0, W - cw)
    image = image.copy()
    image[y:y + ch, x:x + cw] = 0
    return image


# --- Color helpers (OKLab palette remapping) ---

def _extract_color_map(image: np.ndarray) -> dict[tuple, np.ndarray]:
    """Extract unique opaque RGB colors from image."""
    opaque = image[:, :, 3] > 0
    pixels = image[opaque, :3]
    unique = np.unique(pixels, axis=0)
    return {tuple(c): c for c in unique}


def _apply_color_map(image: np.ndarray, old_to_new: dict[tuple, np.ndarray]) -> np.ndarray:
    """Remap image pixels using a color map."""
    result = image.copy()
    H, W = image.shape[:2]
    for y in range(H):
        for x in range(W):
            if image[y, x, 3] == 0:
                continue
            key = tuple(image[y, x, :3])
            if key in old_to_new:
                result[y, x, :3] = old_to_new[key]
    return result


def _hue_shift(image: np.ndarray, palette: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Rotate all colors by a random angle in OKLab a/b plane."""
    colors = _extract_color_map(image)
    if not colors:
        return image, palette

    theta = random.uniform(-math.pi, math.pi)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    rgb_arr = np.array(list(colors.keys()), dtype=np.uint8)
    lab = srgb_to_oklab(rgb_arr)

    # Rotate a/b
    a, b = lab[:, 1].copy(), lab[:, 2].copy()
    lab[:, 1] = a * cos_t - b * sin_t
    lab[:, 2] = a * sin_t + b * cos_t

    new_rgb = oklab_to_srgb(lab)
    old_to_new = {tuple(old): new for old, new in zip(rgb_arr, new_rgb)}

    image = _apply_color_map(image, old_to_new)
    if palette is not None:
        palette = _remap_palette(palette, old_to_new)
    return image, palette


def _saturation_jitter(image: np.ndarray, palette: np.ndarray | None,
                       lo: float, hi: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Scale a/b channels by a random factor."""
    colors = _extract_color_map(image)
    if not colors:
        return image, palette

    factor = random.uniform(lo, hi)
    rgb_arr = np.array(list(colors.keys()), dtype=np.uint8)
    lab = srgb_to_oklab(rgb_arr)
    lab[:, 1:3] *= factor
    new_rgb = oklab_to_srgb(lab)

    old_to_new = {tuple(old): new for old, new in zip(rgb_arr, new_rgb)}
    image = _apply_color_map(image, old_to_new)
    if palette is not None:
        palette = _remap_palette(palette, old_to_new)
    return image, palette


def _brightness_jitter(image: np.ndarray, palette: np.ndarray | None,
                       max_delta: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Shift L channel by a random amount."""
    colors = _extract_color_map(image)
    if not colors:
        return image, palette

    delta = random.uniform(-max_delta, max_delta)
    rgb_arr = np.array(list(colors.keys()), dtype=np.uint8)
    lab = srgb_to_oklab(rgb_arr)
    lab[:, 0] = np.clip(lab[:, 0] + delta, 0.0, 1.0)
    new_rgb = oklab_to_srgb(lab)

    old_to_new = {tuple(old): new for old, new in zip(rgb_arr, new_rgb)}
    image = _apply_color_map(image, old_to_new)
    if palette is not None:
        palette = _remap_palette(palette, old_to_new)
    return image, palette


def _palette_recolor(image: np.ndarray, old_palette: np.ndarray,
                     lospec_palettes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Remap sprite colors to a random Lospec palette by luminance ordering."""
    candidates = [p for p in lospec_palettes if len(p) >= len(old_palette)]
    if not candidates:
        return image, old_palette

    new_palette = random.choice(candidates)
    old_lab = srgb_to_oklab(old_palette)
    new_lab = srgb_to_oklab(new_palette)

    old_order = np.argsort(old_lab[:, 0])
    new_order = np.argsort(new_lab[:, 0])

    old_to_new = {}
    for i, old_idx in enumerate(old_order):
        new_pos = min(int(i * len(new_order) / len(old_order)), len(new_order) - 1)
        old_to_new[tuple(old_palette[old_idx])] = new_palette[new_order[new_pos]]

    image = _apply_color_map(image, old_to_new)
    new_pal = np.array(list(old_to_new.values()), dtype=np.uint8)
    return image, new_pal


def _remap_palette(palette: np.ndarray, old_to_new: dict[tuple, np.ndarray]) -> np.ndarray:
    """Apply a color map to a palette array."""
    result = palette.copy()
    for i in range(len(palette)):
        key = tuple(palette[i])
        if key in old_to_new:
            result[i] = old_to_new[key]
    return result
