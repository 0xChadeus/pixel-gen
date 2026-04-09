"""Dithering algorithms for pixel art with limited palettes."""

import numpy as np

from server.utils.color import srgb_to_oklab, oklab_to_srgb

# 4x4 Bayer ordered dithering matrix (normalized to [0,1])
BAYER_4X4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
], dtype=np.float32) / 16.0


def _nearest_palette_color(pixel_lab: np.ndarray, palette_lab: np.ndarray) -> int:
    """Find index of nearest palette color in OKLab space."""
    dists = np.sum((palette_lab - pixel_lab) ** 2, axis=1)
    return int(np.argmin(dists))


def dither_ordered(image: np.ndarray, palette_rgb: np.ndarray,
                   matrix: np.ndarray | None = None, strength: float = 0.08) -> np.ndarray:
    """Apply ordered (Bayer) dithering with a given palette.

    Args:
        image: (H, W, 4) RGBA uint8
        palette_rgb: (N, 3) RGB uint8 palette
        matrix: Dithering threshold matrix. Default: 4x4 Bayer.
        strength: Dither strength in OKLab L space.

    Returns:
        (H, W, 4) RGBA uint8 dithered image
    """
    if matrix is None:
        matrix = BAYER_4X4

    H, W, _ = image.shape
    result = image.copy()
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]

    img_lab = srgb_to_oklab(rgb)
    pal_lab = srgb_to_oklab(palette_rgb)

    mh, mw = matrix.shape
    for y in range(H):
        for x in range(W):
            if alpha[y, x] == 0:
                continue

            threshold = matrix[y % mh, x % mw] - 0.5  # center around 0
            pixel = img_lab[y, x].copy()
            pixel[0] += threshold * strength  # perturb luminance

            idx = _nearest_palette_color(pixel, pal_lab)
            result[y, x, :3] = palette_rgb[idx]

    return result


def dither_floyd_steinberg(image: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """Apply Floyd-Steinberg error diffusion dithering.

    Args:
        image: (H, W, 4) RGBA uint8
        palette_rgb: (N, 3) RGB uint8

    Returns:
        (H, W, 4) RGBA uint8 dithered image
    """
    H, W, _ = image.shape
    result = image.copy()
    alpha = image[:, :, 3]

    img_lab = srgb_to_oklab(image[:, :, :3]).astype(np.float64)
    pal_lab = srgb_to_oklab(palette_rgb).astype(np.float64)

    for y in range(H):
        for x in range(W):
            if alpha[y, x] == 0:
                continue

            old = img_lab[y, x].copy()
            idx = _nearest_palette_color(old, pal_lab)
            new = pal_lab[idx]
            result[y, x, :3] = palette_rgb[idx]

            error = old - new

            # Distribute error to neighbors
            if x + 1 < W and alpha[y, x + 1] > 0:
                img_lab[y, x + 1] += error * (7.0 / 16.0)
            if y + 1 < H:
                if x - 1 >= 0 and alpha[y + 1, x - 1] > 0:
                    img_lab[y + 1, x - 1] += error * (3.0 / 16.0)
                if alpha[y + 1, x] > 0:
                    img_lab[y + 1, x] += error * (5.0 / 16.0)
                if x + 1 < W and alpha[y + 1, x + 1] > 0:
                    img_lab[y + 1, x + 1] += error * (1.0 / 16.0)

    return result
