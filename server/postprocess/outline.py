"""Outline detection and cleanup for pixel art sprites."""

import numpy as np


def cleanup_outlines(image: np.ndarray) -> np.ndarray:
    """Clean up sprite outlines: fill diagonal-only gaps.

    In pixel art, if two opaque pixels are connected only diagonally
    with transparent pixels on both orthogonal sides, this creates a
    visual gap. This function fills such gaps.

    Args:
        image: (H, W, 4) RGBA uint8

    Returns:
        (H, W, 4) RGBA uint8 with outline gaps filled
    """
    H, W, _ = image.shape
    result = image.copy()
    alpha = image[:, :, 3]
    rgb = image[:, :, :3]

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if alpha[y, x] > 0:
                continue

            # Check for diagonal-only connections
            # Pattern: transparent pixel with opaque diagonal neighbors
            # and transparent orthogonal neighbors
            top = alpha[y - 1, x] > 0
            bot = alpha[y + 1, x] > 0
            left = alpha[y, x - 1] > 0
            right = alpha[y, x + 1] > 0
            tl = alpha[y - 1, x - 1] > 0
            tr = alpha[y - 1, x + 1] > 0
            bl = alpha[y + 1, x - 1] > 0
            br = alpha[y + 1, x + 1] > 0

            # Case 1: top-left and bottom-right connected diagonally
            if tl and br and not top and not left and not bot and not right:
                avg_color = (rgb[y - 1, x - 1].astype(np.int32) + rgb[y + 1, x + 1].astype(np.int32)) // 2
                result[y, x, :3] = avg_color.astype(np.uint8)
                result[y, x, 3] = 255

            # Case 2: top-right and bottom-left
            elif tr and bl and not top and not right and not bot and not left:
                avg_color = (rgb[y - 1, x + 1].astype(np.int32) + rgb[y + 1, x - 1].astype(np.int32)) // 2
                result[y, x, :3] = avg_color.astype(np.uint8)
                result[y, x, 3] = 255

    return result


def find_contour(image: np.ndarray) -> np.ndarray:
    """Find the outer contour of the sprite using Moore-neighbor tracing.

    Args:
        image: (H, W, 4) RGBA uint8

    Returns:
        (H, W) bool mask where True = contour pixel
    """
    alpha = image[:, :, 3] > 0
    H, W = alpha.shape
    contour = np.zeros((H, W), dtype=bool)

    for y in range(H):
        for x in range(W):
            if not alpha[y, x]:
                continue
            # A pixel is on the contour if any of its 4-neighbors is
            # transparent or out of bounds
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W or not alpha[ny, nx]:
                    contour[y, x] = True
                    break

    return contour
