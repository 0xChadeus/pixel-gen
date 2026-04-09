"""Anti-aliasing removal for pixel art.

Detects and removes semi-transparent edge pixels and color blending
artifacts that break the clean pixel grid.
"""

import numpy as np


def snap_alpha(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Snap all alpha values to fully opaque or fully transparent.

    Args:
        image: (H, W, 4) RGBA uint8
        threshold: Alpha values >= threshold become 255, below become 0

    Returns:
        (H, W, 4) RGBA uint8 with binary alpha
    """
    result = image.copy()
    result[:, :, 3] = np.where(result[:, :, 3] >= threshold, 255, 0)
    return result


def remove_aa(image: np.ndarray) -> np.ndarray:
    """Remove anti-aliasing artifacts from pixel art.

    For each opaque pixel, check its 8 neighbors. If the pixel's color
    is unique (appears only once) and lies between two differently-colored
    regions, replace it with the nearest neighbor color. This removes
    the blended transition pixels that AA creates.

    Args:
        image: (H, W, 4) RGBA uint8 (should have binary alpha already)

    Returns:
        (H, W, 4) RGBA uint8 with AA removed
    """
    H, W, _ = image.shape
    result = image.copy()
    rgb = image[:, :, :3].astype(np.float32)
    alpha = image[:, :, 3]

    # Build color frequency map (only opaque pixels)
    opaque_mask = alpha > 0
    opaque_pixels = rgb[opaque_mask]

    if len(opaque_pixels) == 0:
        return result

    # Quantize colors to reduce near-duplicates for frequency counting
    # Round to nearest 8 to group very similar colors
    quantized = (opaque_pixels / 8).astype(np.int32)
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    color_freq = {}
    for u, c in zip(unique, counts):
        color_freq[tuple(u)] = c

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(H):
        for x in range(W):
            if alpha[y, x] == 0:
                continue

            pixel_q = tuple((rgb[y, x] / 8).astype(np.int32))
            freq = color_freq.get(pixel_q, 0)

            # If this color appears very rarely, it might be an AA artifact
            if freq > 3:
                continue

            # Check neighbors
            neighbor_colors = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and alpha[ny, nx] > 0:
                    neighbor_colors.append(rgb[ny, nx])

            if len(neighbor_colors) < 2:
                continue

            # Check if this pixel's color is between two neighbor colors
            # (i.e., it's an interpolation artifact)
            pixel_color = rgb[y, x]
            neighbor_arr = np.array(neighbor_colors)
            dists = np.linalg.norm(neighbor_arr - pixel_color, axis=1)

            # Replace with the closest neighbor
            nearest_idx = dists.argmin()
            avg_dist = dists.mean()

            # Only replace if the pixel is close to its neighbors
            # (true AA artifacts are blends, not outliers)
            if dists[nearest_idx] < 60 and avg_dist < 120:
                result[y, x, :3] = neighbor_colors[nearest_idx].astype(np.uint8)

    return result
