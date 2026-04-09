"""Color quantization in OKLab space using K-means."""

import numpy as np
from sklearn.cluster import KMeans

from server.utils.color import srgb_to_oklab, oklab_to_srgb


def quantize_to_palette(image: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """Map each pixel to the nearest color in the given palette.

    Args:
        image: (H, W, 4) RGBA uint8 image
        palette_rgb: (N, 3) RGB uint8 palette colors

    Returns:
        (H, W, 4) RGBA uint8 image with colors from palette only
    """
    H, W, _ = image.shape
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]

    img_lab = srgb_to_oklab(rgb).reshape(-1, 3)
    pal_lab = srgb_to_oklab(palette_rgb)

    # Find nearest palette color for each pixel (Euclidean in OKLab)
    # Broadcasting: (N_pixels, 1, 3) - (1, N_colors, 3) -> (N_pixels, N_colors)
    dists = np.linalg.norm(img_lab[:, None, :] - pal_lab[None, :, :], axis=2)
    nearest_idx = dists.argmin(axis=1)

    quantized_rgb = palette_rgb[nearest_idx].reshape(H, W, 3)
    return np.dstack([quantized_rgb, alpha])


def quantize_kmeans(image: np.ndarray, num_colors: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Extract an optimal palette via K-means in OKLab, then quantize.

    Args:
        image: (H, W, 4) RGBA uint8 image
        num_colors: Target palette size

    Returns:
        quantized: (H, W, 4) RGBA uint8
        palette: (num_colors, 3) RGB uint8 palette
    """
    H, W, _ = image.shape
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]

    # Only cluster opaque pixels
    opaque_mask = alpha.reshape(-1) > 0
    opaque_rgb = rgb.reshape(-1, 3)[opaque_mask]

    if len(opaque_rgb) == 0:
        palette = np.zeros((num_colors, 3), dtype=np.uint8)
        return image.copy(), palette

    opaque_lab = srgb_to_oklab(opaque_rgb)

    actual_k = min(num_colors, len(np.unique(opaque_rgb, axis=0)))
    if actual_k < num_colors:
        num_colors = actual_k

    kmeans = KMeans(n_clusters=num_colors, n_init=10, max_iter=30, random_state=42)
    kmeans.fit(opaque_lab)

    # Get palette in RGB
    palette_lab = kmeans.cluster_centers_.astype(np.float32)
    palette_rgb = oklab_to_srgb(palette_lab)

    # Quantize the full image to this palette
    return quantize_to_palette(image, palette_rgb), palette_rgb
