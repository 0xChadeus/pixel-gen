"""Grid detection and pixel snapping for cleaning up generated pixel art.

Uses gradient analysis to detect if an image has been upscaled, then
downsamples to true pixel resolution using nearest-neighbor.
"""

import numpy as np
from scipy.signal import find_peaks


def detect_grid_size(image: np.ndarray) -> int:
    """Detect the pixel grid size of an image via gradient peak analysis.

    If the image was upscaled (e.g., 2x, 3x), the gradients will show
    peaks at regular intervals corresponding to the upscale factor.

    Args:
        image: (H, W, 3) or (H, W, 4) uint8 image

    Returns:
        Detected grid size (1 = native, 2 = 2x upscaled, etc.)
    """
    if image.ndim == 3 and image.shape[2] == 4:
        gray = np.mean(image[:, :, :3].astype(np.float32), axis=2)
    elif image.ndim == 3:
        gray = np.mean(image.astype(np.float32), axis=2)
    else:
        gray = image.astype(np.float32)

    H, W = gray.shape

    # Compute horizontal and vertical gradients
    grad_h = np.abs(np.diff(gray, axis=1)).sum(axis=0)  # (W-1,)
    grad_v = np.abs(np.diff(gray, axis=0)).sum(axis=1)  # (H-1,)

    best_grid = 1

    for grad, length in [(grad_h, W), (grad_v, H)]:
        if len(grad) < 8:
            continue

        # Autocorrelation to find periodicity
        grad_centered = grad - grad.mean()
        autocorr = np.correlate(grad_centered, grad_centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]  # positive lags only

        if autocorr[0] == 0:
            continue
        autocorr = autocorr / autocorr[0]

        # Find peaks in autocorrelation
        peaks, props = find_peaks(autocorr, height=0.3, distance=1)

        if len(peaks) > 0:
            # The first strong peak gives the grid period
            candidate = peaks[0]
            if 2 <= candidate <= 8:  # reasonable upscale factors
                if candidate > best_grid:
                    best_grid = candidate

    return best_grid


def downsample_to_grid(image: np.ndarray, grid_size: int) -> np.ndarray:
    """Downsample image by grid_size using geometric median sampling.

    For each grid_size x grid_size block, pick the most common color
    (mode). This preserves pixel art detail better than averaging.

    Args:
        image: (H, W, C) uint8 image
        grid_size: Detected grid size (2, 3, 4, etc.)

    Returns:
        (H//grid_size, W//grid_size, C) uint8 image
    """
    if grid_size <= 1:
        return image.copy()

    H, W = image.shape[:2]
    C = image.shape[2] if image.ndim == 3 else 1
    new_h = H // grid_size
    new_w = W // grid_size

    if image.ndim == 2:
        image = image[:, :, None]

    result = np.zeros((new_h, new_w, C), dtype=np.uint8)

    for y in range(new_h):
        for x in range(new_w):
            block = image[y * grid_size:(y + 1) * grid_size,
                          x * grid_size:(x + 1) * grid_size]
            block_flat = block.reshape(-1, C)

            # Find mode (most common color in block)
            unique, counts = np.unique(block_flat, axis=0, return_counts=True)
            result[y, x] = unique[counts.argmax()]

    if C == 1:
        return result.squeeze(-1)
    return result
