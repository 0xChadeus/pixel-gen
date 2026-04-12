"""Quality filtering for pixel art images.

Rejects images that are not true pixel art: too many colors, AA edges,
wrong resolution, upscaled, etc.
"""

import numpy as np
from PIL import Image


def is_valid_pixel_art(img: np.ndarray, max_colors: int = 256,
                       max_native_size: int = 256, min_size: int = 8,
                       max_aa_ratio: float = 0.05) -> tuple[bool, str]:
    """Check if an image qualifies as true pixel art.

    Args:
        img: (H, W, 4) RGBA uint8
        max_colors: Reject if more unique colors than this
        max_native_size: Reject if native res exceeds this
        min_size: Reject if smaller than this
        max_aa_ratio: Reject if more than this fraction of pixels have semi-transparent alpha

    Returns:
        (valid, reason) where reason explains rejection
    """
    H, W = img.shape[:2]

    # Size checks
    if H < min_size or W < min_size:
        return False, f"too small ({W}x{H})"
    if H > max_native_size or W > max_native_size:
        return False, f"too large ({W}x{H})"

    alpha = img[:, :, 3]
    rgb = img[:, :, :3]

    # Count unique colors (opaque pixels only)
    opaque = alpha > 0
    opaque_rgb = rgb[opaque]
    if len(opaque_rgb) == 0:
        return False, "fully transparent"

    unique_colors = len(np.unique(opaque_rgb, axis=0))
    if unique_colors > max_colors:
        return False, f"too many colors ({unique_colors})"

    # Check for anti-aliasing (semi-transparent pixels)
    semi_transparent = (alpha > 10) & (alpha < 245)
    aa_ratio = semi_transparent.sum() / max(opaque.sum(), 1)
    if aa_ratio > max_aa_ratio:
        return False, f"too much AA ({aa_ratio:.2%} semi-transparent)"

    return True, "ok"


def detect_effective_resolution(img: np.ndarray, threshold: float = 0.55) -> int:
    """Estimate the true pixel resolution of an image that may be upscaled.

    Checks block uniformity at several scales. If most NxN blocks are
    a single color, the image is effectively (img_size / N) resolution.

    Args:
        img: (H, W, 4) RGBA uint8
        threshold: fraction of uniform blocks to consider upscaled

    Returns:
        Effective resolution (max dimension) of the image.
    """
    H, W = img.shape[:2]
    rgb = img[:, :, :3]

    for block_size in [8, 6, 5, 4, 3, 2]:
        if H < block_size * 4 or W < block_size * 4:
            continue  # too few blocks to be meaningful

        uniform = 0
        total = 0
        for y in range(0, H - block_size + 1, block_size):
            for x in range(0, W - block_size + 1, block_size):
                block = rgb[y:y + block_size, x:x + block_size]
                total += 1
                if np.all(block == block[0, 0]):
                    uniform += 1

        if total > 0 and (uniform / total) > threshold:
            effective = max(H // block_size, W // block_size)
            return effective

    return max(H, W)


def filter_directory(input_dir: str, output_dir: str, **kwargs):
    """Filter a directory of images, copying valid ones to output."""
    from pathlib import Path
    import shutil

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0

    for img_path in sorted(in_path.glob("*.png")):
        try:
            img = np.array(Image.open(img_path).convert("RGBA"))
            valid, reason = is_valid_pixel_art(img, **kwargs)
            if valid:
                shutil.copy2(img_path, out_path / img_path.name)
                accepted += 1
            else:
                rejected += 1
        except Exception as e:
            rejected += 1

    print(f"Accepted: {accepted}, Rejected: {rejected}")
    return accepted, rejected
