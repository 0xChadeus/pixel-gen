"""Post-processing pipeline orchestrator.

Runs the full cleanup sequence to ensure generated images are "true" pixel art:
binary alpha, quantized colors, no AA artifacts, clean outlines.
"""

import numpy as np

from server.postprocess.aa_removal import snap_alpha, remove_aa
from server.postprocess.quantize import quantize_to_palette, quantize_kmeans
from server.postprocess.outline import cleanup_outlines
from server.postprocess.dither import dither_ordered, dither_floyd_steinberg


def postprocess(
    image: np.ndarray,
    palette: np.ndarray | None = None,
    num_colors: int = 16,
    dither_mode: str | None = None,
    outline_cleanup: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full post-processing pipeline.

    Args:
        image: (H, W, 4) RGBA uint8 raw model output
        palette: (N, 3) RGB uint8 target palette, or None for auto
        num_colors: Number of colors for auto palette extraction
        dither_mode: None, "ordered", or "floyd_steinberg"
        outline_cleanup: Whether to fill diagonal gaps

    Returns:
        processed: (H, W, 4) RGBA uint8 clean pixel art
        palette_out: (N, 3) RGB uint8 final palette used
    """
    # 1. Snap alpha to binary
    image = snap_alpha(image)

    # 2. Color quantization
    if palette is not None:
        image = quantize_to_palette(image, palette)
        palette_out = palette
    else:
        image, palette_out = quantize_kmeans(image, num_colors)

    # 3. Remove anti-aliasing artifacts
    image = remove_aa(image)

    # 4. Optional dithering (re-quantizes with dither pattern)
    if dither_mode == "ordered":
        image = dither_ordered(image, palette_out)
    elif dither_mode == "floyd_steinberg":
        image = dither_floyd_steinberg(image, palette_out)

    # 5. Outline cleanup
    if outline_cleanup:
        image = cleanup_outlines(image)

    return image, palette_out
