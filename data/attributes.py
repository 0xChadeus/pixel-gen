"""Algorithmic sprite attribute detection.

Pure numpy/PIL analysis — no ML models. Detects sprite type, view direction,
outline style, color characteristics, and background type from pixel data.
"""

import numpy as np
from pathlib import Path

from server.utils.color import srgb_to_oklab
from server.postprocess.outline import find_contour


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Get tight bounding box (y_min, y_max, x_min, x_max) of True pixels."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, 0, 0, 0
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(y_min), int(y_max + 1), int(x_min), int(x_max + 1)


def detect_sprite_type(img: np.ndarray) -> str:
    """Classify sprite as character, object, tile, or effect.

    Args:
        img: (H, W, 4) RGBA uint8

    Returns:
        One of: "character", "object", "tile", "effect"
    """
    H, W = img.shape[:2]
    alpha = img[:, :, 3]
    opaque = alpha > 0
    total_pixels = H * W

    fill_ratio = opaque.sum() / total_pixels

    # Bounding box metrics
    y0, y1, x0, x1 = _bounding_box(opaque)
    bbox_h = max(y1 - y0, 1)
    bbox_w = max(x1 - x0, 1)
    bbox_area = bbox_h * bbox_w
    pixel_density = opaque.sum() / max(bbox_area, 1)
    aspect = bbox_h / max(bbox_w, 1)

    # Tile: fills nearly the entire canvas
    if fill_ratio > 0.90:
        # Check tileability: compare top/bottom and left/right edges
        top = img[0, :, :3]
        bot = img[-1, :, :3]
        left = img[:, 0, :3]
        right = img[:, -1, :3]
        h_match = np.mean(np.abs(top.astype(int) - bot.astype(int))) < 30
        v_match = np.mean(np.abs(left.astype(int) - right.astype(int))) < 30
        if h_match or v_match:
            return "tile"

    # Effect: very sparse AND low density within bounding box (scattered particles, not a solid blob)
    if fill_ratio < 0.10 and pixel_density < 0.30:
        return "effect"

    # Object: small fill but solid within its bbox, roughly square
    if fill_ratio < 0.40 and pixel_density > 0.50 and 0.5 < aspect < 2.0:
        return "object"

    # Character: everything else (typically taller than wide)
    return "character"


def detect_view(img: np.ndarray, source_path: str | None = None) -> str:
    """Detect viewing angle of a sprite.

    Args:
        img: (H, W, 4) RGBA uint8
        source_path: Original filename/path for hint extraction

    Returns:
        One of: "front view", "side view", "3/4 view", "back view", "top-down view"
    """
    # Filename hints (highest priority — LPC dataset encodes views)
    if source_path:
        name = str(source_path).lower()
        if "front" in name or "_down" in name:
            return "front view"
        if "back" in name or "_up" in name:
            return "back view"
        if "side" in name or "_left" in name or "_right" in name:
            return "side view"

    H, W = img.shape[:2]
    alpha = img[:, :, 3]
    opaque = alpha > 0

    if opaque.sum() < 4:
        return "front view"

    # Compute bilateral symmetry
    # Flip the image horizontally and compare
    flipped_alpha = np.flip(opaque, axis=1)
    flipped_rgb = np.flip(img[:, :, :3], axis=1)

    # Alpha symmetry
    both_opaque = opaque & flipped_alpha
    if both_opaque.sum() < 4:
        return "side view"

    alpha_sym = (opaque == flipped_alpha).mean()

    # Color symmetry (only where both sides are opaque)
    orig_colors = img[:, :, :3][both_opaque].astype(np.float32)
    flip_colors = flipped_rgb[both_opaque].astype(np.float32)
    color_diff = np.mean(np.abs(orig_colors - flip_colors)) / 255.0
    color_sym = 1.0 - color_diff

    # Combined symmetry score
    symmetry = alpha_sym * 0.4 + color_sym * 0.6

    if symmetry > 0.85:
        return "front view"
    elif symmetry < 0.55:
        return "side view"
    else:
        return "3/4 view"


def detect_outline_style(img: np.ndarray) -> str:
    """Detect outline style of a sprite.

    Args:
        img: (H, W, 4) RGBA uint8

    Returns:
        One of: "thick outline", "thin outline", "no outline"
    """
    H, W = img.shape[:2]
    alpha = img[:, :, 3]
    opaque = alpha > 0

    if opaque.sum() < 8:
        return "no outline"

    contour = find_contour(img)

    if contour.sum() == 0:
        return "no outline"

    # Compare lightness of contour vs interior
    interior = opaque & ~contour
    if interior.sum() == 0:
        return "no outline"

    rgb = img[:, :, :3]
    contour_pixels = rgb[contour].astype(np.float32)
    interior_pixels = rgb[interior].astype(np.float32)

    contour_lum = np.mean(contour_pixels)
    interior_lum = np.mean(interior_pixels)

    darkness_ratio = contour_lum / max(interior_lum, 1.0)

    if darkness_ratio > 0.8:
        return "no outline"

    # Check outline thickness: erode opaque mask, see if second layer is also dark
    from scipy.ndimage import binary_erosion
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    eroded = binary_erosion(opaque, kernel)
    eroded_2 = binary_erosion(eroded, kernel)
    second_layer = eroded & ~eroded_2

    if second_layer.sum() > 0:
        second_layer_lum = np.mean(rgb[second_layer].astype(np.float32))
        if second_layer_lum / max(interior_lum, 1.0) < 0.7:
            return "thick outline"

    return "thin outline"


def detect_color_characteristics(img: np.ndarray, palette: list | np.ndarray | None) -> dict:
    """Analyze color properties of a sprite.

    Args:
        img: (H, W, 4) RGBA uint8
        palette: (N, 3) RGB palette or list of [r,g,b]

    Returns:
        Dict with color_count, temperature, saturation keys
    """
    alpha = img[:, :, 3]
    rgb = img[:, :, :3]
    opaque = alpha > 0
    opaque_pixels = rgb[opaque]

    if len(opaque_pixels) == 0:
        return {"color_count": 0, "temperature": None, "saturation": None}

    # Color count
    if palette is not None and len(palette) > 0:
        color_count = len(palette)
    else:
        color_count = len(np.unique(opaque_pixels, axis=0))

    # Convert to HSV for temperature and saturation analysis
    from PIL import Image as PILImage
    # Work with opaque pixels in HSV
    hsv_pixels = _rgb_to_hsv(opaque_pixels)
    hues = hsv_pixels[:, 0]       # [0, 360)
    sats = hsv_pixels[:, 1]       # [0, 1]

    # Temperature: warm (reds/oranges/yellows 0-60, 300-360) vs cool (120-270)
    warm_mask = (hues < 60) | (hues > 300)
    cool_mask = (hues > 120) & (hues < 270)
    # Only count sufficiently saturated pixels (grays are neutral)
    sat_threshold = 0.15
    chromatic = sats > sat_threshold
    warm_count = (warm_mask & chromatic).sum()
    cool_count = (cool_mask & chromatic).sum()
    total_chromatic = max(chromatic.sum(), 1)

    if warm_count / total_chromatic > 0.6:
        temperature = "warm palette"
    elif cool_count / total_chromatic > 0.6:
        temperature = "cool palette"
    else:
        temperature = None  # neutral, omit from caption

    # Saturation
    avg_sat = np.mean(sats)
    if avg_sat > 0.6:
        saturation = "vibrant"
    elif avg_sat < 0.2:
        saturation = "desaturated"
    else:
        saturation = None  # muted is default for pixel art, omit

    return {
        "color_count": int(color_count),
        "temperature": temperature,
        "saturation": saturation,
    }


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert (N, 3) RGB uint8 to (N, 3) HSV (hue 0-360, sat 0-1, val 0-1)."""
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[:, 0], rgb_f[:, 1], rgb_f[:, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    hue = np.zeros_like(delta)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    hue[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Saturation
    sat = np.where(cmax > 0, delta / cmax, 0)

    return np.stack([hue, sat, cmax], axis=1)


def detect_background_type(img: np.ndarray) -> str:
    """Detect background type of a sprite.

    Args:
        img: (H, W, 4) RGBA uint8

    Returns:
        One of: "transparent background", "solid background", "scene background"
    """
    H, W = img.shape[:2]
    alpha = img[:, :, 3]

    transparent_ratio = (alpha == 0).sum() / (H * W)
    opaque_ratio = (alpha == 255).sum() / (H * W)

    if transparent_ratio > 0.10:
        return "transparent background"

    if opaque_ratio > 0.95:
        # Check if border is uniform color
        border = np.concatenate([
            img[0, :, :3],       # top row
            img[-1, :, :3],      # bottom row
            img[:, 0, :3],       # left col
            img[:, -1, :3],      # right col
        ])
        color_std = np.std(border.astype(np.float32), axis=0).mean()
        if color_std < 15:
            return "solid background"
        return "scene background"

    return "transparent background"


def detect_all_attributes(img: np.ndarray, palette: list | np.ndarray | None = None,
                          source_path: str | None = None) -> dict:
    """Run all attribute detectors on a sprite.

    Args:
        img: (H, W, 4) RGBA uint8
        palette: Extracted palette (list of [r,g,b] or ndarray)
        source_path: Original filename for hint extraction

    Returns:
        Dict with all detected attributes
    """
    sprite_type = detect_sprite_type(img)
    color_chars = detect_color_characteristics(img, palette)

    # View detection: skip for tiles and effects
    if sprite_type in ("tile", "effect"):
        view = "top-down view" if sprite_type == "tile" else None
    else:
        view = detect_view(img, source_path)

    return {
        "sprite_type": sprite_type,
        "view": view,
        "outline_style": detect_outline_style(img),
        "background": detect_background_type(img),
        **color_chars,
    }
