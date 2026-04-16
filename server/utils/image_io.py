"""Image serialization utilities for WebSocket transport."""

import io
import numpy as np
from PIL import Image


def image_to_rgba_bytes(image: np.ndarray) -> bytes:
    """Convert RGBA uint8 numpy array to raw bytes (row-major, 4 bytes/pixel).

    Args:
        image: (H, W, 4) RGBA uint8

    Returns:
        Raw bytes of length H * W * 4
    """
    assert image.ndim == 3 and image.shape[2] == 4 and image.dtype == np.uint8
    return image.tobytes()


def rgba_bytes_to_image(data: bytes, width: int, height: int) -> np.ndarray:
    """Convert raw RGBA bytes back to numpy array.

    Args:
        data: Raw bytes of length width * height * 4
        width: Image width
        height: Image height

    Returns:
        (height, width, 4) RGBA uint8
    """
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4).copy()


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode RGBA uint8 numpy array as PNG bytes."""
    pil = Image.fromarray(image, "RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_to_image(data: bytes) -> np.ndarray:
    """Decode PNG bytes to RGBA uint8 numpy array."""
    pil = Image.open(io.BytesIO(data)).convert("RGBA")
    return np.array(pil)


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_str = hex_str.lstrip("#")
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def palette_hex_to_array(hex_list: list[str]) -> np.ndarray:
    """Convert list of hex color strings to (N, 3) RGB uint8 array."""
    return np.array([hex_to_rgb(h) for h in hex_list], dtype=np.uint8)
