"""OKLab <-> sRGB color space conversion.

OKLab is perceptually uniform: Euclidean distance corresponds to perceived
color difference. This makes MSE loss and K-means quantization produce
visually meaningful results.

Reference: Bjorn Ottosson's OKLab specification.
"""

import numpy as np
import torch


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB [0,1]."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0,1] to sRGB [0,1]."""
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1.0 / 2.4) - 0.055)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 [0,255] or float [0,1] to OKLab.

    Args:
        rgb: (..., 3) array in sRGB. uint8 [0,255] or float [0,1].

    Returns:
        (..., 3) array in OKLab. L in [0,1], a,b in ~[-0.5, 0.5].
    """
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)

    linear = _srgb_to_linear(rgb)

    # Linear RGB -> LMS (Oklab M1 matrix)
    l = 0.4122214708 * linear[..., 0] + 0.5363325363 * linear[..., 1] + 0.0514459929 * linear[..., 2]
    m = 0.2119034982 * linear[..., 0] + 0.6806995451 * linear[..., 1] + 0.1073969566 * linear[..., 2]
    s = 0.0883024619 * linear[..., 0] + 0.2817188376 * linear[..., 1] + 0.6299787005 * linear[..., 2]

    # Cube root
    l_ = np.cbrt(np.clip(l, 0, None))
    m_ = np.cbrt(np.clip(m, 0, None))
    s_ = np.cbrt(np.clip(s, 0, None))

    # LMS' -> OKLab (Oklab M2 matrix)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return np.stack([L, a, b], axis=-1)


def oklab_to_srgb(lab: np.ndarray, clip: bool = True) -> np.ndarray:
    """Convert OKLab to sRGB uint8 [0,255].

    Args:
        lab: (..., 3) array in OKLab.
        clip: If True, clip output to [0, 255].

    Returns:
        (..., 3) array in sRGB uint8.
    """
    lab = lab.astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # OKLab -> LMS' (inverse M2)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    # Cube
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    # LMS -> linear RGB (inverse M1)
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_out = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    linear = np.stack([r, g, b_out], axis=-1)
    srgb = _linear_to_srgb(linear)

    if clip:
        srgb = np.clip(srgb, 0.0, 1.0)

    return (srgb * 255.0).astype(np.uint8)


# --- Torch versions for use in training/inference pipeline ---

def srgb_to_oklab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB float [0,1] tensor to OKLab.

    Args:
        rgb: (..., 3) tensor in sRGB float [0,1].

    Returns:
        (..., 3) tensor in OKLab.
    """
    # sRGB -> linear
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = torch.clamp(l, min=0).pow(1.0 / 3.0)
    m_ = torch.clamp(m, min=0).pow(1.0 / 3.0)
    s_ = torch.clamp(s, min=0).pow(1.0 / 3.0)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_out = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return torch.stack([L, a, b_out], dim=-1)


def oklab_to_srgb_torch(lab: torch.Tensor) -> torch.Tensor:
    """Convert OKLab tensor to sRGB float [0,1].

    Args:
        lab: (..., 3) tensor in OKLab.

    Returns:
        (..., 3) tensor in sRGB float [0,1], clamped.
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_out = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    linear = torch.stack([r, g, b_out], dim=-1)

    srgb = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * torch.clamp(linear, min=0).pow(1.0 / 2.4) - 0.055,
    )

    return torch.clamp(srgb, 0.0, 1.0)


def normalize_oklab(lab: torch.Tensor) -> torch.Tensor:
    """Normalize OKLab to approximately [-1, 1] range for diffusion model input.

    L: [0,1] -> [-1,1] via 2*L - 1
    a,b: ~[-0.35, 0.35] -> ~[-1,1] via *3 (no clamp to avoid losing saturated colors)
    """
    L = lab[..., 0:1] * 2.0 - 1.0
    ab = lab[..., 1:3] * 3.0
    return torch.cat([L, ab], dim=-1)


def denormalize_oklab(normalized: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1,1] back to OKLab range."""
    L = (normalized[..., 0:1] + 1.0) / 2.0
    ab = normalized[..., 1:3] / 3.0
    return torch.cat([L, ab], dim=-1)
