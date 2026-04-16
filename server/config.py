"""Server configuration."""

from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9847
    checkpoint_path: str = ""
    device: str = "cuda"
    clip_model: str = "openai/clip-vit-large-patch14"

    # Model config
    base_channels: int = 128
    channel_mults: tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 3
    attention_resolutions: tuple[int, ...] = (32, 16)
    num_heads: int = 8
    cond_dim: int = 512
    cross_attn_dim: int = 512
    dropout: float = 0.0  # no dropout at inference
    self_condition: bool = True
    in_channels: int = 4
    sigma_data: float = 0.5

    # Sampler defaults
    default_num_steps: int = 35
    default_guidance_scale: float = 5.0
