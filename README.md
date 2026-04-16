# Pixel Gen

True pixel art generation via EDM diffusion. Not "AI art downscaled to look pixely" -- this produces
clean, grid-aligned pixel art with binary alpha, quantized palettes, and no anti-aliasing artifacts.

## How It Works

A ~134M parameter UNet denoiser operates in **OKLab color space** (perceptually uniform, so MSE loss
and palette quantization produce visually meaningful results). The model is conditioned on:

- **Text** -- CLIP ViT-L/14 pooled + token embeddings (cross-attention)
- **Palette** -- up to 32 OKLab colors processed by a small transformer, enabling palette-locked generation
- **Resolution** -- learned embeddings for 32x32, 64x64, and 128x128
- **Timestep** -- Fourier feature encoding (EDM-style)

All conditioning signals are merged into a single FiLM vector + cross-attention token sequence via
`ConditioningAssembler`.

Sampling uses the **Heun 2nd-order ODE sampler** with classifier-free guidance (default scale 5.0,
35 steps). Self-conditioning feeds the previous denoised estimate back into the network for
improved coherence.

### Post-Processing Pipeline

Raw model output goes through a cleanup pipeline that enforces pixel art constraints:

1. **Alpha snap** -- threshold to binary (fully opaque or fully transparent)
2. **Color quantization** -- snap to a provided palette, or extract one via K-means in OKLab
3. **AA removal** -- detect and replace rare blended-color pixels with their nearest neighbor
4. **Dithering** (optional) -- ordered (Bayer 4x4) or Floyd-Steinberg error diffusion
5. **Outline cleanup** -- fill diagonal-only gaps in sprite contours

### Training

- EDM preconditioning (Karras et al. 2022) with resolution-aware noise schedule
- Weighted MSE loss + optional LPIPS perceptual loss for low-noise samples
- Pixel-art-safe augmentations only: horizontal/vertical flip, palette recoloring via Lospec palettes.
  No rotation, scaling, or shearing
- Gradient checkpointing supported

## Project Structure

```
pixel_gen/
  model/
    unet.py           # EDM UNet (~134M params)
    diffusion.py      # EDM preconditioning, loss, Heun sampler, CFG wrapper
    blocks.py         # ResBlock, FiLM, self/cross-attention, up/downsample
    conditioning.py   # Timestep, palette, resolution, CLIP text conditioning
  training/
    augment.py        # Pixel-art-safe augmentations
  server/
    main.py           # Entry point (WebSocket server)
    config.py         # ServerConfig dataclass
    ws_handler.py     # WebSocket handler (JSON protocol)
    inference/
      pipeline.py     # Text prompt -> pixel art image
      clip_encoder.py # Frozen CLIP text encoder
    postprocess/
      pipeline.py     # Post-processing orchestrator
      quantize.py     # K-means palette extraction + nearest-color quantization
      aa_removal.py   # Anti-aliasing artifact removal
      outline.py      # Outline gap filling + contour detection
      dither.py       # Ordered and Floyd-Steinberg dithering
      grid_snap.py    # Grid detection + downsampling for upscaled inputs
    utils/
      color.py        # OKLab <-> sRGB (numpy + torch)
      image_io.py     # RGBA serialization, PNG encoding, hex parsing
  data/
    lospec_scraper.py # Download palettes from Lospec as GIMP .gpl files
  aseprite_plugin/
    plugin.lua        # Aseprite extension: generate dialog, WebSocket client
    sprite_utils.lua  # Apply generated pixels + read active palette
    json.lua          # Minimal JSON encoder/decoder for Lua
    package.json      # Extension manifest
  scripts/
    build_extension.sh  # Package plugin as .aseprite-extension
    install_plugin.sh   # Symlink plugin into Aseprite's extensions dir
  checkpoints/        # Model weights (gitignored)
  samples/            # Training sample snapshots
```

## Setup

```bash
cd pixel_gen
python -m venv .venv
source .venv/bin/activate

# PyTorch -- install for your GPU first:
# ROCm:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
# CUDA:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Usage

### Inference Server

```bash
python -m server.main --checkpoint checkpoints/model.safetensors --device cuda
```

The server listens on `ws://127.0.0.1:9847` and accepts JSON messages over WebSocket.

#### Generate Request

```json
{
  "action": "generate",
  "prompt": "pixel art knight character, side view",
  "size": 128,
  "palette": ["#1a1c2c", "#5d275d", "#b13e53", "#ef7d57", "#ffcd75"],
  "guidance_scale": 5.0,
  "steps": 35,
  "seed": 42,
  "dither_mode": null,
  "outline_cleanup": true,
  "num_colors": 16
}
```

The server streams `progress` messages, then a `result` JSON header followed by raw RGBA bytes.

### Aseprite Plugin

Install the plugin into Aseprite:

```bash
# Option A: symlink for development
./scripts/install_plugin.sh

# Option B: build a .aseprite-extension package
./scripts/build_extension.sh
# Then: Edit > Preferences > Extensions > Add Extension
```

With the server running, open Aseprite and use **Pixel Gen > Generate Sprite...** from the menu.
The dialog lets you set prompt, size, palette source, guidance scale, sampling steps, seed,
dithering mode, and outline cleanup. Generated sprites are applied directly to the active canvas
with the palette set on the sprite.

### Lospec Palette Scraper

```bash
python -m data.lospec_scraper --output aseprite_plugin/palettes
```

Downloads popular palettes (PICO-8, Endesga-32, Sweetie-16, etc.) as `.gpl` files for use in
training augmentation and the Aseprite plugin.
