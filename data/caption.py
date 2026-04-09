"""Auto-captioning for pixel art sprites using BLIP-2.

Upscales sprites 8x via nearest-neighbor before captioning (BLIP needs
reasonable resolution). Adds structured prefix with resolution and color count.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def caption_with_blip(image_paths: list[Path], device: str = "cuda",
                      batch_size: int = 8) -> dict[str, str]:
    """Generate captions for a batch of images using BLIP-2.

    Args:
        image_paths: List of image file paths
        device: Torch device
        batch_size: Batch size for inference

    Returns:
        Dict mapping filename to caption string
    """
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    captions = {}

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Captioning"):
        batch_paths = image_paths[i:i + batch_size]
        images = []

        for p in batch_paths:
            img = Image.open(p).convert("RGBA")
            # Create white background composite for BLIP
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            rgb = bg.convert("RGB")
            # Upscale 8x for BLIP
            w, h = rgb.size
            rgb = rgb.resize((w * 8, h * 8), Image.NEAREST)
            images.append(rgb)

        inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)

        for j, p in enumerate(batch_paths):
            raw_caption = processor.decode(out[j], skip_special_tokens=True).strip()
            captions[p.name] = raw_caption

    return captions


def enrich_caption(raw_caption: str, size: int, num_colors: int) -> str:
    """Add structured prefix to a raw BLIP caption.

    Example: "32x32 pixel art sprite, 16 colors, a knight holding a sword"
    """
    # Clean up BLIP output
    caption = raw_caption.lower().strip()
    # Remove common BLIP artifacts
    for prefix in ["a picture of ", "an image of ", "a photo of ", "a drawing of "]:
        if caption.startswith(prefix):
            caption = caption[len(prefix):]

    return f"{size}x{size} pixel art sprite, {num_colors} colors, {caption}"


def main():
    parser = argparse.ArgumentParser(description="Auto-caption pixel art sprites")
    parser.add_argument("--data-dir", required=True, help="Processed data directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for size_dir in sorted(data_dir.iterdir()):
        if not size_dir.is_dir():
            continue
        try:
            size = int(size_dir.name)
        except ValueError:
            continue

        image_paths = sorted(size_dir.glob("*.png"))
        if not image_paths:
            continue

        print(f"\nCaptioning {len(image_paths)} images at {size}x{size}...")

        captions = caption_with_blip(image_paths, args.device, args.batch_size)

        # Update metadata files
        for p in image_paths:
            meta_path = p.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = {}

            raw = captions.get(p.name, "pixel art sprite")
            num_colors = len(meta.get("palette", []))
            meta["caption"] = enrich_caption(raw, size, max(num_colors, 1))

            with open(meta_path, "w") as f:
                json.dump(meta, f)

    print("\nDone.")


if __name__ == "__main__":
    main()
