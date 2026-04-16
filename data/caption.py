"""Structured captioning pipeline for pixel art sprites.

Two-layer approach:
1. Florence-2 (0.77B params) for subject description ("a warrior with a sword")
2. Template assembly from algorithmically detected attributes (type, view, outline, etc.)

Usage:
    python -m data.caption --data-dir data/processed
    python -m data.caption --data-dir data/processed --skip-vlm  # attributes only
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def describe_with_florence(image_paths: list[Path], device: str = "cuda",
                           batch_size: int = 16) -> dict[str, str]:
    """Generate subject descriptions using Florence-2.

    Args:
        image_paths: List of sprite PNG paths
        device: Torch device
        batch_size: Inference batch size

    Returns:
        Dict mapping filename to cleaned subject description
    """
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = "microsoft/Florence-2-large"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    model.eval()

    descriptions = {}

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Florence-2"):
        batch_paths = image_paths[i:i + batch_size]
        images = []

        for p in batch_paths:
            img = Image.open(p).convert("RGBA")
            # Composite onto neutral gray background (not white — preserves edge contrast)
            bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
            bg.paste(img, mask=img.split()[3])
            rgb = bg.convert("RGB")
            # Upscale 4x nearest-neighbor for VLM
            w, h = rgb.size
            rgb = rgb.resize((w * 4, h * 4), Image.NEAREST)
            images.append(rgb)

        inputs = processor(
            text=["<CAPTION>"] * len(images),
            images=images,
            return_tensors="pt",
        ).to(device, torch.float16)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=3,
            )

        for j, p in enumerate(batch_paths):
            raw = processor.decode(out[j], skip_special_tokens=True).strip()
            descriptions[p.name] = _clean_vlm_output(raw)

    return descriptions


def _clean_vlm_output(text: str) -> str:
    """Clean up Florence-2 output for use as subject description.

    Strips VLM artifacts, generic prefixes, and redundant pixel art mentions
    since those are added structurally by the template.
    """
    text = text.lower().strip()

    # Remove common VLM prefixes
    for prefix in [
        "the image shows ", "this is ", "a picture of ", "an image of ",
        "a drawing of ", "the image depicts ", "the picture shows ",
        "a pixel art of ", "a pixelated ", "an 8-bit ", "a retro ",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):]

    # Remove redundant style words (we add these structurally)
    for word in ["pixel art", "pixelated", "8-bit", "16-bit", "retro style",
                 "pixel style", "sprite"]:
        text = text.replace(word, "")

    # Remove background descriptions (detected algorithmically)
    for phrase in ["on a white background", "on a black background",
                   "on a gray background", "on a grey background",
                   "with a transparent background", "with no background"]:
        text = text.replace(phrase, "")

    # Clean up extra spaces and trailing punctuation
    text = " ".join(text.split())
    text = text.strip(" .,;:")

    # If too short or generic, return empty (caller will use fallback)
    if len(text) < 3 or text in ("a", "an", "the", "image", "picture"):
        return ""

    return text


def assemble_caption(attributes: dict, subject: str) -> str:
    """Assemble a structured caption from attributes and subject description.

    Args:
        attributes: Dict from detect_all_attributes()
        subject: Cleaned VLM subject description (or empty string)

    Returns:
        Structured caption string
    """
    parts = []

    # Resolution + medium
    parts.append("128x128 pixel art")

    # Color count
    color_count = attributes.get("color_count")
    if color_count and color_count > 0:
        parts.append(f"{color_count} {'color' if color_count == 1 else 'colors'}")

    # Sprite type
    sprite_type = attributes.get("sprite_type", "sprite")
    if sprite_type == "tile":
        parts.append("tile")
    else:
        parts.append(f"{sprite_type} sprite")

    # View (skip for tiles and effects)
    view = attributes.get("view")
    if view and sprite_type not in ("tile", "effect"):
        parts.append(view)

    # Subject description (from VLM or fallback)
    if subject:
        parts.append(subject)

    # Color temperature (only if notable)
    temp = attributes.get("temperature")
    if temp:
        parts.append(temp)

    # Saturation (only if extreme — muted is pixel art default)
    sat = attributes.get("saturation")
    if sat:
        parts.append(sat)

    # Outline style
    outline = attributes.get("outline_style")
    if outline:
        parts.append(outline)

    # Background
    bg = attributes.get("background")
    if bg:
        parts.append(bg)

    return ", ".join(parts)


def assemble_caption_short(subject: str) -> str:
    """Assemble a short caption (mimics how users prompt at inference)."""
    if subject:
        return f"pixel art {subject}"
    return "pixel art sprite"


def main():
    parser = argparse.ArgumentParser(description="Caption pixel art sprites")
    parser.add_argument("--data-dir", required=True, help="Processed data directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip Florence-2, use sprite type as subject fallback")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) / "128"

    image_paths = sorted(data_dir.glob("*.png"))
    if not image_paths:
        print("No images found in data_dir/128/")
        return

    print(f"\nCaptioning {len(image_paths)} images at 128x128...")

    # Get VLM subject descriptions
    if not args.skip_vlm:
        descriptions = describe_with_florence(
            image_paths, args.device, args.batch_size,
        )
    else:
        descriptions = {}

    # Assemble captions from attributes + VLM
    for p in tqdm(image_paths, desc="Assembling captions"):
        meta_path = p.with_suffix(".json")
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        attributes = meta.get("attributes", {})

        # Subject: VLM output, or fallback to sprite type
        subject = descriptions.get(p.name, "")
        if not subject:
            st = attributes.get("sprite_type", "sprite")
            if st == "tile":
                subject = ""
            elif st in ("object", "effect"):
                subject = f"an {st}"
            else:
                subject = f"a {st}"

        caption = assemble_caption(attributes, subject)
        caption_short = assemble_caption_short(subject)

        meta["caption"] = caption
        meta["caption_short"] = caption_short
        if p.name in descriptions:
            meta["vlm_raw"] = descriptions[p.name]

        with open(meta_path, "w") as f:
            json.dump(meta, f)

    print(f"Done: {len(image_paths)} captions written")


if __name__ == "__main__":
    main()
