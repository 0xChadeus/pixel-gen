"""Precompute CLIP text embeddings for all captioned sprites.

Run once after captioning. Saves embeddings as .pt files alongside each image,
eliminating the need for CLIP on GPU during training.

Usage:
    python -m data.cache_embeddings --data-dir data/processed
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def main():
    parser = argparse.ArgumentParser(description="Cache CLIP embeddings to disk")
    parser.add_argument("--data-dir", required=True, help="Processed data directory")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading CLIP: {args.clip_model}")
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    model = CLIPTextModel.from_pretrained(args.clip_model).to(args.device)
    model.eval()
    model.requires_grad_(False)

    data_dir = Path(args.data_dir)

    for size_dir in sorted(data_dir.iterdir()):
        if not size_dir.is_dir():
            continue

        json_files = sorted(size_dir.glob("*.json"))
        if not json_files:
            continue

        print(f"\nCaching {len(json_files)} embeddings in {size_dir.name}/...")

        # Collect all captions
        captions = []
        paths = []
        for jf in json_files:
            with open(jf) as f:
                meta = json.load(f)
            caption = meta.get("caption", "pixel art sprite")
            captions.append(caption)
            paths.append(jf.with_suffix(".emb.pt"))

        # Batch encode
        for i in tqdm(range(0, len(captions), args.batch_size), desc=size_dir.name):
            batch_captions = captions[i:i + args.batch_size]
            batch_paths = paths[i:i + args.batch_size]

            inputs = tokenizer(
                batch_captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(args.device)

            with torch.no_grad():
                outputs = model(**inputs)

            for j, p in enumerate(batch_paths):
                torch.save({
                    "pooled": outputs.pooler_output[j].cpu(),
                    "tokens": outputs.last_hidden_state[j].cpu(),
                }, p)

        print(f"  Saved {len(captions)} embedding files")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    print("\nDone. CLIP is no longer needed during training.")


if __name__ == "__main__":
    main()
