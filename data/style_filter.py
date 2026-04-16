"""CLIP vision-based style filter for pixel art sprites.

Computes a reference embedding from example sprites, then scores
candidate images by cosine similarity to filter for matching style.

Usage:
    python -m data.style_filter --compute-reference
    python -m data.style_filter --input data/raw --output data/raw_filtered --threshold 0.75
"""

import argparse
import csv
import shutil
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL = "openai/clip-vit-large-patch14"
REFERENCE_PATH = "example_sprites/style_reference.pt"


def _load_clip(device: str = "cuda") -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, processor


def _prepare_image(img: Image.Image) -> Image.Image:
    """Composite RGBA sprite onto neutral gray background for CLIP."""
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
    bg.paste(img, mask=img.split()[3])
    return bg.convert("RGB")


def compute_reference_embedding(
    example_dir: str = "example_sprites",
    device: str = "cuda",
    output_path: str = REFERENCE_PATH,
) -> torch.Tensor:
    """Compute mean CLIP image embedding from example sprites."""
    model, processor = _load_clip(device)

    pngs = sorted(Path(example_dir).glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG files in {example_dir}")

    images = [_prepare_image(Image.open(p)) for p in pngs]
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_out = model.vision_model(**inputs)
        features = model.visual_projection(vision_out.pooler_output)

    features = torch.nn.functional.normalize(features, dim=-1)
    mean_emb = features.mean(dim=0)
    mean_emb = torch.nn.functional.normalize(mean_emb, dim=0)

    torch.save(mean_emb.cpu(), output_path)
    print(f"Saved reference embedding ({mean_emb.shape}) to {output_path}")
    print(f"Computed from {len(pngs)} images: {[p.name for p in pngs]}")
    return mean_emb


def load_reference_embedding(path: str = REFERENCE_PATH) -> torch.Tensor:
    return torch.load(path, weights_only=True)


@torch.no_grad()
def score_images(
    image_paths: list[Path],
    reference: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 64,
) -> list[float]:
    """Score each image's cosine similarity to the reference style."""
    model, processor = _load_clip(device)
    reference = reference.to(device)

    scores = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        valid_idx = []
        for j, p in enumerate(batch_paths):
            try:
                images.append(_prepare_image(Image.open(p)))
                valid_idx.append(j)
            except Exception:
                pass

        if not images:
            scores.extend([0.0] * len(batch_paths))
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        vision_out = model.vision_model(**inputs)
        features = model.visual_projection(vision_out.pooler_output)
        features = torch.nn.functional.normalize(features, dim=-1)
        sims = (features @ reference).cpu().tolist()

        # Map back to original batch positions
        batch_scores = [0.0] * len(batch_paths)
        for k, idx in enumerate(valid_idx):
            batch_scores[idx] = sims[k]
        scores.extend(batch_scores)

    return scores


def filter_directory(
    input_dir: str,
    output_dir: str,
    reference_path: str = REFERENCE_PATH,
    threshold: float = 0.75,
    device: str = "cuda",
    batch_size: int = 64,
    save_scores: str | None = None,
) -> tuple[int, int]:
    """Filter images by style similarity, copying passing images to output_dir."""
    reference = load_reference_embedding(reference_path).to(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect all PNGs
    all_pngs = sorted(Path(input_dir).rglob("*.png"))
    if not all_pngs:
        print(f"No PNG files found in {input_dir}")
        return 0, 0

    print(f"Scoring {len(all_pngs)} images against style reference...")
    scores = score_images(all_pngs, reference, device, batch_size)

    accepted = 0
    rejected = 0
    csv_rows = []

    for path, score in tqdm(zip(all_pngs, scores), total=len(all_pngs), desc="Filtering"):
        csv_rows.append((str(path), f"{score:.4f}"))
        if score >= threshold:
            shutil.copy2(path, out / path.name)
            accepted += 1
        else:
            rejected += 1

    if save_scores:
        with open(save_scores, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "score"])
            writer.writerows(csv_rows)
        print(f"Scores saved to {save_scores}")

    print(f"Accepted: {accepted}, Rejected: {rejected} (threshold={threshold})")
    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description="CLIP-based style filter for pixel art")
    parser.add_argument("--compute-reference", action="store_true",
                        help="Compute reference embedding from example sprites")
    parser.add_argument("--example-dir", default="example_sprites")
    parser.add_argument("--input", help="Input directory of raw images to filter")
    parser.add_argument("--output", help="Output directory for accepted images")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-scores", help="Path to save CSV of all scores")
    args = parser.parse_args()

    if args.compute_reference:
        compute_reference_embedding(args.example_dir, args.device)

    if args.input and args.output:
        filter_directory(
            args.input, args.output,
            threshold=args.threshold,
            device=args.device,
            batch_size=args.batch_size,
            save_scores=args.save_scores,
        )


if __name__ == "__main__":
    main()
