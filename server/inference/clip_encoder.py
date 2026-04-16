"""CLIP text encoder wrapper (frozen)."""

import torch
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPEncoder:
    """Wraps Hugging Face CLIP for text encoding. Frozen — no gradients."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 device: str = "cuda"):
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a text prompt into pooled and token-level embeddings.

        Args:
            text: Text prompt string

        Returns:
            pooled: (1, 768) pooled text embedding
            tokens: (1, 77, 768) token-level embeddings
        """
        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        tokens = outputs.last_hidden_state      # (1, 77, 768)
        pooled = outputs.pooler_output           # (1, 768)

        return pooled, tokens

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of text prompts."""
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        return outputs.pooler_output, outputs.last_hidden_state
