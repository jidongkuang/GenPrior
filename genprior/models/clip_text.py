from __future__ import annotations

# OpenAI GitHub CLIP (github.com/openai/CLIP): use clip_backend=openai + clip.tokenize.
import torch
import torch.nn as nn


class OpenAIClipTextEncoder(nn.Module):
    """Expose OpenAI CLIP text outputs with Hugging Face field names."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model

    def forward(self, input_ids: torch.Tensor):
        # OpenAI CLIP uses a fixed context length of 77 tokens.
        text = input_ids
        w = self.clip.token_embedding(text).float()
        w = w + self.clip.positional_embedding.float()
        w = w.permute(1, 0, 2)
        w = self.clip.transformer(w)
        w = w.permute(1, 0, 2)
        w = self.clip.ln_final(w).float()
        pooled = w[torch.arange(w.shape[0], device=w.device), text.argmax(dim=-1)]
        return {"last_hidden_state": w, "pooler_output": pooled}


def load_openai_clip_text(checkpoint_or_name: str, device: str, download_root=None):
    import clip

    kw = {"device": device, "jit": False}
    if download_root:
        kw["download_root"] = download_root
    model, _ = clip.load(checkpoint_or_name, **kw)
    model = model.float()
    model.eval()
    return OpenAIClipTextEncoder(model)
