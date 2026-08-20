"""Prediction utilities shared by ZSL and GZSL inference."""

from __future__ import annotations

import torch


def fuse_distances(
    latent_distances: torch.Tensor,
    visual_distances: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fuse row-normalized latent and visual prototype distances."""
    if latent_distances.shape != visual_distances.shape:
        raise ValueError(
            "latent and visual distance tensors must have identical shapes, "
            f"got {tuple(latent_distances.shape)} and {tuple(visual_distances.shape)}"
        )
    if latent_distances.ndim != 2:
        raise ValueError("distance tensors must have shape [batch, classes]")

    latent_scale = latent_distances.mean(dim=1, keepdim=True) + eps
    visual_scale = visual_distances.mean(dim=1, keepdim=True) + eps
    return latent_distances / latent_scale + visual_distances / visual_scale


__all__ = ["fuse_distances"]
