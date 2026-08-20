"""Checkpoint loading and state-key compatibility for GenPrior."""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn


_STATE_DICT_KEYS = (
    "model",
    "state_dict",
    "model_state_dict",
    "model_state",
    "network",
    "net",
)
_CHECKPOINT_COMPONENT_ALIASES = {
    "visual_adapter": "generative_prototype_projection",
    "motion_prior_projection": "generative_prototype_projection",
    "fusion_gate": "dispersion_gate",
    "sigma_proj": "dispersion_projection",
    "motion_encoder": "visual_encoder",
    "motion_decoder": "visual_decoder",
    "joint_fc": "joint_projection",
}
_EXTERNAL_FROZEN_PREFIXES = (
    "text_encoder.",
    "skeleton_encoder.",
)


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Details of a checkpoint load, including every incompatible key."""

    path: str
    container_key: Optional[str]
    metadata: Mapping[str, Any]
    migrated_keys: Mapping[str, str]
    missing_keys: Tuple[str, ...]
    external_frozen_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]

    @property
    def is_exact(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys

    def summary(self) -> str:
        return (
            f"checkpoint={self.path!r}, container={self.container_key or 'raw'}, "
            f"migrated={len(self.migrated_keys)}, missing={len(self.missing_keys)}, "
            f"external_frozen={len(self.external_frozen_keys)}, "
            f"unexpected={len(self.unexpected_keys)}"
        )


def _is_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(isinstance(key, str) for key in value) and all(
        isinstance(tensor, (torch.Tensor, nn.Parameter)) for tensor in value.values()
    )


def _extract_state_dict(checkpoint: Any) -> Tuple[Mapping[str, torch.Tensor], Optional[str], Dict[str, Any]]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict(), None, {}
    if _is_state_dict(checkpoint):
        return checkpoint, None, {}
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "checkpoint must be a state dict, nn.Module, or mapping containing a state dict"
        )

    for container_key in _STATE_DICT_KEYS:
        candidate = checkpoint.get(container_key)
        if _is_state_dict(candidate):
            metadata = {
                key: value for key, value in checkpoint.items() if key != container_key
            }
            return candidate, container_key, metadata

    available = ", ".join(sorted(str(key) for key in checkpoint.keys()))
    raise ValueError(
        "could not find a model state dict in checkpoint; "
        f"supported container keys are {_STATE_DICT_KEYS}, found: {available}"
    )


def _normalize_state_key(key: str) -> str:
    components = key.split(".")
    while components and components[0] == "module":
        components.pop(0)
    components = [
        _CHECKPOINT_COMPONENT_ALIASES.get(component, component)
        for component in components
    ]
    return ".".join(components)


def normalize_checkpoint_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, str]]:
    """Apply component aliases and reject ambiguous key collisions."""

    migrated = OrderedDict()
    renamed = {}
    for old_key, value in state_dict.items():
        new_key = _normalize_state_key(old_key)
        if new_key in migrated:
            raise ValueError(
                f"checkpoint key collision after migration: {old_key!r} -> {new_key!r}"
            )
        migrated[new_key] = value
        if new_key != old_key:
            renamed[old_key] = new_key
    return migrated, renamed


def _load_file(path: os.PathLike, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model_checkpoint(
    model: nn.Module,
    path: os.PathLike,
    map_location: Any = "cpu",
    *,
    strict: bool = False,
) -> CheckpointLoadReport:
    """Load a checkpoint into ``model`` and report all compatibility details.

    Previous component names are normalized to the public GenPrior names.
    Published checkpoints intentionally omit frozen encoders; those keys are
    reported separately as ``external_frozen_keys``. ``strict=False`` warns on
    other incompatibilities. Set ``strict=True`` to make any trainable missing
    key or unexpected key an error.
    """

    path_string = os.fspath(path)
    checkpoint = _load_file(path_string, map_location)
    state_dict, container_key, metadata = _extract_state_dict(checkpoint)
    migrated_state, migrated_keys = normalize_checkpoint_state_dict(state_dict)
    incompatible = model.load_state_dict(migrated_state, strict=False)
    external_frozen_keys = tuple(
        key
        for key in incompatible.missing_keys
        if key.startswith(_EXTERNAL_FROZEN_PREFIXES)
    )
    missing_keys = tuple(
        key
        for key in incompatible.missing_keys
        if not key.startswith(_EXTERNAL_FROZEN_PREFIXES)
    )

    report = CheckpointLoadReport(
        path=path_string,
        container_key=container_key,
        metadata=metadata,
        migrated_keys=migrated_keys,
        missing_keys=missing_keys,
        external_frozen_keys=external_frozen_keys,
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
    if not report.is_exact:
        details = [report.summary()]
        if report.missing_keys:
            details.append("missing keys: " + ", ".join(report.missing_keys))
        if report.unexpected_keys:
            details.append("unexpected keys: " + ", ".join(report.unexpected_keys))
        message = "; ".join(details)
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return report


__all__ = [
    "CheckpointLoadReport",
    "load_model_checkpoint",
    "normalize_checkpoint_state_dict",
]
