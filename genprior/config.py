"""Command-line configuration for GenPrior experiments."""

from __future__ import annotations

import argparse
import json
from argparse import Action
from collections.abc import Mapping, Sequence
from typing import Any


class JsonObjectAction(Action):
    """Parse a JSON object passed on the command line."""

    def __call__(self, parser, namespace, values, option_string=None):
        value = json.loads(values)
        if not isinstance(value, dict):
            raise argparse.ArgumentError(self, "expected a JSON object")
        setattr(namespace, self.dest, value)


def normalize_config_values(
    parser: argparse.ArgumentParser,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize values loaded from an experiment YAML file."""
    if not isinstance(values, Mapping):
        raise TypeError("expected a YAML mapping at the document root")

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown_keys = sorted(set(values) - set(actions))
    if unknown_keys:
        raise ValueError(f"unknown config keys: {', '.join(unknown_keys)}")
    if "config" in values:
        raise ValueError("the config path must be supplied on the command line")

    normalized: dict[str, Any] = {}
    for key, value in values.items():
        action = actions[key]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise TypeError(f"config key {key!r} must be a boolean")
            normalized_value = value
        elif action.nargs not in (None, "?"):
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise TypeError(f"config key {key!r} must be a sequence")
            if isinstance(action.nargs, int) and len(value) != action.nargs:
                raise ValueError(
                    f"config key {key!r} must contain {action.nargs} values"
                )
            normalized_value = [
                action.type(item) if action.type is not None else item for item in value
            ]
        elif action.type is not None and value is not None:
            try:
                normalized_value = action.type(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"invalid value for config key {key!r}: {value!r}"
                ) from error
        else:
            normalized_value = value

        if action.choices is not None and normalized_value not in action.choices:
            choices = ", ".join(map(str, action.choices))
            raise ValueError(
                f"invalid value for config key {key!r}: {normalized_value!r}; "
                f"choose from {choices}"
            )
        normalized[key] = normalized_value

    return normalized


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate GenPrior",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--split-name", default="ntu60_xsub_seen55_unseen5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")

    parser.add_argument("--num-seen-classes", type=int, default=55)
    parser.add_argument("--num-point", type=int, default=25)
    parser.add_argument("--num-person", type=int, default=2)
    parser.add_argument("--graph", default="genprior.graph.ntu_rgb_d.Graph")
    parser.add_argument("--graph-args", action=JsonObjectAction, default={})

    parser.add_argument("--distribution-alignment-weight", type=float, default=0.1)
    parser.add_argument("--skeleton-feature-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--text-feature-dim", type=int, default=1536)

    parser.add_argument(
        "--use-generated-motion-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable GESF with Dispersion-Gated Feature Fusion",
    )
    parser.add_argument(
        "--use-dispersion-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dispersion-gate-mode",
        choices=("full", "fixed", "prototype_dispersion", "text_prototype"),
        default="full",
        help="dispersion-gated feature fusion ablation mode",
    )
    parser.add_argument("--prior-fusion-weight", type=float, default=0.5)
    parser.add_argument(
        "--generated-motion-path",
        default="data/generated_motions.pkl",
        help="generated HumanML3D-format motions",
    )
    parser.add_argument(
        "--skeleton-encoder-checkpoint",
        default="checkpoints/skeleton_encoder.pt",
    )

    parser.add_argument(
        "--text-encoder-name-or-path",
        default="openai/clip-vit-large-patch14-336",
    )
    parser.add_argument("--clip-backend", choices=("hf", "openai"), default="hf")
    parser.add_argument("--clip-download-root", default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=10)

    parser.add_argument("--feeder", default="genprior.data.ntu.Feeder")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--feature-path")

    parser.add_argument("--semantic-dir", default="semantics")
    parser.add_argument("--max-text-length", type=int, default=60)
    parser.add_argument("--semantic-neighbor-count", type=int, default=2)
    parser.add_argument(
        "--semantic-neighbor-weights",
        type=float,
        nargs=2,
        default=(1.0, 0.4),
        metavar=("ZSL", "GZSL"),
    )
    parser.add_argument("--calibration-factor", type=float, default=0.72)
    parser.add_argument("--setting", choices=("ZSL", "GZSL", "All"), default="ZSL")
    parser.add_argument("--output-dir", default="outputs/default")

    parser.add_argument(
        "--use-gpr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Generative Prototype Refinement",
    )
    parser.add_argument("--gpr-support-ratio", type=float, default=0.4)
    parser.add_argument(
        "--second-refinement-round",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply the calibration-split second GPR round during GZSL evaluation",
    )
    parser.add_argument(
        "--prediction-space",
        choices=("latent", "text", "visual", "fused", "joint"),
        default="fused",
        help="prediction space used for checkpoint selection and reported metrics",
    )
    parser.add_argument(
        "--diagnostic-heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="log prediction spaces other than the configured prediction_space",
    )
    return parser
