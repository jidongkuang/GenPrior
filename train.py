#!/usr/bin/env python3
"""Train GenPrior from a validated YAML experiment configuration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parent
if os.getcwd() != str(REPOSITORY_ROOT):
    os.chdir(REPOSITORY_ROOT)
sys.path.insert(0, str(REPOSITORY_ROOT))

from genprior import config, utils
from genprior.trainer import GenPriorTrainer


def load_yaml_config(
    path: str | os.PathLike[str],
    parser: argparse.ArgumentParser,
) -> dict[str, object]:
    """Safely load one YAML mapping and reject unsupported configuration keys."""
    with open(path, "r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    try:
        return config.normalize_config_values(parser, values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid configuration {path}: {error}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    if any(argument in {"-h", "--help"} for argument in argv):
        return config.get_parser().parse_args(argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", required=True)
    config_args, _ = bootstrap.parse_known_args(argv)

    parser = config.get_parser()
    yaml_values = load_yaml_config(config_args.config, parser)
    parser.set_defaults(**yaml_values)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device)
    utils.init_seed(args.seed)
    trainer = GenPriorTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
