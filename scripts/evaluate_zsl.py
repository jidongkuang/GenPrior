#!/usr/bin/env python3
"""Evaluate one pretrained GenPrior checkpoint in the ZSL setting."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if Path.cwd() != REPOSITORY_ROOT:
    os.chdir(REPOSITORY_ROOT)
sys.path.insert(0, str(REPOSITORY_ROOT))

from genprior import config, utils
from genprior.checkpoint import load_model_checkpoint
from genprior.trainer import GenPriorTrainer


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output-dir', default='outputs/evaluation/zsl')
    cli_args, config_rest = parser.parse_known_args(argv)

    experiment_parser = config.get_parser()
    with open(cli_args.config, 'r', encoding='utf-8') as handle:
        config_values = yaml.safe_load(handle) or {}
    config_values = config.normalize_config_values(experiment_parser, config_values)
    experiment_parser.set_defaults(**config_values)
    experiment_parser.set_defaults(
        setting='ZSL',
        output_dir=cli_args.output_dir,
        device=cli_args.device,
    )
    experiment_args = experiment_parser.parse_args(
        ['--config', cli_args.config, *config_rest]
    )
    experiment_args.setting = 'ZSL'
    experiment_args.output_dir = cli_args.output_dir
    return cli_args, experiment_args


def main() -> None:
    cli_args, args = parse_args()
    torch.cuda.set_device(cli_args.device)
    utils.init_seed(args.seed)

    trainer = GenPriorTrainer(args)
    report = load_model_checkpoint(
        trainer.model,
        cli_args.checkpoint,
        map_location='cpu',
        strict=True,
    )
    trainer.model.cuda()

    # Model construction consumes random numbers; reset before stochastic inference.
    utils.init_seed(args.seed)
    checkpoint_iteration = int(report.metadata.get('iter', 0))
    metrics = trainer.test_zsl(
        checkpoint_iteration,
        save_checkpoints=False,
    )

    result = {
        'config': os.path.abspath(cli_args.config),
        'checkpoint': os.path.abspath(cli_args.checkpoint),
        'checkpoint_iteration': checkpoint_iteration,
        'base_accuracy': float(metrics['base'][args.prediction_space]),
    }
    if 'gpr' in metrics:
        result['accuracy'] = float(metrics['gpr'][args.prediction_space])
    else:
        result['accuracy'] = result['base_accuracy']

    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
