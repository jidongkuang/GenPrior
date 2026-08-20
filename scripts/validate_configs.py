#!/usr/bin/env python3
"""Validate all public GenPrior experiment configurations."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from genprior import config, utils


EXPLICIT_EXPERIMENT_KEYS = {
    "diagnostic_heads",
    "dispersion_gate_mode",
    "gpr_support_ratio",
    "output_dir",
    "prediction_space",
    "prior_fusion_weight",
    "use_dispersion_gate",
    "use_generated_motion_prior",
    "use_gpr",
}


def validate_config(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        raw_values = yaml.safe_load(handle) or {}
    try:
        values = config.normalize_config_values(config.get_parser(), raw_values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error

    missing_keys = sorted(EXPLICIT_EXPERIMENT_KEYS - set(values))
    if missing_keys:
        raise ValueError(f"{path}: missing explicit keys: {', '.join(missing_keys)}")
    if values["diagnostic_heads"] is not False:
        raise ValueError(f"{path}: diagnostic_heads must be false for reported runs")
    if values["use_gpr"] is not True:
        raise ValueError(f"{path}: use_gpr must be true for reported runs")
    if not 0.0 < values["gpr_support_ratio"] <= 1.0:
        raise ValueError(f"{path}: gpr_support_ratio must be in (0, 1]")
    if not values["output_dir"].startswith("outputs/"):
        raise ValueError(f"{path}: output_dir must be under outputs/")

    seen_match = re.search(r"_seen(\d+)_", values["split_name"])
    if seen_match is None:
        raise ValueError(f"{path}: split_name does not encode the seen-class count")
    expected_seen_classes = int(seen_match.group(1))
    if values["num_seen_classes"] != expected_seen_classes:
        raise ValueError(
            f"{path}: num_seen_classes={values['num_seen_classes']} does not match "
            f"split_name ({expected_seen_classes})"
        )

    is_random_split = path.parent.name == "table2"
    is_gzsl = path.parent.name == "table3"
    if values["use_generated_motion_prior"] is is_random_split:
        expected = "false" if is_random_split else "true"
        raise ValueError(
            f"{path}: use_generated_motion_prior must be {expected} for this table"
        )
    if values["use_dispersion_gate"] != values["use_generated_motion_prior"]:
        raise ValueError(
            f"{path}: dispersion gate and generated-motion prior disagree"
        )
    if is_gzsl:
        if "second_refinement_round" not in values:
            raise ValueError(f"{path}: missing explicit key: second_refinement_round")
        if values["setting"] != "GZSL":
            raise ValueError(f"{path}: setting must be GZSL for Table 3")

    utils.import_class(values["feeder"])
    utils.import_class(values["graph"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.configs or sorted((REPOSITORY_ROOT / "configs").glob("**/*.yaml"))
    if not paths:
        parser.error("no configuration files found")

    for path in paths:
        validate_config(path.resolve())
        print(f"OK {os.path.relpath(path.resolve(), REPOSITORY_ROOT)}")
    print(f"Validated {len(paths)} configuration files.")


if __name__ == "__main__":
    main()
