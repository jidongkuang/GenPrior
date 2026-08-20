import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from genprior import config, utils


PREDICTION_SPACES = ("latent", "text", "visual", "fused", "joint")
REJECTED_CONFIG_KEYS = {
    # Raw-data paths from the source research repository.
    "use_features",
    "low_shot",
    "percentage",
}
EXPECTED_CONFIG_SPACES = {
    "configs/table1/ntu120_xsub_unseen10.yaml": "visual",
    "configs/table1/ntu120_xsub_unseen24.yaml": "joint",
    "configs/table1/ntu60_xsub_unseen12.yaml": "visual",
    "configs/table1/ntu60_xsub_unseen5.yaml": "joint",
    "configs/table2/ntu120_xsub_unseen10_split1.yaml": "joint",
    "configs/table2/ntu120_xsub_unseen10_split2.yaml": "fused",
    "configs/table2/ntu120_xsub_unseen10_split3.yaml": "visual",
    "configs/table2/ntu60_xsub_unseen5_split1.yaml": "visual",
    "configs/table2/ntu60_xsub_unseen5_split2.yaml": "fused",
    "configs/table2/ntu60_xsub_unseen5_split3.yaml": "visual",
    "configs/table2/pku51_xsub_unseen5_split1.yaml": "visual",
    "configs/table2/pku51_xsub_unseen5_split2.yaml": "latent",
    "configs/table2/pku51_xsub_unseen5_split3.yaml": "fused",
    "configs/table3/ntu120_xsub_unseen10.yaml": "fused",
    "configs/table3/ntu120_xsub_unseen24.yaml": "fused",
    "configs/table3/ntu60_xsub_unseen12.yaml": "joint",
    "configs/table3/ntu60_xsub_unseen5.yaml": "fused",
}

EXPECTED_SECOND_REFINEMENT = {
    "configs/table3/ntu120_xsub_unseen10.yaml": False,
    "configs/table3/ntu120_xsub_unseen24.yaml": False,
    "configs/table3/ntu60_xsub_unseen12.yaml": False,
    "configs/table3/ntu60_xsub_unseen5.yaml": True,
}


def parse_config(path: Path, *cli_args: str):
    parser = config.get_parser()
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    parser.set_defaults(**config.normalize_config_values(parser, values))
    return parser.parse_args(["--config", str(path), *cli_args])


class PredictionSpaceParserTest(unittest.TestCase):
    def test_parser_default_choices_and_rejection(self):
        parser = config.get_parser()
        parser_keys = {action.dest for action in parser._actions}
        self.assertTrue(REJECTED_CONFIG_KEYS.isdisjoint(parser_keys))
        action = next(
            action for action in parser._actions if action.dest == "prediction_space"
        )

        self.assertEqual(tuple(action.choices), PREDICTION_SPACES)
        self.assertEqual(
            parser.parse_args(["--config", "experiment.yaml"]).prediction_space,
            "fused",
        )
        for prediction_space in PREDICTION_SPACES:
            args = parser.parse_args(
                [
                    "--config",
                    "experiment.yaml",
                    "--prediction-space",
                    prediction_space,
                ]
            )
            self.assertEqual(args.prediction_space, prediction_space)

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                ["--config", "experiment.yaml", "--prediction-space", "unknown"]
            )
        with self.assertRaisesRegex(ValueError, "invalid value.*prediction_space"):
            config.normalize_config_values(
                config.get_parser(), {"prediction_space": "unknown"}
            )

    def test_second_refinement_round_boolean_override(self):
        parser = config.get_parser()
        base_args = ["--config", "experiment.yaml"]
        self.assertFalse(parser.parse_args(base_args).second_refinement_round)
        self.assertTrue(
            parser.parse_args(
                [*base_args, "--second-refinement-round"]
            ).second_refinement_round
        )
        self.assertFalse(
            parser.parse_args(
                [*base_args, "--no-second-refinement-round"]
            ).second_refinement_round
        )


class PredictionSpaceConfigTest(unittest.TestCase):
    def test_all_published_configs_have_the_expected_explicit_mapping(self):
        actual_paths = {
            str(path.relative_to(REPOSITORY_ROOT))
            for path in (REPOSITORY_ROOT / "configs").glob("table*/*.yaml")
        }
        self.assertEqual(actual_paths, set(EXPECTED_CONFIG_SPACES))

        for relative_path, expected_space in EXPECTED_CONFIG_SPACES.items():
            path = REPOSITORY_ROOT / relative_path
            with self.subTest(config=relative_path):
                with path.open("r", encoding="utf-8") as handle:
                    raw_values = yaml.safe_load(handle) or {}
                self.assertTrue(REJECTED_CONFIG_KEYS.isdisjoint(raw_values))
                self.assertIn("prediction_space", raw_values)
                self.assertEqual(raw_values["prediction_space"], expected_space)
                self.assertEqual(parse_config(path).prediction_space, expected_space)
                if relative_path in EXPECTED_SECOND_REFINEMENT:
                    self.assertEqual(raw_values["setting"], "GZSL")
                    self.assertIn("second_refinement_round", raw_values)
                    expected_second_round = EXPECTED_SECOND_REFINEMENT[relative_path]
                    self.assertIs(
                        raw_values["second_refinement_round"], expected_second_round
                    )
                    self.assertIs(
                        parse_config(path).second_refinement_round,
                        expected_second_round,
                    )
                num_classes, unseen, seen, *_ = utils.task_definition(
                    raw_values["split_name"]
                )
                self.assertEqual(len(seen), raw_values["num_seen_classes"])
                self.assertEqual(len(seen) + len(unseen), num_classes)

    def test_command_line_prediction_space_overrides_yaml(self):
        path = REPOSITORY_ROOT / "configs/table1/ntu60_xsub_unseen5.yaml"
        self.assertEqual(parse_config(path).prediction_space, "joint")
        self.assertEqual(
            parse_config(path, "--prediction-space", "text").prediction_space,
            "text",
        )


class PredictionSpaceTrainerTest(unittest.TestCase):
    def test_zsl_checkpoint_selection_uses_configured_space_not_best_head(self):
        import torch

        try:
            from genprior.trainer import GenPriorTrainer
        except ImportError as error:
            self.skipTest(f"optional Shift-GCN extension is unavailable: {error}")

        class FakeModel:
            def eval(self):
                return self

            def train(self):
                return self

            def predict_zsl(self, *unused_args):
                labels = torch.tensor([0, 1, 0, 1])
                return {
                    "predictions": 1 - labels,
                    "diagnostics": {
                        "latent": 1 - labels,
                        "text": torch.tensor([0, 0, 0, 0]),
                        "visual": labels,
                        "joint": 1 - labels,
                    },
                }

        trainer = GenPriorTrainer.__new__(GenPriorTrainer)
        trainer.args = SimpleNamespace(
            prediction_space="text",
            use_gpr=False,
            diagnostic_heads=False,
        )
        trainer.model = FakeModel()
        trainer.unseen_classes = [0, 1]
        trainer.class_semantics = torch.empty(0)
        trainer.test_zsl_label_dict = {10: [1, 0], 11: [0, 1]}
        trainer.zsl_loader = [
            (
                torch.zeros(4, 256),
                torch.tensor([10, 11, 10, 11]),
                None,
            )
        ]
        trainer.best_accuracy = [float("-inf")] * 4
        trainer.best_base_accuracy = float("-inf")
        trainer.best_base_method = ""
        trainer.best_method_name = ""
        trainer.best_gpr_accuracy = float("-inf")
        trainer.logger = SimpleNamespace(info=lambda *unused_args: None)
        saved_checkpoints = []
        trainer._save_checkpoint = lambda filename, iteration: saved_checkpoints.append(
            (filename, iteration)
        )

        with mock.patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
            trainer.test_zsl(nb_iter=7)

        self.assertEqual(trainer.best_base_accuracy, 50.0)
        self.assertEqual(trainer.best_base_method, "text")
        self.assertEqual(trainer.best_accuracy[0], 50.0)
        self.assertEqual(trainer.best_method_name, "text")
        self.assertEqual(saved_checkpoints, [("best_base.pt", 7)])

        trainer.best_accuracy = [float("-inf")] * 4
        trainer.best_base_accuracy = float("-inf")
        trainer.best_base_method = ""
        trainer.best_method_name = ""
        saved_checkpoints.clear()
        with mock.patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
            trainer.test_zsl(nb_iter=8, save_checkpoints=False)

        self.assertEqual(saved_checkpoints, [])


if __name__ == "__main__":
    unittest.main()
