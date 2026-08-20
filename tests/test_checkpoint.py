import torch
from torch import nn

from genprior.checkpoint import load_model_checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generative_prototype_projection = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)


class TinyCompatibilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.generative_prototype_projection = nn.Linear(2, 2)
        self.dispersion_gate = nn.Module()
        self.dispersion_gate.dispersion_projection = nn.Linear(2, 2)
        self.visual_encoder = nn.Linear(2, 2)
        self.visual_decoder = nn.Linear(2, 2)
        self.joint_projection = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)


def test_checkpoint_migrates_legacy_keys_and_allows_external_frozen_weights(tmp_path):
    model = TinyModel()
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": {
                "visual_adapter.weight": torch.ones_like(
                    model.generative_prototype_projection.weight
                ),
                "visual_adapter.bias": torch.zeros_like(
                    model.generative_prototype_projection.bias
                ),
            }
        },
        checkpoint_path,
    )

    report = load_model_checkpoint(model, checkpoint_path, strict=True)

    assert report.is_exact
    assert len(report.migrated_keys) == 2
    assert set(report.external_frozen_keys) == {
        "text_encoder.weight",
        "text_encoder.bias",
    }


def test_checkpoint_migrates_all_historical_component_names(tmp_path):
    model = TinyCompatibilityModel()
    checkpoint_path = tmp_path / "historical_names.pt"
    legacy_to_current = {
        "module.module.motion_prior_projection": "generative_prototype_projection",
        "fusion_gate.sigma_proj": "dispersion_gate.dispersion_projection",
        "motion_encoder": "visual_encoder",
        "motion_decoder": "visual_decoder",
        "joint_fc": "joint_projection",
    }
    state_dict = {}
    expected_values = {}
    for index, (legacy_name, current_name) in enumerate(legacy_to_current.items(), 1):
        current_module = model.get_submodule(current_name)
        weight_value = float(index)
        bias_value = float(-index)
        state_dict[f"{legacy_name}.weight"] = torch.full_like(
            current_module.weight, weight_value
        )
        state_dict[f"{legacy_name}.bias"] = torch.full_like(
            current_module.bias, bias_value
        )
        expected_values[current_name] = (weight_value, bias_value)
    torch.save({"state_dict": state_dict, "epoch": 12}, checkpoint_path)

    report = load_model_checkpoint(model, checkpoint_path, strict=True)

    assert report.is_exact
    assert report.container_key == "state_dict"
    assert report.metadata == {"epoch": 12}
    assert len(report.migrated_keys) == len(state_dict)
    for current_name, (weight_value, bias_value) in expected_values.items():
        current_module = model.get_submodule(current_name)
        torch.testing.assert_close(
            current_module.weight, torch.full_like(current_module.weight, weight_value)
        )
        torch.testing.assert_close(
            current_module.bias, torch.full_like(current_module.bias, bias_value)
        )
