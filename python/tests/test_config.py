"""Tests for proofforge.config.ProofForgeConfig."""

import json
import pathlib

import pytest

from proofforge.code_verify.task_loader import TaskLevel
from proofforge.config import (
    CodeVerifyConfig,
    GRPOTrainingConfig,
    LeanConfig,
    ProofForgeConfig,
)
from proofforge.controller.controller import ControllerConfig
from proofforge.curriculum.expanding_ring import RingConfig
from proofforge.rewards.efficiency import EfficiencyConfig
from proofforge.rewards.shaped import ShapedConfig


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------


class TestProofForgeConfigDefaults:
    """ProofForgeConfig constructs with all sub-configs present by default."""

    def setup_method(self):
        self.config = ProofForgeConfig()

    def test_has_grpo_config(self):
        assert isinstance(self.config.grpo, GRPOTrainingConfig)

    def test_has_lean_config(self):
        assert isinstance(self.config.lean, LeanConfig)

    def test_has_efficiency_config(self):
        assert isinstance(self.config.efficiency, EfficiencyConfig)

    def test_has_controller_config(self):
        assert isinstance(self.config.controller, ControllerConfig)

    def test_has_code_verify_config(self):
        assert isinstance(self.config.code_verify, CodeVerifyConfig)

    def test_has_shaped_config(self):
        assert isinstance(self.config.shaped, ShapedConfig)

    def test_has_ring_config(self):
        assert isinstance(self.config.ring, RingConfig)

    def test_has_output_dir(self):
        assert isinstance(self.config.output_dir, str)

    def test_has_seed(self):
        assert isinstance(self.config.seed, int)

    def test_default_seed_is_42(self):
        assert self.config.seed == 42

    def test_default_output_dir_set(self):
        assert len(self.config.output_dir) > 0


# ---------------------------------------------------------------------------
# to_dict serialisability
# ---------------------------------------------------------------------------


class TestProofForgeConfigToDict:
    """to_dict() produces a fully JSON-safe nested dict."""

    def test_to_dict_returns_dict(self):
        config = ProofForgeConfig()
        assert isinstance(config.to_dict(), dict)

    def test_to_dict_all_values_json_safe(self):
        """json.dumps must not raise on the output of to_dict()."""
        config = ProofForgeConfig()
        d = config.to_dict()
        # to_json uses default=str, so we match that behaviour here
        json.dumps(d, default=str)

    def test_to_dict_contains_top_level_keys(self):
        config = ProofForgeConfig()
        d = config.to_dict()
        for key in ("grpo", "lean", "efficiency", "controller", "code_verify", "shaped", "ring"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_grpo_group_size_preserved(self):
        config = ProofForgeConfig()
        assert config.to_dict()["grpo"]["group_size"] == config.grpo.group_size

    def test_to_dict_seed_preserved(self):
        config = ProofForgeConfig()
        assert config.to_dict()["seed"] == config.seed


# ---------------------------------------------------------------------------
# to_json / from_json round-trip
# ---------------------------------------------------------------------------


class TestProofForgeConfigJsonRoundtrip:
    """to_json() and from_json() produce an equivalent config."""

    def test_roundtrip_preserves_seed(self, tmp_path: pathlib.Path):
        config = ProofForgeConfig()
        path = str(tmp_path / "config.json")
        config.to_json(path)
        loaded = ProofForgeConfig.from_json(path)
        assert loaded.seed == config.seed

    def test_roundtrip_preserves_output_dir(self, tmp_path: pathlib.Path):
        config = ProofForgeConfig()
        path = str(tmp_path / "config.json")
        config.to_json(path)
        loaded = ProofForgeConfig.from_json(path)
        assert loaded.output_dir == config.output_dir

    def test_roundtrip_preserves_grpo_group_size(self, tmp_path: pathlib.Path):
        config = ProofForgeConfig()
        path = str(tmp_path / "config.json")
        config.to_json(path)
        loaded = ProofForgeConfig.from_json(path)
        assert loaded.grpo.group_size == config.grpo.group_size

    def test_roundtrip_preserves_lean_workers(self, tmp_path: pathlib.Path):
        config = ProofForgeConfig()
        path = str(tmp_path / "config.json")
        config.to_json(path)
        loaded = ProofForgeConfig.from_json(path)
        assert loaded.lean.workers == config.lean.workers

    def test_roundtrip_file_is_valid_json(self, tmp_path: pathlib.Path):
        config = ProofForgeConfig()
        path = tmp_path / "config.json"
        config.to_json(str(path))
        with open(path) as f:
            parsed = json.load(f)
        assert isinstance(parsed, dict)

    def test_roundtrip_preserves_code_verify_initial_level(self, tmp_path: pathlib.Path):
        config = ProofForgeConfig()
        path = str(tmp_path / "config.json")
        config.to_json(path)
        loaded = ProofForgeConfig.from_json(path)
        assert loaded.code_verify.initial_level == config.code_verify.initial_level

    def test_roundtrip_modified_seed(self, tmp_path: pathlib.Path):
        config = ProofForgeConfig()
        config.seed = 123
        path = str(tmp_path / "config.json")
        config.to_json(path)
        loaded = ProofForgeConfig.from_json(path)
        assert loaded.seed == 123


# ---------------------------------------------------------------------------
# from_yaml
# ---------------------------------------------------------------------------


class TestProofForgeConfigFromYaml:
    """from_yaml() loads a YAML file correctly."""

    @pytest.fixture(autouse=True)
    def _skip_without_pyyaml(self):
        pytest.importorskip("yaml")

    def test_from_yaml_returns_config(self, tmp_path: pathlib.Path):
        yaml_text = "seed: 99\noutput_dir: /tmp/test\n"
        p = tmp_path / "config.yaml"
        p.write_text(yaml_text)
        config = ProofForgeConfig.from_yaml(str(p))
        assert isinstance(config, ProofForgeConfig)

    def test_from_yaml_applies_seed(self, tmp_path: pathlib.Path):
        p = tmp_path / "config.yaml"
        p.write_text("seed: 77\n")
        config = ProofForgeConfig.from_yaml(str(p))
        assert config.seed == 77

    def test_from_yaml_applies_output_dir(self, tmp_path: pathlib.Path):
        p = tmp_path / "config.yaml"
        p.write_text("output_dir: /custom/path\n")
        config = ProofForgeConfig.from_yaml(str(p))
        assert config.output_dir == "/custom/path"

    def test_from_yaml_applies_nested_grpo(self, tmp_path: pathlib.Path):
        yaml_text = "grpo:\n  group_size: 32\n"
        p = tmp_path / "config.yaml"
        p.write_text(yaml_text)
        config = ProofForgeConfig.from_yaml(str(p))
        assert config.grpo.group_size == 32

    def test_from_yaml_empty_file_uses_defaults(self, tmp_path: pathlib.Path):
        p = tmp_path / "config.yaml"
        p.write_text("{}\n")
        config = ProofForgeConfig.from_yaml(str(p))
        assert config.seed == ProofForgeConfig().seed


# ---------------------------------------------------------------------------
# _from_dict partial and empty overrides
# ---------------------------------------------------------------------------


class TestProofForgeConfigFromDict:
    """_from_dict() correctly applies partial overrides and preserves defaults."""

    def test_partial_override_changes_only_specified_field(self):
        config = ProofForgeConfig._from_dict({"seed": 999})
        assert config.seed == 999
        # All other defaults must be intact
        defaults = ProofForgeConfig()
        assert config.output_dir == defaults.output_dir
        assert config.grpo.group_size == defaults.grpo.group_size

    def test_partial_override_nested_grpo_field(self):
        config = ProofForgeConfig._from_dict({"grpo": {"group_size": 8}})
        assert config.grpo.group_size == 8
        # Other grpo fields preserved
        assert config.grpo.learning_rate == ProofForgeConfig().grpo.learning_rate

    def test_partial_override_nested_lean_field(self):
        config = ProofForgeConfig._from_dict({"lean": {"workers": 4}})
        assert config.lean.workers == 4
        assert config.lean.timeout == ProofForgeConfig().lean.timeout

    def test_partial_override_nested_controller_field(self):
        config = ProofForgeConfig._from_dict(
            {"controller": {"stagnation_patience": 100}}
        )
        assert config.controller.stagnation_patience == 100

    def test_partial_override_nested_ring_field(self):
        config = ProofForgeConfig._from_dict({"ring": {"tiers": 6}})
        assert config.ring.tiers == 6

    def test_partial_override_nested_efficiency_field(self):
        config = ProofForgeConfig._from_dict({"efficiency": {"phase_in_step": 25}})
        assert config.efficiency.phase_in_step == 25

    def test_empty_dict_preserves_all_defaults(self):
        config = ProofForgeConfig._from_dict({})
        defaults = ProofForgeConfig()
        assert config.seed == defaults.seed
        assert config.output_dir == defaults.output_dir
        assert config.grpo.group_size == defaults.grpo.group_size
        assert config.lean.workers == defaults.lean.workers
        assert config.controller.stagnation_patience == defaults.controller.stagnation_patience

    def test_unknown_key_is_silently_ignored(self):
        """Keys not present in the dataclass attributes are ignored."""
        config = ProofForgeConfig._from_dict({"nonexistent_key": "value"})
        assert config.seed == ProofForgeConfig().seed

    def test_unknown_nested_key_is_silently_ignored(self):
        config = ProofForgeConfig._from_dict({"grpo": {"not_a_field": 42}})
        assert config.grpo.group_size == ProofForgeConfig().grpo.group_size


# ---------------------------------------------------------------------------
# Sub-config accessibility
# ---------------------------------------------------------------------------


class TestProofForgeConfigSubConfigAccess:
    """Sub-configs are accessible as typed attributes."""

    def setup_method(self):
        self.config = ProofForgeConfig()

    def test_efficiency_is_accessible(self):
        cfg = self.config.efficiency
        assert cfg is not None
        assert isinstance(cfg, EfficiencyConfig)

    def test_controller_is_accessible(self):
        cfg = self.config.controller
        assert cfg is not None
        assert isinstance(cfg, ControllerConfig)

    def test_ring_is_accessible(self):
        cfg = self.config.ring
        assert cfg is not None
        assert isinstance(cfg, RingConfig)

    def test_shaped_is_accessible(self):
        cfg = self.config.shaped
        assert cfg is not None
        assert isinstance(cfg, ShapedConfig)

    def test_code_verify_is_accessible(self):
        cfg = self.config.code_verify
        assert cfg is not None
        assert isinstance(cfg, CodeVerifyConfig)

    def test_grpo_is_accessible(self):
        assert isinstance(self.config.grpo, GRPOTrainingConfig)

    def test_lean_is_accessible(self):
        assert isinstance(self.config.lean, LeanConfig)


# ---------------------------------------------------------------------------
# CodeVerifyConfig.initial_level TaskLevel conversion
# ---------------------------------------------------------------------------


class TestCodeVerifyInitialLevel:
    """code_verify.initial_level is correctly converted to TaskLevel."""

    def test_default_initial_level_is_task_level(self):
        config = ProofForgeConfig()
        assert isinstance(config.code_verify.initial_level, TaskLevel)

    def test_default_initial_level_is_expression(self):
        config = ProofForgeConfig()
        assert config.code_verify.initial_level == TaskLevel.EXPRESSION

    def test_from_dict_int_zero_converts_to_expression(self):
        config = ProofForgeConfig._from_dict(
            {"code_verify": {"initial_level": 0}}
        )
        assert config.code_verify.initial_level == TaskLevel.EXPRESSION
        assert isinstance(config.code_verify.initial_level, TaskLevel)

    def test_from_dict_int_one_converts_to_function(self):
        config = ProofForgeConfig._from_dict(
            {"code_verify": {"initial_level": 1}}
        )
        assert config.code_verify.initial_level == TaskLevel.FUNCTION
        assert isinstance(config.code_verify.initial_level, TaskLevel)

    def test_from_dict_int_two_converts_to_multi_function(self):
        config = ProofForgeConfig._from_dict(
            {"code_verify": {"initial_level": 2}}
        )
        assert config.code_verify.initial_level == TaskLevel.MULTI_FUNCTION

    def test_from_dict_int_three_converts_to_module(self):
        config = ProofForgeConfig._from_dict(
            {"code_verify": {"initial_level": 3}}
        )
        assert config.code_verify.initial_level == TaskLevel.MODULE

    def test_roundtrip_preserves_non_default_level(self, tmp_path: pathlib.Path):
        config = ProofForgeConfig._from_dict({"code_verify": {"initial_level": 2}})
        path = str(tmp_path / "config.json")
        config.to_json(path)
        loaded = ProofForgeConfig.from_json(path)
        assert loaded.code_verify.initial_level == TaskLevel.MULTI_FUNCTION
        assert isinstance(loaded.code_verify.initial_level, TaskLevel)
