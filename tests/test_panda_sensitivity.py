"""Check that the Panda study changes only the intended command fault."""

import runpy
from pathlib import Path

import pytest

from forge_ci.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_panda_sensitivity_conditions():
    pytest.importorskip("mujoco")
    study = runpy.run_path(str(ROOT / "scripts/panda_sensitivity.py"))
    base = load_config(ROOT / "configs/panda_nominal.yaml")
    base.episodes = 20
    base.seed = 1000
    conditions = list(study["build_conditions"](base))
    assert len(conditions) == 19
    assert conditions[0][0] == "nominal"
    assert sum(config.episodes for _, _, config in conditions) == 380
    for axis, value, config in conditions:
        assert config.environment == base.environment
        assert config.gate == base.gate
        assert config.seed == base.seed
        assert config.episodes == base.episodes
        policy = base.policy.model_dump()
        if axis != "nominal":
            parameter = study["LEVELS"][axis][0]
            policy[parameter] = value
        assert config.policy.model_dump() == policy
