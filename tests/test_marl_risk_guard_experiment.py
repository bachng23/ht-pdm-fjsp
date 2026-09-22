from __future__ import annotations

from pathlib import Path

import numpy as np

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.marl_diagnostic_experiment import INDEPENDENT_MAPPO
from ht_pdm_fjsp.marl_factorial_experiment import COMBINED_MAPPO
from ht_pdm_fjsp.marl_risk_guard_experiment import (
    GUARDED_COMBINED,
    apply_failure_risk_guard,
    defaults,
    mechanism_decision,
)
from ht_pdm_fjsp.models import BenchmarkConfig


def _env() -> MachineAgentsCTDEEnv:
    config = BenchmarkConfig.from_json(Path("configs/minimal_benchmark.json"))
    env = MachineAgentsCTDEEnv(config, include_broadcast_context=True)
    env.reset(seed=123)
    return env


def test_profiles_use_fresh_disjoint_panels() -> None:
    smoke = defaults("smoke")
    full = defaults("full")
    assert full["diagnostic_seeds"] == tuple(range(56_000, 56_200))
    assert set(smoke["diagnostic_seeds"]).isdisjoint(full["diagnostic_seeds"])
    assert set(full["diagnostic_seeds"]).isdisjoint(range(50_000, 50_100))
    assert set(full["diagnostic_seeds"]).isdisjoint(range(55_000, 55_200))


def test_guard_masks_risky_production_only_with_preventive_fallback() -> None:
    env = _env()
    env.core.machines["M1"].effective_age = 20.0
    env.core.machines["M1"].preventive_armed = True
    observation = env._ctde_observation(env.core._observation())
    guarded, info = apply_failure_risk_guard(env, observation, threshold=0.0)
    assert info["opportunity_agents"] >= 1
    assert info["suppressed_actions"]
    for item in info["suppressed_actions"]:
        assert observation["action_masks"][item["agent_index"], item["local_action"]]
        assert not guarded["action_masks"][item["agent_index"], item["local_action"]]
    assert np.all(guarded["action_masks"].any(axis=1))
    env.close()


def test_guard_is_noop_without_feasible_preventive_action() -> None:
    env = _env()
    observation = env._ctde_observation(env.core._observation())
    guarded, info = apply_failure_risk_guard(env, observation, threshold=0.0)
    assert info == {"opportunity_agents": 0, "suppressed_actions": []}
    assert np.array_equal(guarded["action_masks"], observation["action_masks"])
    env.close()


def _contrast(objective: tuple[float, ...], failures: tuple[float, ...]) -> dict:
    return {
        "per_training_seed": {
            str(15_000 + index * 1_000): {
                "objective": objective_delta,
                "failures": failure_delta,
            }
            for index, (objective_delta, failure_delta) in enumerate(
                zip(objective, failures, strict=True)
            )
        }
    }


def test_mechanism_gate_requires_complete_consistent_improvement() -> None:
    contrast = _contrast((-1.0,) * 10, (-0.1,) * 10)
    rows = [
        {
            "condition": GUARDED_COMBINED,
            "train_seed": seed,
            "altered_agent_actions": 1,
        }
        for seed in range(15_000, 25_000, 1_000)
    ]
    audits = {
        condition: {"status": "PASS"}
        for condition in (INDEPENDENT_MAPPO, COMBINED_MAPPO, GUARDED_COMBINED)
    }
    decision = mechanism_decision(contrast, rows, audits)
    assert decision["supports_risk_aware_learning_followup"]
    contrast["per_training_seed"]["15000"]["failures"] = 2.0
    decision = mechanism_decision(contrast, rows, audits)
    assert not decision["supports_risk_aware_learning_followup"]
