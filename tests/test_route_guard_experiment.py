from __future__ import annotations

import numpy as np

from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.route_guard_experiment import (
    _route_indices,
    guard_j1_o1_m2,
    paired_summary,
)


def _episode(condition: str, seed: int, objective: float):
    return {
        "condition": condition,
        "train_seed": 10,
        "seed": seed,
        "objective": objective,
        "makespan": objective / 2,
        "total_tardiness": 1.0,
        "failures": 1.0,
        "preventive_maintenance": 1.0,
        "corrective_maintenance": 1.0,
        "total_cost": objective / 2,
        "decision_count": 20,
        "intervention_activations": int(condition != "original"),
    }


def test_route_guard_suppresses_m2_even_while_m1_is_busy() -> None:
    config = BenchmarkConfig.from_json("configs/minimal_benchmark.json")
    env = HTPdmFjspEnv(config=config)
    env.reset(seed=40_200)
    m1_index, m2_index = _route_indices(env)
    mask = env.action_masks()
    original = mask.copy()
    transformed, applied = guard_j1_o1_m2(env, mask)
    assert applied
    assert transformed[m1_index] == 1
    assert transformed[m2_index] == 0
    assert np.array_equal(mask, original)
    only_m2 = mask.copy()
    only_m2[m1_index] = 0
    transformed, applied = guard_j1_o1_m2(env, only_m2)
    assert applied
    assert transformed[m2_index] == 0
    env.close()


def test_paired_summary_uses_training_seed_means() -> None:
    rows = [
        _episode("original", 1, 10.0),
        _episode("original", 2, 14.0),
        _episode("route_guard_j1_o1_m2", 1, 8.0),
        _episode("route_guard_j1_o1_m2", 2, 10.0),
    ]
    summary = paired_summary(rows, (10,))
    effect = summary["per_training_seed"]["10"]
    assert effect["delta_guard_minus_original"]["objective"] == -3.0
    assert effect["objective_paired_win_rate"] == 1.0
    assert summary["delta_across_training_seeds"]["objective"]["mean"] == -3.0
