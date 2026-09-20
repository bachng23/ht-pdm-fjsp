from __future__ import annotations

from ht_pdm_fjsp.production_context_experiment import (
    CONDITIONS,
    production_context_defaults,
)
from ht_pdm_fjsp.shared_policy_experiment import summarize_architectures


def _row(condition: str, train_seed: int, seed: int, objective: float):
    return {
        "condition": condition,
        "train_seed": train_seed,
        "seed": seed,
        "objective": objective,
        "makespan": objective / 2,
        "total_tardiness": 1.0,
        "failures": 1.0,
        "preventive_maintenance": 1.0,
        "corrective_maintenance": 1.0,
        "total_cost": objective / 2,
    }


def test_production_context_profiles_use_fresh_validation_seeds() -> None:
    smoke = production_context_defaults("smoke")
    full = production_context_defaults("full")
    assert len(smoke["train_seeds"]) == 2
    assert len(full["train_seeds"]) == 5
    assert len(full["validation_seeds"]) == 200
    assert set(full["validation_seeds"]).isdisjoint(range(40_000, 44_000))
    assert set(full["validation_seeds"]).isdisjoint(range(50_000, 50_100))


def test_production_context_summary_uses_shared_entropy_as_control() -> None:
    rows = []
    offsets = {
        "shared_scorer_entropy": 0.0,
        "entity_scorer_entropy": 3.0,
        "production_context_entropy": -2.0,
    }
    for condition in CONDITIONS:
        for train_seed, base in ((10, 10.0), (11, 14.0)):
            for seed in (44_000, 44_001):
                rows.append(_row(condition, train_seed, seed, base + offsets[condition]))
    summary = summarize_architectures(
        rows,
        conditions=CONDITIONS,
        baseline_condition="shared_scorer_entropy",
    )
    comparison = summary["comparisons"][
        "production_context_entropy_minus_shared_scorer_entropy"
    ]
    assert comparison["delta_across_training_seeds"]["objective"]["mean"] == -2.0
    assert comparison["pooled_validation_win_rate"] == 1.0
