from __future__ import annotations

from ht_pdm_fjsp.entity_conditioned_experiment import (
    CONDITIONS,
    _paired_summary,
    entity_defaults,
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


def test_entity_profiles_use_a_fresh_validation_panel() -> None:
    smoke = entity_defaults("smoke")
    full = entity_defaults("full")
    assert len(smoke["train_seeds"]) == 2
    assert len(full["train_seeds"]) == 5
    assert set(full["validation_seeds"]).isdisjoint(range(40_000, 42_000))
    assert set(full["validation_seeds"]).isdisjoint(range(50_000, 50_100))


def test_entity_ablation_is_paired_by_training_and_validation_seed() -> None:
    rows = []
    offsets = {
        "fixed_logits": 0.0,
        "shared_scorer": 2.0,
        "shared_scorer_entropy": 1.0,
        "entity_scorer": -1.0,
        "entity_scorer_entropy": -2.0,
    }
    for condition in CONDITIONS:
        for train_seed, base in ((10, 10.0), (11, 14.0)):
            for seed in (42_000, 42_001):
                rows.append(
                    _row(condition, train_seed, seed, base + offsets[condition])
                )
    summary = summarize_architectures(rows, conditions=CONDITIONS)
    assert summary["training_seed_count"] == 2
    paired = _paired_summary(rows, "entity_scorer", "shared_scorer")
    assert paired["delta_across_training_seeds"]["objective"]["mean"] == -3.0
    assert paired["pooled_validation_win_rate"] == 1.0
