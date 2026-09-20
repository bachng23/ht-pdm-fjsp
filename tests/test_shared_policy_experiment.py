from __future__ import annotations

from ht_pdm_fjsp.shared_policy_experiment import (
    architecture_defaults,
    summarize_architectures,
)


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


def test_architecture_profiles_keep_future_test_separate() -> None:
    smoke = architecture_defaults("smoke")
    full = architecture_defaults("full")
    assert len(smoke["train_seeds"]) == 2
    assert len(full["train_seeds"]) == 5
    assert set(full["validation_seeds"]).isdisjoint(range(50_000, 50_100))


def test_architecture_summary_uses_training_seed_as_replication_unit() -> None:
    rows = []
    for condition, offset in (
        ("fixed_logits", 0.0),
        ("shared_scorer", -2.0),
        ("shared_scorer_entropy", -3.0),
    ):
        for train_seed, base in ((10, 10.0), (11, 14.0)):
            rows.extend(
                _row(condition, train_seed, seed, base + offset)
                for seed in (41_000, 41_001)
            )
    summary = summarize_architectures(rows)
    comparison = summary["comparisons"]["shared_scorer_minus_fixed_logits"]
    assert comparison["delta_across_training_seeds"]["objective"]["mean"] == -2.0
    entropy = summary["comparisons"][
        "shared_scorer_entropy_minus_fixed_logits"
    ]
    assert entropy["delta_across_training_seeds"]["objective"]["mean"] == -3.0
    assert summary["training_seed_count"] == 2
