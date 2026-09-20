from __future__ import annotations

from ht_pdm_fjsp.multi_seed_experiment import aggregate_results, replication_defaults


def _row(policy: str, seed: int, objective: float, train_seed: int | None = None):
    row = {
        "split": "test",
        "policy": policy,
        "seed": seed,
        "objective": objective,
        "makespan": objective / 2,
        "total_tardiness": 1.0,
        "failures": 1.0,
        "preventive_maintenance": 1.0,
        "corrective_maintenance": 1.0,
        "total_cost": objective / 2,
    }
    if train_seed is not None:
        row["train_seed"] = train_seed
    return row


def test_replication_profiles_separate_smoke_and_full() -> None:
    smoke = replication_defaults("smoke")
    full = replication_defaults("full")
    assert len(smoke["train_seeds"]) == 2
    assert len(full["train_seeds"]) == 5
    assert smoke["total_timesteps"] < full["total_timesteps"]


def test_aggregate_uses_training_seed_as_replication_unit() -> None:
    ppo_rows = [
        _row("maskable_ppo", 30_000, 8.0, 10_000),
        _row("maskable_ppo", 30_001, 10.0, 10_000),
        _row("maskable_ppo", 30_000, 12.0, 11_000),
        _row("maskable_ppo", 30_001, 14.0, 11_000),
    ]
    baseline_rows = [
        _row("joint_risk_greedy", 30_000, 15.0),
        _row("joint_risk_greedy", 30_001, 15.0),
    ]
    result = aggregate_results(ppo_rows, baseline_rows)
    assert result["training_seed_count"] == 2
    assert result["ppo_per_training_seed"]["10000"]["objective"] == 9.0
    assert result["ppo_per_training_seed"]["11000"]["objective"] == 13.0
    comparison = result["paired_comparisons"]["joint_risk_greedy"]
    assert comparison["objective_delta_across_training_seeds"]["mean"] == -4.0
