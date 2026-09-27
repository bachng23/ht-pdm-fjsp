from __future__ import annotations

import json
from argparse import Namespace

import pytest

from ht_pdm_fjsp.passive_technician_stress_checkpoint_sweep import (
    ALGORITHMS,
    FULL_BUDGETS,
    checkpoint_changes,
    component_interactions,
    primary_results,
    profile_settings,
    run,
)


def test_protocol_seed_panels_and_checkpoints_are_locked() -> None:
    train_seeds, evaluation_seeds, budgets, settings = profile_settings("full")
    assert train_seeds == (83_100, 83_101, 83_102, 83_103, 83_104)
    assert evaluation_seeds == tuple(range(83_500, 83_600))
    assert budgets == FULL_BUDGETS == (5_000, 10_000, 20_000, 30_000, 40_000, 50_000)
    assert settings.episodes == 50_000
    assert ALGORITHMS == ("qmix", "tqmix_no_queue", "tqmix_no_cf", "tqmix")
    sealed = set(range(201, 301))
    assert set(train_seeds).isdisjoint(evaluation_seeds)
    assert set(train_seeds).isdisjoint(sealed)
    assert set(evaluation_seeds).isdisjoint(sealed)


def _summary_row(
    algorithm: str, train_seed: int, scenario: str, budget: int, objective: float
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "train_seed": train_seed,
        "scenario": scenario,
        "budget": budget,
        "objective_mean": objective,
    }


def test_change_and_interaction_signs_match_locked_estimand() -> None:
    rows = []
    # Full degrades by 10, no-queue by 2, no-CF by 4, and QMIX by 6.
    for seed in range(1, 6):
        for algorithm, final in (
            ("tqmix", 20.0),
            ("tqmix_no_queue", 12.0),
            ("tqmix_no_cf", 14.0),
            ("qmix", 16.0),
        ):
            rows.extend(
                (
                    _summary_row(algorithm, seed, "combined_pressure", 20_000, 10.0),
                    _summary_row(algorithm, seed, "combined_pressure", 50_000, final),
                )
            )
    changes = checkpoint_changes(rows, 20_000, 50_000)
    interactions = component_interactions(changes)
    primary = {row["contrast"]: row for row in primary_results(interactions)}
    assert primary["queue_full_minus_no_queue"]["interaction_mean"] == 8.0
    assert primary["cf_full_minus_no_cf"]["interaction_mean"] == 6.0
    assert primary["queue_full_minus_no_queue"]["positive_seed_count"] == 5
    assert primary["queue_full_minus_no_queue"]["directional_replication"] is True


def test_runner_rejects_nonempty_output_directory(tmp_path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(Namespace(profile="smoke", device="cpu", output_dir=str(output)))


def test_smoke_runner_meets_counts_and_audits(tmp_path) -> None:
    output = run(
        Namespace(profile="smoke", device="cpu", output_dir=str(tmp_path / "smoke"))
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["requested_device"] == "cpu"
    assert manifest["resolved_device"] == "cpu"
    assert manifest["actual_counts"] == {
        "evaluation_rows": 288,
        "checkpoints": 24,
        "training_trajectories": 4,
    }
    assert manifest["sealed_test_evaluated"] is False
    required = (
        "all_expected_rows_present",
        "unique_episode_keys",
        "all_checkpoints_present",
        "all_trajectories_updated",
        "losses_finite",
        "cost_reconciliation_passed",
        "resource_semantics_passed",
        "sealed_test_panel_closed",
        "seed_panels_disjoint",
    )
    assert all(manifest["audits"][name] for name in required)
    for filename in (
        "budget_summary.csv",
        "checkpoint_changes.csv",
        "component_interactions.csv",
        "primary_results.csv",
        "training_progress.csv",
        "episodes.csv",
    ):
        assert (output / filename).is_file()
