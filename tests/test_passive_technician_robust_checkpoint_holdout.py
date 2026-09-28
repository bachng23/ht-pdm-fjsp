from __future__ import annotations

import csv
import hashlib
import json
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from ht_pdm_fjsp.passive_technician_leave_one_out import environment_cells
from ht_pdm_fjsp.passive_technician_robust_checkpoint_holdout import (
    CANDIDATES,
    FULL_EVALUATION_SEEDS,
    SELECTED_CANDIDATE,
    SOURCE_EVALUATION_SEEDS,
    SOURCE_GIT_REVISION,
    SOURCE_TRAIN_SEEDS,
    primary_results,
    profile_settings,
    robust_scores,
    run,
    selection_scores,
)
from ht_pdm_fjsp.passive_technician_stress_checkpoint_sweep import (
    ALGORITHMS,
    FULL_BUDGETS,
    PROTOCOL_VERSION as SOURCE_PROTOCOL_VERSION,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition,
    ValueTrainSettings,
)


SCENARIOS = ("in_distribution", "early_failure", "slow_service", "combined_pressure")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selection_fixture_rows() -> list[dict[str, object]]:
    rows = []
    # Full@40k is balanced at 12 everywhere. Every other candidate has at least
    # one scenario above 12; scenario minima are 10 from specialized candidates.
    for algorithm in ALGORITHMS:
        for seed in SOURCE_TRAIN_SEEDS:
            for scenario_index, scenario in enumerate(SCENARIOS):
                for budget in FULL_BUDGETS:
                    value = 20.0 + scenario_index
                    if algorithm == "tqmix" and budget == 40_000:
                        value = 12.0
                    if algorithm == ALGORITHMS[scenario_index] and budget == 10_000:
                        value = 10.0
                    rows.append(
                        {
                            "algorithm": algorithm,
                            "reporting_name": algorithm,
                            "train_seed": seed,
                            "scenario": scenario,
                            "budget": budget,
                            "evaluation_count": 100,
                            "objective_mean": value,
                        }
                    )
    return rows


def _make_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    cells = environment_cells()
    manifest = {
        "status": "COMPLETED",
        "protocol_version": SOURCE_PROTOCOL_VERSION,
        "git_revision": SOURCE_GIT_REVISION,
        "git_dirty": False,
        "algorithms": list(ALGORITHMS),
        "checkpoint_budgets": list(FULL_BUDGETS),
        "train_seeds": list(SOURCE_TRAIN_SEEDS),
        "evaluation_seeds": list(SOURCE_EVALUATION_SEEDS),
        "expected_counts": {
            "evaluation_rows": 48_000,
            "checkpoints": 120,
            "training_trajectories": 20,
        },
        "actual_counts": {
            "evaluation_rows": 48_000,
            "checkpoints": 120,
            "training_trajectories": 20,
        },
        "sealed_test_evaluated": False,
        "environment_cells": {name: asdict(config) for name, config in cells.items()},
        "audits": {
            "all_expected_rows_present": True,
            "unique_episode_keys": True,
            "all_checkpoints_present": True,
            "all_trajectories_updated": True,
            "losses_finite": True,
            "cost_reconciliation_passed": True,
            "resource_semantics_passed": True,
            "sealed_test_panel_closed": True,
            "seed_panels_disjoint": True,
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_csv(source / "budget_summary.csv", _selection_fixture_rows())
    config = cells["in_distribution"]
    settings = ValueTrainSettings(episodes=50_000, hidden_dim=16, mixer_hidden_dim=8)
    for _, algorithm, budget, _ in CANDIDATES:
        path = (
            source
            / algorithm
            / f"train_seed_{SOURCE_TRAIN_SEEDS[0]}"
            / "checkpoints"
            / f"budget_{budget}"
            / "model.pt"
        )
        PassiveValueDecomposition(config, algorithm, 16, 8).save(
            path, SOURCE_TRAIN_SEEDS[0], settings
        )
    return source


def test_protocol_panels_and_candidate_registry_are_locked() -> None:
    train_seeds, evaluation_seeds = profile_settings("full")
    assert train_seeds == SOURCE_TRAIN_SEEDS
    assert evaluation_seeds == FULL_EVALUATION_SEEDS == tuple(range(83_600, 83_700))
    assert SELECTED_CANDIDATE == ("tqmix", 40_000)
    assert len(CANDIDATES) == 5
    panels = (
        set(SOURCE_TRAIN_SEEDS),
        set(SOURCE_EVALUATION_SEEDS),
        set(FULL_EVALUATION_SEEDS),
        set(range(201, 301)),
    )
    assert all(
        panels[left].isdisjoint(panels[right])
        for left in range(len(panels))
        for right in range(left + 1, len(panels))
    )


def test_selection_rule_reproduces_locked_candidate() -> None:
    rows = [{key: str(value) for key, value in row.items()} for row in _selection_fixture_rows()]
    scores, references, selected = selection_scores(rows, SCENARIOS)
    assert selected == SELECTED_CANDIDATE
    assert references == {scenario: 10.0 for scenario in SCENARIOS}
    selected_row = next(row for row in scores if row["selected"])
    assert selected_row["max_relative_regret"] == pytest.approx(0.2)


def test_primary_estimand_uses_selected_minus_full_50k() -> None:
    rows = []
    for seed in SOURCE_TRAIN_SEEDS:
        rows.extend(
            (
                {
                    "row_type": "training_seed",
                    "candidate_id": "selected_full_40k",
                    "train_seed": seed,
                    "max_relative_regret": 0.2,
                },
                {
                    "row_type": "training_seed",
                    "candidate_id": "full_50k",
                    "train_seed": seed,
                    "max_relative_regret": 0.3,
                },
            )
        )
    result = primary_results(rows)
    aggregate = result[-1]
    assert aggregate["delta_mean"] == pytest.approx(-0.1)
    assert aggregate["selected_better_count"] == 5
    assert aggregate["directional_replication"] is True


def test_runner_rejects_nonempty_output(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(
            Namespace(
                profile="smoke",
                device="cpu",
                source_run=str(source),
                output_dir=str(output),
            )
        )


def test_smoke_runner_preserves_source_and_meets_audits(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    source_manifest = source / "manifest.json"
    before = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    output = run(
        Namespace(
            profile="smoke",
            device="cpu",
            source_run=str(source),
            output_dir=str(tmp_path / "output"),
        )
    )
    after = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    assert before == after
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["resolved_device"] == "cpu"
    assert manifest["selected_candidate"] == {"algorithm": "tqmix", "budget": 40_000}
    assert manifest["actual_counts"] == {
        "evaluation_rows": 60,
        "source_checkpoints": 5,
        "candidate_training_seed_cells": 5,
    }
    required = (
        "source_manifest_valid",
        "selection_reproduced",
        "all_expected_rows_present",
        "unique_episode_keys",
        "all_source_checkpoints_present",
        "source_files_unchanged",
        "all_objectives_finite",
        "cost_reconciliation_passed",
        "resource_semantics_passed",
        "holdout_progress_complete",
        "seed_panels_disjoint",
        "sealed_test_panel_closed",
    )
    assert all(manifest["audits"][name] for name in required)
    with (output / "episodes.csv").open(encoding="utf-8") as stream:
        assert sum(1 for _ in stream) - 1 == 60
    with (output / "selection_scores.csv").open(encoding="utf-8") as stream:
        assert sum(1 for _ in stream) - 1 == 24
    assert (output / "primary_results.csv").is_file()
