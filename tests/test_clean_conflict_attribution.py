from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from ht_pdm_fjsp.clean_conflict_attribution import (
    CELLS,
    POLICIES,
    run,
    seeds_for_profile,
)
from ht_pdm_fjsp.conflict_consequence import (
    SEALED_TEST_SEEDS,
    ConflictConsequenceEnv,
    build_cell_config,
    policy_actions,
)
from ht_pdm_fjsp.parallel_maintenance import ParallelMaintenanceConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "parallel_maintenance.json"


def test_protocol_cells_isolate_request_commitment() -> None:
    assert [cell.committed_request for cell in CELLS] == [False, True]
    assert all(not cell.maintenance_window for cell in CELLS)
    assert all(not cell.strong_substitution for cell in CELLS)
    assert POLICIES == (
        "reactive_fibt",
        "coordinated_preventive",
        "independent_preventive",
    )


def test_fresh_seed_panels_do_not_touch_sealed_panel() -> None:
    assert set(seeds_for_profile("smoke")).isdisjoint(SEALED_TEST_SEEDS)
    assert set(seeds_for_profile("full")).isdisjoint(SEALED_TEST_SEEDS)
    assert set(seeds_for_profile("smoke")).isdisjoint(seeds_for_profile("full"))
    assert len(seeds_for_profile("full")) == 100


def test_clean_policies_share_fastest_preference_but_only_coordinated_reserves() -> None:
    base = ParallelMaintenanceConfig.from_json(CONFIG_PATH)
    cell = CELLS[1]
    config = build_cell_config(base, cell)
    independent = ConflictConsequenceEnv(config, cell)
    coordinated = ConflictConsequenceEnv(config, cell)
    independent.reset(seed=7)
    coordinated.reset(seed=7)
    independent.age[:] = 120.0
    coordinated.age[:] = 120.0

    independent_actions = policy_actions("independent_preventive", independent)
    coordinated_actions = policy_actions("coordinated_preventive", coordinated)
    first_candidate = independent.candidates(preventive=True)[0]
    fastest = min(
        range(independent.technician_count),
        key=lambda technician: (
            independent.expected_duration(first_candidate, technician),
            technician,
        ),
    )
    assert independent_actions[first_candidate] == fastest + 1
    assert coordinated_actions[first_candidate] == fastest + 1
    proposed = [int(action) for action in independent_actions if action]
    reserved = [int(action) for action in coordinated_actions if action]
    assert len(proposed) > len(set(proposed))
    assert len(reserved) == len(set(reserved))


def test_runner_refuses_sealed_seed_before_creating_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="sealed test seeds"):
        run(
            argparse.Namespace(
                config=str(CONFIG_PATH),
                profile="smoke",
                device="cpu",
                seeds=str(SEALED_TEST_SEEDS[0]),
                output_dir=str(output),
            )
        )
    assert not output.exists()


def test_smoke_run_writes_complete_audited_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "clean_conflict_smoke_20260924T000000Z"
    summary = run(
        argparse.Namespace(
            config=str(CONFIG_PATH),
            profile="smoke",
            device="cpu",
            seeds="63990:63992",
            output_dir=str(output),
        )
    )
    manifest = json.loads((output / "clean_conflict_manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 2 * len(POLICIES) * 2
    assert manifest["decision_count"] == manifest["episode_count"] * 168
    assert manifest["sealed_test_evaluated"] is False
    assert summary["gate"]["passed"]
    assert not summary["promotion"]["confirmatory"]
    assert summary["primary_cell"] == "commit1_window0_sub0"
    if (
        summary["cell_summaries"][summary["primary_cell"]][
            "independent_minus_coordinated_greedy"
        ]["mean"]
        <= 0
    ):
        assert summary["explicit_conflict_penalty_share"] is None
    assert all(
        cell["decomposition_error"] <= 1e-6
        for cell in summary["cell_summaries"].values()
    )
    for filename in (
        "source_config.json",
        "resolved_base_config.json",
        "cells.json",
        "episodes.partial.csv",
        "episodes.csv",
        "decisions.csv",
        "summary.json",
    ):
        assert (output / filename).is_file()


def test_runner_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "existing.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="new or empty"):
        run(
            argparse.Namespace(
                config=str(CONFIG_PATH),
                profile="smoke",
                device="cpu",
                seeds="63990:63991",
                output_dir=str(output),
            )
        )
    assert (output / "existing.txt").read_text() == "keep"
