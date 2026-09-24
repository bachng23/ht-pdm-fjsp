from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ht_pdm_fjsp.conflict_consequence import (
    POLICIES,
    WAITING,
    ConflictConsequenceEnv,
    ConsequenceCell,
    build_cell_config,
    policy_actions,
    run,
    selected_parent_config,
)
from ht_pdm_fjsp.parallel_maintenance import ParallelMaintenanceConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "parallel_maintenance.json"


def deterministic_config(cell: ConsequenceCell) -> ParallelMaintenanceConfig:
    base = ParallelMaintenanceConfig.from_json(CONFIG_PATH)
    config = build_cell_config(base, cell)
    return replace(
        config,
        horizon=8,
        technicians=tuple(
            replace(
                technician,
                absence_probability=0.0,
                duration_sigma=tuple(0.0 for _ in technician.duration_sigma),
            )
            for technician in config.technicians
        ),
    )


def test_parent_configuration_is_locked_to_promoted_calibration_cell() -> None:
    base = ParallelMaintenanceConfig.from_json(CONFIG_PATH)
    config = selected_parent_config(base)
    assert config.costs.failure == 36.0
    assert config.maintenance.pm_duration_factor == 0.35
    assert config.maintenance.pm_risk_threshold == 0.005


def test_strong_substitution_makes_every_nonspecialist_slower_and_less_skilled() -> None:
    base = ParallelMaintenanceConfig.from_json(CONFIG_PATH)
    low = build_cell_config(base, ConsequenceCell(False, False, False))
    high = build_cell_config(base, ConsequenceCell(False, False, True))
    for failure_type in range(3):
        specialist = min(
            range(2),
            key=lambda index: low.technicians[index].base_duration[failure_type],
        )
        nonspecialist = 1 - specialist
        specialist_duration = high.technicians[specialist].base_duration[failure_type]
        assert high.technicians[nonspecialist].base_duration[
            failure_type
        ] >= 2.5 * specialist_duration - 1e-12
        assert high.technicians[nonspecialist].base_skill[failure_type] <= 0.25


def test_rejected_committed_request_stops_machine_and_persists() -> None:
    cell = ConsequenceCell(True, False, False)
    env = ConflictConsequenceEnv(deterministic_config(cell), cell)
    env.reset(seed=1)
    env.age[:] = 120.0
    actions = policy_actions("independent_preventive", env)
    assert np.count_nonzero(actions) > env.technician_count
    _, _, info = env.step(actions)
    assert info["proposal_conflicts"] > 0
    rejected = [index for index in range(env.num_agents) if env.pending_kind[index]]
    assert rejected
    assert all(env.mode[index] == WAITING for index in rejected)
    assert all(env.wait[index] > 0 for index in rejected)


def test_rejected_uncommitted_request_keeps_producing() -> None:
    cell = ConsequenceCell(False, False, False)
    env = ConflictConsequenceEnv(deterministic_config(cell), cell)
    env.reset(seed=2)
    env.age[:] = 120.0
    actions = policy_actions("independent_preventive", env)
    env.step(actions)
    rejected = [index for index in range(env.num_agents) if env.pending_kind[index]]
    assert rejected
    assert all(env.mode[index] == 0 for index in rejected)


def test_missed_window_triples_conditional_hazard() -> None:
    cell = ConsequenceCell(False, True, False)
    env = ConflictConsequenceEnv(deterministic_config(cell), cell)
    env.reset(seed=3)
    env.age[0] = 70.0
    base = env.base_failure_probability(0)
    env.overdue[0] = True
    assert env.failure_probability(0) == pytest.approx(1 - (1 - base) ** 3)


def test_coordinated_policies_never_propose_duplicate_technicians() -> None:
    cell = ConsequenceCell(True, True, True)
    env = ConflictConsequenceEnv(deterministic_config(cell), cell)
    env.reset(seed=4)
    env.age[:] = 120.0
    for policy in ("coordinated_preventive", "coordinated_matching"):
        actions = policy_actions(policy, env)
        selected = [int(action) for action in actions if action]
        assert len(selected) == len(set(selected))


def test_smoke_run_writes_audited_factorial_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "conflict_consequence_smoke_20260924T000000Z"
    summary = run(
        argparse.Namespace(
            config=str(CONFIG_PATH),
            profile="smoke",
            seeds="63590:63591",
            output_dir=str(output),
        )
    )
    manifest = json.loads(
        (output / "conflict_consequence_manifest.json").read_text()
    )
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 8 * len(POLICIES)
    assert manifest["decision_count"] == manifest["episode_count"] * 168
    assert manifest["sealed_test_evaluated"] is False
    assert summary["gate"]["passed"]
    assert len(summary["cell_summaries"]) == 8
    for filename in (
        "resolved_base_config.json",
        "cells.json",
        "episodes.partial.csv",
        "episodes.csv",
        "decisions.csv",
        "summary.json",
    ):
        assert (output / filename).is_file()
