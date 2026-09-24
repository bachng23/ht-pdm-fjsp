from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ht_pdm_fjsp.parallel_maintenance import (
    FAILED,
    ParallelMaintenanceConfig,
    ParallelMaintenanceEnv,
)
from ht_pdm_fjsp.parallel_maintenance_calibration import (
    POLICIES,
    CalibrationCell,
    build_cell_config,
    policy_actions,
    run,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "parallel_maintenance.json"


def test_cell_config_changes_only_locked_factors() -> None:
    base = ParallelMaintenanceConfig.from_json(CONFIG_PATH)
    cell = CalibrationCell(36.0, 0.35, 0.005)
    config = build_cell_config(base, cell)
    assert config.costs.failure == 36.0
    assert config.maintenance.pm_duration_factor == 0.35
    assert config.maintenance.pm_risk_threshold == 0.005
    assert config.technicians == base.technicians
    assert config.machines == base.machines
    assert config.costs.downtime == base.costs.downtime


def test_reactive_policy_never_requests_pm_and_independent_policy_can_conflict() -> None:
    config = ParallelMaintenanceConfig.from_json(CONFIG_PATH)
    env = ParallelMaintenanceEnv(config)
    env.reset(seed=1)
    env.age[:] = 120.0
    reactive = policy_actions("reactive_fibt", env)
    independent = policy_actions("independent_preventive", env)
    coordinated = policy_actions("preventive_fibt", env)
    assert np.count_nonzero(reactive) == 0
    assert np.count_nonzero(independent) > 1
    assert len(set(independent[independent > 0])) < np.count_nonzero(independent)
    assert len(set(coordinated[coordinated > 0])) == np.count_nonzero(coordinated)
    env.mode[0] = FAILED
    assert policy_actions("reactive_fibt", env)[0] > 0


def test_summary_promotes_a_cell_only_when_all_mechanism_gates_pass() -> None:
    cell = CalibrationCell(36.0, 0.35, 0.005)
    rows = []
    for policy in POLICIES:
        rows.append(
            {
                "cell_id": cell.cell_id,
                "policy": policy,
                "seed": 1,
                "objective": -110.0 if policy == "preventive_fibt" else -100.0,
                "production": 10.0,
                "downtime": 2.0,
                "failures": 5.0 if policy == "preventive_fibt" else 10.0,
                "preventive": 2.0,
                "corrective": 1.0,
                "waiting": 0.0,
                "proposal_conflicts": 0.0,
                "conflict_steps": 10.0 if policy == "independent_preventive" else 0.0,
                "rework": 0.0,
                "workload_imbalance": 0.0,
                "eligible_demand_steps": 10.0,
                "excess_demand_steps": 2.0,
                "same_best_technician_steps": 5.0,
                "return_objective_error": 0.0,
                "invalid_executions": 0,
                "duplicate_machine_assignments": 0,
                "duplicate_technician_assignments": 0,
            }
        )
    result = summarize(rows, cells=(cell,), seeds=(1,), profile="smoke")
    assert result["selected_candidate_cell"] == cell.cell_id
    assert result["cell_summaries"][cell.cell_id]["qualifies"]
    assert result["gate"]["passed"]


def test_smoke_calibration_writes_complete_audited_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "parallel_calibration_smoke_20260924T000000Z"
    result = run(
        argparse.Namespace(
            config=str(CONFIG_PATH),
            profile="smoke",
            seeds="63490:63491",
            output_dir=str(output),
        )
    )
    manifest = json.loads((output / "calibration_manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 8
    assert manifest["decision_count"] == 8 * 168
    assert manifest["sealed_test_evaluated"] is False
    assert result["gate"]["passed"]
    for filename in (
        "resolved_base_config.json",
        "cells.json",
        "episodes.partial.csv",
        "episodes.csv",
        "decisions.csv",
        "summary.json",
    ):
        assert (output / filename).is_file()
