from __future__ import annotations

import argparse
import json
from pathlib import Path

from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.technician_capacity_screening import (
    REFERENCE,
    build_condition_config,
    condition_specs,
    evaluate_episode,
    run,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def test_builds_two_specialist_and_shared_capacity_variants() -> None:
    base = BenchmarkConfig.from_json(CONFIG_PATH)
    scaled = build_condition_config(base, topology="two_specialists", multiplier=1.5)
    assert len(scaled.technicians) == 2
    assert scaled.technicians[0].preventive_duration["M1"] == 1.5

    shared = build_condition_config(base, topology="one_shared", multiplier=2.0)
    assert len(shared.technicians) == 1
    technician = shared.technicians[0]
    assert technician.technician_id == "T_SHARED"
    assert set(technician.eligible_machines) == {f"M{index}" for index in range(1, 7)}
    assert technician.preventive_duration["M4"] == 2.0
    assert technician.corrective_duration["M4"] == 4.0


def test_episode_exposes_queue_and_utilization_endpoints() -> None:
    base = BenchmarkConfig.from_json(CONFIG_PATH)
    config = build_condition_config(base, topology="one_shared", multiplier=2.0)
    row = evaluate_episode(
        config,
        condition="one_shared_x2_0",
        topology="one_shared",
        duration_multiplier=2.0,
        seed=61_430,
    )
    assert row["technician_count"] == 1
    assert row["objective"] == row["total_cost"]
    assert row["maintenance_wait_time"] >= 0
    assert 0 <= row["technician_utilization"] <= 1
    assert row["any_queue_wait"] in {0, 1}


def test_summary_uses_locked_moderate_binding_selection_rule() -> None:
    rows = []
    for condition, topology, multiplier in condition_specs():
        for seed in range(1, 11):
            incidence = (
                condition == "one_shared_x1_5" and seed <= 4
            ) or (
                condition == "one_shared_x2_0" and seed <= 2
            )
            row = {
                "condition": condition,
                "topology": topology,
                "duration_multiplier": multiplier,
                "seed": seed,
                "any_queue_wait": int(incidence),
                "any_preventive_wait": int(incidence),
                "any_corrective_wait": 0,
            }
            for metric in (
                "objective", "makespan", "schedule_end", "failures",
                "preventive_maintenance", "corrective_maintenance",
                "maintenance_time", "failure_wait_time", "preventive_wait_time",
                "maintenance_wait_time", "technician_utilization", "processed_events",
            ):
                row[metric] = 0.0
            row["maintenance_wait_time"] = 1.0 if incidence else 0.0
            row["preventive_wait_time"] = row["maintenance_wait_time"]
            row["technician_utilization"] = 0.5
            rows.append(row)
    summary = summarize(rows)
    assert summary["selected_condition"] == "one_shared_x2_0"
    assert summary["per_condition"][REFERENCE]["moderate_binding_gate"] is False


def test_smoke_run_writes_completed_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "technician_capacity_smoke_20260923T000000Z"
    run(argparse.Namespace(
        profile="smoke",
        config=str(CONFIG_PATH),
        seeds="61430:61432",
        output_dir=str(output),
    ))
    manifest = json.loads((output / "technician_capacity_manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 12
    assert manifest["future_test_panel_opened"] is False
    assert {
        "benchmark_config.json",
        "condition_configs.json",
        "technician_capacity_episodes.csv",
        "technician_capacity_episodes.partial.csv",
        "technician_capacity_summary.json",
    }.issubset(set(manifest["output_files"]))
