from __future__ import annotations

import argparse
import json
from pathlib import Path

from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.technician_conflict_factorial import (
    FACTORS,
    POLICIES,
    FactorSpec,
    LiteratureFactorialSimulator,
    build_condition_config,
    condition_specs,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def test_factorial_enumerates_every_binary_combination_once() -> None:
    specs = condition_specs()
    assert len(specs) == 32
    assert len({spec.bits for spec in specs}) == 32
    assert specs[0].bits == (0, 0, 0, 0, 0)
    assert specs[-1].bits == (1, 1, 1, 1, 1)


def test_overlap_and_duration_factors_are_orthogonal() -> None:
    base = BenchmarkConfig.from_json(CONFIG_PATH)
    overlap = build_condition_config(base, FactorSpec(False, False, True, False, False))
    pressured = build_condition_config(base, FactorSpec(False, False, True, False, True))
    machines = {machine.machine_id for machine in base.machines}
    assert all(set(technician.eligible_machines) == machines for technician in overlap.technicians)
    for before, after in zip(overlap.technicians, pressured.technicians):
        for machine_id in machines:
            assert after.duration(machine_id, "preventive") == 2.0 * before.duration(machine_id, "preventive")
            assert after.duration(machine_id, "corrective") == 2.0 * before.duration(machine_id, "corrective")


def test_both_assignment_policies_complete_and_pass_feasibility_audit() -> None:
    base = BenchmarkConfig.from_json(CONFIG_PATH)
    spec = FactorSpec(True, True, True, True, True)
    config = build_condition_config(base, spec)
    results = {
        policy: LiteratureFactorialSimulator(config, spec, policy).run(seed=61_700)[0]
        for policy in POLICIES
    }
    assert all(row["feasibility_audit_passed"] == 1 for row in results.values())
    assert all(row["invalid_executions"] == 0 for row in results.values())
    assert all(row["duplicate_operation_executions"] == 0 for row in results.values())
    assert all(row["duplicate_technician_executions"] == 0 for row in results.values())
    assert results["conflict_aware_matcher"]["technician_conflicts"] == 0


def test_smoke_run_writes_complete_factorial_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "technician_factorial_smoke_20260924T000000Z"
    run(
        argparse.Namespace(
            profile="smoke",
            device="cpu",
            config=str(CONFIG_PATH),
            seeds="61700:61702",
            output_dir=str(output),
        )
    )
    manifest = json.loads((output / "technician_factorial_manifest.json").read_text())
    summary = json.loads((output / "technician_factorial_summary.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 32 * len(POLICIES) * 2
    assert manifest["confirmation_panel_opened"] is False
    assert manifest["factor_order"] == list(FACTORS)
    assert summary["gate"]["run_valid"]
    assert len(summary["per_condition"]) == 32
    assert len(summary["factorial_effects"]) == (5 + 10) * 4
    assert {
        "benchmark_config.json",
        "resolved_conditions.json",
        "technician_factorial_episodes.partial.csv",
        "technician_factorial_episodes.csv",
        "technician_factorial_coordination.partial.csv",
        "technician_factorial_coordination.csv",
        "technician_factorial_effects.csv",
        "technician_factorial_summary.json",
    }.issubset(set(manifest["output_files"]))
