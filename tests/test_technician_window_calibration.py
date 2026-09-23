from __future__ import annotations

import argparse
import json
from pathlib import Path

from ht_pdm_fjsp.technician_window_calibration import (
    POLICIES,
    grouped_releases,
    run,
    window_conditions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def test_grouped_releases_preserve_two_rounds_and_form_expected_waves() -> None:
    releases = grouped_releases(2, 1.0)
    assert releases["M1"] == releases["M2"] == (12.0, 36.0)
    assert releases["M3"] == releases["M4"] == (13.0, 37.0)
    assert releases["M5"] == releases["M6"] == (14.0, 38.0)
    assert grouped_releases(6, 0.0) == {
        machine_id: (12.0, 36.0)
        for machine_id in ("M1", "M2", "M3", "M4", "M5", "M6")
    }


def test_calibration_grid_has_one_reference_and_ten_unique_candidates() -> None:
    conditions = window_conditions()
    assert len(conditions) == 11
    assert sum(item.reference for item in conditions) == 1
    assert len({item.name for item in conditions}) == len(conditions)
    candidates = [item for item in conditions if not item.reference]
    assert len(candidates) == 10
    assert sum(item.group_size == 6 for item in candidates) == 1


def test_smoke_run_writes_complete_calibration_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "technician_window_smoke_20260924T000000Z"
    run(
        argparse.Namespace(
            profile="smoke",
            device="cpu",
            config=str(CONFIG_PATH),
            seeds="62300:62302",
            output_dir=str(output),
        )
    )
    manifest = json.loads((output / "technician_window_manifest.json").read_text())
    summary = json.loads((output / "technician_window_summary.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 11 * len(POLICIES) * 2
    assert manifest["confirmation_panel_opened"] is False
    assert summary["gate"]["run_valid"]
    assert len(summary["per_condition"]) == 11
    assert not summary["per_condition"]["parent_reference"]["selection_eligible"]
    assert {
        "benchmark_config.json",
        "resolved_window_conditions.json",
        "technician_window_episodes.partial.csv",
        "technician_window_episodes.csv",
        "technician_window_coordination.partial.csv",
        "technician_window_coordination.csv",
        "technician_window_summary.json",
    }.issubset(set(manifest["output_files"]))
