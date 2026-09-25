from __future__ import annotations

import csv
import json
from argparse import Namespace

from ht_pdm_fjsp.passive_technician_v2_calibration import candidate_configs, run


def test_locked_factorial_contains_twelve_valid_candidates() -> None:
    candidates = candidate_configs()
    assert len(candidates) == 12
    assert set(config.machines for config in candidates.values()) == {4, 6}
    assert all(config.technicians == 2 for config in candidates.values())
    assert all(config.horizon == 36 for config in candidates.values())
    assert all(config.failure_probability == 0.30 for config in candidates.values())
    assert all(len(config.initial_ages) == config.machines for config in candidates.values())
    for config in candidates.values():
        config.validate()


def test_smoke_calibration_activates_choice_and_writes_artifacts(tmp_path) -> None:
    output = tmp_path / "passive_v2_calibration_smoke_20260925T000000Z"
    result = run(Namespace(profile="smoke", device="cpu", output_dir=str(output)))

    assert result == output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "candidate_summary.json").read_text())
    selected = json.loads((output / "selected_candidate.json").read_text())
    resolved = json.loads((output / "resolved_candidates.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["hard_gate_passed"] is True
    assert manifest["candidate_count"] == 12
    assert manifest["episode_count"] == 108
    assert manifest["sealed_test_evaluated"] is False
    assert summary["mechanics_passed"] is True
    assert summary["choice_activated"] is True
    assert summary["authoritative_selection"] is False
    assert selected["authoritative"] is False
    assert len(summary["candidates"]) == 12
    assert len(resolved["fingerprint"]) == 64
    assert len(resolved["candidates"]) == 12
    assert all("mechanics_gate" in candidate for candidate in summary["candidates"])

    with (output / "episodes.csv").open() as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 108
    with (output / "decisions.csv").open() as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 108 * 36
    with (output / "pairwise.csv").open() as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 12 * 3
    for name in (
        "resolved_candidates.json",
        "episodes.partial.csv",
        "coordination.csv",
        "candidate_summary.csv",
    ):
        assert (output / name).is_file()
