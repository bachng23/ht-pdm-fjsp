from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ht_pdm_fjsp.technician_cluster_mixture import (
    POLICIES,
    mixture_conditions,
    mixture_uniform,
    resolved_releases,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def test_mixture_grid_has_two_references_and_seven_candidates() -> None:
    conditions = mixture_conditions()
    assert [item.cluster_probability for item in conditions] == [
        0.0,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        1.0,
    ]
    assert sum(item.reference for item in conditions) == 2
    assert len({item.name for item in conditions}) == 9


def test_seed_mixture_draw_is_deterministic_and_assignments_are_nested() -> None:
    conditions = mixture_conditions()
    for seed in range(62_800, 62_820):
        uniform = mixture_uniform(seed)
        assert 0.0 <= uniform < 1.0
        assert uniform == mixture_uniform(seed)
        assignments = [
            resolved_releases(condition, seed)[0] == "skill_clustered"
            for condition in conditions
        ]
        assert assignments[0] is False
        assert assignments[-1] is True
        assert assignments == sorted(assignments)


def test_smoke_run_writes_paired_schedule_and_complete_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "technician_mixture_smoke_20260924T000000Z"
    run(
        argparse.Namespace(
            profile="smoke",
            device="cpu",
            config=str(CONFIG_PATH),
            seeds="62800:62802",
            output_dir=str(output),
        )
    )
    manifest = json.loads((output / "technician_mixture_manifest.json").read_text())
    summary = json.loads((output / "technician_mixture_summary.json").read_text())
    with (output / "technician_mixture_episodes.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 9 * len(POLICIES) * 2
    assert manifest["confirmation_panel_opened"] is False
    assert summary["gate"]["run_valid"]
    assert len(summary["per_condition"]) == 9
    for condition in {row["condition"] for row in episodes}:
        for seed in {row["seed"] for row in episodes}:
            rows = [
                row
                for row in episodes
                if row["condition"] == condition and row["seed"] == seed
            ]
            assert len({row["schedule_name"] for row in rows}) == 1
            assert len({row["mixture_uniform"] for row in rows}) == 1
    assert {
        "benchmark_config.json",
        "resolved_mixture_conditions.json",
        "technician_mixture_episodes.partial.csv",
        "technician_mixture_episodes.csv",
        "technician_mixture_coordination.partial.csv",
        "technician_mixture_coordination.csv",
        "technician_mixture_summary.json",
    }.issubset(set(manifest["output_files"]))
