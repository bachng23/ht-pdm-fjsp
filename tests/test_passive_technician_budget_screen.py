import argparse
import csv
import json
from pathlib import Path

from ht_pdm_fjsp.passive_technician_budget_screen import ALGORITHMS, BUDGETS, run


def test_smoke_budget_screen_writes_checkpoints_and_schema(tmp_path: Path) -> None:
    output = tmp_path / "budget_screen_smoke_20260925T000000Z"
    run(argparse.Namespace(profile="smoke", device="cpu", output_dir=str(output)))

    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["sealed_test_evaluated"] is False
    assert manifest["budgets"] == [8, 16, 32]
    assert manifest["checkpoint_count"] == len(ALGORITHMS) * 3
    assert summary["audits"]["all_expected_rows_present"]
    assert summary["audits"]["all_learned_train_seeds_recorded"]
    assert summary["audits"]["sealed_test_panel_closed"]

    with (output / "episodes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {"policy", "budget", "seed", "train_seed", "objective"}.issubset(rows[0])
    assert all(row["train_seed"] for row in rows if row["policy"] in ALGORITHMS)
    for algorithm in ALGORITHMS:
        for budget in (8, 16, 32):
            suffix = "model.json" if algorithm == "independent_q" else "model.pt"
            assert (output / algorithm / "train_seed_11" / f"budget_{budget}" / suffix).is_file()


def test_full_budget_constants_are_locked() -> None:
    assert BUDGETS == (5_000, 10_000, 20_000)
