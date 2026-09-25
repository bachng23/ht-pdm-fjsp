import argparse
import csv
import json
from pathlib import Path

import torch

import ht_pdm_fjsp.passive_technician_budget_screen as budget_screen
from ht_pdm_fjsp.passive_technician_budget_screen import (
    ALGORITHMS,
    BUDGETS,
    _profile,
    _summary,
    run,
)
from ht_pdm_fjsp.passive_technician_baselines import PPOSettings, stress_config


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


def test_replication_profile_uses_fresh_training_seeds_and_locked_panels() -> None:
    train_seeds, evaluation_seeds, budgets = _profile("replication")

    assert train_seeds == tuple(range(14, 24))
    assert evaluation_seeds == tuple(range(101, 201))
    assert budgets == BUDGETS


def test_summary_uses_training_seeds_for_paired_budget_interval() -> None:
    rows = []
    for train_seed, delta in ((14, -2.0), (15, -3.0), (16, -4.0)):
        for budget, objective in ((5_000, 10.0), (20_000, 10.0 + delta)):
            rows.append(
                {
                    "policy": "centralized_ppo",
                    "budget": budget,
                    "seed": 101,
                    "train_seed": train_seed,
                    "objective": objective,
                    "failures": 0,
                    "jobs": 1,
                    "collisions": 0,
                    "waiting": 0,
                }
            )

    result = _summary(rows, (14, 15, 16), (101,), (5_000, 20_000))
    contrast = result["paired_objective_contrasts"]["centralized_ppo"]

    assert contrast["training_seed_count"] == 3
    assert contrast["training_seed_deltas"] == [-2.0, -3.0, -4.0]
    assert contrast["mean_delta_high_minus_low"] == -3.0
    assert contrast["supports_objective_reduction"]


def test_centralized_ppo_builds_returns_on_requested_device(tmp_path: Path, monkeypatch) -> None:
    requested_device = torch.device("cpu")
    observed_devices: list[torch.device | None] = []
    original_returns = budget_screen._returns

    def tracked_returns(rewards, gamma, device=None):
        observed_devices.append(device)
        return original_returns(rewards, gamma, device=device)

    monkeypatch.setattr(budget_screen, "_returns", tracked_returns)
    budget_screen._train_centralized_ppo_checkpoints(
        stress_config(),
        seed=11,
        budgets=(1,),
        root=tmp_path / "centralized",
        settings=PPOSettings(episodes=1, update_epochs=1),
        device=requested_device,
    )

    assert observed_devices == [requested_device]
