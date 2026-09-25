import argparse
import csv
import json
from pathlib import Path

import pytest
import torch

from ht_pdm_fjsp.passive_technician_confirmatory import (
    CONFIRMATORY_ALGORITHMS,
    FULL_BUDGETS,
    FULL_EVALUATION_SEEDS,
    FULL_TRAIN_SEEDS,
    SEALED_TEST_SEEDS,
    _paired_effects,
    _profile,
    exact_sign_flip_p,
    holm_adjust,
    paired_bootstrap_interval,
    run,
    scenario_configs,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition,
    evaluate_value_decomposition,
)
from ht_pdm_fjsp.passive_technician_baselines import evaluate_fixed


def test_confirmatory_protocol_is_locked_to_ten_training_seeds() -> None:
    train, evaluation, budgets = _profile("full")
    assert train == tuple(range(21, 31)) == FULL_TRAIN_SEEDS
    assert evaluation == tuple(range(1001, 1101)) == FULL_EVALUATION_SEEDS
    assert budgets == (20_000, 50_000) == FULL_BUDGETS
    assert SEALED_TEST_SEEDS == tuple(range(2001, 2101))
    assert CONFIRMATORY_ALGORITHMS == (
        "qmix",
        "qmix_queue",
        "qmix_counterfactual",
        "tqmix",
    )


def test_reporting_scenarios_keep_model_dimensions_fixed() -> None:
    scenarios = scenario_configs()
    assert tuple(scenarios) == (
        "in_distribution",
        "early_failure",
        "slow_service",
        "combined_pressure",
    )
    dimensions = {
        (config.machines, config.technicians, config.observation_dim)
        for config in scenarios.values()
    }
    assert len(dimensions) == 1
    assert scenarios["combined_pressure"].failure_age == 4
    assert scenarios["combined_pressure"].service_time == ((3, 5), (5, 3), (4, 4))


def test_exact_sign_flip_and_holm_statistics() -> None:
    assert exact_sign_flip_p([-1.0] * 10) == pytest.approx(2 / 1024)
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.2})
    first = paired_bootstrap_interval([-3.0, -2.0, -1.0], key="fixed", replicates=500)
    second = paired_bootstrap_interval([-3.0, -2.0, -1.0], key="fixed", replicates=500)
    assert first == second
    assert first[1] < 0.0


def test_paired_effects_use_training_seeds_as_independent_units() -> None:
    rows = []
    objectives = {
        "qmix": 10.0,
        "qmix_queue": 9.0,
        "qmix_counterfactual": 8.0,
        "tqmix": 11.0,
    }
    for policy, objective in objectives.items():
        for train_seed in FULL_TRAIN_SEEDS:
            rows.append(
                {
                    "policy": policy,
                    "budget": 50_000,
                    "scenario": "in_distribution",
                    "train_seed": train_seed,
                    "objective_mean": objective,
                }
            )
    effects = {row["policy"]: row for row in _paired_effects(rows)}
    counterfactual = effects["qmix_counterfactual"]
    assert counterfactual["training_seed_count"] == 10
    assert counterfactual["mean_difference"] == -2.0
    assert counterfactual["ci95_low"] == -2.0
    assert counterfactual["ci95_high"] == -2.0
    assert counterfactual["exact_sign_flip_p"] == pytest.approx(2 / 1024)
    assert counterfactual["seed_wins"] == 10


def test_checkpoint_can_evaluate_all_reporting_scenarios(tmp_path: Path) -> None:
    scenarios = scenario_configs()
    training = scenarios["in_distribution"]
    model = PassiveValueDecomposition(training, "qmix_counterfactual", 16, 8)
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint, seed=21)
    loaded = PassiveValueDecomposition.load(checkpoint, training, torch.device("cpu"))
    for config in scenarios.values():
        row = evaluate_value_decomposition(config, loaded, 1001, 21, torch.device("cpu"))
        assert row["invalid_requests"] == 0
        assert row["objective"] >= 0


def test_fixed_references_are_feasible_in_all_reporting_scenarios() -> None:
    for config in scenario_configs().values():
        for policy in ("random_feasible", "skill_aware_fifo"):
            row = evaluate_fixed(config, policy, seed=1001)
            assert row["invalid_requests"] == 0


def test_confirmatory_smoke_writes_complete_artifact(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    result = run(
        argparse.Namespace(profile="smoke", device="cpu", output_dir=str(output))
    )
    assert result == output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 120
    assert manifest["checkpoint_count"] == 8
    assert manifest["sealed_test_evaluated"] is False
    assert manifest["hard_gate_passed"] is True
    assert summary["audits"]["all_expected_rows_present"] is True
    assert summary["audits"]["complete_factorial_coverage"] is True
    assert summary["audits"]["no_final_joint_action_collapse"] is None
    assert summary["primary_hypothesis"]["supported"] is False
    assert len(summary["paired_effects"]) == 24
    required = {
        "benchmark_config.json",
        "resolved_config.json",
        "training_progress.partial.csv",
        "training_progress.csv",
        "episodes.partial.csv",
        "episodes.csv",
        "coordination.csv",
        "seed_summary.csv",
        "scenario_summary.csv",
        "paired_effects.csv",
        "summary.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    with (output / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 120
    assert {row["scenario"] for row in rows} == set(scenario_configs())
