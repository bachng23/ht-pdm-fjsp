"""Long, budget-matched comparison of passive-technician MARL families."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_baselines import evaluate_fixed, stress_config
from ht_pdm_fjsp.passive_technician_budget_screen import _evaluate_q, _train_q_checkpoints
from ht_pdm_fjsp.passive_technician_marl import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    IndependentQ,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition,
    ValueTrainSettings,
    evaluate_value_decomposition,
    train_value_decomposition_checkpoints,
)


ALGORITHMS = ("independent_q", "vdn", "qmix", "tqmix")
VALUE_ALGORITHMS = ("independent_q", "vdn", "qmix", "tqmix")
FIXED_POLICIES = ("random_feasible", "skill_aware_fifo")
FULL_BUDGETS = (5_000, 10_000, 20_000, 50_000)
SMOKE_BUDGETS = (8, 16, 32)
FIELDNAMES = (
    "cell",
    "policy",
    "budget",
    "seed",
    "train_seed",
    "objective",
    "failures",
    "jobs",
    "collisions",
    "waiting",
    "invalid_requests",
    "busy_requests",
    "unique_joint_actions",
    "defer_fraction",
    "action_steps",
    "request_count",
)


def environment_cells() -> dict[str, Any]:
    base = stress_config()
    return {
        "in_distribution": base,
        "early_failure": type(base)(
            **{**asdict(base), "horizon": 18, "failure_age": 4, "failure_probability": 0.60}
        ),
        "slow_service": type(base)(
            **{**asdict(base), "horizon": 18, "service_time": ((3, 5), (5, 3), (4, 4))}
        ),
        "combined_pressure": type(base)(
            **{
                **asdict(base),
                "horizon": 18,
                "failure_age": 4,
                "failure_probability": 0.60,
                "service_time": ((3, 5), (5, 3), (4, 4)),
            }
        ),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: tuple[str, ...] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    if fieldnames is not None:
        names = list(fieldnames)
    else:
        names = []
        for row in rows:
            for name in row:
                if name not in names:
                    names.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _profile(profile: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if profile == "smoke":
        return (11,), (101, 102, 103), SMOKE_BUDGETS
    if profile == "full":
        return tuple(range(11, 16)), tuple(range(101, 201)), FULL_BUDGETS
    raise ValueError(profile)


def _row(row: dict[str, Any], cell: str, policy: str, budget: int, train_seed: int | str) -> dict[str, Any]:
    return {
        "cell": cell,
        "policy": policy,
        "budget": budget,
        "seed": row.get("seed", ""),
        "train_seed": train_seed,
        "objective": row.get("objective", 0.0),
        "failures": row.get("failures", 0),
        "jobs": row.get("jobs", 0),
        "collisions": row.get("collisions", 0),
        "waiting": row.get("waiting", 0),
        "invalid_requests": row.get("invalid_requests", 0),
        "busy_requests": row.get("busy_requests", 0),
        "unique_joint_actions": row.get("unique_joint_actions", 0),
        "defer_fraction": row.get("defer_fraction", 0.0),
        "action_steps": row.get("action_steps", 0),
        "request_count": row.get("request_count", 0),
    }


def _summary(
    rows: list[dict[str, Any]],
    train_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["cell"]), str(row["policy"]), int(row["budget"]), str(row["train_seed"])), []
        ).append(row)
    budget_summary: list[dict[str, Any]] = []
    for (cell, policy, budget, train_seed), group in sorted(grouped.items()):
        budget_summary.append(
            {
                "cell": cell,
                "policy": policy,
                "budget": budget,
                "train_seed": train_seed,
                "evaluation_count": len(group),
                "objective_mean": statistics.fmean(float(row["objective"]) for row in group),
                "failures_mean": statistics.fmean(float(row["failures"]) for row in group),
                "jobs_mean": statistics.fmean(float(row["jobs"]) for row in group),
                "collisions_mean": statistics.fmean(float(row["collisions"]) for row in group),
                "waiting_mean": statistics.fmean(float(row["waiting"]) for row in group),
                "invalid_requests_mean": statistics.fmean(float(row["invalid_requests"]) for row in group),
                "busy_requests_mean": statistics.fmean(float(row["busy_requests"]) for row in group),
                "unique_joint_actions_mean": statistics.fmean(float(row["unique_joint_actions"]) for row in group),
                "defer_fraction_mean": statistics.fmean(float(row["defer_fraction"]) for row in group),
            }
        )
    final_budget = max(budgets)
    final_by_cell_policy: dict[str, dict[str, Any]] = {}
    for cell in environment_cells():
        for policy in (*ALGORITHMS, *FIXED_POLICIES):
            budget = 0 if policy in FIXED_POLICIES else final_budget
            selected = [
                row for row in budget_summary
                if row["cell"] == cell and row["policy"] == policy and int(row["budget"]) == budget
            ]
            if selected:
                means = [float(row["objective_mean"]) for row in selected]
                final_by_cell_policy[f"{cell}/{policy}"] = {
                    "cell": cell,
                    "policy": policy,
                    "training_seed_count": len(means),
                    "objective_mean": statistics.fmean(means),
                    "objective_std_across_train_seeds": statistics.stdev(means) if len(means) > 1 else 0.0,
                    "training_seed_means": means,
                }
    expected = (
        len(environment_cells()) * len(FIXED_POLICIES) * len(evaluation_seeds)
        + len(environment_cells()) * len(VALUE_ALGORITHMS) * len(train_seeds) * len(budgets) * len(evaluation_seeds)
    )
    return {
        "purpose": "Stage 1 learning-curve comparison across four environment cells.",
        "primary_metric": "mean objective cost on the common evaluation panel",
        "budget_summary": budget_summary,
        "final_budget": final_budget,
        "final_by_cell_policy": final_by_cell_policy,
        "audits": {
            "expected_episode_count": expected,
            "episode_count": len(rows),
            "all_expected_rows_present": len(rows) == expected,
            "invalid_requests_zero": all(float(row["invalid_requests"]) == 0 for row in rows),
            "nonnegative_objectives": all(float(row["objective"]) >= 0 for row in rows),
            "sealed_test_panel_closed": True,
            "all_learned_train_seeds_recorded": all(
                str(row["train_seed"]) for row in rows if row["policy"] in ALGORITHMS
            ),
        },
    }


def _evaluate(
    policy_name: str,
    config,
    policy,
    seed: int,
    train_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if policy_name == "independent_q":
        return _evaluate_q(config, policy, seed, train_seed)
    if policy_name == "vdn" or policy_name == "qmix" or policy_name == "tqmix":
        return evaluate_value_decomposition(config, policy, seed, train_seed, device)
    raise ValueError(f"unknown policy {policy_name}")


def run(args: argparse.Namespace) -> Path:
    train_seeds, evaluation_seeds, budgets = _profile(args.profile)
    cells = environment_cells()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    max_budget = max(budgets)
    value_settings = ValueTrainSettings(episodes=max_budget)
    manifest = {
        "status": "RUNNING",
        "experiment": "passive_technician_long_comparison",
        "profile": args.profile,
        "algorithms": list(ALGORITHMS),
        "fixed_policies": list(FIXED_POLICIES),
        "environment_cells": {name: asdict(config) for name, config in cells.items()},
        "checkpoint_budgets": list(budgets),
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "sealed_test_seeds": list(range(201, 301)),
        "hypothesis": "RA-QMIX will achieve lower mean objective cost than independent Q-learning, VDN, and standard QMIX as training budget increases, consistently across the four locked environment cells.",
        "value_settings": asdict(value_settings),
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "requested_device": args.device,
        "resolved_device": str(device),
        "git_revision": _git_revision(),
        "sealed_test_evaluated": False,
        "started_at": datetime.now(UTC).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "benchmark_config.json").write_text(
        json.dumps(
            {"environment_cells": {name: asdict(config) for name, config in cells.items()}, "train_seeds": train_seeds, "evaluation_seeds": evaluation_seeds, "sealed_test_seeds": tuple(range(201, 301)), "budgets": budgets, "value_settings": asdict(value_settings)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    rows: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    for cell_name, config in cells.items():
        for seed in tqdm(evaluation_seeds, desc=f"fixed policies/{cell_name}", unit="episode"):
            for policy in FIXED_POLICIES:
                rows.append(_row(evaluate_fixed(config, policy, seed), cell_name, policy, 0, ""))
    _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)

    for cell_name, config in cells.items():
        for algorithm in VALUE_ALGORITHMS:
            for train_seed in tqdm(train_seeds, desc=f"{cell_name}/{algorithm}", unit="seed"):
                root = output / cell_name / algorithm / f"train_seed_{train_seed}"
                if algorithm == "independent_q":
                    checkpoints, training_rows = _train_q_checkpoints(config, train_seed, budgets, root)
                else:
                    checkpoints, training_rows = train_value_decomposition_checkpoints(
                        config, algorithm, train_seed, budgets, root, value_settings, device
                    )
                progress.extend({**row, "cell": cell_name, "policy": algorithm} for row in training_rows)
                for budget in budgets:
                    checkpoint = checkpoints[budget]
                    if algorithm == "independent_q":
                        policy = IndependentQ.load(checkpoint, config, seed=train_seed)
                    else:
                        policy = PassiveValueDecomposition.load(checkpoint, config, device)
                    for seed in tqdm(
                        evaluation_seeds,
                        desc=f"evaluate {cell_name}/{algorithm}/{train_seed}/{budget}",
                        unit="episode",
                        leave=False,
                    ):
                        rows.append(_row(_evaluate(algorithm, config, policy, seed, train_seed, device), cell_name, algorithm, budget, train_seed))
                    _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)

    _write_csv(rows, output / "episodes.csv", FIELDNAMES)
    _write_csv(rows, output / "coordination.csv", FIELDNAMES)
    _write_csv(progress, output / "training_progress.csv")
    summary = _summary(rows, train_seeds, evaluation_seeds, budgets)
    _write_csv(summary["budget_summary"], output / "budget_summary.csv")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest.update(
        {
            "status": "COMPLETED",
            "finished_at": datetime.now(UTC).isoformat(),
            "episode_count": len(rows),
            "checkpoint_count": sum(1 for path in output.rglob("model.*") if path.is_file()),
            "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()),
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
