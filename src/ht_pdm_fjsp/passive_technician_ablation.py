"""Factorial ablation of the technician-aware QMIX components."""

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
from ht_pdm_fjsp.passive_technician_budget_screen import (
    _evaluate_q,
    _train_q_checkpoints,
)
from ht_pdm_fjsp.passive_technician_marl import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    IndependentQ,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    VALUE_ALGORITHMS,
    PassiveValueDecomposition,
    ValueTrainSettings,
    evaluate_value_decomposition,
    train_value_decomposition_checkpoints,
)


ABLATION_ALGORITHMS = (
    "qmix",
    "qmix_edge",
    "qmix_queue",
    "qmix_counterfactual",
    "tqmix",
)
FIXED_POLICIES = ("random_feasible", "skill_aware_fifo")
BUDGETS = (20_000, 50_000)
SMOKE_BUDGETS = (8, 16)
FIELDNAMES = (
    "policy", "budget", "seed", "train_seed", "objective", "failures", "jobs",
    "collisions", "waiting", "invalid_requests", "busy_requests",
    "unique_joint_actions", "defer_fraction", "action_steps", "request_count",
)


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: tuple[str, ...] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    names = list(fieldnames) if fieldnames else list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _profile(profile: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if profile == "smoke":
        return (11,), (101, 102, 103), SMOKE_BUDGETS
    if profile == "full":
        return (11, 12, 13), tuple(range(101, 201)), BUDGETS
    raise ValueError(profile)


def _row(row: dict[str, Any], policy: str, budget: int, train_seed: int | str) -> dict[str, Any]:
    return {
        field: row.get(field, "" if field in {"seed", "train_seed"} else 0)
        for field in FIELDNAMES
    } | {"policy": policy, "budget": budget, "train_seed": train_seed}


def _summary(rows: list[dict[str, Any]], train_seeds: tuple[int, ...], eval_seeds: tuple[int, ...], budgets: tuple[int, ...]) -> dict[str, Any]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["policy"]), int(row["budget"]), str(row["train_seed"])), []).append(row)
    budget_summary = []
    for (policy, budget, train_seed), group in sorted(groups.items()):
        budget_summary.append({
            "policy": policy, "budget": budget, "train_seed": train_seed,
            "evaluation_count": len(group),
            "objective_mean": statistics.fmean(float(r["objective"]) for r in group),
            "failures_mean": statistics.fmean(float(r["failures"]) for r in group),
            "jobs_mean": statistics.fmean(float(r["jobs"]) for r in group),
            "collisions_mean": statistics.fmean(float(r["collisions"]) for r in group),
            "waiting_mean": statistics.fmean(float(r["waiting"]) for r in group),
            "busy_requests_mean": statistics.fmean(float(r["busy_requests"]) for r in group),
            "unique_joint_actions_mean": statistics.fmean(float(r["unique_joint_actions"]) for r in group),
            "defer_fraction_mean": statistics.fmean(float(r["defer_fraction"]) for r in group),
        })
    final_budget = max(budgets)
    final_by_policy = {}
    for policy in ABLATION_ALGORITHMS:
        selected = [r for r in budget_summary if r["policy"] == policy and int(r["budget"]) == final_budget]
        means = [float(r["objective_mean"]) for r in selected]
        if means:
            final_by_policy[policy] = {
                "training_seed_means": means,
                "objective_mean": statistics.fmean(means),
                "objective_std_across_train_seeds": statistics.stdev(means) if len(means) > 1 else 0.0,
            }
    for policy in FIXED_POLICIES:
        selected = [r for r in budget_summary if r["policy"] == policy]
        if selected:
            final_by_policy[policy] = {"objective_mean": float(selected[0]["objective_mean"]), "training_seed_means": []}
    expected = len(FIXED_POLICIES) * len(eval_seeds) + len(ABLATION_ALGORITHMS) * len(train_seeds) * len(budgets) * len(eval_seeds)
    return {
        "purpose": "Component ablation for technician-aware QMIX.",
        "component_mapping": {
            "qmix": [],
            "qmix_edge": ["technician_edge_q_head"],
            "qmix_queue": ["queue_conditioned_mixer"],
            "qmix_counterfactual": ["counterfactual_consistency_loss"],
            "tqmix": ["technician_edge_q_head", "queue_conditioned_mixer", "counterfactual_consistency_loss"],
        },
        "budget_summary": budget_summary,
        "final_budget": final_budget,
        "final_by_policy": final_by_policy,
        "audits": {
            "expected_episode_count": expected,
            "episode_count": len(rows),
            "all_expected_rows_present": len(rows) == expected,
            "invalid_requests_zero": all(float(r["invalid_requests"]) == 0 for r in rows),
            "nonnegative_objectives": all(float(r["objective"]) >= 0 for r in rows),
            "sealed_test_panel_closed": True,
            "all_train_seeds_recorded": all(str(r["train_seed"]) for r in rows if r["policy"] in ABLATION_ALGORITHMS),
        },
    }


def run(args: argparse.Namespace) -> Path:
    train_seeds, eval_seeds, budgets = _profile(args.profile)
    config = stress_config()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = ValueTrainSettings(episodes=max(budgets))
    manifest = {
        "status": "RUNNING", "experiment": "passive_technician_ablation", "profile": args.profile,
        "algorithms": list(ABLATION_ALGORITHMS), "fixed_policies": list(FIXED_POLICIES),
        "budgets": list(budgets), "train_seeds": list(train_seeds), "evaluation_seeds": list(eval_seeds),
        "stress_config": asdict(config), "value_settings": asdict(settings),
        "objective_version": OBJECTIVE_VERSION, "observation_version": OBSERVATION_VERSION,
        "requested_device": args.device, "resolved_device": str(device), "git_revision": _git_revision(),
        "sealed_test_evaluated": False, "started_at": datetime.now(UTC).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "benchmark_config.json").write_text(json.dumps({"stress_config": asdict(config), "budgets": budgets, "train_seeds": train_seeds, "evaluation_seeds": eval_seeds, "value_settings": asdict(settings)}, indent=2, sort_keys=True) + "\n")
    rows: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    for seed in tqdm(eval_seeds, desc="fixed policies", unit="episode"):
        for policy in FIXED_POLICIES:
            rows.append(_row(evaluate_fixed(config, policy, seed), policy, 0, ""))
    _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)
    for algorithm in ABLATION_ALGORITHMS:
        for train_seed in tqdm(train_seeds, desc=algorithm, unit="seed"):
            root = output / algorithm / f"train_seed_{train_seed}"
            checkpoints, training_rows = train_value_decomposition_checkpoints(config, algorithm, train_seed, budgets, root, settings, device)
            progress.extend(training_rows)
            for budget in budgets:
                model = PassiveValueDecomposition.load(checkpoints[budget], config, device)
                for seed in tqdm(eval_seeds, desc=f"evaluate {algorithm}/{train_seed}/{budget}", unit="episode", leave=False):
                    rows.append(_row(evaluate_value_decomposition(config, model, seed, train_seed, device), algorithm, budget, train_seed))
                _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)
    _write_csv(rows, output / "episodes.csv", FIELDNAMES)
    _write_csv(rows, output / "coordination.csv", FIELDNAMES)
    _write_csv(progress, output / "training_progress.csv")
    summary = _summary(rows, train_seeds, eval_seeds, budgets)
    _write_csv(summary["budget_summary"], output / "budget_summary.csv")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest.update({"status": "COMPLETED", "finished_at": datetime.now(UTC).isoformat(), "episode_count": len(rows), "checkpoint_count": sum(1 for p in output.rglob("model.*") if p.is_file()), "outputs": sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())})
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
