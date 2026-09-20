"""Compare fixed-logit PPO with shared per-action scoring on fresh validation seeds."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any

from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.multi_seed_experiment import _stats
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.rl_experiment import (
    ExperimentSettings,
    evaluate_ppo,
    resolve_device,
    train_model,
)
from ht_pdm_fjsp.shared_action_policy import SharedActionMaskablePolicy


CONDITIONS = ("fixed_logits", "shared_scorer", "shared_scorer_entropy")
METRICS = (
    "objective",
    "makespan",
    "total_tardiness",
    "failures",
    "preventive_maintenance",
    "corrective_maintenance",
    "total_cost",
)


def architecture_defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "total_timesteps": 256,
            "n_envs": 1,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 2,
            "train_seeds": (10_000, 11_000),
            "validation_seeds": tuple(range(41_000, 41_005)),
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "n_envs": 4,
            "n_steps": 1024,
            "batch_size": 256,
            "n_epochs": 10,
            "train_seeds": (10_000, 11_000, 12_000, 13_000, 14_000),
            "validation_seeds": tuple(range(41_000, 41_200)),
        }
    raise ValueError(f"Unknown profile: {profile}")


def summarize_architectures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_seeds = sorted({int(row["train_seed"]) for row in rows})
    validation_seeds = sorted({int(row["seed"]) for row in rows})
    per_condition: dict[str, Any] = {}
    seed_means: dict[str, dict[int, dict[str, float]]] = {}
    for condition in CONDITIONS:
        selected_condition = [row for row in rows if row["condition"] == condition]
        condition_seed_means: dict[int, dict[str, float]] = {}
        for train_seed in train_seeds:
            selected = [
                row
                for row in selected_condition
                if int(row["train_seed"]) == train_seed
            ]
            if len(selected) != len(validation_seeds):
                raise ValueError(
                    f"Incomplete validation panel for {condition}/{train_seed}."
                )
            condition_seed_means[train_seed] = {
                metric: fmean(float(row[metric]) for row in selected)
                for metric in METRICS
            }
        seed_means[condition] = condition_seed_means
        per_condition[condition] = {
            "per_training_seed": {
                str(seed): values for seed, values in condition_seed_means.items()
            },
            "across_training_seeds": {
                metric: _stats(
                    condition_seed_means[seed][metric] for seed in train_seeds
                )
                for metric in METRICS
            },
        }

    comparisons: dict[str, Any] = {}
    baseline_lookup = {
        (int(row["train_seed"]), int(row["seed"])): row
        for row in rows
        if row["condition"] == "fixed_logits"
    }
    for condition in CONDITIONS[1:]:
        replicate_deltas = {
            metric: [
                seed_means[condition][seed][metric]
                - seed_means["fixed_logits"][seed][metric]
                for seed in train_seeds
            ]
            for metric in METRICS
        }
        selected = [row for row in rows if row["condition"] == condition]
        objective_deltas = [
            float(row["objective"])
            - float(
                baseline_lookup[(int(row["train_seed"]), int(row["seed"]))][
                    "objective"
                ]
            )
            for row in selected
        ]
        comparisons[f"{condition}_minus_fixed_logits"] = {
            "delta_across_training_seeds": {
                metric: _stats(values) for metric, values in replicate_deltas.items()
            },
            "pooled_validation_objective_delta": _stats(objective_deltas),
            "pooled_validation_win_rate": fmean(value < 0 for value in objective_deltas),
        }
    return {
        "primary_endpoint": "mean validation objective by training seed",
        "replication_unit": "independent PPO training seed",
        "validation_seed_count": len(validation_seeds),
        "training_seed_count": len(train_seeds),
        "per_condition": per_condition,
        "comparisons": comparisons,
        "future_test_panel_opened": False,
    }


def run_architecture_experiment(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_run).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((source_dir / "run_manifest.json").read_text())
    if source_manifest.get("status") != "COMPLETED":
        raise ValueError("Source fixed-logit PPO run is not marked COMPLETED.")
    defaults = architecture_defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds)
        if args.train_seeds
        else defaults["train_seeds"]
    )
    available_source_seeds = set(int(seed) for seed in source_manifest["train_seeds"])
    if not train_seeds or not set(train_seeds).issubset(available_source_seeds):
        raise ValueError("Requested seeds are not all available in source baseline run.")
    validation_seeds = tuple(
        parse_seeds(args.validation_seeds)
        if args.validation_seeds
        else defaults["validation_seeds"]
    )
    forbidden = (
        set(source_manifest["common_settings"]["validation_seeds"])
        | set(source_manifest["common_settings"]["test_seeds"])
        | set(range(40_000, 40_400))
        | set(range(50_000, 50_100))
    )
    if not validation_seeds or set(validation_seeds) & forbidden:
        raise ValueError("Architecture validation seeds overlap an existing panel.")
    resolved_device = resolve_device(args.device)
    config = BenchmarkConfig.from_json(args.config)
    common = {
        "profile": args.profile,
        "total_timesteps": args.total_timesteps or defaults["total_timesteps"],
        "n_envs": args.n_envs or defaults["n_envs"],
        "n_steps": args.n_steps or defaults["n_steps"],
        "batch_size": args.batch_size or defaults["batch_size"],
        "n_epochs": args.n_epochs or defaults["n_epochs"],
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "validation_seeds": validation_seeds,
        "test_seeds": (),
        "device": resolved_device,
        "include_slow_baselines": False,
    }
    if common["n_steps"] * common["n_envs"] % common["batch_size"]:
        raise ValueError("n_steps * n_envs must be divisible by batch_size.")

    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "source_run": str(source_dir),
        "conditions": CONDITIONS,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "common_settings": common,
        "shared_entropy_coefficient": args.shared_entropy_coefficient,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "completed_training_cells": [],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "stable_baselines3": version("stable-baselines3"),
            "sb3_contrib": version("sb3-contrib"),
            "torch": version("torch"),
        },
    }
    manifest_path = output_dir / "architecture_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    training_seconds: dict[str, float] = {}

    baseline_progress = tqdm(
        train_seeds,
        desc="Fixed-logit validation",
        unit="model",
        disable=args.no_progress,
    )
    for train_seed in baseline_progress:
        baseline_progress.set_postfix(train_seed=str(train_seed))
        model_path = source_dir / f"train_seed_{train_seed}" / "maskable_ppo.zip"
        model = MaskablePPO.load(model_path, device=resolved_device)
        evaluated = evaluate_ppo(
            model,
            config,
            validation_seeds,
            split="architecture_validation",
            show_progress=False,
        )
        for row in evaluated:
            row["policy"] = "fixed_logits"
            row["condition"] = "fixed_logits"
            row["train_seed"] = train_seed
        rows.extend(evaluated)
        del model
    baseline_progress.close()
    _write_csv(rows, output_dir / "architecture_episodes.partial.csv")

    cells = [(condition, seed) for condition in CONDITIONS[1:] for seed in train_seeds]
    progress = tqdm(
        cells,
        desc="Shared-action PPO training",
        unit="model",
        disable=args.no_progress,
    )
    for condition, train_seed in progress:
        progress.set_postfix(condition=condition, train_seed=str(train_seed))
        cell_dir = output_dir / condition / f"train_seed_{train_seed}"
        settings = ExperimentSettings(train_seed=train_seed, **common)
        ent_coef = (
            args.shared_entropy_coefficient
            if condition == "shared_scorer_entropy"
            else 0.0
        )
        model, elapsed = train_model(
            config,
            settings,
            cell_dir,
            show_progress=not args.no_progress,
            policy=SharedActionMaskablePolicy,
            policy_kwargs={
                "global_hidden_dim": args.global_hidden_dim,
                "action_hidden_dim": args.action_hidden_dim,
                "scorer_hidden_dim": args.scorer_hidden_dim,
            },
            ent_coef=ent_coef,
        )
        training_seconds[f"{condition}/{train_seed}"] = elapsed
        evaluated = evaluate_ppo(
            model,
            config,
            validation_seeds,
            split="architecture_validation",
            show_progress=False,
        )
        for row in evaluated:
            row["policy"] = condition
            row["condition"] = condition
            row["train_seed"] = train_seed
        rows.extend(evaluated)
        del model
        _write_csv(rows, output_dir / "architecture_episodes.partial.csv")
        manifest["completed_training_cells"].append(
            {"condition": condition, "train_seed": train_seed}
        )
        manifest["training_seconds"] = training_seconds
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    progress.close()

    expected = len(CONDITIONS) * len(train_seeds) * len(validation_seeds)
    unique = {
        (row["condition"], row["train_seed"], row["seed"]) for row in rows
    }
    if len(rows) != expected or len(unique) != expected:
        raise ValueError("Architecture validation panel is incomplete.")
    summary = summarize_architectures(rows)
    _write_csv(rows, output_dir / "architecture_episodes.csv")
    (output_dir / "architecture_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["status"] = "COMPLETED"
    manifest["episode_count"] = len(rows)
    manifest["training_seconds"] = training_seconds
    manifest["output_files"] = sorted(path.name for path in output_dir.iterdir())
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/minimal_benchmark.json")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--train-seeds")
    parser.add_argument("--validation-seeds")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--shared-entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--global-hidden-dim", type=int, default=128)
    parser.add_argument("--action-hidden-dim", type=int, default=64)
    parser.add_argument("--scorer-hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run_architecture_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
