"""Compare shared PPO scorers with and without entity-linked action context."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
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
from ht_pdm_fjsp.shared_policy_experiment import METRICS, summarize_architectures


CONDITIONS = (
    "fixed_logits",
    "shared_scorer",
    "shared_scorer_entropy",
    "entity_scorer",
    "entity_scorer_entropy",
)
SOURCE_CONDITIONS = CONDITIONS[:3]
ENTITY_CONDITIONS = CONDITIONS[3:]


def entity_defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "total_timesteps": 256,
            "n_envs": 1,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 2,
            "train_seeds": (10_000, 11_000),
            "validation_seeds": tuple(range(42_000, 42_005)),
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "n_envs": 4,
            "n_steps": 1024,
            "batch_size": 256,
            "n_epochs": 10,
            "train_seeds": (10_000, 11_000, 12_000, 13_000, 14_000),
            "validation_seeds": tuple(range(42_000, 42_200)),
        }
    raise ValueError(f"Unknown profile: {profile}")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _paired_summary(
    rows: list[dict[str, Any]], treatment: str, control: str
) -> dict[str, Any]:
    train_seeds = sorted({int(row["train_seed"]) for row in rows})
    validation_seeds = sorted({int(row["seed"]) for row in rows})
    lookup = {
        (str(row["condition"]), int(row["train_seed"]), int(row["seed"])): row
        for row in rows
    }
    seed_deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
    pooled_objective: list[float] = []
    for train_seed in train_seeds:
        for metric in METRICS:
            values = [
                float(lookup[(treatment, train_seed, seed)][metric])
                - float(lookup[(control, train_seed, seed)][metric])
                for seed in validation_seeds
            ]
            seed_deltas[metric].append(fmean(values))
            if metric == "objective":
                pooled_objective.extend(values)
    return {
        "treatment": treatment,
        "control": control,
        "replication_unit": "independent PPO training seed",
        "delta_across_training_seeds": {
            metric: _stats(values) for metric, values in seed_deltas.items()
        },
        "per_training_seed_objective_delta": {
            str(seed): value
            for seed, value in zip(
                train_seeds, seed_deltas["objective"], strict=True
            )
        },
        "pooled_validation_objective_delta": _stats(pooled_objective),
        "pooled_validation_win_rate": fmean(value < 0 for value in pooled_objective),
    }


def _load_manifest(path: Path, filename: str) -> dict[str, Any]:
    manifest_path = path / filename
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"Source run is not marked COMPLETED: {path}")
    return manifest


def run_entity_experiment(args: argparse.Namespace) -> Path:
    fixed_dir = Path(args.fixed_source_run).expanduser().resolve()
    shared_dir = Path(args.shared_source_run).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_manifest = _load_manifest(fixed_dir, "run_manifest.json")
    shared_manifest = _load_manifest(shared_dir, "architecture_manifest.json")
    defaults = entity_defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds)
        if args.train_seeds
        else defaults["train_seeds"]
    )
    fixed_seeds = {int(seed) for seed in fixed_manifest["train_seeds"]}
    shared_seeds = {int(seed) for seed in shared_manifest["train_seeds"]}
    if not train_seeds or not set(train_seeds).issubset(fixed_seeds & shared_seeds):
        raise ValueError("Requested seeds are not available in both source runs.")
    validation_seeds = tuple(
        parse_seeds(args.validation_seeds)
        if args.validation_seeds
        else defaults["validation_seeds"]
    )
    forbidden = (
        set(fixed_manifest["common_settings"]["validation_seeds"])
        | set(fixed_manifest["common_settings"]["test_seeds"])
        | set(shared_manifest["validation_seeds"])
        | set(range(40_000, 42_000))
        | set(range(50_000, 50_100))
    )
    if not validation_seeds or set(validation_seeds) & forbidden:
        raise ValueError("Entity validation seeds overlap an existing panel.")

    config_path = Path(args.config).expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = BenchmarkConfig.from_json(config_path)
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)
    resolved_device = resolve_device(args.device)
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
        "git_commit": _git_commit(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "fixed_source_run": str(fixed_dir),
        "shared_source_run": str(shared_dir),
        "conditions": CONDITIONS,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "common_settings": common,
        "entity_entropy_coefficient": args.entity_entropy_coefficient,
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
    manifest_path = output_dir / "entity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    training_seconds: dict[str, float] = {}

    source_cells = [
        (condition, seed) for condition in SOURCE_CONDITIONS for seed in train_seeds
    ]
    source_progress = tqdm(
        source_cells,
        desc="Source policy validation",
        unit="model",
        disable=args.no_progress,
    )
    for condition, train_seed in source_progress:
        source_progress.set_postfix(condition=condition, train_seed=str(train_seed))
        if condition == "fixed_logits":
            model_path = fixed_dir / f"train_seed_{train_seed}" / "maskable_ppo.zip"
        else:
            model_path = (
                shared_dir / condition / f"train_seed_{train_seed}" / "maskable_ppo.zip"
            )
        model = MaskablePPO.load(model_path, device=resolved_device)
        evaluated = evaluate_ppo(
            model,
            config,
            validation_seeds,
            split="entity_validation",
            show_progress=False,
        )
        for row in evaluated:
            row["policy"] = condition
            row["condition"] = condition
            row["train_seed"] = train_seed
        rows.extend(evaluated)
        del model
        _write_csv(rows, output_dir / "entity_episodes.partial.csv")
    source_progress.close()

    entity_cells = [
        (condition, seed) for condition in ENTITY_CONDITIONS for seed in train_seeds
    ]
    progress = tqdm(
        entity_cells,
        desc="Entity-conditioned PPO training",
        unit="model",
        disable=args.no_progress,
    )
    for condition, train_seed in progress:
        progress.set_postfix(condition=condition, train_seed=str(train_seed))
        cell_dir = output_dir / condition / f"train_seed_{train_seed}"
        settings = ExperimentSettings(train_seed=train_seed, **common)
        ent_coef = (
            args.entity_entropy_coefficient
            if condition == "entity_scorer_entropy"
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
                "extra_action_feature_keys": ("action_context",),
            },
            ent_coef=ent_coef,
            env_kwargs={"include_action_context": True},
        )
        training_seconds[f"{condition}/{train_seed}"] = elapsed
        evaluated = evaluate_ppo(
            model,
            config,
            validation_seeds,
            split="entity_validation",
            show_progress=False,
            env_kwargs={"include_action_context": True},
        )
        for row in evaluated:
            row["policy"] = condition
            row["condition"] = condition
            row["train_seed"] = train_seed
        rows.extend(evaluated)
        del model
        _write_csv(rows, output_dir / "entity_episodes.partial.csv")
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
    unique = {(row["condition"], row["train_seed"], row["seed"]) for row in rows}
    if len(rows) != expected or len(unique) != expected:
        raise ValueError("Entity-conditioned validation panel is incomplete.")
    summary = summarize_architectures(rows, conditions=CONDITIONS)
    summary["primary_endpoint"] = (
        "mean validation objective by training seed for entity-vs-shared ablations"
    )
    summary["ablations"] = {
        "entity_scorer_minus_shared_scorer": _paired_summary(
            rows, "entity_scorer", "shared_scorer"
        ),
        "entity_scorer_entropy_minus_shared_scorer_entropy": _paired_summary(
            rows, "entity_scorer_entropy", "shared_scorer_entropy"
        ),
    }
    _write_csv(rows, output_dir / "entity_episodes.csv")
    (output_dir / "entity_summary.json").write_text(
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
    parser.add_argument("--fixed-source-run", required=True)
    parser.add_argument("--shared-source-run", required=True)
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
    parser.add_argument("--entity-entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--global-hidden-dim", type=int, default=128)
    parser.add_argument("--action-hidden-dim", type=int, default=64)
    parser.add_argument("--scorer-hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run_entity_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
