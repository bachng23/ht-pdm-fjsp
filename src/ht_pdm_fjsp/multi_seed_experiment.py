"""Run independent MaskablePPO training replicates and shared-seed evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.rl_experiment import (
    ExperimentSettings,
    evaluate_baselines,
    evaluate_ppo,
    profile_defaults,
    resolve_device,
    train_model,
)


def replication_defaults(profile: str) -> dict[str, Any]:
    base = profile_defaults(profile)
    if profile == "smoke":
        return {
            **base,
            "total_timesteps": 256,
            "train_seeds": (10_000, 11_000),
            "validation_seeds": (20_000, 20_001),
            "test_seeds": (30_000, 30_001, 30_002),
        }
    return {
        **base,
        "train_seeds": (10_000, 11_000, 12_000, 13_000, 14_000),
    }


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: Iterable[float]) -> dict[str, float]:
    items = [float(value) for value in values]
    return {
        "mean": fmean(items),
        "std": pstdev(items),
        "min": min(items),
        "max": max(items),
    }


def aggregate_results(
    ppo_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = (
        "objective",
        "makespan",
        "total_tardiness",
        "failures",
        "preventive_maintenance",
        "corrective_maintenance",
        "total_cost",
    )
    test_ppo = [row for row in ppo_rows if row["split"] == "test"]
    train_seeds = sorted({int(row["train_seed"]) for row in test_ppo})
    per_training_seed: dict[str, Any] = {}
    for train_seed in train_seeds:
        selected = [
            row for row in test_ppo if int(row["train_seed"]) == train_seed
        ]
        per_training_seed[str(train_seed)] = {
            metric: fmean(float(row[metric]) for row in selected)
            for metric in metrics
        }
    ppo_across_training_seeds = {
        metric: _stats(per_training_seed[str(seed)][metric] for seed in train_seeds)
        for metric in metrics
    }
    baseline_summary: dict[str, Any] = {}
    for policy in sorted({str(row["policy"]) for row in baseline_rows}):
        selected = [row for row in baseline_rows if row["policy"] == policy]
        baseline_summary[policy] = {
            metric: _stats(float(row[metric]) for row in selected)
            for metric in metrics
        }
    baseline_lookup = {
        (str(row["policy"]), int(row["seed"])): row for row in baseline_rows
    }
    paired: dict[str, Any] = {}
    for policy in sorted({str(row["policy"]) for row in baseline_rows}):
        seed_level_deltas: dict[str, list[float]] = {metric: [] for metric in metrics}
        replicate_objective_means: list[float] = []
        for train_seed in train_seeds:
            selected = [
                row for row in test_ppo if int(row["train_seed"]) == train_seed
            ]
            objective_deltas: list[float] = []
            for row in selected:
                baseline = baseline_lookup[(policy, int(row["seed"]))]
                for metric in metrics:
                    delta = float(row[metric]) - float(baseline[metric])
                    seed_level_deltas[metric].append(delta)
                    if metric == "objective":
                        objective_deltas.append(delta)
            replicate_objective_means.append(fmean(objective_deltas))
        paired[policy] = {
            "delta_ppo_minus_baseline": {
                metric: _stats(values) for metric, values in seed_level_deltas.items()
            },
            "objective_delta_across_training_seeds": _stats(
                replicate_objective_means
            ),
            "pooled_objective_win_rate": fmean(
                value < 0.0 for value in seed_level_deltas["objective"]
            ),
        }
    return {
        "replication_unit": "independent PPO training seed",
        "evaluation_unit": "held-out stochastic environment seed",
        "training_seed_count": len(train_seeds),
        "test_seed_count": len({int(row["seed"]) for row in test_ppo}),
        "ppo_per_training_seed": per_training_seed,
        "ppo_across_training_seeds": ppo_across_training_seeds,
        "baselines": baseline_summary,
        "paired_comparisons": paired,
    }


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.rsplit("_", maxsplit=2)[1])


def run_replicates(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = replication_defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds)
        if args.train_seeds
        else defaults["train_seeds"]
    )
    validation_seeds = tuple(
        parse_seeds(args.validation_seeds)
        if args.validation_seeds
        else defaults["validation_seeds"]
    )
    test_seeds = tuple(
        parse_seeds(args.test_seeds) if args.test_seeds else defaults["test_seeds"]
    )
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
        "test_seeds": test_seeds,
        "device": resolved_device,
        "include_slow_baselines": (
            defaults["include_slow_baselines"] and not args.skip_slow_baselines
        ),
    }
    if common["n_steps"] * common["n_envs"] % common["batch_size"]:
        raise ValueError("n_steps * n_envs must be divisible by batch_size.")
    config = BenchmarkConfig.from_json(args.config)
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "train_seeds": train_seeds,
        "completed_train_seeds": [],
        "common_settings": common,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "gymnasium": version("gymnasium"),
            "stable_baselines3": version("stable-baselines3"),
            "sb3_contrib": version("sb3-contrib"),
            "torch": version("torch"),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ppo_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    training_seconds: dict[str, float] = {}
    progress = tqdm(
        train_seeds,
        desc="Independent PPO replicates",
        unit="model",
        disable=args.no_progress,
    )
    for train_seed in progress:
        progress.set_postfix(train_seed=train_seed)
        seed_dir = output_dir / f"train_seed_{train_seed}"
        settings = ExperimentSettings(train_seed=train_seed, **common)
        model, elapsed = train_model(
            config, settings, seed_dir, show_progress=not args.no_progress
        )
        training_seconds[str(train_seed)] = elapsed
        for split, seeds in (("validation", validation_seeds), ("test", test_seeds)):
            rows = evaluate_ppo(
                model,
                config,
                seeds,
                split=split,
                show_progress=not args.no_progress,
            )
            for row in rows:
                row["train_seed"] = train_seed
            ppo_rows.extend(rows)
        del model
        if not args.skip_checkpoint_evaluation:
            checkpoint_paths = sorted(
                (seed_dir / "checkpoints").glob("maskable_ppo_*_steps.zip"),
                key=_checkpoint_step,
            )
            for checkpoint_path in tqdm(
                checkpoint_paths,
                desc=f"Checkpoints {train_seed}",
                unit="checkpoint",
                disable=args.no_progress,
            ):
                checkpoint_model = MaskablePPO.load(
                    checkpoint_path, device=resolved_device
                )
                rows = evaluate_ppo(
                    checkpoint_model,
                    config,
                    validation_seeds,
                    split="checkpoint_validation",
                    show_progress=False,
                )
                for row in rows:
                    row["train_seed"] = train_seed
                    row["checkpoint_steps"] = _checkpoint_step(checkpoint_path)
                checkpoint_rows.extend(rows)
                del checkpoint_model
        _write_table(ppo_rows, output_dir / "ppo_episodes.partial.csv")
        _write_table(
            checkpoint_rows, output_dir / "checkpoint_validation.partial.csv"
        )
        manifest["completed_train_seeds"].append(train_seed)
        manifest["training_seconds"] = training_seconds
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    baseline_rows = evaluate_baselines(
        config,
        test_seeds,
        include_slow=bool(common["include_slow_baselines"]),
        show_progress=not args.no_progress,
    )
    _write_table(ppo_rows, output_dir / "ppo_episodes.csv")
    _write_table(checkpoint_rows, output_dir / "checkpoint_validation.csv")
    _write_table(baseline_rows, output_dir / "baseline_episodes.csv")
    summary = aggregate_results(ppo_rows, baseline_rows)
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["status"] = "COMPLETED"
    manifest["training_seconds"] = training_seconds
    manifest["output_files"] = sorted(path.name for path in output_dir.iterdir())
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/minimal_benchmark.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--train-seeds")
    parser.add_argument("--validation-seeds")
    parser.add_argument("--test-seeds")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-slow-baselines", action="store_true")
    parser.add_argument("--skip-checkpoint-evaluation", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run_replicates(build_parser().parse_args())


if __name__ == "__main__":
    main()
