"""Run a paired counterfactual diagnostic for the J1-O1 machine route."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.multi_seed_experiment import _stats
from ht_pdm_fjsp.ppo_diagnostics import _write_csv, trace_policy
from ht_pdm_fjsp.rl_experiment import resolve_device


METRICS = (
    "objective",
    "makespan",
    "total_tardiness",
    "failures",
    "preventive_maintenance",
    "corrective_maintenance",
    "total_cost",
    "decision_count",
)


def _route_indices(env: HTPdmFjspEnv) -> tuple[int, int]:
    matches: dict[str, int] = {}
    for index, descriptor in enumerate(env.actions):
        if (
            descriptor.kind == "production"
            and descriptor.job_id == "J1"
            and descriptor.operation_id == "O1"
            and descriptor.operation_index == 0
        ):
            matches[str(descriptor.machine_id)] = index
    if set(matches) != {"M1", "M2"}:
        raise ValueError("The benchmark does not have both J1-O1 machine routes.")
    return matches["M1"], matches["M2"]


def guard_j1_o1_m2(
    env: HTPdmFjspEnv, mask: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Suppress J1-O1 on M2, forcing the policy to wait for its M1 route."""

    _, m2_index = _route_indices(env)
    transformed = mask.copy()
    applied = bool(mask[m2_index])
    if applied:
        transformed[m2_index] = 0
    return transformed, applied


def paired_summary(
    episode_rows: list[dict[str, Any]], train_seeds: Iterable[int]
) -> dict[str, Any]:
    """Summarize paired guard-minus-original effects without pseudoreplication."""

    lookup = {
        (str(row["condition"]), int(row["train_seed"]), int(row["seed"])): row
        for row in episode_rows
    }
    if len(lookup) != len(episode_rows):
        raise ValueError("Counterfactual episode table contains duplicate keys.")
    condition_summary: dict[str, Any] = {}
    per_training_seed: dict[str, Any] = {}
    for condition in ("original", "route_guard_j1_o1_m2"):
        rows = [row for row in episode_rows if row["condition"] == condition]
        condition_summary[condition] = {
            metric: _stats(float(row[metric]) for row in rows) for metric in METRICS
        }
    replicate_deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
    pooled_deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for train_seed in sorted(set(int(seed) for seed in train_seeds)):
        keys = sorted(
            seed
            for condition, candidate_seed, seed in lookup
            if condition == "original" and candidate_seed == train_seed
        )
        seed_deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
        activations: list[float] = []
        for seed in keys:
            original = lookup[("original", train_seed, seed)]
            guarded = lookup[("route_guard_j1_o1_m2", train_seed, seed)]
            activations.append(float(guarded["intervention_activations"]))
            for metric in METRICS:
                delta = float(guarded[metric]) - float(original[metric])
                seed_deltas[metric].append(delta)
                pooled_deltas[metric].append(delta)
        means = {metric: fmean(values) for metric, values in seed_deltas.items()}
        for metric, mean in means.items():
            replicate_deltas[metric].append(mean)
        per_training_seed[str(train_seed)] = {
            "paired_episode_count": len(keys),
            "mean_intervention_activations": fmean(activations),
            "delta_guard_minus_original": means,
            "objective_paired_win_rate": fmean(
                value < 0.0 for value in seed_deltas["objective"]
            ),
        }
    return {
        "primary_endpoint": "objective delta: route guard minus original",
        "replication_unit": "independent PPO training seed",
        "condition_summary_over_episodes": condition_summary,
        "per_training_seed": per_training_seed,
        "delta_across_training_seeds": {
            metric: _stats(values) for metric, values in replicate_deltas.items()
        },
        "pooled_paired_delta": {
            metric: _stats(values) for metric, values in pooled_deltas.items()
        },
        "pooled_objective_win_rate": fmean(
            value < 0.0 for value in pooled_deltas["objective"]
        ),
    }


def run_route_guard_experiment(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_run).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((source_dir / "run_manifest.json").read_text())
    if source_manifest.get("status") != "COMPLETED":
        raise ValueError("Source replication run is not marked COMPLETED.")
    available_train_seeds = tuple(int(seed) for seed in source_manifest["train_seeds"])
    train_seeds = tuple(
        parse_seeds(args.train_seeds) if args.train_seeds else available_train_seeds
    )
    if not train_seeds or not set(train_seeds).issubset(available_train_seeds):
        raise ValueError("Requested training seeds are not all present in source run.")
    diagnostic_seeds = tuple(
        parse_seeds(args.diagnostic_seeds)
        if args.diagnostic_seeds
        else tuple(range(40_200, 40_400))
    )
    forbidden = (
        set(source_manifest["common_settings"]["validation_seeds"])
        | set(source_manifest["common_settings"]["test_seeds"])
        | set(range(40_000, 40_200))
        | set(range(50_000, 50_100))
    )
    if set(diagnostic_seeds) & forbidden:
        raise ValueError("Counterfactual seeds overlap an existing diagnostic/test panel.")

    resolved_device = resolve_device(args.device)
    config = BenchmarkConfig.from_json(args.config)
    probe = HTPdmFjspEnv(config=config)
    probe.reset(seed=diagnostic_seeds[0])
    route_indices = _route_indices(probe)
    probe.close()
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "source_run": str(source_dir),
        "intervention": (
            "suppress J1-O1 -> M2 whenever feasible, forcing the M1 route"
        ),
        "route_action_indices": {"M1": route_indices[0], "M2": route_indices[1]},
        "conditions": ["original", "route_guard_j1_o1_m2"],
        "train_seeds": train_seeds,
        "diagnostic_seeds": diagnostic_seeds,
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "stable_baselines3": version("stable-baselines3"),
            "sb3_contrib": version("sb3-contrib"),
            "torch": version("torch"),
        },
    }
    manifest_path = output_dir / "route_guard_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=2 * len(train_seeds) * len(diagnostic_seeds),
        desc="PPO route guard",
        unit="episode",
        disable=args.no_progress,
    )
    for train_seed in train_seeds:
        model_path = source_dir / f"train_seed_{train_seed}" / "maskable_ppo.zip"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        model = MaskablePPO.load(model_path, device=resolved_device)
        for condition, transform in (
            ("original", None),
            ("route_guard_j1_o1_m2", guard_j1_o1_m2),
        ):
            progress.set_postfix(train_seed=str(train_seed), condition=condition)
            episodes, decisions = trace_policy(
                model,
                config,
                diagnostic_seeds,
                train_seed=train_seed,
                condition=condition,
                mask_transform=transform,
                progress=progress,
            )
            episode_rows.extend(episodes)
            decision_rows.extend(decisions)
        _write_csv(episode_rows, output_dir / "route_guard_episodes.partial.csv")
        _write_csv(decision_rows, output_dir / "route_guard_decisions.partial.csv")
        del model
    progress.close()

    expected = 2 * len(train_seeds) * len(diagnostic_seeds)
    unique = {
        (row["condition"], row["train_seed"], row["seed"]) for row in episode_rows
    }
    if len(episode_rows) != expected or len(unique) != expected:
        raise ValueError("Counterfactual paired episode panel is incomplete.")
    if any(int(row["invalid_actions"]) for row in episode_rows):
        raise AssertionError("A counterfactual policy selected an invalid action.")

    summary = paired_summary(episode_rows, train_seeds)
    _write_csv(episode_rows, output_dir / "route_guard_episodes.csv")
    _write_csv(decision_rows, output_dir / "route_guard_decisions.csv")
    (output_dir / "route_guard_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["status"] = "COMPLETED"
    manifest["episode_count"] = len(episode_rows)
    manifest["decision_count"] = len(decision_rows)
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
    parser.add_argument("--train-seeds")
    parser.add_argument("--diagnostic-seeds")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run_route_guard_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
