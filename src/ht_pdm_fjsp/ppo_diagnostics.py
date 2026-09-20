"""Diagnose trained PPO policies without opening the future held-out test panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from ht_pdm_fjsp.rl_experiment import resolve_device


TRAIN_METRICS = {
    "rollout_reward": "rollout/ep_rew_mean",
    "rollout_length": "rollout/ep_len_mean",
    "explained_variance": "train/explained_variance",
    "entropy": "train/entropy_loss",
    "approx_kl": "train/approx_kl",
    "clip_fraction": "train/clip_fraction",
    "policy_gradient_loss": "train/policy_gradient_loss",
    "value_loss": "train/value_loss",
}
REWARD_COMPONENTS = (
    "makespan",
    "tardiness",
    "preventive",
    "corrective",
    "downtime",
    "invalid",
)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _metric_series(
    rows: list[dict[str, str]], key: str
) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for row in rows:
        raw = row.get(key, "")
        if raw in {None, ""}:
            continue
        value = float(raw)
        if key == "train/entropy_loss":
            value = -value
        values.append((int(float(row["time/total_timesteps"])), value))
    return values


def _linear_slope_per_100k(series: list[tuple[int, float]]) -> float:
    if len(series) < 2:
        return 0.0
    xs = [step / 100_000.0 for step, _ in series]
    ys = [value for _, value in series]
    x_mean = fmean(xs)
    y_mean = fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0.0:
        return 0.0
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(xs, ys)
    ) / denominator


def summarize_training_log(path: Path, *, train_seed: int) -> dict[str, Any]:
    """Summarize early, best, final, and late-window training dynamics."""

    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"Empty training log: {path}")
    summary: dict[str, Any] = {
        "train_seed": train_seed,
        "training_log_rows": len(rows),
        "final_total_timesteps": int(float(rows[-1]["time/total_timesteps"])),
    }
    for label, column in TRAIN_METRICS.items():
        series = _metric_series(rows, column)
        if not series:
            summary[f"{label}_observations"] = 0
            continue
        late_count = max(1, math.ceil(len(series) * 0.2))
        late = series[-late_count:]
        summary.update(
            {
                f"{label}_observations": len(series),
                f"{label}_first": series[0][1],
                f"{label}_final": series[-1][1],
                f"{label}_late_mean": fmean(value for _, value in late),
                f"{label}_late_slope_per_100k": _linear_slope_per_100k(late),
            }
        )
        if label == "rollout_reward":
            best_step, best_value = max(series, key=lambda item: item[1])
            summary["rollout_reward_best"] = best_value
            summary["rollout_reward_best_step"] = best_step
            summary["rollout_reward_final_gap_from_best"] = (
                series[-1][1] - best_value
            )
    return summary


def _feasible_kind_counts(env: HTPdmFjspEnv, mask: np.ndarray) -> dict[str, int]:
    counts = {kind: 0 for kind in ("advance", "production", "preventive", "corrective")}
    for index in np.flatnonzero(mask):
        counts[env.actions[int(index)].kind] += 1
    return counts


def trace_policy(
    model: Any,
    config: BenchmarkConfig,
    seeds: Iterable[int],
    *,
    train_seed: int,
    progress: tqdm[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate a deterministic policy and retain decision-level diagnostics."""

    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for seed in seeds:
        env = HTPdmFjspEnv(config=config)
        observation, _ = env.reset(seed=int(seed))
        episode_return = 0.0
        action_counts = {
            kind: 0
            for kind in ("advance", "production", "preventive", "corrective")
        }
        selected_risks: list[float] = []
        feasible_counts: list[int] = []
        preventive_opportunities = 0
        invalid_actions = 0
        last_info: dict[str, Any] | None = None
        while not env._done:
            mask = env.action_masks()
            feasible_by_kind = _feasible_kind_counts(env, mask)
            preventive_opportunities += int(feasible_by_kind["preventive"] > 0)
            feasible_counts.append(int(mask.sum()))
            action, _ = model.predict(
                observation,
                action_masks=mask,
                deterministic=True,
            )
            action_index = int(np.asarray(action).item())
            descriptor = env.actions[action_index]
            action_counts[descriptor.kind] += 1
            failure_probability = (
                env.failure_probability(descriptor)
                if descriptor.kind == "production"
                else math.nan
            )
            if descriptor.kind == "production":
                selected_risks.append(failure_probability)
            age_ratio = math.nan
            if descriptor.machine_id is not None:
                machine_id = str(descriptor.machine_id)
                age_ratio = (
                    env.machines[machine_id].effective_age
                    / env.machine_specs[machine_id].weibull_eta
                )
            time_before = env.now
            failures_before = env.failures
            observation, reward, _, _, info = env.step(action_index)
            invalid_actions += int(info["invalid_action"])
            episode_return += reward
            components = info["reward_components"]
            decision_rows.append(
                {
                    "split": "diagnostic",
                    "train_seed": train_seed,
                    "seed": int(seed),
                    "decision_index": env.decision_count,
                    "time_before": time_before,
                    "time_after": env.now,
                    "action_index": action_index,
                    "action_kind": descriptor.kind,
                    "job_id": descriptor.job_id or "",
                    "operation_id": descriptor.operation_id or "",
                    "machine_id": descriptor.machine_id or "",
                    "technician_id": descriptor.technician_id or "",
                    "feasible_actions": int(mask.sum()),
                    **{
                        f"feasible_{kind}": count
                        for kind, count in feasible_by_kind.items()
                    },
                    "selected_failure_probability": failure_probability,
                    "selected_machine_age_ratio": age_ratio,
                    "reward": reward,
                    **{
                        f"reward_{name}": float(components[name])
                        for name in REWARD_COMPONENTS
                    },
                    "failure_events": env.failures - failures_before,
                    "invalid_action": bool(info["invalid_action"]),
                }
            )
            last_info = info
        result = env.result("maskable_ppo")
        objective = float(result.metrics["objective"])
        if not np.isclose(episode_return, -objective):
            raise AssertionError("PPO reward/objective identity failed.")
        if last_info is None:
            raise AssertionError("A completed episode had no decisions.")
        cumulative = last_info["cumulative_reward_components"]
        episode_rows.append(
            {
                "split": "diagnostic",
                "policy": "maskable_ppo",
                "train_seed": train_seed,
                "seed": int(seed),
                "episode_return": episode_return,
                **result.metrics,
                **{f"actions_{kind}": count for kind, count in action_counts.items()},
                "mean_feasible_actions": fmean(feasible_counts),
                "min_feasible_actions": min(feasible_counts),
                "max_feasible_actions": max(feasible_counts),
                "preventive_opportunity_decisions": preventive_opportunities,
                "preventive_take_rate": (
                    action_counts["preventive"] / preventive_opportunities
                    if preventive_opportunities
                    else 0.0
                ),
                "mean_selected_production_risk": (
                    fmean(selected_risks) if selected_risks else 0.0
                ),
                "max_selected_production_risk": (
                    max(selected_risks) if selected_risks else 0.0
                ),
                "high_risk_production_fraction": (
                    fmean(
                        risk >= config.preventive_probability_threshold
                        for risk in selected_risks
                    )
                    if selected_risks
                    else 0.0
                ),
                "invalid_actions": invalid_actions,
                **{
                    f"cumulative_reward_{name}": float(cumulative[name])
                    for name in REWARD_COMPONENTS
                },
            }
        )
        env.close()
        if progress is not None:
            progress.update(1)
    return episode_rows, decision_rows


def summarize_diagnostics(
    episode_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    metric_names = (
        "objective",
        "makespan",
        "total_tardiness",
        "failures",
        "preventive_maintenance",
        "corrective_maintenance",
        "total_cost",
        "decision_count",
        "actions_advance",
        "actions_production",
        "actions_preventive",
        "actions_corrective",
        "mean_feasible_actions",
        "preventive_opportunity_decisions",
        "preventive_take_rate",
        "mean_selected_production_risk",
        "max_selected_production_risk",
        "high_risk_production_fraction",
    )
    train_seeds = sorted({int(row["train_seed"]) for row in episode_rows})
    per_seed: dict[str, Any] = {}
    for train_seed in train_seeds:
        selected = [
            row for row in episode_rows if int(row["train_seed"]) == train_seed
        ]
        per_seed[str(train_seed)] = {
            "diagnostic_episode_count": len(selected),
            "source_test_objective": source_summary["ppo_per_training_seed"][
                str(train_seed)
            ]["objective"],
            "diagnostic_metrics": {
                metric: _stats(float(row[metric]) for row in selected)
                for metric in metric_names
            },
            "reward_components": {
                name: _stats(
                    float(row[f"cumulative_reward_{name}"]) for row in selected
                )
                for name in REWARD_COMPONENTS
            },
        }
    return {
        "replication_unit": "independent PPO training seed",
        "diagnostic_unit": "fresh stochastic environment seed",
        "training_seed_count": len(train_seeds),
        "diagnostic_seed_count": len(
            {int(row["seed"]) for row in episode_rows}
        ),
        "future_test_panel_opened": False,
        "training_dynamics": training_rows,
        "per_training_seed": per_seed,
    }


def run_diagnostics(args: argparse.Namespace) -> Path:
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
        else tuple(range(40_000, 40_200))
    )
    source_validation = set(source_manifest["common_settings"]["validation_seeds"])
    source_test = set(source_manifest["common_settings"]["test_seeds"])
    future_test = set(range(50_000, 50_100))
    if (
        set(diagnostic_seeds) & source_validation
        or set(diagnostic_seeds) & source_test
        or set(diagnostic_seeds) & future_test
    ):
        raise ValueError("Diagnostic seeds overlap a validation or test panel.")

    resolved_device = resolve_device(args.device)
    config = BenchmarkConfig.from_json(args.config)
    source_summary = json.loads((source_dir / "aggregate_summary.json").read_text())
    training_rows: list[dict[str, Any]] = []
    model_paths: dict[int, Path] = {}
    for train_seed in train_seeds:
        seed_dir = source_dir / f"train_seed_{train_seed}"
        model_path = seed_dir / "maskable_ppo.zip"
        log_path = seed_dir / "training_log" / "progress.csv"
        if not model_path.is_file() or not log_path.is_file():
            raise FileNotFoundError(f"Missing model or training log for {train_seed}.")
        model_paths[train_seed] = model_path
        training = summarize_training_log(log_path, train_seed=train_seed)
        training["source_test_objective"] = source_summary[
            "ppo_per_training_seed"
        ][str(train_seed)]["objective"]
        training_rows.append(training)

    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "source_run": str(source_dir),
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
    manifest_path = output_dir / "diagnostic_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(training_rows, output_dir / "training_dynamics.csv")

    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(train_seeds) * len(diagnostic_seeds),
        desc="PPO diagnostics",
        unit="episode",
        disable=args.no_progress,
    )
    for train_seed in train_seeds:
        progress.set_postfix(train_seed=str(train_seed))
        model = MaskablePPO.load(model_paths[train_seed], device=resolved_device)
        episodes, decisions = trace_policy(
            model,
            config,
            diagnostic_seeds,
            train_seed=train_seed,
            progress=progress,
        )
        episode_rows.extend(episodes)
        decision_rows.extend(decisions)
        _write_csv(episode_rows, output_dir / "diagnostic_episodes.partial.csv")
        _write_csv(decision_rows, output_dir / "diagnostic_decisions.partial.csv")
        del model
    progress.close()

    expected = len(train_seeds) * len(diagnostic_seeds)
    if len(episode_rows) != expected:
        raise ValueError("Diagnostic episode panel is incomplete.")
    if len({(row["train_seed"], row["seed"]) for row in episode_rows}) != expected:
        raise ValueError("Diagnostic episode panel contains duplicate pairs.")
    if any(int(row["invalid_actions"]) for row in episode_rows):
        raise AssertionError("A masked PPO policy selected an invalid action.")

    summary = summarize_diagnostics(episode_rows, training_rows, source_summary)
    _write_csv(episode_rows, output_dir / "diagnostic_episodes.csv")
    _write_csv(decision_rows, output_dir / "diagnostic_decisions.csv")
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["status"] = "COMPLETED"
    manifest["diagnostic_episode_count"] = len(episode_rows)
    manifest["diagnostic_decision_count"] = len(decision_rows)
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
    run_diagnostics(build_parser().parse_args())


if __name__ == "__main__":
    main()
