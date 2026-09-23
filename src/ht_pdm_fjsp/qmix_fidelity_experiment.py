"""Audit feed-forward QMIX against a recurrent baseline-fidelity variant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ht_pdm_fjsp.advanced_baselines import (
    HealthThresholdPolicy,
    MaskedSPTPolicy,
    rollout_policy,
)
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.recurrent_qmix import (
    RecurrentQMIXPolicy,
    RecurrentQMIXSettings,
    train_recurrent_qmix,
)
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.technician_capacity_screening import build_condition_config
from ht_pdm_fjsp.three_way_marl_experiment import evaluate_episode
from ht_pdm_fjsp.value_decomposition import (
    ValueDecompositionPolicy,
    ValueLearningSettings,
    train_value_policy,
)


FF_QMIX = "ff_qmix"
RECURRENT_QMIX = "recurrent_qmix"
CONDITIONS = (FF_QMIX, RECURRENT_QMIX)
SEALED_TEST_SEEDS = tuple(range(62_000, 62_100))
SUMMARY_METRICS = (
    "objective",
    "makespan",
    "total_cost",
    "failures",
    "preventive_maintenance",
    "corrective_maintenance",
    "maintenance_wait_time",
    "maximum_rejection_streak",
)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profile(name: str) -> dict[str, Any]:
    if name == "smoke":
        return {
            "total_timesteps": 600,
            "train_seeds": (76_100,),
            "evaluation_seeds": tuple(range(61_990, 61_993)),
            "n_envs": 4,
            "replay_capacity": 600,
            "learning_starts": 32,
            "ff_batch_size": 16,
            "episode_batch_size": 2,
        }
    if name == "full":
        return {
            "total_timesteps": 300_000,
            "train_seeds": (76_000, 77_000, 78_000),
            "evaluation_seeds": tuple(range(61_700, 61_750)),
            "n_envs": 8,
            "replay_capacity": 20_000,
            "learning_starts": 2_000,
            "ff_batch_size": 256,
            "episode_batch_size": 8,
        }
    raise ValueError(f"Unknown profile: {name}")


def _checkpoint_targets(total_timesteps: int) -> tuple[int, ...]:
    if total_timesteps % 3:
        raise ValueError("total_timesteps must be divisible by three")
    return tuple(total_timesteps * part // 3 for part in range(1, 4))


def _checkpoint_path(cell: Path, target: int) -> Path:
    return cell / "checkpoints" / f"model_{target}_steps.pt"


def _reward_sanity(
    config: BenchmarkConfig, seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policies = (MaskedSPTPolicy(), HealthThresholdPolicy())
    for policy in tqdm(policies, desc="Reward sanity policies", unit="policy"):
        for seed in tqdm(
            seeds, desc=policy.name, unit="episode", leave=False
        ):
            result, episode_return = rollout_policy(
                HTPdmFjspEnv(config), policy, seed=seed
            )
            rows.append(
                {
                    "policy": policy.name,
                    "seed": seed,
                    "episode_return": episode_return,
                    **result.metrics,
                }
            )
    return rows


def _group_mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def summarize(
    episodes: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    coordination: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    *,
    train_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    checkpoints: tuple[int, ...],
    profile: str,
) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_points: list[dict[str, Any]] = []
        for train_seed in train_seeds:
            for checkpoint in checkpoints:
                selected = [
                    row
                    for row in episodes
                    if row["condition"] == condition
                    and int(row["train_seed"]) == train_seed
                    and int(row["checkpoint_steps"]) == checkpoint
                ]
                selected_decisions = [
                    row
                    for row in decisions
                    if row["condition"] == condition
                    and int(row["train_seed"]) == train_seed
                    and int(row["checkpoint_steps"]) == checkpoint
                ]
                proposals = sum(int(row["proposals"]) for row in selected_decisions)
                rejected = sum(int(row["rejected"]) for row in selected_decisions)
                production_conflicts = sum(
                    int(row["production_conflicts"])
                    for row in selected_decisions
                )
                technician_conflicts = sum(
                    int(row["technician_conflicts"])
                    for row in selected_decisions
                )
                joint_steps = len(selected_decisions)
                condition_points.append(
                    {
                        "train_seed": train_seed,
                        "checkpoint_steps": checkpoint,
                        "episodes": len(selected),
                        **{
                            f"{metric}_mean": _group_mean(selected, metric)
                            for metric in SUMMARY_METRICS
                        },
                        "rejections_per_1000_joint_steps": (
                            1_000 * rejected / joint_steps
                        ),
                        "production_conflicts_per_1000_joint_steps": (
                            1_000 * production_conflicts / joint_steps
                        ),
                        "technician_conflicts_per_1000_joint_steps": (
                            1_000 * technician_conflicts / joint_steps
                        ),
                        "proposal_acceptance_ratio": (
                            (proposals - rejected) / proposals if proposals else 1.0
                        ),
                    }
                )
        curves[condition] = condition_points

    final_checkpoint = checkpoints[-1]
    paired_by_seed: list[dict[str, Any]] = []
    for train_seed in train_seeds:
        seed_row: dict[str, Any] = {"train_seed": train_seed}
        for metric in SUMMARY_METRICS:
            recurrent = [
                row for row in episodes
                if row["condition"] == RECURRENT_QMIX
                and int(row["train_seed"]) == train_seed
                and int(row["checkpoint_steps"]) == final_checkpoint
            ]
            feed_forward = [
                row for row in episodes
                if row["condition"] == FF_QMIX
                and int(row["train_seed"]) == train_seed
                and int(row["checkpoint_steps"]) == final_checkpoint
            ]
            seed_row[f"{metric}_delta_recurrent_minus_ff"] = (
                _group_mean(recurrent, metric) - _group_mean(feed_forward, metric)
            )
        paired_by_seed.append(seed_row)

    invalid = sum(
        int(row[key])
        for row in coordination
        for key in (
            "invalid_executions",
            "duplicate_operation_executions",
            "duplicate_technician_executions",
        )
    )
    expected = (
        len(CONDITIONS)
        * len(train_seeds)
        * len(checkpoints)
        * len(evaluation_seeds)
    )
    objective_deltas = [
        row["objective_delta_recurrent_minus_ff"] for row in paired_by_seed
    ]
    checks = {
        "all_expected_evaluations_completed": len(episodes) == expected,
        "all_coordination_audits_passed": invalid == 0,
        "reward_objective_identity_checked": len(reward_rows)
        == 2 * len(evaluation_seeds),
        "sealed_test_panel_remains_closed": True,
    }
    if profile == "full":
        checks["promotion_mean_objective_noninferior"] = (
            statistics.fmean(objective_deltas) <= 0
        )
        checks["promotion_two_of_three_training_seeds_noninferior"] = (
            sum(delta <= 0 for delta in objective_deltas) >= 2
        )
    sanity: dict[str, Any] = {}
    for policy in ("masked_spt", "health_threshold"):
        selected = [row for row in reward_rows if row["policy"] == policy]
        sanity[policy] = {
            "episodes": len(selected),
            "objective_mean": _group_mean(selected, "objective"),
            "preventive_maintenance_mean": _group_mean(
                selected, "preventive_maintenance"
            ),
        }
    return {
        "experiment": "E1 QMIX baseline fidelity",
        "primary_contrast": "recurrent_qmix - ff_qmix at 300k",
        "curves": curves,
        "final_paired_deltas_by_training_seed": paired_by_seed,
        "final_objective_delta": {
            "mean": statistics.fmean(objective_deltas),
            "sample_std": (
                statistics.stdev(objective_deltas)
                if len(objective_deltas) > 1
                else None
            ),
            "training_seed_count": len(objective_deltas),
        },
        "reward_sanity": sanity,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _profile(args.profile)
    total_timesteps = protocol["total_timesteps"]
    train_seeds = protocol["train_seeds"]
    evaluation_seeds = protocol["evaluation_seeds"]
    if set(evaluation_seeds) & set(SEALED_TEST_SEEDS):
        raise ValueError("The sealed future test panel cannot be opened")
    if total_timesteps % protocol["n_envs"]:
        raise ValueError("total_timesteps must be divisible by n_envs")
    checkpoints = _checkpoint_targets(total_timesteps)
    base_path = Path(args.config).resolve()
    raw_config = base_path.read_bytes()
    base = BenchmarkConfig.from_json(base_path)
    config = build_condition_config(base, topology="two_specialists", multiplier=2.0)
    scaled_payload = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    (output_dir / "scaled_config.json").write_text(scaled_payload)
    device = resolve_device(args.device)
    common = {
        "total_timesteps": total_timesteps,
        "replay_capacity": protocol["replay_capacity"],
        "learning_starts": protocol["learning_starts"],
        "train_frequency": 4,
        "gradient_steps": 1,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "target_update_interval": max(32, total_timesteps // 100),
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_fraction": 0.5,
        "hidden_dim": 128,
        "mixer_hidden_dim": 64,
        "device": device,
        "n_envs": protocol["n_envs"],
    }
    ff_settings = ValueLearningSettings(
        **common, batch_size=protocol["ff_batch_size"]
    )
    recurrent_settings = RecurrentQMIXSettings(
        **common,
        episode_batch_size=protocol["episode_batch_size"],
        agent_embedding_dim=16,
        checkpoint_targets=checkpoints,
    )
    manifest_path = output_dir / "qmix_fidelity_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "git_commit": _git_revision(),
        "base_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "scaled_config_sha256": hashlib.sha256(scaled_payload.encode()).hexdigest(),
        "capacity_condition": "two_specialists_x2_0",
        "conditions": CONDITIONS,
        "train_seeds": train_seeds,
        "evaluation_seeds": evaluation_seeds,
        "reserved_future_test_seeds": SEALED_TEST_SEEDS,
        "future_test_panel_opened": False,
        "checkpoint_targets": checkpoints,
        "ff_settings": asdict(ff_settings),
        "recurrent_settings": asdict(recurrent_settings),
        "requested_device": args.device,
        "resolved_device": device,
        "completed_training_cells": [],
        "completed_evaluation_cells": [],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "torch")},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    reward_rows = _reward_sanity(config, evaluation_seeds)
    _write_csv(reward_rows, output_dir / "reward_sanity_episodes.csv")
    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    coordination: list[dict[str, Any]] = []
    training_seconds: dict[str, float] = {}
    for condition in tqdm(CONDITIONS, desc="QMIX conditions", unit="condition"):
        for train_seed in tqdm(
            train_seeds, desc=condition, unit="training-seed", leave=False
        ):
            cell = output_dir / condition / f"train_seed_{train_seed}"
            if condition == FF_QMIX:
                _, elapsed = train_value_policy(
                    config,
                    ff_settings,
                    cell,
                    algorithm="qmix",
                    train_seed=train_seed,
                    show_progress=True,
                    checkpoint_targets=checkpoints,
                )
            else:
                _, elapsed = train_recurrent_qmix(
                    config,
                    recurrent_settings,
                    cell,
                    train_seed=train_seed,
                    show_progress=True,
                )
            cell_key = f"{condition}:{train_seed}"
            training_seconds[cell_key] = elapsed
            manifest["completed_training_cells"].append(cell_key)
            manifest["training_seconds"] = training_seconds
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            for checkpoint in tqdm(
                checkpoints,
                desc=f"Evaluate {cell_key}",
                unit="checkpoint",
                leave=False,
            ):
                path = _checkpoint_path(cell, checkpoint)
                policy = (
                    ValueDecompositionPolicy.load(path, device=device)
                    if condition == FF_QMIX
                    else RecurrentQMIXPolicy.load(path, device=device)
                )
                for seed in tqdm(
                    evaluation_seeds,
                    desc=f"{condition}/{checkpoint}",
                    unit="episode",
                    leave=False,
                ):
                    episode, episode_decisions, audit = evaluate_episode(
                        policy,
                        config,
                        condition=condition,
                        train_seed=train_seed,
                        seed=seed,
                        device=device,
                    )
                    episode["checkpoint_steps"] = checkpoint
                    audit["checkpoint_steps"] = checkpoint
                    for row in episode_decisions:
                        row["checkpoint_steps"] = checkpoint
                    episodes.append(episode)
                    decisions.extend(episode_decisions)
                    coordination.append(audit)
                manifest["completed_evaluation_cells"].append(
                    f"{cell_key}:{checkpoint}"
                )
                _write_csv(episodes, output_dir / "qmix_fidelity_episodes.partial.csv")
                _write_csv(decisions, output_dir / "qmix_fidelity_decisions.partial.csv")
                _write_csv(
                    coordination,
                    output_dir / "qmix_fidelity_coordination.partial.csv",
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )

    summary = summarize(
        episodes,
        decisions,
        coordination,
        reward_rows,
        train_seeds=train_seeds,
        evaluation_seeds=evaluation_seeds,
        checkpoints=checkpoints,
        profile=args.profile,
    )
    _write_csv(episodes, output_dir / "qmix_fidelity_episodes.csv")
    _write_csv(decisions, output_dir / "qmix_fidelity_decisions.csv")
    _write_csv(coordination, output_dir / "qmix_fidelity_coordination.csv")
    (output_dir / "qmix_fidelity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        status="COMPLETED",
        completed_at_utc=datetime.now(UTC).isoformat(),
        episode_count=len(episodes),
        decision_count=len(decisions),
        training_seconds=training_seconds,
        gate=summary["gate"],
        output_files=sorted(path.name for path in output_dir.iterdir()),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/brandimarte_mk01_ht_pdm.json")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
