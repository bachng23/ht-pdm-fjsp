"""Baseline RL scan on the validated committed technician-conflict kernel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.centralized_joint_dqn import (
    CentralizedJointDQNPolicy,
    train_centralized_joint_dqn,
)
from ht_pdm_fjsp.committed_conflict_rl import COMMITTED_CELL, CommittedConflictCTDEEnv
from ht_pdm_fjsp.conflict_consequence import (
    SEALED_TEST_SEEDS,
    ConsequenceCell,
    build_cell_config,
    evaluate_episode as evaluate_heuristic_episode,
)
from ht_pdm_fjsp.parallel_maintenance import ParallelMaintenanceConfig
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.value_decomposition import (
    ValueDecompositionPolicy,
    ValueLearningSettings,
    train_value_policy,
)


ALGORITHMS = ("iql", "vdn", "qmix", "centralized_dqn")
HEURISTICS = ("reactive_fibt", "independent_preventive", "coordinated_preventive")
METRICS = (
    "objective",
    "production",
    "downtime",
    "failures",
    "preventive",
    "corrective",
    "waiting",
    "proposal_conflicts",
    "conflict_steps",
    "rejected_requests",
    "committed_wait_steps",
    "missed_windows",
    "overdue_steps",
    "rework",
    "workload_imbalance",
)


def _profile(name: str) -> dict[str, Any]:
    if name == "smoke":
        return {
            "total_timesteps": 1_200,
            "train_seeds": (91_900,),
            "evaluation_seeds": tuple(range(64_390, 64_393)),
            "n_envs": 4,
            "replay_capacity": 1_200,
            "learning_starts": 128,
            "batch_size": 32,
            "train_frequency": 8,
            "target_update_interval": 200,
            "checkpoints": (300, 600, 1_200),
        }
    if name == "full":
        return {
            "total_timesteps": 200_000,
            "train_seeds": (91_000, 92_000, 93_000),
            "evaluation_seeds": tuple(range(64_200, 64_250)),
            "n_envs": 8,
            "replay_capacity": 50_000,
            "learning_starts": 5_000,
            "batch_size": 256,
            "train_frequency": 8,
            "target_update_interval": 2_000,
            "checkpoints": (50_000, 100_000, 200_000),
        }
    raise ValueError(f"Unknown profile: {name}")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _episode_from_info(
    info: dict[str, Any],
    *,
    algorithm: str,
    train_seed: int,
    checkpoint_steps: int,
    seed: int,
) -> dict[str, Any]:
    utilization = info["technician_utilization"]
    return {
        "algorithm": algorithm,
        "train_seed": train_seed,
        "checkpoint_steps": checkpoint_steps,
        "seed": seed,
        **{
            key: info[key]
            for key in (
                *METRICS,
                "episode_return",
                "early_pm",
                "return_objective_error",
                "invalid_executions",
                "duplicate_machine_assignments",
                "duplicate_technician_assignments",
            )
        },
        "technician_0_utilization": utilization[0],
        "technician_1_utilization": utilization[1],
    }


def evaluate_learned_episode(
    policy: ValueDecompositionPolicy | CentralizedJointDQNPolicy,
    base_config: ParallelMaintenanceConfig,
    *,
    algorithm: str,
    train_seed: int,
    checkpoint_steps: int,
    seed: int,
    device: str,
    cell: ConsequenceCell = COMMITTED_CELL,
    record_decisions: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    env = CommittedConflictCTDEEnv(base_config, cell)
    observation, _ = env.reset(seed=seed)
    decisions: list[dict[str, Any]] = []
    done = False
    info: dict[str, Any] = {}
    while not done:
        actions = policy.act(observation, deterministic=True, device=device)
        observation, _, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        if record_decisions:
            decision = env.core.last_decision
            decisions.append(
                {
                    "algorithm": algorithm,
                    "train_seed": train_seed,
                    "checkpoint_steps": checkpoint_steps,
                    "seed": seed,
                    "step": decision["step"],
                    "actions": json.dumps(decision["actions"]),
                    "accepted": json.dumps(decision["accepted"]),
                    "rejected": json.dumps(decision["rejected"]),
                    "proposal_conflicts": decision["proposal_conflicts"],
                    "incremental_cost": decision["incremental_cost"],
                    "reward": decision["reward"],
                }
            )
    episode = _episode_from_info(
        info,
        algorithm=algorithm,
        train_seed=train_seed,
        checkpoint_steps=checkpoint_steps,
        seed=seed,
    )
    audit = {
        "algorithm": algorithm,
        "train_seed": train_seed,
        "checkpoint_steps": checkpoint_steps,
        "seed": seed,
        "return_objective_error": info["return_objective_error"],
        "invalid_executions": info["invalid_executions"],
        "duplicate_machine_assignments": info["duplicate_machine_assignments"],
        "duplicate_technician_assignments": info[
            "duplicate_technician_assignments"
        ],
    }
    env.close()
    return episode, decisions, audit


def summarize(
    learned: list[dict[str, Any]],
    heuristics: list[dict[str, Any]],
    coordination: list[dict[str, Any]],
    *,
    profile: str,
    train_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    checkpoints: tuple[int, ...],
    checkpoint_files_complete: bool,
    training_artifacts_complete: bool,
) -> dict[str, Any]:
    curves: dict[str, list[dict[str, Any]]] = {}
    for algorithm in ALGORITHMS:
        points = []
        for checkpoint in checkpoints:
            selected = [
                row
                for row in learned
                if row["algorithm"] == algorithm
                and int(row["checkpoint_steps"]) == checkpoint
            ]
            points.append(
                {
                    "checkpoint_steps": checkpoint,
                    "episodes": len(selected),
                    **{f"{metric}_mean": _mean(selected, metric) for metric in METRICS},
                }
            )
        curves[algorithm] = points

    heuristic_summary = {
        policy: {
            "episodes": len([row for row in heuristics if row["policy"] == policy]),
            **{
                f"{metric}_mean": _mean(
                    [row for row in heuristics if row["policy"] == policy], metric
                )
                for metric in METRICS
            },
        }
        for policy in HEURISTICS
    }
    final_checkpoint = checkpoints[-1]
    per_training_seed: list[dict[str, Any]] = []
    for train_seed in train_seeds:
        row: dict[str, Any] = {"train_seed": train_seed}
        for algorithm in ALGORITHMS:
            selected = [
                item
                for item in learned
                if item["algorithm"] == algorithm
                and int(item["train_seed"]) == train_seed
                and int(item["checkpoint_steps"]) == final_checkpoint
            ]
            for metric in METRICS:
                row[f"{algorithm}_{metric}"] = _mean(selected, metric)
        per_training_seed.append(row)

    contrasts: dict[str, Any] = {}
    iql_values = [float(row["iql_objective"]) for row in per_training_seed]
    for algorithm in ("vdn", "qmix", "centralized_dqn"):
        deltas = [
            float(row[f"{algorithm}_objective"]) - float(row["iql_objective"])
            for row in per_training_seed
        ]
        contrasts[f"{algorithm}_minus_iql"] = {
            "mean": statistics.fmean(deltas),
            "training_seed_deltas": deltas,
            "wins": sum(delta < 0 for delta in deltas),
            "relative_improvement": -statistics.fmean(deltas)
            / max(abs(statistics.fmean(iql_values)), 1e-9),
        }
    qmix_minus_vdn = [
        float(row["qmix_objective"]) - float(row["vdn_objective"])
        for row in per_training_seed
    ]
    contrasts["qmix_minus_vdn"] = {
        "mean": statistics.fmean(qmix_minus_vdn),
        "training_seed_deltas": qmix_minus_vdn,
        "qmix_wins": sum(delta < 0 for delta in qmix_minus_vdn),
    }
    h1 = contrasts["centralized_dqn_minus_iql"]
    cooperative = min(
        (contrasts["vdn_minus_iql"], contrasts["qmix_minus_iql"]),
        key=lambda item: item["mean"],
    )
    expected_learned = (
        len(ALGORITHMS)
        * len(train_seeds)
        * len(checkpoints)
        * len(evaluation_seeds)
    )
    expected_heuristics = len(HEURISTICS) * len(evaluation_seeds)
    audits = {
        "all_expected_learned_evaluations_completed": len(learned)
        == expected_learned,
        "all_expected_heuristic_evaluations_completed": len(heuristics)
        == expected_heuristics,
        "all_checkpoint_files_present": checkpoint_files_complete,
        "all_training_artifacts_present": training_artifacts_complete,
        "reward_objective_identity": all(
            float(row["return_objective_error"]) <= 1e-6 for row in learned
        )
        and all(float(row["return_objective_error"]) <= 1e-6 for row in heuristics),
        "no_invalid_executions": all(
            int(row["invalid_executions"]) == 0 for row in coordination
        ),
        "no_duplicate_machine_assignments": all(
            int(row["duplicate_machine_assignments"]) == 0
            for row in coordination
        ),
        "no_duplicate_technician_assignments": all(
            int(row["duplicate_technician_assignments"]) == 0
            for row in coordination
        ),
        "sealed_test_panel_remains_closed": not any(
            int(row["seed"]) in SEALED_TEST_SEEDS
            for row in (*learned, *heuristics)
        ),
    }
    return {
        "purpose": "Committed-conflict RL baseline scan; not an algorithm claim.",
        "profile": profile,
        "primary_checkpoint": final_checkpoint,
        "curves": curves,
        "heuristic_references": heuristic_summary,
        "final_metrics_by_training_seed": per_training_seed,
        "final_contrasts": contrasts,
        "hypotheses": {
            "H1_centralized_learnability": h1["mean"] < 0
            and h1["wins"] >= 2,
            "H2_cooperative_factorization": cooperative["mean"] < 0
            and cooperative["wins"] >= 2,
            "H3_qmix_better_than_vdn": contrasts["qmix_minus_vdn"]["mean"] < 0,
        },
        "audits": audits,
        "gate": {"passed": all(audits.values())},
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    protocol = _profile(args.profile)
    if set(protocol["evaluation_seeds"]) & set(SEALED_TEST_SEEDS):
        raise ValueError("The sealed future test panel cannot be opened")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.config).resolve()
    raw_config = base_path.read_bytes()
    base = ParallelMaintenanceConfig.from_json(base_path)
    resolved = build_cell_config(base, COMMITTED_CELL)
    resolved_payload = json.dumps(resolved.to_dict(), indent=2, sort_keys=True) + "\n"
    (output_dir / "source_config.json").write_bytes(raw_config)
    (output_dir / "resolved_config.json").write_text(resolved_payload)
    device = resolve_device(args.device)
    settings = ValueLearningSettings(
        total_timesteps=protocol["total_timesteps"],
        replay_capacity=protocol["replay_capacity"],
        learning_starts=protocol["learning_starts"],
        batch_size=protocol["batch_size"],
        train_frequency=protocol["train_frequency"],
        gradient_steps=1,
        learning_rate=3e-4,
        gamma=0.99,
        target_update_interval=protocol["target_update_interval"],
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_fraction=0.60,
        hidden_dim=128,
        mixer_hidden_dim=64,
        device=device,
        n_envs=protocol["n_envs"],
    )
    manifest_path = output_dir / "committed_rl_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "purpose": "committed_conflict_rl_baseline_scan",
        "profile": args.profile,
        "started_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "source_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "resolved_config_sha256": hashlib.sha256(resolved_payload.encode()).hexdigest(),
        "cell": asdict(COMMITTED_CELL) | {"cell_id": COMMITTED_CELL.cell_id},
        "algorithms": list(ALGORITHMS),
        "heuristics": list(HEURISTICS),
        "settings": asdict(settings),
        "checkpoints": list(protocol["checkpoints"]),
        "train_seeds": list(protocol["train_seeds"]),
        "evaluation_seeds": list(protocol["evaluation_seeds"]),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
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

    heuristic_rows: list[dict[str, Any]] = []
    for policy in tqdm(HEURISTICS, desc="Heuristic references", unit="policy"):
        for seed in tqdm(
            protocol["evaluation_seeds"],
            desc=policy,
            unit="episode",
            leave=False,
        ):
            episode, _ = evaluate_heuristic_episode(
                resolved, COMMITTED_CELL, policy, seed
            )
            heuristic_rows.append(episode)
    _write_csv(heuristic_rows, output_dir / "heuristic_episodes.csv")

    learned_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    coordination_rows: list[dict[str, Any]] = []
    training_seconds: dict[str, float] = {}
    env_factory = lambda: CommittedConflictCTDEEnv(base)
    for algorithm in tqdm(ALGORITHMS, desc="RL baselines", unit="algorithm"):
        for train_seed in tqdm(
            protocol["train_seeds"],
            desc=algorithm,
            unit="training-seed",
            leave=False,
        ):
            cell_dir = output_dir / algorithm / f"train_seed_{train_seed}"
            if algorithm == "centralized_dqn":
                _, elapsed = train_centralized_joint_dqn(
                    settings,
                    cell_dir,
                    train_seed=train_seed,
                    show_progress=True,
                    checkpoint_targets=protocol["checkpoints"],
                    env_factory=env_factory,
                )
            else:
                _, elapsed = train_value_policy(
                    base,
                    settings,
                    cell_dir,
                    algorithm=algorithm,
                    train_seed=train_seed,
                    show_progress=True,
                    checkpoint_targets=protocol["checkpoints"],
                    env_factory=env_factory,
                )
            cell_key = f"{algorithm}:{train_seed}"
            training_seconds[cell_key] = elapsed
            manifest["completed_training_cells"].append(cell_key)
            manifest["training_seconds"] = training_seconds
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            for checkpoint in tqdm(
                protocol["checkpoints"],
                desc=f"Evaluate {cell_key}",
                unit="checkpoint",
                leave=False,
            ):
                checkpoint_path = (
                    cell_dir / "checkpoints" / f"model_{checkpoint}_steps.pt"
                )
                policy = (
                    CentralizedJointDQNPolicy.load(checkpoint_path, device=device)
                    if algorithm == "centralized_dqn"
                    else ValueDecompositionPolicy.load(checkpoint_path, device=device)
                )
                for seed in tqdm(
                    protocol["evaluation_seeds"],
                    desc=f"{algorithm}/{checkpoint}",
                    unit="episode",
                    leave=False,
                ):
                    episode, decisions, audit = evaluate_learned_episode(
                        policy,
                        base,
                        algorithm=algorithm,
                        train_seed=train_seed,
                        checkpoint_steps=checkpoint,
                        seed=seed,
                        device=device,
                    )
                    learned_rows.append(episode)
                    decision_rows.extend(decisions)
                    coordination_rows.append(audit)
                manifest["completed_evaluation_cells"].append(
                    f"{cell_key}:{checkpoint}"
                )
                _write_csv(
                    learned_rows, output_dir / "learned_episodes.partial.csv"
                )
                _write_csv(decision_rows, output_dir / "decisions.partial.csv")
                _write_csv(
                    coordination_rows, output_dir / "coordination.partial.csv"
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )

    checkpoint_files_complete = all(
        (
            output_dir
            / algorithm
            / f"train_seed_{train_seed}"
            / "checkpoints"
            / f"model_{checkpoint}_steps.pt"
        ).is_file()
        for algorithm in ALGORITHMS
        for train_seed in protocol["train_seeds"]
        for checkpoint in protocol["checkpoints"]
    )
    training_artifacts_complete = all(
        (
            output_dir / algorithm / f"train_seed_{train_seed}" / filename
        ).is_file()
        for algorithm in ALGORITHMS
        for train_seed in protocol["train_seeds"]
        for filename in (
            "settings.json",
            "training_progress.csv",
            "training_episodes.csv",
            "model.pt",
        )
    )
    summary = summarize(
        learned_rows,
        heuristic_rows,
        coordination_rows,
        profile=args.profile,
        train_seeds=protocol["train_seeds"],
        evaluation_seeds=protocol["evaluation_seeds"],
        checkpoints=protocol["checkpoints"],
        checkpoint_files_complete=checkpoint_files_complete,
        training_artifacts_complete=training_artifacts_complete,
    )
    _write_csv(learned_rows, output_dir / "learned_episodes.csv")
    _write_csv(decision_rows, output_dir / "decisions.csv")
    _write_csv(coordination_rows, output_dir / "coordination.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        {
            "status": "COMPLETED" if summary["gate"]["passed"] else "FAILED",
            "finished_at": datetime.now(UTC).isoformat(),
            "training_seconds": training_seconds,
            "learned_episode_count": len(learned_rows),
            "heuristic_episode_count": len(heuristic_rows),
            "decision_count": len(decision_rows),
            "coordination_audit_count": len(coordination_rows),
            "gate": summary["gate"],
            "hypotheses": summary["hypotheses"],
            "outputs": sorted(
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*")
                if path.is_file()
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not summary["gate"]["passed"]:
        raise RuntimeError("Committed-conflict RL audit gate failed")
    print(json.dumps({"gate": summary["gate"], "hypotheses": summary["hypotheses"]}, indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/parallel_maintenance.json")
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
