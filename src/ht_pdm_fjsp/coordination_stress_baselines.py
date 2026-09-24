"""Screen coordination pressure, then run fixed-budget RL baselines on one cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
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
from ht_pdm_fjsp.committed_conflict_rl import CommittedConflictCTDEEnv
from ht_pdm_fjsp.committed_conflict_rl_experiment import (
    ALGORITHMS,
    METRICS,
    _git_revision,
    _write_csv,
    evaluate_learned_episode,
)
from ht_pdm_fjsp.conflict_consequence import (
    SEALED_TEST_SEEDS,
    ConflictConsequenceEnv,
    ConsequenceCell,
    build_cell_config,
    evaluate_episode,
)
from ht_pdm_fjsp.parallel_maintenance import ParallelMaintenanceConfig
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.value_decomposition import (
    ValueDecompositionPolicy,
    ValueLearningSettings,
    train_value_policy,
)


CELLS = (
    ConsequenceCell(True, False, False),
    ConsequenceCell(True, True, False),
    ConsequenceCell(True, False, True),
)
SCREEN_POLICIES = (
    "random_feasible",
    "independent_preventive",
    "coordinated_preventive",
)


def profile_settings(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "screen_seeds": tuple(range(64_900, 64_903)),
            "evaluation_seeds": tuple(range(64_910, 64_912)),
            "train_seeds": (94_900,),
            "total_timesteps": 40,
            "checkpoints": (20, 40),
            "n_envs": 4,
            "replay_capacity": 40,
            "learning_starts": 8,
            "batch_size": 4,
            "target_update_interval": 8,
        }
    if profile == "full":
        return {
            "screen_seeds": tuple(range(64_500, 64_600)),
            "evaluation_seeds": tuple(range(64_600, 64_700)),
            "train_seeds": tuple(range(94_000, 94_010)),
            "total_timesteps": 500_000,
            "checkpoints": (100_000, 300_000, 500_000),
            "n_envs": 8,
            "replay_capacity": 50_000,
            "learning_starts": 5_000,
            "batch_size": 256,
            "target_update_interval": 2_000,
        }
    raise ValueError(f"Unknown profile: {profile}")


def random_feasible_episode(
    config: ParallelMaintenanceConfig,
    cell: ConsequenceCell,
    seed: int,
) -> dict[str, Any]:
    """Use an independent policy RNG; never consume environment shock draws."""
    env = ConflictConsequenceEnv(config, cell)
    env.reset(seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 91_417]))
    done = False
    info: dict[str, Any] = {}
    while not done:
        masks = env.action_masks()
        available = {
            technician
            for technician in range(env.technician_count)
            if env.technician_available(technician)
        }
        actions = np.zeros(env.num_agents, dtype=np.int64)
        for machine in rng.permutation(env.num_agents):
            choices = [0] + [
                technician + 1
                for technician in sorted(available)
                if masks[machine, technician + 1]
            ]
            action = int(rng.choice(choices))
            actions[machine] = action
            if action:
                available.remove(action - 1)
        _, done, info = env.step(actions)
    return {
        "cell_id": cell.cell_id,
        **asdict(cell),
        "policy": "random_feasible",
        "seed": seed,
        **{key: value for key, value in info.items() if key != "technician_utilization"},
        **{
            f"technician_{index}_utilization": value
            for index, value in enumerate(info["technician_utilization"])
        },
    }


def paired_interval(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("A paired interval requires observations")
    mean = statistics.fmean(values)
    # Student-t critical values for the locked screen and training panels.
    critical = {3: 4.302653, 10: 2.262157, 100: 1.984217}.get(len(values), 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def summarize_screen(
    rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    *,
    profile: str,
) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    qualified: list[tuple[float, str]] = []
    for cell in CELLS:
        groups = {
            policy: {
                int(row["seed"]): row
                for row in rows
                if row["cell_id"] == cell.cell_id and row["policy"] == policy
            }
            for policy in SCREEN_POLICIES
        }
        if any(set(group) != set(seeds) for group in groups.values()):
            raise AssertionError(f"Incomplete heuristic screen for {cell.cell_id}")
        gaps = [
            float(groups["independent_preventive"][seed]["objective"])
            - float(groups["coordinated_preventive"][seed]["objective"])
            for seed in seeds
        ]
        paired = paired_interval(gaps)
        averages = {
            policy: {
                metric: statistics.fmean(float(group[seed][metric]) for seed in seeds)
                for metric in METRICS
            }
            for policy, group in groups.items()
        }
        coordinated = averages["coordinated_preventive"]
        independent = averages["independent_preventive"]
        relative_gap = paired["mean"] / max(abs(coordinated["objective"]), 1e-9)
        worse_rate = sum(gap > 0 for gap in gaps) / len(gaps)
        incidence = independent["conflict_steps"] / 168.0
        checks = {
            "relative_gap_at_least_5_percent": relative_gap >= 0.05,
            "paired_95_percent_lower_above_zero": paired["lower"] > 0,
            "independent_worse_on_70_percent": worse_rate >= 0.70,
            "conflict_step_incidence_at_least_2p5_percent": incidence >= 0.025,
            "coordinated_zero_conflicts": coordinated["proposal_conflicts"] == 0,
            "random_worse_than_coordinated": averages["random_feasible"]["objective"]
            > coordinated["objective"],
            "window_mechanism_active_if_enabled": (
                not cell.maintenance_window
                or any(
                    averages[policy]["missed_windows"] > 0
                    or averages[policy]["overdue_steps"] > 0
                    for policy in SCREEN_POLICIES
                )
            ),
        }
        audit = {
            "reward_identity": all(
                float(row["return_objective_error"]) <= 1e-6
                for policy in SCREEN_POLICIES
                for row in groups[policy].values()
            ),
            "no_invalid_or_duplicate_execution": all(
                int(row[key]) == 0
                for policy in SCREEN_POLICIES
                for row in groups[policy].values()
                for key in (
                    "invalid_executions",
                    "duplicate_machine_assignments",
                    "duplicate_technician_assignments",
                )
            ),
        }
        qualifies = all(checks.values()) and all(audit.values())
        if qualifies:
            qualified.append((relative_gap, cell.cell_id))
        by_cell[cell.cell_id] = {
            "factors": asdict(cell),
            "paired_independent_minus_coordinated": paired,
            "relative_gap": relative_gap,
            "independent_worse_rate": worse_rate,
            "independent_conflict_step_incidence": incidence,
            "policy_metrics": averages,
            "checks": checks,
            "audit": audit,
            "qualifies": qualifies,
        }
    selected = sorted(qualified, key=lambda item: (-item[0], item[1]))[0][1] if qualified else None
    audits = {
        "expected_episode_count": len(rows) == len(CELLS) * len(SCREEN_POLICIES) * len(seeds),
        "all_cells_audited": all(all(item["audit"].values()) for item in by_cell.values()),
        "sealed_panel_closed": not set(seeds).intersection(SEALED_TEST_SEEDS),
    }
    return {
        "profile": profile,
        "selection_rule": "largest qualifying relative paired gap; cell-id tie break",
        "selected_cell": selected,
        "cell_summaries": by_cell,
        "audits": audits,
        "gate": {"passed": all(audits.values())},
    }


def summarize_rl(
    learned: list[dict[str, Any]],
    heuristics: list[dict[str, Any]],
    coordination: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    checkpoint_files_complete: bool,
    training_artifacts_complete: bool,
) -> dict[str, Any]:
    final_checkpoint = settings["checkpoints"][-1]
    final = [row for row in learned if row["checkpoint_steps"] == final_checkpoint]
    reference = statistics.fmean(
        float(row["objective"])
        for row in heuristics
        if row["policy"] == "coordinated_preventive"
    )
    reference_scale = max(abs(reference), 1e-9)
    per_seed: list[dict[str, Any]] = []
    for train_seed in settings["train_seeds"]:
        record: dict[str, Any] = {"train_seed": train_seed}
        for algorithm in ALGORITHMS:
            group = [
                row for row in final
                if row["algorithm"] == algorithm and row["train_seed"] == train_seed
            ]
            if len(group) != len(settings["evaluation_seeds"]):
                raise AssertionError("Incomplete final checkpoint evaluation")
            record[algorithm] = {
                metric: statistics.fmean(float(row[metric]) for row in group)
                for metric in METRICS
            }
        per_seed.append(record)
    curves = {
        algorithm: [
            {
                "checkpoint_steps": checkpoint,
                "episodes": len(group := [
                    row for row in learned
                    if row["algorithm"] == algorithm
                    and row["checkpoint_steps"] == checkpoint
                ]),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in group)
                    for metric in METRICS
                },
            }
            for checkpoint in settings["checkpoints"]
        ]
        for algorithm in ALGORITHMS
    }
    final_means = {
        algorithm: statistics.fmean(
            record[algorithm]["objective"] for record in per_seed
        )
        for algorithm in ALGORITHMS
    }
    vs_reference = {}
    for algorithm in ALGORITHMS:
        seed_gaps = [
            record[algorithm]["objective"] - reference for record in per_seed
        ]
        vs_reference[algorithm] = {
            "mean_gap": statistics.fmean(seed_gaps),
            "relative_gap": statistics.fmean(seed_gaps) / reference_scale,
            "training_seed_gap_interval_95": paired_interval(seed_gaps),
            "training_seed_gaps": seed_gaps,
        }
    comparisons = {}
    for left, right in (
        ("centralized_dqn", "iql"),
        ("vdn", "iql"),
        ("qmix", "iql"),
        ("vdn", "qmix"),
    ):
        differences = [
            record[left]["objective"] - record[right]["objective"]
            for record in per_seed
        ]
        comparisons[f"{left}_minus_{right}"] = {
            **paired_interval(differences),
            "training_seed_differences": differences,
            "left_wins": sum(delta < 0 for delta in differences),
        }
    audits = {
        "learned_episode_count": len(learned) == len(ALGORITHMS)
        * len(settings["train_seeds"])
        * len(settings["checkpoints"])
        * len(settings["evaluation_seeds"]),
        "heuristic_episode_count": len(heuristics) == len(SCREEN_POLICIES)
        * len(settings["evaluation_seeds"]),
        "coordination_episode_count": len(coordination) == len(learned),
        "checkpoint_files_complete": checkpoint_files_complete,
        "training_artifacts_complete": training_artifacts_complete,
        "reward_identity": all(
            float(row["return_objective_error"]) <= 1e-6
            for row in (*learned, *heuristics)
        ),
        "no_invalid_or_duplicate_execution": all(
            int(row[key]) == 0
            for row in (*coordination, *heuristics)
            for key in (
                "invalid_executions",
                "duplicate_machine_assignments",
                "duplicate_technician_assignments",
            )
        ),
        "sealed_panel_closed": not any(
            int(row["seed"]) in SEALED_TEST_SEEDS
            for row in (*learned, *heuristics)
        ),
    }
    return {
        "primary_checkpoint": final_checkpoint,
        "reference_objective": reference,
        "curves": curves,
        "per_training_seed": per_seed,
        "final_mean_objective": final_means,
        "vs_reference": vs_reference,
        "paired_algorithm_differences": comparisons,
        "hypotheses": {
            "H2_factorization_reaches_reference": min(
                vs_reference[algorithm]["relative_gap"]
                for algorithm in ("vdn", "qmix")
            ) <= 0.05,
            "H3_diagnostic_bottleneck_pattern": (
                vs_reference["centralized_dqn"]["relative_gap"] <= 0.05
                and all(
                    vs_reference[algorithm]["relative_gap"] > 0.05
                    for algorithm in ("vdn", "qmix")
                )
            ),
        },
        "audits": audits,
        "gate": {"passed": all(audits.values())},
    }


def _settings(protocol: dict[str, Any], device: str) -> ValueLearningSettings:
    return ValueLearningSettings(
        total_timesteps=protocol["total_timesteps"],
        replay_capacity=protocol["replay_capacity"],
        learning_starts=protocol["learning_starts"],
        batch_size=protocol["batch_size"],
        train_frequency=8,
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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> Path:
    protocol = profile_settings(args.profile)
    if set(protocol["screen_seeds"] + protocol["evaluation_seeds"]) & set(SEALED_TEST_SEEDS):
        raise ValueError("The sealed future test panel cannot be opened")
    if set(protocol["screen_seeds"]) & set(protocol["evaluation_seeds"]):
        raise ValueError("Screen and RL evaluation seeds must be disjoint")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    raw_config = config_path.read_bytes()
    base = ParallelMaintenanceConfig.from_json(config_path)
    device = resolve_device(args.device)
    (output_dir / "source_config.json").write_bytes(raw_config)
    screen_configs = {
        cell.cell_id: build_cell_config(base, cell).to_dict() for cell in CELLS
    }
    screen_configs_payload = json.dumps(screen_configs, indent=2, sort_keys=True) + "\n"
    (output_dir / "screen_configs.json").write_text(screen_configs_payload)
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "SCREENING",
        "purpose": "coordination_stress_baseline_diagnostic",
        "profile": args.profile,
        "started_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "source_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "screen_configs_sha256": hashlib.sha256(screen_configs_payload.encode()).hexdigest(),
        "cells": [asdict(cell) | {"cell_id": cell.cell_id} for cell in CELLS],
        "screen_policies": list(SCREEN_POLICIES),
        "algorithms": list(ALGORITHMS),
        "screen_seeds": list(protocol["screen_seeds"]),
        "evaluation_seeds": list(protocol["evaluation_seeds"]),
        "train_seeds": list(protocol["train_seeds"]),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "checkpoints": list(protocol["checkpoints"]),
        "requested_device": args.device,
        "resolved_device": device,
        "settings": asdict(_settings(protocol, device)),
        "completed_training_cells": [],
        "completed_evaluation_cells": [],
        "completed_screen_cells": [],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "torch")},
        },
    }
    _save_json(manifest_path, manifest)

    screen_rows: list[dict[str, Any]] = []
    for cell in tqdm(CELLS, desc="Screen cells", unit="cell"):
        config = build_cell_config(base, cell)
        for policy in tqdm(SCREEN_POLICIES, desc=cell.cell_id, unit="policy", leave=False):
            for seed in tqdm(protocol["screen_seeds"], desc=policy, unit="episode", leave=False):
                episode = (
                    random_feasible_episode(config, cell, seed)
                    if policy == "random_feasible"
                    else evaluate_episode(config, cell, policy, seed)[0]
                )
                screen_rows.append(episode)
        _write_csv(screen_rows, output_dir / "screen_episodes.partial.csv")
        manifest["completed_screen_cells"].append(cell.cell_id)
        _save_json(manifest_path, manifest)
    _write_csv(screen_rows, output_dir / "screen_episodes.csv")
    screen_summary = summarize_screen(screen_rows, protocol["screen_seeds"], profile=args.profile)
    _save_json(output_dir / "screen_summary.json", screen_summary)
    if not screen_summary["gate"]["passed"]:
        manifest["status"] = "FAILED"
        manifest["gate"] = screen_summary["gate"]
        _save_json(manifest_path, manifest)
        raise RuntimeError("Heuristic screen audit gate failed")
    selected_id = (
        CELLS[0].cell_id if args.profile == "smoke" else screen_summary["selected_cell"]
    )
    manifest["selected_cell"] = selected_id
    manifest["smoke_selection_override"] = args.profile == "smoke"
    manifest["screen_qualifying_cells"] = [
        name for name, item in screen_summary["cell_summaries"].items()
        if item["qualifies"]
    ]
    if selected_id is None:
        _save_json(output_dir / "summary.json", {
            "outcome": "SCREEN_ONLY",
            "selected_cell": None,
            "screen_hypothesis_H1": False,
            "gate": screen_summary["gate"],
        })
        manifest.update({
            "status": "COMPLETED",
            "outcome": "SCREEN_ONLY",
            "finished_at": datetime.now(UTC).isoformat(),
            "screen_episode_count": len(screen_rows),
            "gate": screen_summary["gate"],
            "outputs": sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()),
        })
        _save_json(manifest_path, manifest)
        print("No cell met the locked coordination-pressure rule; RL training skipped.")
        return output_dir

    cell = next(item for item in CELLS if item.cell_id == selected_id)
    resolved = build_cell_config(base, cell)
    resolved_payload = json.dumps(resolved.to_dict(), indent=2, sort_keys=True) + "\n"
    (output_dir / "resolved_config.json").write_text(resolved_payload)
    manifest["resolved_config_sha256"] = hashlib.sha256(resolved_payload.encode()).hexdigest()
    manifest["status"] = "TRAINING"
    _save_json(manifest_path, manifest)
    heuristic_rows: list[dict[str, Any]] = []
    for policy in tqdm(SCREEN_POLICIES, desc="RL-panel heuristics", unit="policy"):
        for seed in tqdm(protocol["evaluation_seeds"], desc=policy, unit="episode", leave=False):
            episode = (
                random_feasible_episode(resolved, cell, seed)
                if policy == "random_feasible"
                else evaluate_episode(resolved, cell, policy, seed)[0]
            )
            heuristic_rows.append(episode)
    _write_csv(heuristic_rows, output_dir / "heuristic_episodes.csv")

    settings = _settings(protocol, device)
    env_factory = lambda: CommittedConflictCTDEEnv(base, cell)
    learned_rows: list[dict[str, Any]] = []
    coordination_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    training_seconds: dict[str, float] = {}
    for algorithm in tqdm(ALGORITHMS, desc="RL baselines", unit="algorithm"):
        for train_seed in tqdm(protocol["train_seeds"], desc=algorithm, unit="seed", leave=False):
            run_dir = output_dir / algorithm / f"train_seed_{train_seed}"
            if algorithm == "centralized_dqn":
                _, elapsed = train_centralized_joint_dqn(
                    settings, run_dir, train_seed=train_seed,
                    show_progress=True, checkpoint_targets=protocol["checkpoints"],
                    env_factory=env_factory,
                )
            else:
                _, elapsed = train_value_policy(
                    base, settings, run_dir, algorithm=algorithm,
                    train_seed=train_seed, show_progress=True,
                    checkpoint_targets=protocol["checkpoints"],
                    env_factory=env_factory,
                )
            training_seconds[f"{algorithm}:{train_seed}"] = elapsed
            manifest["completed_training_cells"].append(f"{algorithm}:{train_seed}")
            manifest["training_seconds"] = training_seconds
            _save_json(manifest_path, manifest)
            for checkpoint in tqdm(protocol["checkpoints"], desc=f"Evaluate {algorithm}:{train_seed}", unit="checkpoint", leave=False):
                model_path = run_dir / "checkpoints" / f"model_{checkpoint}_steps.pt"
                policy = (
                    CentralizedJointDQNPolicy.load(model_path, device=device)
                    if algorithm == "centralized_dqn"
                    else ValueDecompositionPolicy.load(model_path, device=device)
                )
                for seed in tqdm(protocol["evaluation_seeds"], desc=f"{algorithm}/{checkpoint}", unit="episode", leave=False):
                    trace = (
                        checkpoint == protocol["checkpoints"][-1]
                        and train_seed == protocol["train_seeds"][0]
                        and seed == protocol["evaluation_seeds"][0]
                    )
                    episode, decisions, audit = evaluate_learned_episode(
                        policy, base, algorithm=algorithm, train_seed=train_seed,
                        checkpoint_steps=checkpoint, seed=seed, device=device,
                        cell=cell, record_decisions=trace,
                    )
                    episode["cell_id"] = selected_id
                    audit["cell_id"] = selected_id
                    learned_rows.append(episode)
                    coordination_rows.append(audit)
                    decision_rows.extend(decisions)
                manifest["completed_evaluation_cells"].append(f"{algorithm}:{train_seed}:{checkpoint}")
                _write_csv(learned_rows, output_dir / "learned_episodes.partial.csv")
                _write_csv(coordination_rows, output_dir / "coordination.partial.csv")
                _save_json(manifest_path, manifest)

    checkpoint_files_complete = all(
        (output_dir / algorithm / f"train_seed_{seed}" / "checkpoints" / f"model_{checkpoint}_steps.pt").is_file()
        for algorithm in ALGORITHMS
        for seed in protocol["train_seeds"]
        for checkpoint in protocol["checkpoints"]
    )
    training_artifacts_complete = all(
        (output_dir / algorithm / f"train_seed_{seed}" / filename).is_file()
        for algorithm in ALGORITHMS
        for seed in protocol["train_seeds"]
        for filename in ("settings.json", "training_progress.csv", "training_episodes.csv", "model.pt")
    )
    summary = summarize_rl(
        learned_rows, heuristic_rows, coordination_rows,
        settings=protocol,
        checkpoint_files_complete=checkpoint_files_complete,
        training_artifacts_complete=training_artifacts_complete,
    )
    summary["selected_cell"] = selected_id
    summary["screen_hypothesis_H1"] = screen_summary["selected_cell"] is not None
    _write_csv(learned_rows, output_dir / "learned_episodes.csv")
    _write_csv(coordination_rows, output_dir / "coordination.csv")
    _write_csv(decision_rows, output_dir / "decisions.csv")
    _save_json(output_dir / "summary.json", summary)
    manifest.update({
        "status": "COMPLETED" if summary["gate"]["passed"] else "FAILED",
        "finished_at": datetime.now(UTC).isoformat(),
        "screen_episode_count": len(screen_rows),
        "heuristic_episode_count": len(heuristic_rows),
        "learned_episode_count": len(learned_rows),
        "coordination_audit_count": len(coordination_rows),
        "decision_count": len(decision_rows),
        "gate": summary["gate"],
        "hypotheses": summary["hypotheses"],
        "outputs": sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()),
    })
    _save_json(manifest_path, manifest)
    if not summary["gate"]["passed"]:
        raise RuntimeError("Coordination-stress RL audit gate failed")
    print(json.dumps({"selected_cell": selected_id, "hypotheses": summary["hypotheses"], "gate": summary["gate"]}, indent=2, sort_keys=True))
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
