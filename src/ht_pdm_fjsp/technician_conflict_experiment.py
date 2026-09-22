"""Measure joint technician conflicts on the extended Brandimarte MK01 instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from statistics import NormalDist, fmean, pstdev, stdev
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.simulator import audit_result


REACTIVE = "reactive_joint_spt"
THRESHOLD_018 = "threshold_018_joint_spt"
THRESHOLD_005 = "threshold_005_joint_spt"
CONDITIONS: dict[str, float | None] = {
    REACTIVE: None,
    THRESHOLD_018: 0.18,
    THRESHOLD_005: 0.05,
}


def default_seeds(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(61_400, 61_405))
    if profile == "full":
        return tuple(range(61_500, 61_700))
    raise ValueError(f"Unknown profile: {profile}")


def _valid_descriptors(
    env: MachineAgentsCTDEEnv,
    observation: dict[str, np.ndarray],
    agent_index: int,
) -> list[tuple[int, Any]]:
    return [
        (local_action, env.core.actions[global_action])
        for local_action, global_action in enumerate(
            env.local_action_catalogs[agent_index]
        )
        if global_action is not None
        and bool(observation["action_masks"][agent_index, local_action])
    ]


def choose_joint_actions(
    env: MachineAgentsCTDEEnv,
    observation: dict[str, np.ndarray],
    *,
    preventive_threshold: float | None,
    reasons: list[str] | None = None,
) -> np.ndarray:
    """Choose independently with corrective priority and local SPT routing."""

    actions: list[int] = []
    for agent_index in range(env.num_agents):
        valid = _valid_descriptors(env, observation, agent_index)
        corrective = [item for item in valid if item[1].kind == "corrective"]
        if corrective:
            choice = min(
                corrective,
                key=lambda item: (
                    env.technician_duration(item[1]),
                    str(item[1].technician_id),
                ),
            )
            actions.append(choice[0])
            if reasons is not None:
                reasons.append("corrective")
            continue

        production = [item for item in valid if item[1].kind == "production"]
        selected_production: tuple[int, Any] | None = None
        if production:
            selected_production = min(
                production,
                key=lambda item: (
                    env.core._alternative(item[1]).processing_time,
                    str(item[1].job_id),
                    int(item[1].operation_index),
                ),
            )

        preventive = [item for item in valid if item[1].kind == "preventive"]
        if (
            preventive_threshold is not None
            and selected_production is not None
            and preventive
            and env.core.failure_probability(selected_production[1])
            >= preventive_threshold
        ):
            choice = min(
                preventive,
                key=lambda item: (
                    env.technician_duration(item[1]),
                    str(item[1].technician_id),
                ),
            )
            actions.append(choice[0])
            if reasons is not None:
                reasons.append("threshold_preventive")
        elif selected_production is not None:
            actions.append(selected_production[0])
            if reasons is not None:
                reasons.append("production")
        elif bool(observation["action_masks"][agent_index, 0]):
            actions.append(0)
            if reasons is not None:
                reasons.append("wait")
        elif preventive:
            choice = min(
                preventive,
                key=lambda item: (
                    env.technician_duration(item[1]),
                    str(item[1].technician_id),
                ),
            )
            actions.append(choice[0])
            if reasons is not None:
                reasons.append("forced_preventive")
        else:
            valid_indices = np.flatnonzero(observation["action_masks"][agent_index])
            if not len(valid_indices):
                raise AssertionError("Machine agent has no valid action")
            actions.append(int(valid_indices[0]))
            if reasons is not None:
                reasons.append("fallback")
    return np.asarray(actions, dtype=np.int64)


def _proposal_stats(
    env: MachineAgentsCTDEEnv, actions: np.ndarray, reasons: list[str]
) -> dict[str, int]:
    kinds: Counter[str] = Counter()
    technicians: Counter[str] = Counter()
    for agent_index, local_action in enumerate(actions):
        global_action = env.local_to_global(agent_index, int(local_action))
        if global_action is None:
            kinds["wait"] += 1
            continue
        descriptor = env.core.actions[global_action]
        kinds[descriptor.kind] += 1
        if descriptor.technician_id is not None:
            technicians[str(descriptor.technician_id)] += 1
    return {
        "proposed_production": kinds["production"],
        "proposed_preventive": kinds["preventive"],
        "proposed_corrective": kinds["corrective"],
        "proposed_wait": kinds["wait"],
        "t1_proposals": technicians["T1"],
        "t2_proposals": technicians["T2"],
        "contested_technicians": sum(count > 1 for count in technicians.values()),
        "duplicate_technician_proposals": sum(
            max(0, count - 1) for count in technicians.values()
        ),
        "threshold_preventive_proposals": reasons.count("threshold_preventive"),
        "forced_preventive_proposals": reasons.count("forced_preventive"),
    }


def _zero_totals() -> dict[str, int]:
    return {
        "joint_steps": 0,
        "proposals": 0,
        "accepted": 0,
        "waits": 0,
        "production_conflicts": 0,
        "technician_conflicts": 0,
        "rejected": 0,
        "invalid_executions": 0,
        "duplicate_operation_executions": 0,
        "duplicate_technician_executions": 0,
        "proposed_production": 0,
        "proposed_preventive": 0,
        "proposed_corrective": 0,
        "proposed_wait": 0,
        "t1_proposals": 0,
        "t2_proposals": 0,
        "contested_technicians": 0,
        "duplicate_technician_proposals": 0,
        "threshold_preventive_proposals": 0,
        "forced_preventive_proposals": 0,
    }


def evaluate_episode(
    config: BenchmarkConfig,
    *,
    condition: str,
    seed: int,
    threshold: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    env = MachineAgentsCTDEEnv(config)
    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    decision_rows: list[dict[str, Any]] = []
    totals = _zero_totals()
    joint_step = 0
    while not env._done:
        reasons: list[str] = []
        actions = choose_joint_actions(
            env,
            observation,
            preventive_threshold=threshold,
            reasons=reasons,
        )
        if len(reasons) != env.num_agents:
            raise AssertionError("Missing independent-action reason label")
        proposed = _proposal_stats(env, actions, reasons)
        simulation_time = env.core.now
        observation, reward, _, _, info = env.step(actions)
        resolution = info["coordination"]
        if (
            proposed["duplicate_technician_proposals"]
            != resolution["technician_conflicts"]
        ):
            raise AssertionError("Technician proposal accounting mismatch")
        for key, value in {**resolution, **proposed}.items():
            totals[key] += int(value)
        decision_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "joint_step": joint_step,
                "simulation_time": simulation_time,
                **proposed,
                **resolution,
            }
        )
        episode_return += float(reward)
        joint_step += 1

    result = env.result(condition)
    audit_result(config, result)
    if not np.isclose(episode_return, -float(result.metrics["objective"])):
        raise AssertionError("Episode return does not equal negative objective")
    if totals["technician_conflicts"] != env.coordination_totals[
        "technician_conflicts"
    ]:
        raise AssertionError("Episode coordination totals mismatch")
    maintenance_proposals = (
        totals["proposed_preventive"] + totals["proposed_corrective"]
    )
    episode_row = {
        "condition": condition,
        "threshold": "" if threshold is None else threshold,
        "seed": seed,
        "episode_return": episode_return,
        **result.metrics,
        **totals,
        "maintenance_proposals": maintenance_proposals,
        "technician_conflict_rate": (
            totals["technician_conflicts"] / maintenance_proposals
            if maintenance_proposals
            else 0.0
        ),
        "rejection_rate": (
            totals["rejected"] / totals["proposals"]
            if totals["proposals"]
            else 0.0
        ),
    }
    coordination_row = {
        "condition": condition,
        "seed": seed,
        **totals,
        "maintenance_proposals": maintenance_proposals,
    }
    env.close()
    return episode_row, decision_rows, coordination_row


def _describe(values: Iterable[float]) -> dict[str, float]:
    data = list(values)
    return {
        "mean": fmean(data),
        "std": pstdev(data),
        "min": min(data),
        "max": max(data),
    }


def _student_t_critical_975(degrees_of_freedom: int) -> float:
    """Return a two-sided 95% critical value without adding SciPy."""

    if degrees_of_freedom < 1:
        raise ValueError("A paired interval requires at least two seeds")
    table = (
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    if degrees_of_freedom <= len(table):
        return table[degrees_of_freedom - 1]
    z = NormalDist().inv_cdf(0.975)
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def _paired_contrast(
    episode_rows: list[dict[str, Any]], treatment: str, control: str
) -> dict[str, Any]:
    metrics = (
        "objective",
        "makespan",
        "failures",
        "preventive_maintenance",
        "maintenance_proposals",
        "technician_conflicts",
        "rejection_rate",
    )
    treatment_rows = {
        int(row["seed"]): row
        for row in episode_rows
        if row["condition"] == treatment
    }
    control_rows = {
        int(row["seed"]): row
        for row in episode_rows
        if row["condition"] == control
    }
    if treatment_rows.keys() != control_rows.keys():
        raise AssertionError("Paired conditions do not have identical seed panels")
    seeds = sorted(treatment_rows)
    if len(seeds) < 2:
        raise ValueError("Paired contrasts require at least two seeds")
    critical = _student_t_critical_975(len(seeds) - 1)
    output: dict[str, Any] = {}
    for metric in metrics:
        deltas = [
            float(treatment_rows[seed][metric])
            - float(control_rows[seed][metric])
            for seed in seeds
        ]
        mean = fmean(deltas)
        standard_deviation = stdev(deltas)
        half_width = critical * standard_deviation / math.sqrt(len(deltas))
        output[metric] = {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "lower_95": mean - half_width,
            "upper_95": mean + half_width,
            "negative_seeds": sum(delta < 0 for delta in deltas),
            "zero_seeds": sum(delta == 0 for delta in deltas),
            "positive_seeds": sum(delta > 0 for delta in deltas),
        }
    return {
        "treatment": treatment,
        "control": control,
        "seed_count": len(seeds),
        "degrees_of_freedom": len(seeds) - 1,
        "critical_value": critical,
        "metrics": output,
    }


def summarize(
    episode_rows: list[dict[str, Any]],
    coordination_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    per_condition: dict[str, Any] = {}
    audit_checks: dict[str, Any] = {}
    for condition in CONDITIONS:
        episodes = [row for row in episode_rows if row["condition"] == condition]
        coordination = [
            row for row in coordination_rows if row["condition"] == condition
        ]
        totals = {
            key: sum(int(row[key]) for row in coordination)
            for key in _zero_totals()
        }
        maintenance_proposals = sum(
            int(row["maintenance_proposals"]) for row in coordination
        )
        audit = {
            "zero_invalid_executions": totals["invalid_executions"] == 0,
            "zero_duplicate_operation_executions": totals[
                "duplicate_operation_executions"
            ]
            == 0,
            "zero_duplicate_technician_executions": totals[
                "duplicate_technician_executions"
            ]
            == 0,
        }
        audit_checks[condition] = audit
        per_condition[condition] = {
            "episodes": len(episodes),
            "episodes_with_technician_conflict": sum(
                int(row["technician_conflicts"]) > 0 for row in episodes
            ),
            "totals": totals,
            "maintenance_proposals": maintenance_proposals,
            "technician_conflict_rate": (
                totals["technician_conflicts"] / maintenance_proposals
                if maintenance_proposals
                else 0.0
            ),
            "rejected_proposal_rate": (
                totals["rejected"] / totals["proposals"]
                if totals["proposals"]
                else 0.0
            ),
            "metrics": {
                metric: _describe(float(row[metric]) for row in episodes)
                for metric in (
                    "objective",
                    "makespan",
                    "failures",
                    "preventive_maintenance",
                    "corrective_maintenance",
                    "technician_conflicts",
                    "production_conflicts",
                    "maintenance_proposals",
                )
            },
            "audit": audit,
        }

    intended = per_condition[THRESHOLD_018]
    stress = per_condition[THRESHOLD_005]
    seed_count = intended["episodes"]
    gate_checks = {
        "all_expected_episodes_completed": len(episode_rows) == 3 * seed_count,
        "all_feasibility_audits_passed": all(
            all(checks.values()) for checks in audit_checks.values()
        ),
        "intended_conflicts_in_at_least_ten_percent_of_seeds": intended[
            "episodes_with_technician_conflict"
        ]
        >= math.ceil(0.10 * seed_count),
        "intended_conflict_rate_at_least_one_percent": intended[
            "technician_conflict_rate"
        ]
        >= 0.01,
        "intended_rejected_proposal_rate_at_most_twenty_percent": intended[
            "rejected_proposal_rate"
        ]
        <= 0.20,
        "stress_has_at_least_intended_maintenance_proposals": stress[
            "maintenance_proposals"
        ]
        >= intended["maintenance_proposals"],
        "stress_has_at_least_intended_technician_conflicts": stress["totals"][
            "technician_conflicts"
        ]
        >= intended["totals"]["technician_conflicts"],
    }
    return {
        "primary_endpoint": "joint technician conflicts under threshold 0.18",
        "per_condition": per_condition,
        "paired_contrasts": {
            "threshold_018_minus_reactive": _paired_contrast(
                episode_rows, THRESHOLD_018, REACTIVE
            ),
            "threshold_005_minus_threshold_018": _paired_contrast(
                episode_rows, THRESHOLD_005, THRESHOLD_018
            ),
        },
        "gate": {
            "checks": gate_checks,
            "supports_marl_coordination_experiment": all(gate_checks.values()),
            "note": "Development mechanism gate; not an algorithm-performance claim.",
        },
    }


def _write_partial(
    output_dir: Path,
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    coordination_rows: list[dict[str, Any]],
) -> None:
    _write_csv(episode_rows, output_dir / "technician_conflict_episodes.partial.csv")
    _write_csv(
        decision_rows, output_dir / "technician_conflict_decisions.partial.csv"
    )
    _write_csv(
        coordination_rows,
        output_dir / "technician_conflict_coordination.partial.csv",
    )


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    raw_config = config_path.read_bytes()
    config = BenchmarkConfig.from_json(config_path)
    if config.instance_name != "ht_pdm_fjsp_brandimarte_mk01_v1":
        raise ValueError("This pilot requires the frozen Brandimarte MK01 extension")
    seeds = tuple(parse_seeds(args.seeds)) if args.seeds else default_seeds(args.profile)
    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    manifest_path = output_dir / "technician_conflict_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "profile": args.profile,
        "git_commit": _git_revision(),
        "instance_name": config.instance_name,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "conditions": CONDITIONS,
        "seeds": list(seeds),
        "reserved_future_seeds": list(range(62_000, 62_100)),
        "future_test_panel_opened": False,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    coordination_rows: list[dict[str, Any]] = []
    total = len(CONDITIONS) * len(seeds)
    with tqdm(total=total, desc="Joint technician-conflict pilot", unit="episode") as progress:
        for condition, threshold in CONDITIONS.items():
            for seed in seeds:
                progress.set_postfix(condition=condition, seed=seed)
                episode, decisions, coordination = evaluate_episode(
                    config,
                    condition=condition,
                    seed=seed,
                    threshold=threshold,
                )
                episode_rows.append(episode)
                decision_rows.extend(decisions)
                coordination_rows.append(coordination)
                if len(episode_rows) % 10 == 0 or len(episode_rows) == total:
                    _write_partial(
                        output_dir, episode_rows, decision_rows, coordination_rows
                    )
                progress.update(1)

    summary = summarize(episode_rows, coordination_rows)
    _write_csv(episode_rows, output_dir / "technician_conflict_episodes.csv")
    _write_csv(decision_rows, output_dir / "technician_conflict_decisions.csv")
    _write_csv(
        coordination_rows, output_dir / "technician_conflict_coordination.csv"
    )
    (output_dir / "technician_conflict_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        {
            "status": "COMPLETED",
            "episode_count": len(episode_rows),
            "decision_count": len(decision_rows),
            "gate": summary["gate"],
            "output_files": sorted(path.name for path in output_dir.iterdir()),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--config", default="configs/brandimarte_mk01_ht_pdm.json"
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    output_dir = run(build_parser().parse_args())
    summary = json.loads(
        (output_dir / "technician_conflict_summary.json").read_text()
    )
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
