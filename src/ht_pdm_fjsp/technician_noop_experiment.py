"""Diagnose technician conflicts after replacing forced maintenance with safe no-op."""

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
from statistics import fmean, stdev
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.simulator import audit_result
from ht_pdm_fjsp.technician_conflict_experiment import (
    _describe,
    _student_t_critical_975,
    choose_joint_actions,
)


LEGACY_018 = "legacy_threshold_018_joint_spt"
NOOP_REACTIVE = "noop_reactive_joint_spt"
NOOP_018 = "noop_threshold_018_joint_spt"
NOOP_005 = "noop_threshold_005_joint_spt"
CONDITIONS: dict[str, dict[str, Any]] = {
    LEGACY_018: {"threshold": 0.18, "wait_policy": "legacy"},
    NOOP_REACTIVE: {"threshold": None, "wait_policy": "safe_noop"},
    NOOP_018: {"threshold": 0.18, "wait_policy": "safe_noop"},
    NOOP_005: {"threshold": 0.05, "wait_policy": "safe_noop"},
}

CONFLICT_SOURCE_KEYS = (
    "forced_technician_conflicts",
    "preventive_preventive_conflicts",
    "corrective_corrective_conflicts",
    "preventive_corrective_conflicts",
    "other_technician_conflicts",
)


def default_seeds(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(61_410, 61_415))
    if profile == "full":
        return tuple(range(61_500, 61_700))
    raise ValueError(f"Unknown profile: {profile}")


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


def _reason_class(reason: str) -> str:
    if reason == "forced_preventive":
        return "forced"
    if reason == "threshold_preventive":
        return "preventive"
    if reason == "corrective":
        return "corrective"
    return "other"


def _conflict_sources(
    env: MachineAgentsCTDEEnv, actions: np.ndarray, reasons: list[str]
) -> dict[str, int]:
    """Classify every rejected technician proposal using resolver order."""

    priority_start = env.coordination_totals["joint_steps"] % env.num_agents
    proposed: list[tuple[int, str, str]] = []
    for agent_index, local_action in enumerate(actions):
        global_action = env.local_to_global(agent_index, int(local_action))
        if global_action is None:
            continue
        descriptor = env.core.actions[global_action]
        if descriptor.kind in {"preventive", "corrective"}:
            proposed.append(
                (
                    agent_index,
                    str(descriptor.technician_id),
                    _reason_class(reasons[agent_index]),
                )
            )
    proposed.sort(key=lambda item: (item[0] - priority_start) % env.num_agents)
    accepted_reason: dict[str, str] = {}
    output = {key: 0 for key in CONFLICT_SOURCE_KEYS}
    for _, technician_id, reason in proposed:
        if technician_id not in accepted_reason:
            accepted_reason[technician_id] = reason
            continue
        pair = {accepted_reason[technician_id], reason}
        if "forced" in pair:
            key = "forced_technician_conflicts"
        elif pair == {"preventive"}:
            key = "preventive_preventive_conflicts"
        elif pair == {"corrective"}:
            key = "corrective_corrective_conflicts"
        elif pair == {"preventive", "corrective"}:
            key = "preventive_corrective_conflicts"
        else:
            key = "other_technician_conflicts"
        output[key] += 1
    return output


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
        **{key: 0 for key in CONFLICT_SOURCE_KEYS},
    }


def evaluate_episode(
    config: BenchmarkConfig,
    *,
    condition: str,
    seed: int,
    threshold: float | None,
    wait_policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    env = MachineAgentsCTDEEnv(config, wait_policy=wait_policy)
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
        proposed = _proposal_stats(env, actions, reasons)
        sources = _conflict_sources(env, actions, reasons)
        simulation_time = env.core.now
        observation, reward, _, _, info = env.step(actions)
        resolution = info["coordination"]
        if proposed["duplicate_technician_proposals"] != resolution[
            "technician_conflicts"
        ]:
            raise AssertionError("Technician proposal accounting mismatch")
        if sum(sources.values()) != resolution["technician_conflicts"]:
            raise AssertionError("Technician conflict-source accounting mismatch")
        for key, value in {**resolution, **proposed, **sources}.items():
            totals[key] += int(value)
        decision_rows.append(
            {
                "condition": condition,
                "wait_policy": wait_policy,
                "seed": seed,
                "joint_step": joint_step,
                "simulation_time": simulation_time,
                **proposed,
                **sources,
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
    valid_conflicts = totals["technician_conflicts"] - totals[
        "forced_technician_conflicts"
    ]
    shared = {
        "condition": condition,
        "wait_policy": wait_policy,
        "seed": seed,
        **totals,
        "maintenance_proposals": maintenance_proposals,
        "valid_technician_conflicts": valid_conflicts,
    }
    episode_row = {
        **shared,
        "threshold": "" if threshold is None else threshold,
        "episode_return": episode_return,
        **result.metrics,
        "valid_conflict_rate": (
            valid_conflicts / maintenance_proposals if maintenance_proposals else 0.0
        ),
        "rejection_rate": (
            totals["rejected"] / totals["proposals"] if totals["proposals"] else 0.0
        ),
    }
    coordination_row = shared
    env.close()
    return episode_row, decision_rows, coordination_row


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
        "valid_technician_conflicts",
        "forced_preventive_proposals",
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
    episode_rows: list[dict[str, Any]], coordination_rows: list[dict[str, Any]]
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
        valid_conflicts = totals["technician_conflicts"] - totals[
            "forced_technician_conflicts"
        ]
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
            "conflict_sources_sum_to_total": sum(
                totals[key] for key in CONFLICT_SOURCE_KEYS
            )
            == totals["technician_conflicts"],
        }
        audit_checks[condition] = audit
        per_condition[condition] = {
            "episodes": len(episodes),
            "episodes_with_valid_technician_conflict": sum(
                int(row["valid_technician_conflicts"]) > 0 for row in episodes
            ),
            "totals": totals,
            "maintenance_proposals": maintenance_proposals,
            "valid_technician_conflicts": valid_conflicts,
            "valid_conflict_rate": (
                valid_conflicts / maintenance_proposals
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
                    "valid_technician_conflicts",
                    "production_conflicts",
                    "maintenance_proposals",
                )
            },
            "audit": audit,
        }

    intended = per_condition[NOOP_018]
    noop_conditions = [per_condition[name] for name in CONDITIONS if name != LEGACY_018]
    seed_count = intended["episodes"]
    gate_checks = {
        "all_expected_episodes_completed": len(episode_rows) == 4 * seed_count,
        "all_feasibility_and_accounting_audits_passed": all(
            all(checks.values()) for checks in audit_checks.values()
        ),
        "safe_noop_has_zero_forced_preventive_proposals": all(
            item["totals"]["forced_preventive_proposals"] == 0
            for item in noop_conditions
        ),
        "safe_noop_has_zero_forced_technician_conflicts": all(
            item["totals"]["forced_technician_conflicts"] == 0
            for item in noop_conditions
        ),
        "intended_valid_conflicts_in_at_least_ten_percent_of_seeds": intended[
            "episodes_with_valid_technician_conflict"
        ]
        >= math.ceil(0.10 * seed_count),
        "intended_valid_conflict_rate_at_least_one_percent": intended[
            "valid_conflict_rate"
        ]
        >= 0.01,
        "intended_rejected_proposal_rate_at_most_twenty_percent": intended[
            "rejected_proposal_rate"
        ]
        <= 0.20,
    }
    return {
        "primary_endpoint": (
            "valid joint technician conflicts under safe-noop threshold 0.18"
        ),
        "per_condition": per_condition,
        "paired_contrasts": {
            "noop_018_minus_legacy_018": _paired_contrast(
                episode_rows, NOOP_018, LEGACY_018
            ),
            "noop_018_minus_noop_reactive": _paired_contrast(
                episode_rows, NOOP_018, NOOP_REACTIVE
            ),
            "noop_005_minus_noop_018": _paired_contrast(
                episode_rows, NOOP_005, NOOP_018
            ),
        },
        "gate": {
            "checks": gate_checks,
            "supports_clean_marl_coordination_experiment": all(
                gate_checks.values()
            ),
            "note": "Development mechanism gate; not an algorithm-performance claim.",
        },
    }


def _write_partial(
    output_dir: Path,
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    coordination_rows: list[dict[str, Any]],
) -> None:
    _write_csv(episode_rows, output_dir / "technician_noop_episodes.partial.csv")
    _write_csv(decision_rows, output_dir / "technician_noop_decisions.partial.csv")
    _write_csv(
        coordination_rows, output_dir / "technician_noop_coordination.partial.csv"
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
        raise ValueError(
            "This diagnostic requires the frozen Brandimarte MK01 extension"
        )
    seeds = (
        tuple(parse_seeds(args.seeds))
        if args.seeds
        else default_seeds(args.profile)
    )
    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    manifest_path = output_dir / "technician_noop_manifest.json"
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
    with tqdm(
        total=total, desc="Safe-noop conflict diagnostic", unit="episode"
    ) as progress:
        for condition, settings in CONDITIONS.items():
            for seed in seeds:
                progress.set_postfix(condition=condition, seed=seed)
                episode, decisions, coordination = evaluate_episode(
                    config,
                    condition=condition,
                    seed=seed,
                    threshold=settings["threshold"],
                    wait_policy=settings["wait_policy"],
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
    _write_csv(episode_rows, output_dir / "technician_noop_episodes.csv")
    _write_csv(decision_rows, output_dir / "technician_noop_decisions.csv")
    _write_csv(coordination_rows, output_dir / "technician_noop_coordination.csv")
    (output_dir / "technician_noop_summary.json").write_text(
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
    parser.add_argument("--config", default="configs/brandimarte_mk01_ht_pdm.json")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    output_dir = run(build_parser().parse_args())
    summary = json.loads((output_dir / "technician_noop_summary.json").read_text())
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
