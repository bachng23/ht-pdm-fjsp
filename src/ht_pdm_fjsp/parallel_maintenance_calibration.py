"""Calibrate PM value and technician contention before changing MARL methods."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.parallel_maintenance import (
    FAILED,
    MAINTENANCE,
    ParallelMaintenanceConfig,
    ParallelMaintenanceEnv,
)


POLICIES = (
    "reactive_fibt",
    "preventive_fibt",
    "preventive_balanced",
    "independent_preventive",
)
SEALED_TEST_SEEDS = tuple(range(63_200, 63_300))


@dataclass(frozen=True)
class CalibrationCell:
    failure_penalty: float
    pm_duration_factor: float
    pm_risk_threshold: float

    @property
    def cell_id(self) -> str:
        failure = f"{self.failure_penalty:g}".replace(".", "p")
        duration = f"{self.pm_duration_factor:.2f}".replace(".", "p")
        threshold = f"{self.pm_risk_threshold:.3f}".replace(".", "p")
        return f"f{failure}_pm{duration}_risk{threshold}"


def cells_for_profile(profile: str) -> tuple[CalibrationCell, ...]:
    if profile == "smoke":
        return (
            CalibrationCell(18.0, 0.65, 0.020),
            CalibrationCell(36.0, 0.35, 0.005),
        )
    if profile == "full":
        return tuple(
            CalibrationCell(failure, duration, threshold)
            for failure in (18.0, 36.0, 54.0)
            for duration in (0.35, 0.50, 0.65)
            for threshold in (0.005, 0.020)
        )
    raise ValueError(f"Unknown profile: {profile}")


def seeds_for_profile(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(63_490, 63_493))
    if profile == "full":
        return tuple(range(63_400, 63_450))
    raise ValueError(f"Unknown profile: {profile}")


def build_cell_config(
    base: ParallelMaintenanceConfig, cell: CalibrationCell
) -> ParallelMaintenanceConfig:
    config = replace(
        base,
        maintenance=replace(
            base.maintenance,
            pm_duration_factor=cell.pm_duration_factor,
            pm_risk_threshold=cell.pm_risk_threshold,
        ),
        costs=replace(base.costs, failure=cell.failure_penalty),
    )
    config.validate()
    return config


def _maintenance_candidates(
    env: ParallelMaintenanceEnv, *, include_preventive: bool
) -> list[int]:
    candidates = []
    for machine in range(env.num_agents):
        if env.mode[machine] == MAINTENANCE:
            continue
        if env.mode[machine] == FAILED or (
            include_preventive
            and env.conditional_failure_probability(machine)
            >= env.config.maintenance.pm_risk_threshold
        ):
            candidates.append(machine)
    candidates.sort(
        key=lambda machine: (
            0 if env.mode[machine] == FAILED else 1,
            -env.wait[machine],
            -env.conditional_failure_probability(machine),
            machine,
        )
    )
    return candidates


def policy_actions(name: str, env: ParallelMaintenanceEnv) -> np.ndarray:
    include_preventive = name != "reactive_fibt"
    candidates = _maintenance_candidates(
        env, include_preventive=include_preventive
    )
    available = [
        technician
        for technician in range(env.technician_count)
        if env.technician_available(technician)
    ]
    actions = np.zeros(env.num_agents, dtype=np.int64)
    if name == "independent_preventive":
        for machine in candidates:
            if available:
                technician = min(
                    available,
                    key=lambda item: (env.expected_duration(machine, item), item),
                )
                actions[machine] = technician + 1
        return actions
    if name not in {
        "reactive_fibt",
        "preventive_fibt",
        "preventive_balanced",
    }:
        raise ValueError(f"Unknown policy: {name}")
    for machine in candidates:
        if not available:
            break
        if name == "preventive_balanced":
            technician = min(
                available,
                key=lambda item: (
                    env.technician_busy_time[item],
                    env.expected_duration(machine, item),
                    item,
                ),
            )
        else:
            technician = min(
                available,
                key=lambda item: (env.expected_duration(machine, item), item),
            )
        actions[machine] = technician + 1
        available.remove(technician)
    return actions


def demand_metrics(env: ParallelMaintenanceEnv) -> dict[str, float | int]:
    candidates = _maintenance_candidates(env, include_preventive=True)
    available = [
        technician
        for technician in range(env.technician_count)
        if env.technician_available(technician)
    ]
    demand = len(candidates)
    excess = max(0, demand - len(available))
    preferred: list[int] = []
    if available:
        for machine in candidates:
            preferred.append(
                min(
                    available,
                    key=lambda item: (env.expected_duration(machine, item), item),
                )
            )
    same_best_excess = sum(
        max(0, preferred.count(technician) - 1) for technician in set(preferred)
    )
    return {
        "eligible_demand": demand,
        "available_technicians": len(available),
        "excess_demand": excess,
        "same_best_technician_excess": same_best_excess,
    }


def evaluate_episode(
    config: ParallelMaintenanceConfig,
    cell: CalibrationCell,
    policy: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = ParallelMaintenanceEnv(config)
    env.reset(seed=seed)
    decisions: list[dict[str, Any]] = []
    eligible_steps = 0
    excess_steps = 0
    excess_total = 0
    same_best_steps = 0
    done = False
    while not done:
        demand = demand_metrics(env)
        eligible_steps += int(demand["eligible_demand"] > 0)
        excess_steps += int(demand["excess_demand"] > 0)
        excess_total += int(demand["excess_demand"])
        same_best_steps += int(demand["same_best_technician_excess"] > 0)
        actions = policy_actions(policy, env)
        _, _, done, _, info = env.step(actions)
        decisions.append(
            {
                "cell_id": cell.cell_id,
                "policy": policy,
                "seed": seed,
                "step": env.last_decision["step"],
                **demand,
                "actions": json.dumps(env.last_decision["actions"]),
                "accepted": json.dumps(env.last_decision["accepted"]),
                "proposal_conflicts": env.last_decision["proposal_conflicts"],
                "failures": env.last_decision["failures"],
                "reworks": env.last_decision["reworks"],
                "incremental_cost": env.last_decision["incremental_cost"],
                "reward": env.last_decision["reward"],
            }
        )
    utilization = info["technician_utilization"]
    episode = {
        "cell_id": cell.cell_id,
        "failure_penalty": cell.failure_penalty,
        "pm_duration_factor": cell.pm_duration_factor,
        "pm_risk_threshold": cell.pm_risk_threshold,
        "policy": policy,
        "seed": seed,
        "objective": info["objective"],
        "episode_return": info["episode_return"],
        "production": info["production"],
        "downtime": info["downtime"],
        "failures": info["failures"],
        "preventive": info["preventive"],
        "corrective": info["corrective"],
        "waiting": info["waiting"],
        "proposal_conflicts": info["proposal_conflicts"],
        "conflict_steps": info["conflict_steps"],
        "rework": info["rework"],
        "early_pm": info["early_pm"],
        "workload_imbalance": info["workload_imbalance"],
        "technician_0_utilization": utilization[0],
        "technician_1_utilization": utilization[1],
        "eligible_demand_steps": eligible_steps,
        "excess_demand_steps": excess_steps,
        "excess_demand_total": excess_total,
        "same_best_technician_steps": same_best_steps,
        "return_objective_error": info["return_objective_error"],
        "invalid_executions": info["invalid_executions"],
        "duplicate_machine_assignments": info["duplicate_machine_assignments"],
        "duplicate_technician_assignments": info[
            "duplicate_technician_assignments"
        ],
    }
    env.close()
    return episode, decisions


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def summarize(
    episode_rows: list[dict[str, Any]],
    *,
    cells: tuple[CalibrationCell, ...],
    seeds: tuple[int, ...],
    profile: str,
) -> dict[str, Any]:
    cell_summaries: dict[str, Any] = {}
    qualifying: list[CalibrationCell] = []
    for cell in cells:
        selected = [row for row in episode_rows if row["cell_id"] == cell.cell_id]
        by_policy = {
            policy: [row for row in selected if row["policy"] == policy]
            for policy in POLICIES
        }
        reactive_by_seed = {
            int(row["seed"]): row for row in by_policy["reactive_fibt"]
        }
        preventive = by_policy["preventive_fibt"]
        objective_deltas = [
            float(row["objective"])
            - float(reactive_by_seed[int(row["seed"])]["objective"])
            for row in preventive
        ]
        reactive_objective = _mean(by_policy["reactive_fibt"], "objective")
        objective_improvement = -statistics.fmean(objective_deltas) / max(
            abs(reactive_objective), 1e-9
        )
        reactive_failures = _mean(by_policy["reactive_fibt"], "failures")
        preventive_failures = _mean(preventive, "failures")
        failure_reduction = (reactive_failures - preventive_failures) / max(
            reactive_failures, 1e-9
        )
        independent_conflict_incidence = _mean(
            by_policy["independent_preventive"], "conflict_steps"
        ) / cell_config_horizon(selected)
        coordinated_conflicts = sum(
            int(row["proposal_conflicts"]) for row in preventive
        )
        checks = {
            "objective_improvement_at_least_5_percent": objective_improvement
            >= 0.05,
            "failure_reduction_at_least_20_percent": failure_reduction >= 0.20,
            "independent_conflict_incidence_in_2_to_20_percent": 0.02
            <= independent_conflict_incidence
            <= 0.20,
            "coordinated_preventive_has_zero_conflicts": coordinated_conflicts
            == 0,
        }
        if all(checks.values()):
            qualifying.append(cell)
        policy_metrics: dict[str, Any] = {}
        for policy, rows in by_policy.items():
            policy_metrics[policy] = {
                metric: _mean(rows, metric)
                for metric in (
                    "objective",
                    "production",
                    "downtime",
                    "failures",
                    "preventive",
                    "corrective",
                    "waiting",
                    "proposal_conflicts",
                    "conflict_steps",
                    "rework",
                    "workload_imbalance",
                    "eligible_demand_steps",
                    "excess_demand_steps",
                    "same_best_technician_steps",
                )
            }
        cell_summaries[cell.cell_id] = {
            "factors": asdict(cell),
            "paired_preventive_minus_reactive_objective": {
                "mean": statistics.fmean(objective_deltas),
                "median": statistics.median(objective_deltas),
                "preventive_win_rate": sum(delta < 0 for delta in objective_deltas)
                / len(objective_deltas),
                "relative_improvement": objective_improvement,
            },
            "failure_reduction": failure_reduction,
            "independent_conflict_step_incidence": independent_conflict_incidence,
            "policy_metrics": policy_metrics,
            "promotion_checks": checks,
            "qualifies": all(checks.values()),
        }

    def distance(cell: CalibrationCell) -> float:
        return (
            abs(cell.failure_penalty - 18.0) / 18.0
            + abs(cell.pm_duration_factor - 0.65) / 0.65
            + abs(cell.pm_risk_threshold - 0.020) / 0.020
        )

    selected_cell = None
    if qualifying:
        selected = min(
            qualifying,
            key=lambda cell: (
                distance(cell),
                -cell_summaries[cell.cell_id][
                    "paired_preventive_minus_reactive_objective"
                ]["relative_improvement"],
                cell.cell_id,
            ),
        )
        selected_cell = selected.cell_id
    expected = len(cells) * len(POLICIES) * len(seeds)
    audits = {
        "expected_episodes_completed": len(episode_rows) == expected,
        "reward_objective_identity": all(
            float(row["return_objective_error"]) <= 1e-6 for row in episode_rows
        ),
        "no_invalid_executions": all(
            int(row["invalid_executions"]) == 0 for row in episode_rows
        ),
        "no_duplicate_machine_assignments": all(
            int(row["duplicate_machine_assignments"]) == 0 for row in episode_rows
        ),
        "no_duplicate_technician_assignments": all(
            int(row["duplicate_technician_assignments"]) == 0
            for row in episode_rows
        ),
        "sealed_test_panel_remains_closed": not any(
            int(row["seed"]) in SEALED_TEST_SEEDS for row in episode_rows
        ),
    }
    return {
        "purpose": "Simulator calibration diagnostic, not an algorithm claim.",
        "profile": profile,
        "cell_summaries": cell_summaries,
        "qualifying_cells": sorted(cell.cell_id for cell in qualifying),
        "selected_candidate_cell": selected_cell,
        "hypotheses": {
            "H1_at_least_one_cell_has_5_percent_pm_benefit": any(
                summary["promotion_checks"][
                    "objective_improvement_at_least_5_percent"
                ]
                for summary in cell_summaries.values()
            ),
            "H2_at_least_one_cell_also_reduces_failures_20_percent": any(
                summary["promotion_checks"][
                    "objective_improvement_at_least_5_percent"
                ]
                and summary["promotion_checks"][
                    "failure_reduction_at_least_20_percent"
                ]
                for summary in cell_summaries.values()
            ),
            "H3_at_least_one_cell_meets_coordination_pressure_gate": any(
                summary["promotion_checks"][
                    "independent_conflict_incidence_in_2_to_20_percent"
                ]
                and summary["promotion_checks"][
                    "coordinated_preventive_has_zero_conflicts"
                ]
                for summary in cell_summaries.values()
            ),
        },
        "audits": audits,
        "gate": {"passed": all(audits.values())},
    }


def cell_config_horizon(rows: list[dict[str, Any]]) -> int:
    if not rows:
        raise ValueError("Cannot infer horizon from empty cell rows")
    # Every configured episode in this benchmark has the locked 168-step horizon.
    return 168


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = ParallelMaintenanceConfig.from_json(args.config)
    cells = cells_for_profile(args.profile)
    seeds = (
        tuple(parse_seeds(args.seeds))
        if args.seeds
        else seeds_for_profile(args.profile)
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "calibration_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "purpose": "parallel_maintenance_calibration_diagnostic",
        "profile": args.profile,
        "git_revision": _git_revision(),
        "started_at": datetime.now(UTC).isoformat(),
        "config": str(Path(args.config).resolve()),
        "cells": [asdict(cell) | {"cell_id": cell.cell_id} for cell in cells],
        "policies": list(POLICIES),
        "evaluation_seeds": list(seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "gymnasium": version("gymnasium"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "resolved_base_config.json").write_text(
        json.dumps(base.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "cells.json").write_text(
        json.dumps(manifest["cells"], indent=2, sort_keys=True) + "\n"
    )
    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    total = len(cells) * len(POLICIES) * len(seeds)
    progress = tqdm(total=total, desc="Calibration evaluation", unit="episode")
    try:
        for cell in cells:
            config = build_cell_config(base, cell)
            for policy in POLICIES:
                for seed in seeds:
                    episode, episode_decisions = evaluate_episode(
                        config, cell, policy, seed
                    )
                    episodes.append(episode)
                    decisions.extend(episode_decisions)
                    progress.update(1)
                _write_csv(episodes, output_dir / "episodes.partial.csv")
        progress.close()
        summary = summarize(
            episodes, cells=cells, seeds=seeds, profile=args.profile
        )
        _write_csv(episodes, output_dir / "episodes.csv")
        _write_csv(decisions, output_dir / "decisions.csv")
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        manifest.update(
            {
                "status": "COMPLETED" if summary["gate"]["passed"] else "FAILED",
                "finished_at": datetime.now(UTC).isoformat(),
                "episode_count": len(episodes),
                "decision_count": len(decisions),
                "audit_gate": summary["gate"],
                "selected_candidate_cell": summary["selected_candidate_cell"],
                "outputs": sorted(
                    str(path.relative_to(output_dir))
                    for path in output_dir.rglob("*")
                    if path.is_file()
                ),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if not summary["gate"]["passed"]:
            raise RuntimeError("Calibration audit gate failed")
        return summary
    except Exception:
        progress.close()
        manifest.update(
            {"status": "FAILED", "finished_at": datetime.now(UTC).isoformat()}
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/parallel_maintenance.json")
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
