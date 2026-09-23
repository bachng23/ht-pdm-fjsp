"""Calibrate preventive-maintenance release synchronization for technician conflict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping

from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.technician_conflict_factorial import (
    POLICIES,
    UNAVAILABLE_INTERVALS,
    WINDOW_RELEASES,
    FactorSpec,
    LiteratureFactorialSimulator,
    _describe,
    _t_critical_975,
    build_condition_config,
)


SPEC = FactorSpec(True, True, True, True, False)
MACHINE_ORDER = ("M1", "M2", "M3", "M4", "M5", "M6")
ROUND_RELEASES = (12.0, 36.0)
WINDOW_WIDTH = 8.0


@dataclass(frozen=True)
class WindowCondition:
    name: str
    group_size: int
    inter_wave_gap: float | None
    releases: Mapping[str, tuple[float, ...]]
    reference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "inter_wave_gap": self.inter_wave_gap,
            "window_width": WINDOW_WIDTH,
            "releases": {key: list(value) for key, value in self.releases.items()},
            "reference": self.reference,
        }


def grouped_releases(group_size: int, gap: float) -> dict[str, tuple[float, ...]]:
    if group_size not in (1, 2, 3, 6):
        raise ValueError("group_size must be one of 1, 2, 3, or 6")
    if gap < 0.0:
        raise ValueError("gap must be non-negative")
    return {
        machine_id: tuple(
            base + (machine_index // group_size) * gap for base in ROUND_RELEASES
        )
        for machine_index, machine_id in enumerate(MACHINE_ORDER)
    }


def window_conditions() -> tuple[WindowCondition, ...]:
    output = [
        WindowCondition(
            "parent_reference",
            1,
            None,
            {key: tuple(value) for key, value in WINDOW_RELEASES.items()},
            True,
        )
    ]
    for group_size in (1, 2, 3):
        for gap in (0.5, 1.0, 2.0):
            gap_code = str(gap).replace(".", "p")
            output.append(
                WindowCondition(
                    f"group{group_size}_gap{gap_code}",
                    group_size,
                    gap,
                    grouped_releases(group_size, gap),
                )
            )
    output.append(WindowCondition("group6_sync", 6, 0.0, grouped_releases(6, 0.0)))
    return tuple(output)


def default_seeds(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(62_300, 62_303))
    if profile == "full":
        return tuple(range(62_400, 62_500))
    raise ValueError(f"Unknown profile: {profile}")


def _paired_interval(
    rows: list[dict[str, Any]], condition: str, metric: str
) -> dict[str, float]:
    greedy = {
        int(row["seed"]): float(row[metric])
        for row in rows
        if row["condition"] == condition and row["policy"] == POLICIES[0]
    }
    matcher = {
        int(row["seed"]): float(row[metric])
        for row in rows
        if row["condition"] == condition and row["policy"] == POLICIES[1]
    }
    if greedy.keys() != matcher.keys():
        raise AssertionError("Policy rows do not share identical seeds")
    deltas = [greedy[seed] - matcher[seed] for seed in sorted(greedy)]
    mean = fmean(deltas)
    half = _t_critical_975(len(deltas) - 1) * stdev(deltas) / math.sqrt(len(deltas))
    return {"mean": mean, "lower_95": mean - half, "upper_95": mean + half}


SUMMARY_METRICS = (
    "technician_conflicts",
    "technician_conflict_rate",
    "avoidable_conflict_fraction",
    "maintenance_request_onsets",
    "capacity_unavoidable_wait_units",
    "busy_or_offshift_claims",
    "maintenance_wait_time",
    "window_deadline_misses",
    "production_conflicts",
    "rejection_rate",
    "objective",
    "makespan",
    "failures",
    "preventive_maintenance",
    "corrective_maintenance",
)


def summarize(
    rows: list[dict[str, Any]], conditions: tuple[WindowCondition, ...]
) -> dict[str, Any]:
    per_condition: dict[str, Any] = {}
    candidates: list[tuple[float, float, float, int, float, str]] = []
    for condition in conditions:
        policy_summaries: dict[str, Any] = {}
        for policy in POLICIES:
            episodes = [
                row
                for row in rows
                if row["condition"] == condition.name and row["policy"] == policy
            ]
            totals = {
                metric: sum(float(row[metric]) for row in episodes)
                for metric in (
                    "technician_conflicts",
                    "maintenance_proposals",
                    "avoidable_conflict_units",
                    "maintenance_request_onsets",
                    "capacity_unavoidable_wait_units",
                    "busy_or_offshift_claims",
                )
            }
            policy_summaries[policy] = {
                "episodes": len(episodes),
                "episode_conflict_incidence": fmean(
                    float(row["technician_conflicts"] > 0) for row in episodes
                ),
                "aggregate_conflict_rate": (
                    totals["technician_conflicts"] / totals["maintenance_proposals"]
                    if totals["maintenance_proposals"]
                    else 0.0
                ),
                "aggregate_avoidable_fraction": (
                    min(
                        1.0,
                        totals["avoidable_conflict_units"]
                        / totals["technician_conflicts"],
                    )
                    if totals["technician_conflicts"]
                    else 0.0
                ),
                "capacity_wait_per_request_onset": (
                    totals["capacity_unavoidable_wait_units"]
                    / totals["maintenance_request_onsets"]
                    if totals["maintenance_request_onsets"]
                    else 0.0
                ),
                "busy_claim_rate": (
                    totals["busy_or_offshift_claims"]
                    / totals["maintenance_proposals"]
                    if totals["maintenance_proposals"]
                    else 0.0
                ),
                "metrics": {
                    metric: _describe(float(row[metric]) for row in episodes)
                    for metric in SUMMARY_METRICS
                },
                "audit": {
                    "all_feasibility_audits_passed": all(
                        int(row["feasibility_audit_passed"]) == 1 for row in episodes
                    ),
                    "zero_invalid_executions": all(
                        int(row["invalid_executions"]) == 0 for row in episodes
                    ),
                    "zero_duplicate_operation_executions": all(
                        int(row["duplicate_operation_executions"]) == 0
                        for row in episodes
                    ),
                    "zero_duplicate_technician_executions": all(
                        int(row["duplicate_technician_executions"]) == 0
                        for row in episodes
                    ),
                },
            }
        greedy = policy_summaries[POLICIES[0]]
        matcher = policy_summaries[POLICIES[1]]
        greedy_mean = greedy["metrics"]["technician_conflicts"]["mean"]
        matcher_mean = matcher["metrics"]["technician_conflicts"]["mean"]
        conflict_reduction = 1.0 - matcher_mean / greedy_mean if greedy_mean else 0.0
        checks = {
            "incidence_between_20_and_40_percent": (
                0.20 <= greedy["episode_conflict_incidence"] <= 0.40
            ),
            "conflict_rate_between_2_and_6_percent": (
                0.02 <= greedy["aggregate_conflict_rate"] <= 0.06
            ),
            "avoidable_fraction_at_least_70_percent": (
                greedy["aggregate_avoidable_fraction"] >= 0.70
            ),
            "matcher_reduction_at_least_50_percent": conflict_reduction >= 0.50,
            "capacity_wait_per_onset_at_most_0p35": (
                greedy["capacity_wait_per_request_onset"] <= 0.35
            ),
            "busy_claim_rate_at_most_35_percent": greedy["busy_claim_rate"] <= 0.35,
            "all_audits_pass": all(
                all(audit.values())
                for audit in (greedy["audit"], matcher["audit"])
            ),
        }
        qualifies = not condition.reference and all(checks.values())
        if qualifies:
            candidates.append(
                (
                    abs(greedy["episode_conflict_incidence"] - 0.30),
                    -greedy["aggregate_avoidable_fraction"],
                    greedy["capacity_wait_per_request_onset"],
                    condition.group_size,
                    float(condition.inter_wave_gap),
                    condition.name,
                )
            )
        per_condition[condition.name] = {
            "schedule": condition.to_dict(),
            "selection_eligible": not condition.reference,
            "policies": policy_summaries,
            "greedy_minus_matcher": {
                metric: _paired_interval(rows, condition.name, metric)
                for metric in (
                    "technician_conflicts",
                    "maintenance_wait_time",
                    "objective",
                    "makespan",
                )
            },
            "conflict_reduction_fraction": conflict_reduction,
            "selection_checks": checks,
            "qualifies": qualifies,
        }
    selected = min(candidates)[-1] if candidates else None
    expected = len(rows) // (len(conditions) * len(POLICIES))
    global_checks = {
        "all_expected_episodes_completed": (
            len(rows) == len(conditions) * len(POLICIES) * expected
        ),
        "all_feasibility_audits_passed": all(
            int(row["feasibility_audit_passed"]) == 1 for row in rows
        ),
        "zero_invalid_executions": all(
            int(row["invalid_executions"]) == 0 for row in rows
        ),
        "zero_duplicate_executions": all(
            int(row["duplicate_operation_executions"]) == 0
            and int(row["duplicate_technician_executions"]) == 0
            for row in rows
        ),
    }
    return {
        "primary_endpoint": (
            "episode incidence of technician-proposal conflict under independent greedy"
        ),
        "target_incidence": 0.30,
        "selected_condition": selected,
        "selection_rule": (
            "Apply all locked coordination and anti-pathology gates; minimize distance "
            "to 30% incidence, then maximize avoidability, minimize capacity wait, "
            "group size, gap, and condition name."
        ),
        "per_condition": per_condition,
        "gate": {
            "checks": global_checks,
            "scientific_selection_found": selected is not None,
            "run_valid": all(global_checks.values()),
            "note": "Development calibration; not an RL algorithm-performance claim.",
        },
    }


def run(args: argparse.Namespace) -> Path:
    if args.device != "cpu":
        raise ValueError("This simulator-only calibration is CPU-only")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    raw_config = config_path.read_bytes()
    base = BenchmarkConfig.from_json(config_path)
    if base.instance_name != "ht_pdm_fjsp_brandimarte_mk01_v1":
        raise ValueError("This experiment requires the frozen Brandimarte MK01 extension")
    if not math.isclose(base.preventive_probability_threshold, 0.18):
        raise ValueError("This experiment requires the locked PM threshold 0.18")
    seeds = tuple(parse_seeds(args.seeds)) if args.seeds else default_seeds(args.profile)
    conditions = window_conditions()
    config = build_condition_config(base, SPEC)

    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    (output_dir / "resolved_window_conditions.json").write_text(
        json.dumps(
            {
                "fixed_factors": SPEC.to_dict(),
                "availability_intervals": UNAVAILABLE_INTERVALS,
                "conditions": {item.name: item.to_dict() for item in conditions},
                "config": config.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest_path = output_dir / "technician_window_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "profile": args.profile,
        "device": args.device,
        "git_commit": _git_revision(),
        "instance_name": base.instance_name,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "fixed_factors": SPEC.to_dict(),
        "conditions": [item.name for item in conditions],
        "policies": list(POLICIES),
        "seeds": list(seeds),
        "reserved_confirmation_seeds": list(range(62_600, 62_800)),
        "confirmation_panel_opened": False,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    episode_rows: list[dict[str, Any]] = []
    coordination_rows: list[dict[str, Any]] = []
    total = len(conditions) * len(POLICIES) * len(seeds)
    with tqdm(total=total, desc="Technician-window calibration", unit="episode") as progress:
        for condition in conditions:
            for policy in POLICIES:
                simulator = LiteratureFactorialSimulator(
                    config,
                    SPEC,
                    policy,
                    window_releases=condition.releases,
                    window_width=WINDOW_WIDTH,
                    unavailable_intervals=UNAVAILABLE_INTERVALS,
                )
                for seed in seeds:
                    progress.set_postfix(
                        condition=condition.name, policy=policy, seed=seed
                    )
                    episode, coordination = simulator.run(seed=seed)
                    for row in (episode, coordination):
                        row["condition"] = condition.name
                        row["group_size"] = condition.group_size
                        row["inter_wave_gap"] = condition.inter_wave_gap
                        row["reference"] = int(condition.reference)
                    episode_rows.append(episode)
                    coordination_rows.append(coordination)
                    if len(episode_rows) % 16 == 0 or len(episode_rows) == total:
                        _write_csv(
                            episode_rows,
                            output_dir / "technician_window_episodes.partial.csv",
                        )
                        _write_csv(
                            coordination_rows,
                            output_dir / "technician_window_coordination.partial.csv",
                        )
                    progress.update(1)

    summary = summarize(episode_rows, conditions)
    _write_csv(episode_rows, output_dir / "technician_window_episodes.csv")
    _write_csv(coordination_rows, output_dir / "technician_window_coordination.csv")
    (output_dir / "technician_window_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        {
            "status": "COMPLETED",
            "episode_count": len(episode_rows),
            "selected_condition": summary["selected_condition"],
            "gate": summary["gate"],
            "output_files": sorted(path.name for path in output_dir.iterdir()),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--config", default="configs/brandimarte_mk01_ht_pdm.json")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    output_dir = run(build_parser().parse_args())
    summary = json.loads((output_dir / "technician_window_summary.json").read_text())
    print(
        json.dumps(
            {
                "selected_condition": summary["selected_condition"],
                "gate": summary["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Completed. Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
