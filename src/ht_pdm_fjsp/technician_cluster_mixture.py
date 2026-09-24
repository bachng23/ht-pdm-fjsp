"""Calibrate an episode-level mixture of natural and skill-clustered PM windows."""

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
from statistics import NormalDist, fmean, stdev
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
from ht_pdm_fjsp.technician_window_calibration import grouped_releases


SPEC = FactorSpec(True, True, True, True, False)
WINDOW_WIDTH = 8.0
MIXTURE_PROBABILITIES = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 1.0)
CLUSTERED_RELEASES = grouped_releases(3, 0.5)
MIXTURE_HASH_NAMESPACE = "technician-cluster-mixture-v1"


@dataclass(frozen=True)
class MixtureCondition:
    name: str
    cluster_probability: float
    reference: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_probability": self.cluster_probability,
            "reference": self.reference,
        }


def mixture_conditions() -> tuple[MixtureCondition, ...]:
    return tuple(
        MixtureCondition(
            name=f"cluster_p{round(probability * 100):03d}",
            cluster_probability=probability,
            reference=probability in (0.0, 1.0),
        )
        for probability in MIXTURE_PROBABILITIES
    )


def mixture_uniform(seed: int) -> float:
    digest = hashlib.sha256(f"{MIXTURE_HASH_NAMESPACE}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def resolved_releases(
    condition: MixtureCondition, seed: int
) -> tuple[str, Mapping[str, tuple[float, ...]], float]:
    uniform = mixture_uniform(seed)
    if uniform < condition.cluster_probability:
        return "skill_clustered", CLUSTERED_RELEASES, uniform
    return "natural_parent", WINDOW_RELEASES, uniform


def default_seeds(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(62_800, 62_803))
    if profile == "full":
        return tuple(range(62_900, 63_100))
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
    return _interval(
        [greedy[seed] - matcher[seed] for seed in sorted(greedy)]
    )


def _paired_to_reference_interval(
    rows: list[dict[str, Any]], condition: str, metric: str
) -> dict[str, float]:
    reference = {
        int(row["seed"]): float(row[metric])
        for row in rows
        if row["condition"] == "cluster_p000" and row["policy"] == POLICIES[0]
    }
    candidate = {
        int(row["seed"]): float(row[metric])
        for row in rows
        if row["condition"] == condition and row["policy"] == POLICIES[0]
    }
    if reference.keys() != candidate.keys():
        raise AssertionError("Mixture rows do not share identical seeds")
    return _interval(
        [candidate[seed] - reference[seed] for seed in sorted(reference)]
    )


def _interval(values: list[float]) -> dict[str, float]:
    mean = fmean(values)
    half = _t_critical_975(len(values) - 1) * stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lower_95": mean - half, "upper_95": mean + half}


def _wilson_interval(successes: int, total: int) -> dict[str, float]:
    z = NormalDist().inv_cdf(0.975)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {"lower_95": center - half, "upper_95": center + half}


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
    rows: list[dict[str, Any]], conditions: tuple[MixtureCondition, ...]
) -> dict[str, Any]:
    reference_rows = [
        row
        for row in rows
        if row["condition"] == "cluster_p000" and row["policy"] == POLICIES[0]
    ]
    reference_objective = fmean(float(row["objective"]) for row in reference_rows)
    reference_makespan = fmean(float(row["makespan"]) for row in reference_rows)
    per_condition: dict[str, Any] = {}
    candidates: list[tuple[float, float, float, float, str]] = []

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
            conflict_episodes = sum(
                float(row["technician_conflicts"]) > 0 for row in episodes
            )
            policy_summaries[policy] = {
                "episodes": len(episodes),
                "realized_cluster_fraction": fmean(
                    float(row["clustered_schedule"]) for row in episodes
                ),
                "episode_conflict_incidence": conflict_episodes / len(episodes),
                "episode_conflict_incidence_wilson_95": _wilson_interval(
                    conflict_episodes, len(episodes)
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
        greedy_conflicts = greedy["metrics"]["technician_conflicts"]["mean"]
        matcher_conflicts = matcher["metrics"]["technician_conflicts"]["mean"]
        conflict_reduction = (
            1.0 - matcher_conflicts / greedy_conflicts if greedy_conflicts else 0.0
        )
        objective_ratio = greedy["metrics"]["objective"]["mean"] / reference_objective
        makespan_ratio = greedy["metrics"]["makespan"]["mean"] / reference_makespan
        checks = {
            "incidence_between_25_and_30_percent": (
                0.25 <= greedy["episode_conflict_incidence"] <= 0.30
            ),
            "conflict_rate_between_2_and_6_percent": (
                0.02 <= greedy["aggregate_conflict_rate"] <= 0.06
            ),
            "avoidable_fraction_at_least_70_percent": (
                greedy["aggregate_avoidable_fraction"] >= 0.70
            ),
            "matcher_reduction_at_least_50_percent": conflict_reduction >= 0.50,
            "capacity_wait_per_onset_at_most_0p25": (
                greedy["capacity_wait_per_request_onset"] <= 0.25
            ),
            "busy_claim_rate_at_most_35_percent": greedy["busy_claim_rate"] <= 0.35,
            "objective_at_most_5_percent_worse_than_p0": objective_ratio <= 1.05,
            "makespan_at_most_5_percent_worse_than_p0": makespan_ratio <= 1.05,
            "all_audits_pass": all(
                all(audit.values()) for audit in (greedy["audit"], matcher["audit"])
            ),
        }
        qualifies = not condition.reference and all(checks.values())
        if qualifies:
            candidates.append(
                (
                    abs(greedy["episode_conflict_incidence"] - 0.275),
                    -greedy["aggregate_avoidable_fraction"],
                    objective_ratio - 1.0,
                    condition.cluster_probability,
                    condition.name,
                )
            )
        per_condition[condition.name] = {
            "mixture": condition.to_dict(),
            "selection_eligible": not condition.reference,
            "policies": policy_summaries,
            "objective_ratio_to_p0": objective_ratio,
            "makespan_ratio_to_p0": makespan_ratio,
            "greedy_minus_matcher": {
                metric: _paired_interval(rows, condition.name, metric)
                for metric in (
                    "technician_conflicts",
                    "maintenance_wait_time",
                    "objective",
                    "makespan",
                )
            },
            "greedy_minus_p0": {
                metric: _paired_to_reference_interval(rows, condition.name, metric)
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
        "paired_policies_share_realized_schedule": all(
            len(
                {
                    (row["schedule_name"], float(row["mixture_uniform"]))
                    for row in rows
                    if row["condition"] == condition.name
                    and int(row["seed"]) == seed
                }
            )
            == 1
            for condition in conditions
            for seed in {int(row["seed"]) for row in rows}
        ),
    }
    return {
        "primary_endpoint": (
            "episode incidence of technician-proposal conflict under independent greedy"
        ),
        "target_incidence_range": [0.25, 0.30],
        "selected_condition": selected,
        "selection_rule": (
            "Apply all locked coordination, distortion, and audit gates; minimize "
            "distance to 27.5% incidence, then maximize avoidability, minimize "
            "objective degradation, probability, and condition name."
        ),
        "per_condition": per_condition,
        "gate": {
            "checks": global_checks,
            "scientific_selection_found": selected is not None,
            "run_valid": all(global_checks.values()),
            "note": "Development mixture calibration; not an RL performance claim.",
        },
    }


def run(args: argparse.Namespace) -> Path:
    if args.device != "cpu":
        raise ValueError("This simulator-only mixture calibration is CPU-only")
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
    conditions = mixture_conditions()
    config = build_condition_config(base, SPEC)

    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    (output_dir / "resolved_mixture_conditions.json").write_text(
        json.dumps(
            {
                "fixed_factors": SPEC.to_dict(),
                "window_width": WINDOW_WIDTH,
                "natural_releases": WINDOW_RELEASES,
                "clustered_releases": CLUSTERED_RELEASES,
                "availability_intervals": UNAVAILABLE_INTERVALS,
                "mixture_hash_namespace": MIXTURE_HASH_NAMESPACE,
                "conditions": {item.name: item.to_dict() for item in conditions},
                "config": config.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest_path = output_dir / "technician_mixture_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "profile": args.profile,
        "device": args.device,
        "git_commit": _git_revision(),
        "instance_name": base.instance_name,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "fixed_factors": SPEC.to_dict(),
        "mixture_hash_namespace": MIXTURE_HASH_NAMESPACE,
        "conditions": [item.name for item in conditions],
        "policies": list(POLICIES),
        "seeds": list(seeds),
        "reserved_confirmation_seeds": list(range(63_200, 63_500)),
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
    with tqdm(total=total, desc="Technician-cluster mixture", unit="episode") as progress:
        for condition in conditions:
            for policy in POLICIES:
                for seed in seeds:
                    schedule_name, releases, uniform = resolved_releases(condition, seed)
                    progress.set_postfix(
                        condition=condition.name,
                        policy=policy,
                        seed=seed,
                        schedule=schedule_name,
                    )
                    simulator = LiteratureFactorialSimulator(
                        config,
                        SPEC,
                        policy,
                        window_releases=releases,
                        window_width=WINDOW_WIDTH,
                        unavailable_intervals=UNAVAILABLE_INTERVALS,
                    )
                    episode, coordination = simulator.run(seed=seed)
                    for row in (episode, coordination):
                        row["condition"] = condition.name
                        row["cluster_probability"] = condition.cluster_probability
                        row["mixture_uniform"] = uniform
                        row["schedule_name"] = schedule_name
                        row["clustered_schedule"] = int(
                            schedule_name == "skill_clustered"
                        )
                        row["reference"] = int(condition.reference)
                    episode_rows.append(episode)
                    coordination_rows.append(coordination)
                    if len(episode_rows) % 16 == 0 or len(episode_rows) == total:
                        _write_csv(
                            episode_rows,
                            output_dir / "technician_mixture_episodes.partial.csv",
                        )
                        _write_csv(
                            coordination_rows,
                            output_dir / "technician_mixture_coordination.partial.csv",
                        )
                    progress.update(1)

    summary = summarize(episode_rows, conditions)
    _write_csv(episode_rows, output_dir / "technician_mixture_episodes.csv")
    _write_csv(coordination_rows, output_dir / "technician_mixture_coordination.csv")
    (output_dir / "technician_mixture_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        {
            "status": "COMPLETED",
            "episode_count": len(episode_rows),
            "selected_condition": summary["selected_condition"],
            "gate": summary["gate"],
            "realized_cluster_counts": {
                condition.name: sum(
                    resolved_releases(condition, seed)[0] == "skill_clustered"
                    for seed in seeds
                )
                for condition in conditions
            },
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
    summary = json.loads((output_dir / "technician_mixture_summary.json").read_text())
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
