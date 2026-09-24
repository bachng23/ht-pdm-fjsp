"""Clean attribution of technician-collision loss under request commitment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.conflict_consequence import (
    SEALED_TEST_SEEDS,
    ConsequenceCell,
    build_cell_config,
    evaluate_episode,
    selected_parent_config,
)
from ht_pdm_fjsp.parallel_maintenance import ParallelMaintenanceConfig


POLICIES = (
    "reactive_fibt",
    "coordinated_preventive",
    "independent_preventive",
)
CELLS = (
    ConsequenceCell(False, False, False),
    ConsequenceCell(True, False, False),
)
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
    "early_pm",
    "workload_imbalance",
)


def seeds_for_profile(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(63_990, 63_993))
    if profile == "full":
        return tuple(range(63_800, 63_900))
    raise ValueError(f"Unknown profile: {profile}")


def _critical_value(count: int) -> float:
    values = {2: 12.706, 3: 4.303, 50: 2.009, 100: 1.984}
    return values.get(count, 1.96)


def _paired_interval(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    half = 0.0
    if len(values) >= 2:
        half = (
            _critical_value(len(values))
            * statistics.stdev(values)
            / math.sqrt(len(values))
        )
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _policy_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {metric: _mean(rows, metric) for metric in METRICS}


def _objective_weights(config: ParallelMaintenanceConfig) -> dict[str, float]:
    costs = config.costs
    return {
        "proposal_conflicts": costs.conflict,
        "preventive": costs.preventive,
        "corrective": costs.corrective,
        "downtime": costs.downtime,
        "failures": costs.failure,
        "waiting": costs.waiting,
        "rework": costs.rework,
        "early_pm": costs.early_pm,
        "production": -costs.production_credit,
    }


def summarize(
    episodes: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    profile: str,
    config: ParallelMaintenanceConfig,
) -> dict[str, Any]:
    by_key = {
        (row["cell_id"], row["policy"], int(row["seed"])): row
        for row in episodes
    }
    weights = _objective_weights(config)
    cells: dict[str, Any] = {}
    paired_gaps: dict[str, list[float]] = {}
    for cell in CELLS:
        gap = [
            float(by_key[cell.cell_id, "independent_preventive", seed]["objective"])
            - float(by_key[cell.cell_id, "coordinated_preventive", seed]["objective"])
            for seed in seeds
        ]
        paired_gaps[cell.cell_id] = gap
        interval = _paired_interval(gap)
        grouped = {
            policy: [by_key[cell.cell_id, policy, seed] for seed in seeds]
            for policy in POLICIES
        }
        coordinated_objective = _mean(grouped["coordinated_preventive"], "objective")
        reactive_objective = _mean(grouped["reactive_fibt"], "objective")
        coordinated_minus_reactive = statistics.fmean(
            float(by_key[cell.cell_id, "coordinated_preventive", seed]["objective"])
            - float(by_key[cell.cell_id, "reactive_fibt", seed]["objective"])
            for seed in seeds
        )
        decomposition: dict[str, float] = {}
        for metric, weight in weights.items():
            decomposition[metric] = weight * statistics.fmean(
                float(by_key[cell.cell_id, "independent_preventive", seed][metric])
                - float(by_key[cell.cell_id, "coordinated_preventive", seed][metric])
                for seed in seeds
            )
        decomposition_sum = sum(decomposition.values())
        cells[cell.cell_id] = {
            "factors": asdict(cell),
            "independent_minus_coordinated_greedy": {
                **interval,
                "median": statistics.median(gap),
                "independent_worse_rate": sum(value > 0 for value in gap) / len(gap),
                "relative_gap": interval["mean"]
                / max(abs(coordinated_objective), 1e-9),
            },
            "independent_conflict_step_incidence": _mean(
                grouped["independent_preventive"], "conflict_steps"
            )
            / config.horizon,
            "coordinated_minus_reactive_mean": coordinated_minus_reactive,
            "coordinated_relative_improvement_over_reactive": (
                -coordinated_minus_reactive / max(abs(reactive_objective), 1e-9)
            ),
            "objective_gap_decomposition": decomposition,
            "decomposition_sum": decomposition_sum,
            "decomposition_error": abs(decomposition_sum - interval["mean"]),
            "policy_metrics": {
                policy: _policy_metrics(rows) for policy, rows in grouped.items()
            },
        }

    low_id, high_id = (cell.cell_id for cell in CELLS)
    interaction_values = [
        high - low
        for low, high in zip(paired_gaps[low_id], paired_gaps[high_id], strict=True)
    ]
    commitment_interaction = _paired_interval(interaction_values)
    committed = cells[high_id]
    primary = committed["independent_minus_coordinated_greedy"]
    explicit_conflict = committed["objective_gap_decomposition"]["proposal_conflicts"]
    conflict_share = (
        explicit_conflict / primary["mean"] if primary["mean"] > 0 else None
    )
    coordinated_conflicts = sum(
        float(by_key[cell.cell_id, "coordinated_preventive", seed][
            "proposal_conflicts"
        ])
        for cell in CELLS
        for seed in seeds
    )
    expected = len(CELLS) * len(POLICIES) * len(seeds)
    audits = {
        "expected_episodes_completed": len(episodes) == expected,
        "reward_objective_identity": all(
            float(row["return_objective_error"]) <= 1e-6 for row in episodes
        ),
        "objective_decomposition_identity": all(
            cell["decomposition_error"] <= 1e-6 for cell in cells.values()
        ),
        "no_invalid_executions": all(
            int(row["invalid_executions"]) == 0 for row in episodes
        ),
        "no_duplicate_machine_assignments": all(
            int(row["duplicate_machine_assignments"]) == 0 for row in episodes
        ),
        "no_duplicate_technician_assignments": all(
            int(row["duplicate_technician_assignments"]) == 0 for row in episodes
        ),
        "sealed_test_panel_remains_closed": not any(
            int(row["seed"]) in SEALED_TEST_SEEDS for row in episodes
        ),
    }
    checks = {
        "committed_conflict_incidence_in_2p5_to_10_percent": (
            0.025 <= committed["independent_conflict_step_incidence"] <= 0.10
        ),
        "committed_relative_gap_at_least_5_percent": primary["relative_gap"] >= 0.05,
        "committed_paired_95_percent_lower_bound_above_zero": primary["lower"] > 0,
        "committed_independent_worse_on_70_percent_of_seeds": (
            primary["independent_worse_rate"] >= 0.70
        ),
        "commitment_interaction_lower_bound_above_zero": (
            commitment_interaction["lower"] > 0
        ),
        "coordinated_improves_reactive_by_5_percent": (
            committed["coordinated_relative_improvement_over_reactive"] >= 0.05
        ),
        "coordinated_policy_has_zero_conflicts": coordinated_conflicts == 0,
        "explicit_conflict_penalty_below_half_of_gap": (
            conflict_share is not None and conflict_share < 0.50
        ),
    }
    confirmatory = profile == "full"
    return {
        "purpose": "Clean technician-conflict attribution; not an RL algorithm claim.",
        "profile": profile,
        "primary_cell": high_id,
        "cell_summaries": cells,
        "commitment_interaction": commitment_interaction,
        "explicit_conflict_penalty_share": conflict_share,
        "hypotheses": {
            "H1_clean_committed_loss": (
                primary["relative_gap"] >= 0.05
                and primary["lower"] > 0
                and primary["independent_worse_rate"] >= 0.70
            ),
            "H2_commitment_amplifies_clean_gap": commitment_interaction["lower"] > 0,
            "H3_operational_loss_dominates_explicit_penalty": (
                conflict_share is not None and conflict_share < 0.50
            ),
        },
        "promotion": {
            "confirmatory": confirmatory,
            "checks": checks,
            "passed": confirmatory and all(checks.values()) and all(audits.values()),
        },
        "audits": audits,
        "gate": {"passed": all(audits.values())},
    }


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
    if args.device != "cpu":
        raise ValueError("This simulator diagnostic supports only --device cpu")
    seeds = tuple(parse_seeds(args.seeds)) if args.seeds else seeds_for_profile(args.profile)
    overlap = sorted(set(seeds).intersection(SEALED_TEST_SEEDS))
    if overlap:
        raise ValueError(f"Refusing to evaluate sealed test seeds: {overlap}")
    base = ParallelMaintenanceConfig.from_json(args.config)
    resolved = selected_parent_config(base)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "clean_conflict_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "purpose": "clean_technician_conflict_attribution",
        "profile": args.profile,
        "device": args.device,
        "git_revision": _git_revision(),
        "started_at": datetime.now(UTC).isoformat(),
        "config": str(Path(args.config).resolve()),
        "cells": [asdict(cell) | {"cell_id": cell.cell_id} for cell in CELLS],
        "policies": list(POLICIES),
        "evaluation_seeds": list(seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "source_config.json").write_text(
        json.dumps(base.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "resolved_base_config.json").write_text(
        json.dumps(resolved.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "cells.json").write_text(
        json.dumps(manifest["cells"], indent=2, sort_keys=True) + "\n"
    )
    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(CELLS) * len(POLICIES) * len(seeds),
        desc="Clean conflict attribution",
        unit="episode",
    )
    try:
        for cell in CELLS:
            config = build_cell_config(base, cell)
            for policy in POLICIES:
                for seed in seeds:
                    episode, trace = evaluate_episode(config, cell, policy, seed)
                    episodes.append(episode)
                    decisions.extend(trace)
                    progress.update(1)
                _write_csv(episodes, output_dir / "episodes.partial.csv")
        progress.close()
        summary = summarize(
            episodes, seeds=seeds, profile=args.profile, config=resolved
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
                "promotion": summary["promotion"],
                "outputs": sorted(
                    str(path.relative_to(output_dir))
                    for path in output_dir.rglob("*")
                    if path.is_file()
                ),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if not summary["gate"]["passed"]:
            raise RuntimeError("Clean conflict attribution audit gate failed")
        return summary
    except Exception:
        progress.close()
        manifest.update({"status": "FAILED", "finished_at": datetime.now(UTC).isoformat()})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/parallel_maintenance.json")
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--seeds")
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
