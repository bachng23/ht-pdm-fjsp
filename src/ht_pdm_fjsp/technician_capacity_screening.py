"""Screen queue-aware technician capacity on the Brandimarte MK01 extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from statistics import NormalDist, fmean, pstdev, stdev
from typing import Any, Iterable

from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig, Technician
from ht_pdm_fjsp.policies import HealthThresholdSPTPolicy
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.simulator import Simulator


TOPOLOGIES = ("two_specialists", "one_shared")
DURATION_MULTIPLIERS = (1.0, 1.5, 2.0)
REFERENCE = "two_specialists_x1_0"
WAIT_TOLERANCE = 1e-9
SUMMARY_METRICS = (
    "objective",
    "makespan",
    "schedule_end",
    "failures",
    "preventive_maintenance",
    "corrective_maintenance",
    "maintenance_time",
    "failure_wait_time",
    "preventive_wait_time",
    "maintenance_wait_time",
    "technician_utilization",
    "processed_events",
)
CONTRAST_METRICS = (
    "objective",
    "makespan",
    "maintenance_wait_time",
    "technician_utilization",
    "failures",
    "preventive_maintenance",
)


def default_seeds(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(61_430, 61_435))
    if profile == "full":
        return tuple(range(61_500, 61_700))
    raise ValueError(f"Unknown profile: {profile}")


def condition_name(topology: str, multiplier: float) -> str:
    return f"{topology}_x{multiplier:.1f}".replace(".", "_")


def condition_specs() -> tuple[tuple[str, str, float], ...]:
    return tuple(
        (condition_name(topology, multiplier), topology, multiplier)
        for topology in TOPOLOGIES
        for multiplier in DURATION_MULTIPLIERS
    )


def _scaled_technician(technician: Technician, multiplier: float) -> Technician:
    return replace(
        technician,
        preventive_duration={
            machine: duration * multiplier
            for machine, duration in technician.preventive_duration.items()
        },
        corrective_duration={
            machine: duration * multiplier
            for machine, duration in technician.corrective_duration.items()
        },
    )


def _fastest_value(
    technicians: tuple[Technician, ...], machine_id: str, kind: str
) -> tuple[float, float]:
    eligible = [
        technician
        for technician in technicians
        if machine_id in technician.eligible_machines
    ]
    if not eligible:
        raise ValueError(f"No technician covers {machine_id}")
    chosen = min(
        eligible,
        key=lambda technician: (
            technician.duration(machine_id, kind),
            technician.technician_id,
        ),
    )
    return (
        chosen.duration(machine_id, kind),
        chosen.restoration(machine_id, kind),
    )


def build_condition_config(
    base: BenchmarkConfig, *, topology: str, multiplier: float
) -> BenchmarkConfig:
    if topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology: {topology}")
    if multiplier <= 0 or not math.isfinite(multiplier):
        raise ValueError("Duration multiplier must be finite and positive")
    if topology == "two_specialists":
        technicians = tuple(
            _scaled_technician(technician, multiplier)
            for technician in base.technicians
        )
    else:
        machine_ids = tuple(machine.machine_id for machine in base.machines)
        preventive = {
            machine: _fastest_value(base.technicians, machine, "preventive")
            for machine in machine_ids
        }
        corrective = {
            machine: _fastest_value(base.technicians, machine, "corrective")
            for machine in machine_ids
        }
        technicians = (
            Technician(
                technician_id="T_SHARED",
                eligible_machines=machine_ids,
                preventive_duration={
                    machine: preventive[machine][0] * multiplier
                    for machine in machine_ids
                },
                corrective_duration={
                    machine: corrective[machine][0] * multiplier
                    for machine in machine_ids
                },
                preventive_restoration={
                    machine: preventive[machine][1] for machine in machine_ids
                },
                corrective_restoration={
                    machine: corrective[machine][1] for machine in machine_ids
                },
            ),
        )
    config = replace(base, technicians=technicians)
    config.validate()
    return config


def evaluate_episode(
    config: BenchmarkConfig,
    *,
    condition: str,
    topology: str,
    duration_multiplier: float,
    seed: int,
) -> dict[str, Any]:
    policy = HealthThresholdSPTPolicy(config.preventive_probability_threshold)
    result = Simulator(config).run(policy, seed=seed)
    metrics = result.metrics
    return {
        "condition": condition,
        "topology": topology,
        "duration_multiplier": duration_multiplier,
        "technician_count": len(config.technicians),
        "threshold": config.preventive_probability_threshold,
        "seed": seed,
        "objective": metrics["total_cost"],
        "any_queue_wait": int(float(metrics["maintenance_wait_time"]) > WAIT_TOLERANCE),
        "any_preventive_wait": int(float(metrics["preventive_wait_time"]) > WAIT_TOLERANCE),
        "any_corrective_wait": int(float(metrics["failure_wait_time"]) > WAIT_TOLERANCE),
        **metrics,
    }


def _describe(values: Iterable[float]) -> dict[str, float]:
    data = sorted(values)
    if not data:
        raise ValueError("Cannot summarize an empty sample")

    def quantile(probability: float) -> float:
        position = probability * (len(data) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return data[lower]
        return data[lower] + (position - lower) * (data[upper] - data[lower])

    return {
        "mean": fmean(data),
        "std": pstdev(data),
        "min": data[0],
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "max": data[-1],
    }


def _t_critical_975(degrees_of_freedom: int) -> float:
    table = (
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
        2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
        2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
        2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
    )
    if degrees_of_freedom < 1:
        raise ValueError("Paired intervals require at least two seeds")
    if degrees_of_freedom <= len(table):
        return table[degrees_of_freedom - 1]
    z = NormalDist().inv_cdf(0.975)
    df = float(degrees_of_freedom)
    return z + (z**3 + z) / (4.0 * df) + (
        5.0 * z**5 + 16.0 * z**3 + 3.0 * z
    ) / (96.0 * df**2)


def _paired_contrast(
    rows: list[dict[str, Any]], treatment: str, control: str
) -> dict[str, Any]:
    treatment_rows = {
        int(row["seed"]): row for row in rows if row["condition"] == treatment
    }
    control_rows = {
        int(row["seed"]): row for row in rows if row["condition"] == control
    }
    if treatment_rows.keys() != control_rows.keys():
        raise AssertionError("Conditions do not share an identical seed panel")
    seeds = sorted(treatment_rows)
    critical = _t_critical_975(len(seeds) - 1)
    metrics: dict[str, Any] = {}
    for metric in CONTRAST_METRICS:
        deltas = [
            float(treatment_rows[seed][metric])
            - float(control_rows[seed][metric])
            for seed in seeds
        ]
        mean = fmean(deltas)
        half_width = critical * stdev(deltas) / math.sqrt(len(deltas))
        metrics[metric] = {
            "mean": mean,
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
        "metrics": metrics,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    specs = condition_specs()
    per_condition: dict[str, Any] = {}
    candidates: list[tuple[float, int, float, str]] = []
    for condition, topology, multiplier in specs:
        episodes = [row for row in rows if row["condition"] == condition]
        queue_incidence = fmean(float(row["any_queue_wait"]) for row in episodes)
        utilization = fmean(float(row["technician_utilization"]) for row in episodes)
        mean_wait = fmean(float(row["maintenance_wait_time"]) for row in episodes)
        moderate = (
            0.10 <= queue_incidence <= 0.40
            and 0.0 < utilization <= 0.85
            and mean_wait > 0.0
        )
        if moderate:
            candidates.append(
                (
                    abs(queue_incidence - 0.25),
                    0 if topology == "two_specialists" else 1,
                    multiplier,
                    condition,
                )
            )
        per_condition[condition] = {
            "episodes": len(episodes),
            "topology": topology,
            "duration_multiplier": multiplier,
            "queue_incidence": queue_incidence,
            "preventive_wait_incidence": fmean(
                float(row["any_preventive_wait"]) for row in episodes
            ),
            "corrective_wait_incidence": fmean(
                float(row["any_corrective_wait"]) for row in episodes
            ),
            "moderate_binding_gate": moderate,
            "metrics": {
                metric: _describe(float(row[metric]) for row in episodes)
                for metric in SUMMARY_METRICS
            },
        }
    selected = min(candidates)[3] if candidates else None
    monotonic: dict[str, Any] = {}
    for topology in TOPOLOGIES:
        names = [condition_name(topology, value) for value in DURATION_MULTIPLIERS]
        incidences = [per_condition[name]["queue_incidence"] for name in names]
        waits = [
            per_condition[name]["metrics"]["maintenance_wait_time"]["mean"]
            for name in names
        ]
        monotonic[topology] = {
            "conditions": names,
            "queue_incidence_non_decreasing": incidences == sorted(incidences),
            "mean_wait_non_decreasing": waits == sorted(waits),
        }
    expected_per_condition = len(rows) // len(specs)
    checks = {
        "all_expected_episodes_completed": len(rows)
        == len(specs) * expected_per_condition
        and all(
            per_condition[name]["episodes"] == expected_per_condition
            for name, _, _ in specs
        ),
        "at_least_one_moderate_binding_condition": selected is not None,
        "selected_utilization_not_saturated": selected is not None
        and per_condition[selected]["metrics"]["technician_utilization"]["mean"]
        <= 0.85,
        "selected_has_positive_queue_wait": selected is not None
        and per_condition[selected]["metrics"]["maintenance_wait_time"]["mean"] > 0,
    }
    contrasts = {
        f"{condition}_minus_{REFERENCE}": _paired_contrast(rows, condition, REFERENCE)
        for condition, _, _ in specs
        if condition != REFERENCE
    }
    return {
        "primary_endpoint": "fraction of episodes with positive maintenance queue wait",
        "selection_rule": (
            "Pass queue incidence [0.10, 0.40], mean utilization (0, 0.85], "
            "and positive mean wait; minimize distance to 0.25, then prefer "
            "two specialists and lower duration multiplier."
        ),
        "selected_condition": selected,
        "per_condition": per_condition,
        "paired_contrasts": contrasts,
        "mechanism_checks": monotonic,
        "gate": {
            "checks": checks,
            "supports_next_marl_experiment": all(checks.values()),
            "note": "Development mechanism gate; not an algorithm-performance claim.",
        },
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    raw_config = config_path.read_bytes()
    base = BenchmarkConfig.from_json(config_path)
    if base.instance_name != "ht_pdm_fjsp_brandimarte_mk01_v1":
        raise ValueError("This screening requires the frozen Brandimarte MK01 extension")
    if not math.isclose(base.preventive_probability_threshold, 0.18):
        raise ValueError("This screening requires the locked threshold 0.18")
    seeds = tuple(parse_seeds(args.seeds)) if args.seeds else default_seeds(args.profile)
    specs = condition_specs()
    configs = {
        condition: build_condition_config(base, topology=topology, multiplier=multiplier)
        for condition, topology, multiplier in specs
    }
    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    (output_dir / "condition_configs.json").write_text(
        json.dumps(
            {name: config.to_dict() for name, config in configs.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest_path = output_dir / "technician_capacity_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "profile": args.profile,
        "git_commit": _git_revision(),
        "instance_name": base.instance_name,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "conditions": [name for name, _, _ in specs],
        "seeds": list(seeds),
        "reserved_future_seeds": list(range(62_000, 62_100)),
        "future_test_panel_opened": False,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    rows: list[dict[str, Any]] = []
    total = len(specs) * len(seeds)
    with tqdm(total=total, desc="Technician-capacity screening", unit="episode") as progress:
        for condition, topology, multiplier in specs:
            config = configs[condition]
            for seed in seeds:
                progress.set_postfix(condition=condition, seed=seed)
                rows.append(
                    evaluate_episode(
                        config,
                        condition=condition,
                        topology=topology,
                        duration_multiplier=multiplier,
                        seed=seed,
                    )
                )
                if len(rows) % 10 == 0 or len(rows) == total:
                    _write_csv(
                        rows, output_dir / "technician_capacity_episodes.partial.csv"
                    )
                progress.update(1)

    summary = summarize(rows)
    _write_csv(rows, output_dir / "technician_capacity_episodes.csv")
    (output_dir / "technician_capacity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        {
            "status": "COMPLETED",
            "episode_count": len(rows),
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
    parser.add_argument("--config", default="configs/brandimarte_mk01_ht_pdm.json")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    output_dir = run(build_parser().parse_args())
    summary = json.loads((output_dir / "technician_capacity_summary.json").read_text())
    print(json.dumps({
        "selected_condition": summary["selected_condition"],
        "gate": summary["gate"],
    }, indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
