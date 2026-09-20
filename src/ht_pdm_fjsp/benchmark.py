"""Reproducible multi-seed benchmark runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable

from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.policies import HealthThresholdSPTPolicy, ProductionFirstSPTPolicy
from ht_pdm_fjsp.simulator import SimulationResult, Simulator


def parse_seeds(specification: str) -> list[int]:
    if ":" in specification:
        parts = [int(item) for item in specification.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError("Seed range must be start:stop or start:stop:step.")
        seeds = list(range(*parts))
    else:
        seeds = [int(item) for item in specification.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique.")
    return seeds


def run_panel(config: BenchmarkConfig, seeds: Iterable[int]) -> list[SimulationResult]:
    simulator = Simulator(config)
    policies = (
        ProductionFirstSPTPolicy(),
        HealthThresholdSPTPolicy(config.preventive_probability_threshold),
    )
    return [
        simulator.run(policy, seed=seed)
        for seed in seeds
        for policy in policies
    ]


def summarize(results: list[SimulationResult]) -> dict[str, object]:
    grouped: dict[str, list[SimulationResult]] = {}
    for result in results:
        grouped.setdefault(result.policy, []).append(result)
    summary: dict[str, object] = {}
    for policy, rows in sorted(grouped.items()):
        metrics: dict[str, dict[str, float]] = {}
        for metric in rows[0].metrics:
            values = [float(row.metrics[metric]) for row in rows]
            metrics[metric] = {
                "mean": fmean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        summary[policy] = {"episodes": len(rows), "metrics": metrics}
    return summary


def write_outputs(
    config: BenchmarkConfig,
    results: list[SimulationResult],
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config_payload = json.dumps(config.to_dict(), indent=2, sort_keys=True)
    (destination / "config_snapshot.json").write_text(
        config_payload + "\n", encoding="utf-8"
    )
    fingerprint = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "instance_name": config.instance_name,
                "config_sha256": fingerprint,
                "policies": sorted({item.policy for item in results}),
                "seeds": sorted({item.seed for item in results}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metric_names = list(results[0].metrics)
    with (destination / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["instance_name", "policy", "seed", *metric_names],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "instance_name": result.instance_name,
                    "policy": result.policy,
                    "seed": result.seed,
                    **result.metrics,
                }
            )
    (destination / "summary.json").write_text(
        json.dumps(summarize(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "traces.json").write_text(
        json.dumps([item.to_dict() for item in results], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0:20")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BenchmarkConfig.from_json(args.config)
    results = run_panel(config, parse_seeds(args.seeds))
    write_outputs(config, results, args.output_dir)
    print(json.dumps(summarize(results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
