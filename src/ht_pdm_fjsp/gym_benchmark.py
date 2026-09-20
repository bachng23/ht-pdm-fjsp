"""Evaluate Gymnasium baselines on a common seed panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev

from ht_pdm_fjsp.advanced_baselines import (
    CPSATReactivePolicy,
    HealthThresholdPolicy,
    JointRiskGreedyPolicy,
    MaskedSPTPolicy,
    RollingHorizonPolicy,
    rollout_policy,
)
from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.simulator import SimulationResult


def default_policies(config: BenchmarkConfig):
    return (
        MaskedSPTPolicy(),
        HealthThresholdPolicy(config.preventive_probability_threshold),
        JointRiskGreedyPolicy(),
        CPSATReactivePolicy(
            probability_threshold=config.preventive_probability_threshold
        ),
        RollingHorizonPolicy(depth=2, branch_width=7, scenario_count=4),
    )


def run_gym_panel(
    config: BenchmarkConfig, seeds: list[int]
) -> list[tuple[SimulationResult, float]]:
    outputs: list[tuple[SimulationResult, float]] = []
    for policy in default_policies(config):
        for seed in seeds:
            outputs.append(
                rollout_policy(HTPdmFjspEnv(config=config), policy, seed=seed)
            )
    return outputs


def summarize(outputs: list[tuple[SimulationResult, float]]) -> dict[str, object]:
    grouped: dict[str, list[tuple[SimulationResult, float]]] = {}
    for result, episode_return in outputs:
        grouped.setdefault(result.policy, []).append((result, episode_return))
    summary: dict[str, object] = {}
    for policy, rows in sorted(grouped.items()):
        metric_summary: dict[str, dict[str, float]] = {}
        for metric in rows[0][0].metrics:
            values = [float(result.metrics[metric]) for result, _ in rows]
            metric_summary[metric] = {
                "mean": fmean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        returns = [episode_return for _, episode_return in rows]
        summary[policy] = {
            "episodes": len(rows),
            "episode_return": {
                "mean": fmean(returns),
                "std": pstdev(returns),
                "min": min(returns),
                "max": max(returns),
            },
            "metrics": metric_summary,
        }
    return summary


def paired_comparisons(
    outputs: list[tuple[SimulationResult, float]],
) -> dict[str, dict[str, float]]:
    """Compare objective values on identical seeds; negative delta is better."""

    by_policy_seed = {
        (result.policy, result.seed): float(result.metrics["objective"])
        for result, _ in outputs
    }
    seeds = sorted(
        seed for policy, seed in by_policy_seed if policy == "masked_spt"
    )
    policies = sorted({policy for policy, _ in by_policy_seed} - {"masked_spt"})
    comparisons: dict[str, dict[str, float]] = {}
    for policy in policies:
        deltas = [
            by_policy_seed[(policy, seed)] - by_policy_seed[("masked_spt", seed)]
            for seed in seeds
        ]
        comparisons[policy] = {
            "mean_objective_delta": fmean(deltas),
            "std_objective_delta": pstdev(deltas),
            "win_rate": sum(delta < 0.0 for delta in deltas) / len(deltas),
            "tie_rate": sum(delta == 0.0 for delta in deltas) / len(deltas),
        }
    return comparisons


def write_outputs(
    config: BenchmarkConfig,
    outputs: list[tuple[SimulationResult, float]],
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config_payload = json.dumps(config.to_dict(), indent=2, sort_keys=True)
    (destination / "config_snapshot.json").write_text(
        config_payload + "\n", encoding="utf-8"
    )
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "instance_name": config.instance_name,
                "config_sha256": hashlib.sha256(
                    config_payload.encode("utf-8")
                ).hexdigest(),
                "policies": sorted({result.policy for result, _ in outputs}),
                "seeds": sorted({result.seed for result, _ in outputs}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metric_names = list(outputs[0][0].metrics)
    with (destination / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["instance_name", "policy", "seed", "episode_return", *metric_names],
        )
        writer.writeheader()
        for result, episode_return in outputs:
            writer.writerow(
                {
                    "instance_name": result.instance_name,
                    "policy": result.policy,
                    "seed": result.seed,
                    "episode_return": episode_return,
                    **result.metrics,
                }
            )
    (destination / "summary.json").write_text(
        json.dumps(summarize(outputs), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "paired_comparisons.json").write_text(
        json.dumps(paired_comparisons(outputs), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "traces.json").write_text(
        json.dumps(
            [
                {**result.to_dict(), "episode_return": episode_return}
                for result, episode_return in outputs
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0:20")
    args = parser.parse_args()
    config = BenchmarkConfig.from_json(args.config)
    outputs = run_gym_panel(config, parse_seeds(args.seeds))
    write_outputs(config, outputs, args.output_dir)
    print(json.dumps(summarize(outputs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
