"""Confirmatory multi-scenario ablation for passive shared technicians."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import shutil
import statistics
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_baselines import evaluate_fixed, stress_config
from ht_pdm_fjsp.passive_technician_marl import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    PassiveConfig,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition,
    ValueTrainSettings,
    evaluate_value_decomposition,
    train_value_decomposition_checkpoints,
)


CONFIRMATORY_ALGORITHMS = (
    "qmix",
    "qmix_queue",
    "qmix_counterfactual",
    "tqmix",
)
FIXED_POLICIES = ("random_feasible", "skill_aware_fifo")
FULL_TRAIN_SEEDS = tuple(range(21, 31))
FULL_EVALUATION_SEEDS = tuple(range(1001, 1101))
SEALED_TEST_SEEDS = tuple(range(2001, 2101))
FULL_BUDGETS = (20_000, 50_000)
SMOKE_BUDGETS = (8, 16)
PRIMARY_POLICY = "qmix_counterfactual"
REFERENCE_POLICY = "qmix"
PRIMARY_SCENARIO = "in_distribution"
BOOTSTRAP_REPLICATES = 20_000
FIELDNAMES = (
    "policy",
    "budget",
    "scenario",
    "seed",
    "train_seed",
    "objective",
    "failures",
    "jobs",
    "collisions",
    "waiting",
    "invalid_requests",
    "busy_requests",
    "unique_joint_actions",
    "defer_fraction",
    "action_steps",
    "request_count",
)
MECHANISM_METRICS = (
    "objective",
    "failures",
    "jobs",
    "collisions",
    "waiting",
    "invalid_requests",
    "busy_requests",
    "unique_joint_actions",
    "defer_fraction",
)


def scenario_configs() -> dict[str, PassiveConfig]:
    base = stress_config()
    slow_service = ((3, 5), (5, 3), (4, 4))
    return {
        "in_distribution": base,
        "early_failure": replace(
            base,
            horizon=18,
            failure_age=4,
            failure_probability=0.60,
        ),
        "slow_service": replace(
            base,
            horizon=18,
            service_time=slow_service,
        ),
        "combined_pressure": replace(
            base,
            horizon=18,
            failure_age=4,
            failure_probability=0.60,
            service_time=slow_service,
        ),
    }


def _profile(profile: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if profile == "smoke":
        return (FULL_TRAIN_SEEDS[0],), FULL_EVALUATION_SEEDS[:3], SMOKE_BUDGETS
    if profile == "full":
        return FULL_TRAIN_SEEDS, FULL_EVALUATION_SEEDS, FULL_BUDGETS
    raise ValueError(profile)


def _git_state() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _write_csv(
    rows: Iterable[dict[str, Any]],
    path: Path,
    fieldnames: tuple[str, ...] | list[str],
) -> None:
    materialized = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(materialized)


def _append_csv(
    rows: list[dict[str, Any]],
    path: Path,
    fieldnames: tuple[str, ...] | list[str],
) -> None:
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _episode_row(
    result: dict[str, Any],
    policy: str,
    budget: int,
    scenario: str,
    train_seed: int | str,
) -> dict[str, Any]:
    row = {
        field: result.get(field, "" if field in {"seed", "train_seed"} else 0)
        for field in FIELDNAMES
    }
    row.update(
        policy=policy,
        budget=budget,
        scenario=scenario,
        train_seed=train_seed,
    )
    return row


def _percentile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def paired_bootstrap_interval(
    differences: list[float],
    *,
    key: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("paired differences must not be empty")
    values = np.asarray(differences, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def exact_sign_flip_p(differences: list[float]) -> float:
    if not differences:
        raise ValueError("paired differences must not be empty")
    observed = abs(statistics.fmean(differences))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(statistics.fmean(sign * value for sign, value in zip(signs, differences)))
        extreme += int(permuted >= observed - 1e-12)
        total += 1
    return extreme / total


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def _seed_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["policy"] not in CONFIRMATORY_ALGORITHMS:
            continue
        key = (
            str(row["policy"]),
            int(row["budget"]),
            str(row["scenario"]),
            str(row["train_seed"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (policy, budget, scenario, train_seed), group in sorted(groups.items()):
        result: dict[str, Any] = {
            "policy": policy,
            "budget": budget,
            "scenario": scenario,
            "train_seed": int(train_seed),
            "evaluation_count": len(group),
        }
        for metric in MECHANISM_METRICS:
            result[f"{metric}_mean"] = statistics.fmean(float(row[metric]) for row in group)
        output.append(result)
    return output


def _scenario_summaries(
    rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    learned_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        key = str(row["policy"]), int(row["budget"]), str(row["scenario"])
        learned_groups.setdefault(key, []).append(row)
    for (policy, budget, scenario), group in sorted(learned_groups.items()):
        values = [float(row["objective_mean"]) for row in group]
        output.append(
            {
                "policy": policy,
                "budget": budget,
                "scenario": scenario,
                "training_seed_count": len(values),
                "evaluation_count_per_seed": int(group[0]["evaluation_count"]),
                "objective_mean": statistics.fmean(values),
                "objective_std_across_train_seeds": statistics.stdev(values)
                if len(values) > 1
                else 0.0,
            }
        )
    fixed_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["policy"] in FIXED_POLICIES:
            fixed_groups.setdefault((str(row["policy"]), str(row["scenario"])), []).append(row)
    for (policy, scenario), group in sorted(fixed_groups.items()):
        values = [float(row["objective"]) for row in group]
        output.append(
            {
                "policy": policy,
                "budget": 0,
                "scenario": scenario,
                "training_seed_count": 0,
                "evaluation_count_per_seed": len(values),
                "objective_mean": statistics.fmean(values),
                "objective_std_across_train_seeds": "",
            }
        )
    return output


def _paired_effects(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["policy"]),
            int(row["budget"]),
            str(row["scenario"]),
            int(row["train_seed"]),
        ): float(row["objective_mean"])
        for row in seed_rows
    }
    combinations = sorted(
        {
            (int(row["budget"]), str(row["scenario"]))
            for row in seed_rows
        }
    )
    output: list[dict[str, Any]] = []
    for budget, scenario in combinations:
        family: list[dict[str, Any]] = []
        raw_p: dict[str, float] = {}
        reference_seeds = sorted(
            seed
            for policy, row_budget, row_scenario, seed in lookup
            if policy == REFERENCE_POLICY
            and row_budget == budget
            and row_scenario == scenario
        )
        for policy in CONFIRMATORY_ALGORITHMS:
            if policy == REFERENCE_POLICY:
                continue
            differences = [
                lookup[(policy, budget, scenario, seed)]
                - lookup[(REFERENCE_POLICY, budget, scenario, seed)]
                for seed in reference_seeds
            ]
            low, high = paired_bootstrap_interval(
                differences,
                key=f"{policy}:{budget}:{scenario}",
            )
            p_value = exact_sign_flip_p(differences)
            raw_p[policy] = p_value
            family.append(
                {
                    "policy": policy,
                    "reference": REFERENCE_POLICY,
                    "budget": budget,
                    "scenario": scenario,
                    "training_seed_count": len(differences),
                    "mean_difference": statistics.fmean(differences),
                    "std_difference": statistics.stdev(differences)
                    if len(differences) > 1
                    else 0.0,
                    "median_difference": statistics.median(differences),
                    "ci95_low": low,
                    "ci95_high": high,
                    "exact_sign_flip_p": p_value,
                    "seed_wins": sum(value < 0 for value in differences),
                    "seed_ties": sum(value == 0 for value in differences),
                }
            )
        adjusted = holm_adjust(raw_p)
        for row in family:
            row["holm_adjusted_p"] = adjusted[str(row["policy"])]
        output.extend(family)
    return output


def _summary(
    rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    effect_rows: list[dict[str, Any]],
    *,
    profile: str,
    train_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    checkpoint_count: int,
) -> dict[str, Any]:
    scenarios = scenario_configs()
    expected_rows = (
        len(FIXED_POLICIES) * len(scenarios) * len(eval_seeds)
        + len(CONFIRMATORY_ALGORITHMS)
        * len(train_seeds)
        * len(budgets)
        * len(scenarios)
        * len(eval_seeds)
    )
    expected_checkpoints = (
        len(CONFIRMATORY_ALGORITHMS) * len(train_seeds) * len(budgets)
    )
    final_budget = max(budgets)
    final_base = [
        row
        for row in seed_rows
        if int(row["budget"]) == final_budget
        and row["scenario"] == PRIMARY_SCENARIO
    ]
    coverage = {
        (str(row["policy"]), int(row["train_seed"]), int(row["budget"]), str(row["scenario"]))
        for row in seed_rows
    }
    expected_coverage = {
        (policy, seed, budget, scenario)
        for policy in CONFIRMATORY_ALGORITHMS
        for seed in train_seeds
        for budget in budgets
        for scenario in scenarios
    }
    no_collapse = all(float(row["unique_joint_actions_mean"]) > 1.0 for row in final_base)
    audits = {
        "expected_episode_count": expected_rows,
        "episode_count": len(rows),
        "all_expected_rows_present": len(rows) == expected_rows,
        "expected_checkpoint_count": expected_checkpoints,
        "checkpoint_count": checkpoint_count,
        "all_checkpoints_present": checkpoint_count == expected_checkpoints,
        "complete_factorial_coverage": coverage == expected_coverage,
        "invalid_requests_zero": all(float(row["invalid_requests"]) == 0.0 for row in rows),
        "finite_nonnegative_objectives": all(
            math.isfinite(float(row["objective"])) and float(row["objective"]) >= 0.0
            for row in rows
        ),
        "no_final_joint_action_collapse": no_collapse if profile == "full" else None,
        "sealed_test_panel_closed": True,
    }
    required_audits = (
        "all_expected_rows_present",
        "all_checkpoints_present",
        "complete_factorial_coverage",
        "invalid_requests_zero",
        "finite_nonnegative_objectives",
        "sealed_test_panel_closed",
    ) + (("no_final_joint_action_collapse",) if profile == "full" else ())
    hard_gate = all(bool(audits[key]) for key in required_audits)
    primary = next(
        row
        for row in effect_rows
        if row["policy"] == PRIMARY_POLICY
        and int(row["budget"]) == final_budget
        and row["scenario"] == PRIMARY_SCENARIO
    )
    directional_scenarios = sum(
        float(row["mean_difference"]) < 0.0
        for row in effect_rows
        if row["policy"] == PRIMARY_POLICY and int(row["budget"]) == final_budget
    )
    hypothesis = {
        "primary_policy": PRIMARY_POLICY,
        "reference_policy": REFERENCE_POLICY,
        "scenario": PRIMARY_SCENARIO,
        "budget": final_budget,
        "mean_difference": primary["mean_difference"],
        "ci95_low": primary["ci95_low"],
        "ci95_high": primary["ci95_high"],
        "exact_sign_flip_p": primary["exact_sign_flip_p"],
        "supported": profile == "full"
        and float(primary["mean_difference"]) < 0.0
        and float(primary["ci95_high"]) < 0.0
        and float(primary["exact_sign_flip_p"]) < 0.05,
        "counterfactual_directional_scenarios": directional_scenarios,
        "robust_direction_supported": profile == "full" and directional_scenarios >= 3,
    }
    return {
        "purpose": "Confirmatory ten-seed multi-scenario passive-technician ablation.",
        "profile": profile,
        "primary_hypothesis": hypothesis,
        "audits": audits,
        "hard_gate_passed": hard_gate,
        "scenario_summary": scenario_rows,
        "paired_effects": effect_rows,
        "interpretation_unit": "paired training-seed mean over common evaluation seeds",
        "sealed_test_evaluated": False,
    }


def run(args: argparse.Namespace) -> Path:
    train_seeds, eval_seeds, budgets = _profile(args.profile)
    scenarios = scenario_configs()
    train_config = scenarios[PRIMARY_SCENARIO]
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = ValueTrainSettings(episodes=max(budgets))
    revision, dirty = _git_state()
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "experiment": "passive_technician_confirmatory",
        "profile": args.profile,
        "algorithms": list(CONFIRMATORY_ALGORITHMS),
        "fixed_policies": list(FIXED_POLICIES),
        "budgets": list(budgets),
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(eval_seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "scenarios": {name: asdict(config) for name, config in scenarios.items()},
        "training_scenario": PRIMARY_SCENARIO,
        "value_settings": asdict(settings),
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "requested_device": args.device,
        "resolved_device": str(device),
        "git_revision": revision,
        "git_dirty": dirty,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "started_at": datetime.now(UTC).isoformat(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    benchmark = {
        "protocol": "docs/passive_technician_confirmatory_plan.md",
        "algorithms": CONFIRMATORY_ALGORITHMS,
        "fixed_policies": FIXED_POLICIES,
        "full_train_seeds": FULL_TRAIN_SEEDS,
        "full_evaluation_seeds": FULL_EVALUATION_SEEDS,
        "sealed_test_seeds": SEALED_TEST_SEEDS,
        "full_budgets": FULL_BUDGETS,
        "scenarios": {name: asdict(config) for name, config in scenarios.items()},
        "primary": {
            "policy": PRIMARY_POLICY,
            "reference": REFERENCE_POLICY,
            "scenario": PRIMARY_SCENARIO,
            "budget": FULL_BUDGETS[-1],
        },
    }
    (output / "benchmark_config.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved = benchmark | {
        "profile": args.profile,
        "train_seeds": train_seeds,
        "evaluation_seeds": eval_seeds,
        "budgets": budgets,
        "device": str(device),
        "value_settings": asdict(settings),
    }
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for scenario, config in scenarios.items():
        for seed in tqdm(
            eval_seeds,
            desc=f"fixed/{scenario}",
            unit="episode",
            leave=False,
        ):
            for policy in FIXED_POLICIES:
                rows.append(
                    _episode_row(
                        evaluate_fixed(config, policy, seed),
                        policy,
                        0,
                        scenario,
                        "",
                    )
                )
    _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)

    progress_partial = output / "training_progress.partial.csv"
    progress_fields = ("policy", "train_seed", "episode", "objective", "loss", "epsilon")
    for algorithm in CONFIRMATORY_ALGORITHMS:
        for train_seed in tqdm(train_seeds, desc=algorithm, unit="seed"):
            root = output / algorithm / f"train_seed_{train_seed}"
            checkpoints, progress = train_value_decomposition_checkpoints(
                train_config,
                algorithm,
                train_seed,
                budgets,
                root,
                settings,
                device,
            )
            _append_csv(progress, progress_partial, progress_fields)
            for budget in budgets:
                model = PassiveValueDecomposition.load(
                    checkpoints[budget], train_config, device
                )
                for scenario, config in scenarios.items():
                    for seed in tqdm(
                        eval_seeds,
                        desc=f"eval {algorithm}/{train_seed}/{budget}/{scenario}",
                        unit="episode",
                        leave=False,
                    ):
                        rows.append(
                            _episode_row(
                                evaluate_value_decomposition(
                                    config, model, seed, train_seed, device
                                ),
                                algorithm,
                                budget,
                                scenario,
                                train_seed,
                            )
                        )
                _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)

    _write_csv(rows, output / "episodes.csv", FIELDNAMES)
    _write_csv(rows, output / "coordination.csv", FIELDNAMES)
    shutil.copyfile(progress_partial, output / "training_progress.csv")
    seed_rows = _seed_summaries(rows)
    scenario_rows = _scenario_summaries(rows, seed_rows)
    effect_rows = _paired_effects(seed_rows)
    _write_csv(seed_rows, output / "seed_summary.csv", tuple(seed_rows[0].keys()))
    _write_csv(
        scenario_rows,
        output / "scenario_summary.csv",
        tuple(scenario_rows[0].keys()),
    )
    _write_csv(
        effect_rows,
        output / "paired_effects.csv",
        tuple(effect_rows[0].keys()),
    )
    checkpoint_count = sum(1 for path in output.rglob("model.pt") if path.is_file())
    summary = _summary(
        rows,
        seed_rows,
        scenario_rows,
        effect_rows,
        profile=args.profile,
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        budgets=budgets,
        checkpoint_count=checkpoint_count,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest.update(
        {
            "status": "COMPLETED",
            "finished_at": datetime.now(UTC).isoformat(),
            "episode_count": len(rows),
            "checkpoint_count": checkpoint_count,
            "hard_gate_passed": summary["hard_gate_passed"],
            "outputs": sorted(
                str(path.relative_to(output))
                for path in output.rglob("*")
                if path.is_file()
            ),
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
