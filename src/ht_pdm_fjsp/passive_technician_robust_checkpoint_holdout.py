"""Locked holdout validation for robust RA-QMIX checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_leave_one_out import (
    COST_TOLERANCE,
    SEALED_TEST_SEEDS,
    _budget_summary,
    _git_state,
    _runtime_metadata,
    _write_csv,
    _write_json,
    environment_cells,
    evaluate_diagnostic_episode,
)
from ht_pdm_fjsp.passive_technician_marl import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
)
from ht_pdm_fjsp.passive_technician_stress_checkpoint_sweep import (
    ALGORITHMS as SOURCE_ALGORITHMS,
    FULL_BUDGETS as SOURCE_BUDGETS,
    PROTOCOL_VERSION as SOURCE_PROTOCOL_VERSION,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition,
)


PROTOCOL_VERSION = "ra_qmix_robust_checkpoint_holdout_v1"
SOURCE_GIT_REVISION = "72bd25efa63f8df843d0dec1cd7ca7f81ece75f6"
SOURCE_TRAIN_SEEDS = (83_100, 83_101, 83_102, 83_103, 83_104)
SOURCE_EVALUATION_SEEDS = tuple(range(83_500, 83_600))
FULL_EVALUATION_SEEDS = tuple(range(83_600, 83_700))
SMOKE_EVALUATION_SEEDS = (83_700, 83_701, 83_702)
SELECTED_CANDIDATE = ("tqmix", 40_000)
PRIMARY_COMPARATOR = ("tqmix", 50_000)
REPORTING_NAMES = {
    "qmix": "Standard QMIX",
    "tqmix_no_queue": "RA-QMIX - queue",
    "tqmix_no_cf": "RA-QMIX - CF",
    "tqmix": "Full RA-QMIX",
}
CANDIDATES = (
    ("selected_full_40k", "tqmix", 40_000, "selected"),
    ("full_50k", "tqmix", 50_000, "primary_comparator"),
    ("qmix_50k", "qmix", 50_000, "baseline"),
    ("no_queue_50k", "tqmix_no_queue", 50_000, "ablation_comparator"),
    ("no_cf_50k", "tqmix_no_cf", 50_000, "parameter_matched_comparator"),
)


def profile_settings(profile: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if profile == "smoke":
        return (SOURCE_TRAIN_SEEDS[:1], SMOKE_EVALUATION_SEEDS)
    if profile == "full":
        return (SOURCE_TRAIN_SEEDS, FULL_EVALUATION_SEEDS)
    raise ValueError(profile)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_checkpoint(source: Path, algorithm: str, train_seed: int, budget: int) -> Path:
    return (
        source
        / algorithm
        / f"train_seed_{train_seed}"
        / "checkpoints"
        / f"budget_{budget}"
        / "model.pt"
    )


def selection_scores(
    source_summary: list[dict[str, str]], scenarios: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, float], tuple[str, int]]:
    expected_keys = {
        (algorithm, train_seed, scenario, budget)
        for algorithm in SOURCE_ALGORITHMS
        for train_seed in SOURCE_TRAIN_SEEDS
        for scenario in scenarios
        for budget in SOURCE_BUDGETS
    }
    values = {
        (
            row["algorithm"],
            int(row["train_seed"]),
            row["scenario"],
            int(row["budget"]),
        ): float(row["objective_mean"])
        for row in source_summary
    }
    if set(values) != expected_keys:
        missing = len(expected_keys - set(values))
        extra = len(set(values) - expected_keys)
        raise ValueError(f"source budget summary keys mismatch: missing={missing}, extra={extra}")

    means = {
        (algorithm, budget, scenario): statistics.fmean(
            values[algorithm, train_seed, scenario, budget]
            for train_seed in SOURCE_TRAIN_SEEDS
        )
        for algorithm in SOURCE_ALGORITHMS
        for budget in SOURCE_BUDGETS
        for scenario in scenarios
    }
    references = {
        scenario: min(
            means[algorithm, budget, scenario]
            for algorithm in SOURCE_ALGORITHMS
            for budget in SOURCE_BUDGETS
        )
        for scenario in scenarios
    }
    result: list[dict[str, Any]] = []
    ranked: list[tuple[float, float, int, str]] = []
    for algorithm in SOURCE_ALGORITHMS:
        for budget in SOURCE_BUDGETS:
            regrets = {
                scenario: means[algorithm, budget, scenario] / references[scenario] - 1.0
                for scenario in scenarios
            }
            max_regret = max(regrets.values())
            mean_regret = statistics.fmean(regrets.values())
            ranked.append((max_regret, mean_regret, budget, algorithm))
            result.append(
                {
                    "algorithm": algorithm,
                    "reporting_name": REPORTING_NAMES[algorithm],
                    "budget": budget,
                    **{
                        f"{scenario}_objective_mean": means[algorithm, budget, scenario]
                        for scenario in scenarios
                    },
                    **{
                        f"{scenario}_relative_regret": regrets[scenario]
                        for scenario in scenarios
                    },
                    "max_relative_regret": max_regret,
                    "mean_relative_regret": mean_regret,
                    "selected": False,
                }
            )
    selected_rank = min(ranked)
    selected = (selected_rank[3], selected_rank[2])
    for row in result:
        row["selected"] = (row["algorithm"], int(row["budget"])) == selected
    return result, references, selected


def robust_scores(
    summary_rows: list[dict[str, Any]], references: dict[str, float]
) -> list[dict[str, Any]]:
    candidate_ids = {
        (algorithm, budget): candidate_id
        for candidate_id, algorithm, budget, _ in CANDIDATES
    }
    means = {
        (str(row["algorithm"]), int(row["budget"]), int(row["train_seed"]), str(row["scenario"])): float(row["objective_mean"])
        for row in summary_rows
    }
    scenarios = tuple(references)
    result: list[dict[str, Any]] = []
    grouped: dict[str, list[tuple[float, float]]] = {}
    for candidate_id, algorithm, budget, role in CANDIDATES:
        for train_seed in sorted({key[2] for key in means if key[0] == algorithm and key[1] == budget}):
            regrets = {
                scenario: means[algorithm, budget, train_seed, scenario]
                / references[scenario]
                - 1.0
                for scenario in scenarios
            }
            max_regret = max(regrets.values())
            mean_regret = statistics.fmean(regrets.values())
            grouped.setdefault(candidate_id, []).append((max_regret, mean_regret))
            result.append(
                {
                    "row_type": "training_seed",
                    "candidate_id": candidate_id,
                    "algorithm": algorithm,
                    "reporting_name": REPORTING_NAMES[algorithm],
                    "budget": budget,
                    "role": role,
                    "train_seed": train_seed,
                    **{f"{scenario}_relative_regret": regrets[scenario] for scenario in scenarios},
                    "max_relative_regret": max_regret,
                    "mean_relative_regret": mean_regret,
                    "max_regret_mean": "",
                    "max_regret_sd": "",
                    "mean_regret_mean": "",
                    "mean_regret_sd": "",
                }
            )
    for candidate_id, pairs in grouped.items():
        algorithm, budget = next(key for key, value in candidate_ids.items() if value == candidate_id)
        role = next(row[3] for row in CANDIDATES if row[0] == candidate_id)
        maxima = [pair[0] for pair in pairs]
        averages = [pair[1] for pair in pairs]
        result.append(
            {
                "row_type": "across_training_seeds",
                "candidate_id": candidate_id,
                "algorithm": algorithm,
                "reporting_name": REPORTING_NAMES[algorithm],
                "budget": budget,
                "role": role,
                "train_seed": "",
                **{f"{scenario}_relative_regret": "" for scenario in scenarios},
                "max_relative_regret": "",
                "mean_relative_regret": "",
                "max_regret_mean": statistics.fmean(maxima),
                "max_regret_sd": statistics.stdev(maxima) if len(maxima) > 1 else 0.0,
                "mean_regret_mean": statistics.fmean(averages),
                "mean_regret_sd": statistics.stdev(averages) if len(averages) > 1 else 0.0,
            }
        )
    return result


def primary_results(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_seed = {
        (row["candidate_id"], int(row["train_seed"])): float(row["max_relative_regret"])
        for row in score_rows
        if row["row_type"] == "training_seed"
    }
    train_seeds = sorted(seed for candidate, seed in per_seed if candidate == "selected_full_40k")
    result: list[dict[str, Any]] = []
    deltas: list[float] = []
    for train_seed in train_seeds:
        selected = per_seed["selected_full_40k", train_seed]
        comparator = per_seed["full_50k", train_seed]
        delta = selected - comparator
        deltas.append(delta)
        result.append(
            {
                "row_type": "training_seed",
                "train_seed": train_seed,
                "selected_candidate": "selected_full_40k",
                "comparator_candidate": "full_50k",
                "selected_max_regret": selected,
                "comparator_max_regret": comparator,
                "delta_selected_minus_comparator": delta,
                "delta_mean": "",
                "delta_sd": "",
                "delta_min": "",
                "delta_max": "",
                "selected_better_count": "",
                "directional_replication": "",
            }
        )
    selected_better_count = sum(delta < 0 for delta in deltas)
    required_count = 4 if len(deltas) == 5 else len(deltas)
    result.append(
        {
            "row_type": "across_training_seeds",
            "train_seed": "",
            "selected_candidate": "selected_full_40k",
            "comparator_candidate": "full_50k",
            "selected_max_regret": "",
            "comparator_max_regret": "",
            "delta_selected_minus_comparator": "",
            "delta_mean": statistics.fmean(deltas),
            "delta_sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "selected_better_count": selected_better_count,
            "directional_replication": statistics.fmean(deltas) < 0
            and selected_better_count >= required_count,
        }
    )
    return result


def _validate_source(source: Path, scenarios: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest_path = source / "manifest.json"
    summary_path = source / "budget_summary.csv"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("source manifest.json and budget_summary.csv are required")
    manifest = _read_json(manifest_path)
    required_audits = (
        "all_expected_rows_present",
        "unique_episode_keys",
        "all_checkpoints_present",
        "all_trajectories_updated",
        "losses_finite",
        "cost_reconciliation_passed",
        "resource_semantics_passed",
        "sealed_test_panel_closed",
        "seed_panels_disjoint",
    )
    locked_counts = {
        "evaluation_rows": 48_000,
        "checkpoints": 120,
        "training_trajectories": 20,
    }
    checks = (
        manifest.get("status") == "COMPLETED",
        manifest.get("protocol_version") == SOURCE_PROTOCOL_VERSION,
        manifest.get("git_revision") == SOURCE_GIT_REVISION,
        manifest.get("git_dirty") is False,
        tuple(manifest.get("algorithms", ())) == SOURCE_ALGORITHMS,
        tuple(manifest.get("checkpoint_budgets", ())) == SOURCE_BUDGETS,
        tuple(manifest.get("train_seeds", ())) == SOURCE_TRAIN_SEEDS,
        tuple(manifest.get("evaluation_seeds", ())) == SOURCE_EVALUATION_SEEDS,
        manifest.get("expected_counts") == locked_counts,
        manifest.get("actual_counts") == locked_counts,
        manifest.get("sealed_test_evaluated") is False,
        all(manifest.get("audits", {}).get(name) is True for name in required_audits),
        set(manifest.get("environment_cells", {})) == set(scenarios),
    )
    if not all(checks):
        raise ValueError("source manifest does not match the locked completed experiment")
    return manifest, _read_csv(summary_path)


def run(args: argparse.Namespace) -> Path:
    train_seeds, evaluation_seeds = profile_settings(args.profile)
    source = Path(args.source_run).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    panels = (
        set(SOURCE_TRAIN_SEEDS),
        set(SOURCE_EVALUATION_SEEDS),
        set(evaluation_seeds),
        set(SEALED_TEST_SEEDS),
    )
    if any(panels[left] & panels[right] for left in range(4) for right in range(left + 1, 4)):
        raise ValueError("training, development, holdout, and sealed seed panels must be disjoint")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    cells = environment_cells()
    scenarios = tuple(cells)
    source_manifest, source_summary = _validate_source(source, scenarios)
    scores, references, selected = selection_scores(source_summary, scenarios)
    if selected != SELECTED_CANDIDATE:
        raise RuntimeError(f"locked selection mismatch: expected {SELECTED_CANDIDATE}, got {selected}")

    required_checkpoints = [
        _source_checkpoint(source, algorithm, train_seed, budget)
        for _, algorithm, budget, _ in CANDIDATES
        for train_seed in train_seeds
    ]
    missing = [str(path) for path in required_checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} locked source checkpoints: {missing[:3]}")
    source_files = [source / "manifest.json", source / "budget_summary.csv", *required_checkpoints]
    source_hashes_before = {str(path): _sha256(path) for path in source_files}

    output.mkdir(parents=True, exist_ok=True)
    revision, dirty = _git_state()
    expected_rows = len(CANDIDATES) * len(train_seeds) * len(scenarios) * len(evaluation_seeds)
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "experiment": "ra_qmix_robust_checkpoint_holdout",
        "protocol_version": PROTOCOL_VERSION,
        "profile": args.profile,
        "hypothesis": "Development-selected Full RA-QMIX@40k has lower frozen-reference worst-scenario regret than Full RA-QMIX@50k on the fresh holdout panel.",
        "primary_endpoint": "paired per-training-seed difference in maximum frozen-reference relative regret: selected_full_40k minus full_50k",
        "directional_replication_rule": "mean primary delta < 0 and selected checkpoint better in at least 4 of 5 training seeds",
        "selection_rule": "minimize (maximum scenario regret, mean scenario regret, checkpoint budget, algorithm ID) over the 24 development candidates",
        "selected_candidate": {"algorithm": selected[0], "budget": selected[1]},
        "candidates": [
            {"candidate_id": item[0], "algorithm": item[1], "budget": item[2], "role": item[3]}
            for item in CANDIDATES
        ],
        "source_run": str(source),
        "source_protocol_version": source_manifest["protocol_version"],
        "source_git_revision": source_manifest["git_revision"],
        "source_train_seeds": list(SOURCE_TRAIN_SEEDS),
        "source_development_evaluation_seeds": list(SOURCE_EVALUATION_SEEDS),
        "evaluated_train_seeds": list(train_seeds),
        "holdout_evaluation_seeds": list(evaluation_seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "frozen_scenario_references": references,
        "environment_cells": {name: asdict(config) for name, config in cells.items()},
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "requested_device": args.device,
        "resolved_device": str(device),
        "torch_cpu_threads": torch.get_num_threads(),
        "git_revision": revision,
        "git_dirty": dirty,
        "runtime": _runtime_metadata(),
        "stopping_rule": "Evaluate the five locked candidates once; stop on missing/mutated source, selection mismatch, error, non-finite metric, or audit failure.",
        "expected_counts": {
            "evaluation_rows": expected_rows,
            "source_checkpoints": len(required_checkpoints),
            "candidate_training_seed_cells": len(CANDIDATES) * len(train_seeds),
        },
        "started_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "source_manifest_snapshot.json", source_manifest)
    _write_json(
        output / "benchmark_config.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "environment_cells": manifest["environment_cells"],
            "objective_version": OBJECTIVE_VERSION,
            "observation_version": OBSERVATION_VERSION,
        },
    )
    _write_json(
        output / "resolved_config.json",
        {
            "profile": args.profile,
            "source_run": str(source),
            "train_seeds": train_seeds,
            "evaluation_seeds": evaluation_seeds,
            "candidates": manifest["candidates"],
            "frozen_scenario_references": references,
            "device": str(device),
        },
    )

    try:
        _write_csv(output / "selection_scores.csv", scores)
        _write_csv(
            output / "scenario_references.csv",
            [{"scenario": scenario, "reference_objective": value} for scenario, value in references.items()],
        )
        _write_csv(output / "candidate_registry.csv", manifest["candidates"])
        inventory = [
            {
                "candidate_id": candidate_id,
                "algorithm": algorithm,
                "budget": budget,
                "train_seed": train_seed,
                "path": str(_source_checkpoint(source, algorithm, train_seed, budget)),
                "sha256": source_hashes_before[str(_source_checkpoint(source, algorithm, train_seed, budget))],
            }
            for candidate_id, algorithm, budget, _ in CANDIDATES
            for train_seed in train_seeds
        ]
        _write_csv(output / "source_checkpoint_inventory.csv", inventory)

        episode_rows: list[dict[str, Any]] = []
        progress_rows: list[dict[str, Any]] = []
        train_config = cells["in_distribution"]
        for candidate_id, algorithm, budget, role in CANDIDATES:
            for train_seed in tqdm(
                train_seeds, desc=f"holdout seeds/{candidate_id}", unit="training-seed"
            ):
                checkpoint = _source_checkpoint(source, algorithm, train_seed, budget)
                model = PassiveValueDecomposition.load(checkpoint, train_config, device)
                if model.algorithm != algorithm:
                    raise ValueError(f"checkpoint algorithm mismatch at {checkpoint}")
                for scenario, config in cells.items():
                    for eval_seed in tqdm(
                        evaluation_seeds,
                        desc=f"evaluate {candidate_id}/{train_seed}/{scenario}",
                        unit="episode",
                        leave=False,
                    ):
                        row = evaluate_diagnostic_episode(
                            config,
                            model,
                            eval_seed,
                            train_seed,
                            scenario,
                            budget,
                            device,
                        )
                        episode_rows.append(
                            {"candidate_id": candidate_id, "candidate_role": role, **row}
                        )
                    progress = {
                        "candidate_id": candidate_id,
                        "algorithm": algorithm,
                        "budget": budget,
                        "train_seed": train_seed,
                        "scenario": scenario,
                        "completed_evaluation_episodes": len(evaluation_seeds),
                    }
                    progress_rows.append(progress)
                    _append_csv(output / "evaluation_progress.csv", [progress])
                    _write_csv(output / "episodes.partial.csv", episode_rows)

        _write_csv(output / "episodes.csv", episode_rows)
        coordination_rows = [
            {
                "candidate_id": row["candidate_id"],
                "algorithm": row["algorithm"],
                "budget": row["budget"],
                "train_seed": row["train_seed"],
                "scenario": row["scenario"],
                "eval_seed": row["eval_seed"],
                "invalid_requests": row["invalid_requests"],
                "collisions": row["collisions"],
                "cost_reconciliation_error": row["cost_reconciliation_error"],
                "cost_reconciled": abs(float(row["cost_reconciliation_error"])) <= COST_TOLERANCE,
                "resource_semantics_valid": all(
                    0.0 <= float(row[f"technician_{index}_utilization"]) <= 1.0
                    for index in range(train_config.technicians)
                ),
            }
            for row in episode_rows
        ]
        _write_csv(output / "coordination.csv", coordination_rows)
        budget_summary = _budget_summary(episode_rows)
        score_rows = robust_scores(budget_summary, references)
        primary = primary_results(score_rows)
        _write_csv(output / "budget_summary.csv", budget_summary)
        _write_csv(output / "robust_scores.csv", score_rows)
        _write_csv(output / "primary_results.csv", primary)

        keys = [
            (
                row["candidate_id"],
                row["train_seed"],
                row["scenario"],
                row["eval_seed"],
            )
            for row in episode_rows
        ]
        source_hashes_after = {str(path): _sha256(path) for path in source_files}
        audits = {
            "source_manifest_valid": True,
            "selection_reproduced": selected == SELECTED_CANDIDATE,
            "expected_evaluation_rows": expected_rows,
            "actual_evaluation_rows": len(episode_rows),
            "all_expected_rows_present": len(episode_rows) == expected_rows,
            "unique_episode_keys": len(keys) == len(set(keys)),
            "expected_source_checkpoints": len(required_checkpoints),
            "actual_source_checkpoints": len(inventory),
            "all_source_checkpoints_present": len(inventory) == len(required_checkpoints),
            "source_files_unchanged": source_hashes_before == source_hashes_after,
            "all_objectives_finite": all(math.isfinite(float(row["objective"])) for row in episode_rows),
            "cost_reconciliation_passed": all(row["cost_reconciled"] for row in coordination_rows),
            "resource_semantics_passed": all(row["resource_semantics_valid"] for row in coordination_rows),
            "holdout_progress_complete": len(progress_rows) == len(CANDIDATES) * len(train_seeds) * len(scenarios),
            "seed_panels_disjoint": not any(
                panels[left] & panels[right]
                for left in range(4)
                for right in range(left + 1, 4)
            ),
            "sealed_test_panel_closed": not any(int(row["eval_seed"]) in SEALED_TEST_SEEDS for row in episode_rows),
        }
        required_audits = (
            "source_manifest_valid",
            "selection_reproduced",
            "all_expected_rows_present",
            "unique_episode_keys",
            "all_source_checkpoints_present",
            "source_files_unchanged",
            "all_objectives_finite",
            "cost_reconciliation_passed",
            "resource_semantics_passed",
            "holdout_progress_complete",
            "seed_panels_disjoint",
            "sealed_test_panel_closed",
        )
        if not all(audits[name] for name in required_audits):
            raise RuntimeError(f"holdout audit failed: {audits}")
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "primary_endpoint": manifest["primary_endpoint"],
            "selected_candidate": manifest["selected_candidate"],
            "frozen_scenario_references": references,
            "primary_results": primary,
            "audits": audits,
        }
        _write_json(output / "summary.json", summary)
        manifest.update(
            {
                "status": "COMPLETED",
                "finished_at": datetime.now(UTC).isoformat(),
                "actual_counts": {
                    "evaluation_rows": len(episode_rows),
                    "source_checkpoints": len(inventory),
                    "candidate_training_seed_cells": len(CANDIDATES) * len(train_seeds),
                },
                "audits": audits,
                "outputs": sorted(
                    str(path.relative_to(output))
                    for path in output.rglob("*")
                    if path.is_file()
                ),
            }
        )
        _write_json(output / "manifest.json", manifest)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return output
    except BaseException as error:
        manifest.update(
            {
                "status": "FAILED",
                "finished_at": datetime.now(UTC).isoformat(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        _write_json(output / "manifest.json", manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
