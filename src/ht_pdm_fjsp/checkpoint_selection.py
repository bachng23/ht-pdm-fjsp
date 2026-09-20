"""Select PPO checkpoints on validation seeds and evaluate them on held-out tests."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.multi_seed_experiment import _stats, aggregate_results
from ht_pdm_fjsp.rl_experiment import evaluate_ppo, resolve_device


METRICS = (
    "objective",
    "makespan",
    "total_tardiness",
    "failures",
    "preventive_maintenance",
    "corrective_maintenance",
    "total_cost",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_checkpoints(
    validation_rows: Iterable[dict[str, Any]],
    *,
    train_seeds: Iterable[int],
    validation_seeds: Iterable[int],
) -> list[dict[str, Any]]:
    """Select minimum mean validation objective, breaking ties by earlier step."""

    expected_train = tuple(sorted(set(int(seed) for seed in train_seeds)))
    expected_validation = set(int(seed) for seed in validation_seeds)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        train_seed = int(row["train_seed"])
        if train_seed not in expected_train:
            continue
        if str(row["split"]) != "checkpoint_validation":
            raise ValueError("Checkpoint table contains a non-validation row.")
        grouped[(train_seed, int(row["checkpoint_steps"]))].append(row)

    selections: list[dict[str, Any]] = []
    for train_seed in expected_train:
        candidates: list[dict[str, Any]] = []
        for (candidate_seed, step), rows in grouped.items():
            if candidate_seed != train_seed:
                continue
            observed = [int(row["seed"]) for row in rows]
            if len(observed) != len(set(observed)):
                raise ValueError(
                    f"Duplicate validation episode for train seed {train_seed}, "
                    f"checkpoint {step}."
                )
            if set(observed) != expected_validation:
                raise ValueError(
                    f"Incomplete validation panel for train seed {train_seed}, "
                    f"checkpoint {step}."
                )
            candidates.append(
                {
                    "train_seed": train_seed,
                    "checkpoint_steps": step,
                    "validation_episodes": len(rows),
                    "mean_validation_objective": fmean(
                        float(row["objective"]) for row in rows
                    ),
                }
            )
        if not candidates:
            raise ValueError(f"No validation checkpoints for train seed {train_seed}.")
        selections.append(
            min(
                candidates,
                key=lambda row: (
                    float(row["mean_validation_objective"]),
                    int(row["checkpoint_steps"]),
                ),
            )
        )
    return selections


def compare_selected_with_final(
    selected_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return paired selected-minus-final summaries on identical test episodes."""

    final_lookup = {
        (int(row["train_seed"]), int(row["seed"])): row for row in final_rows
    }
    if len(final_lookup) != len(final_rows):
        raise ValueError("Final PPO table contains duplicate train/test seed pairs.")
    deltas: dict[str, list[float]] = {metric: [] for metric in METRICS}
    objective_by_train: dict[int, list[float]] = defaultdict(list)
    for row in selected_rows:
        key = (int(row["train_seed"]), int(row["seed"]))
        if key not in final_lookup:
            raise ValueError(f"Missing paired final PPO episode: {key}.")
        final = final_lookup[key]
        for metric in METRICS:
            delta = float(row[metric]) - float(final[metric])
            deltas[metric].append(delta)
            if metric == "objective":
                objective_by_train[key[0]].append(delta)
    if len(selected_rows) != len(final_lookup):
        raise ValueError("Selected and final PPO tables are not the same size.")
    replicate_means = [
        fmean(objective_by_train[seed]) for seed in sorted(objective_by_train)
    ]
    return {
        "delta_selected_minus_final": {
            metric: _stats(values) for metric, values in deltas.items()
        },
        "objective_delta_across_training_seeds": _stats(replicate_means),
        "pooled_objective_win_rate": fmean(value < 0 for value in deltas["objective"]),
    }


def _selected_checkpoint_path(
    source_dir: Path, *, train_seed: int, checkpoint_steps: int
) -> Path:
    return (
        source_dir
        / f"train_seed_{train_seed}"
        / "checkpoints"
        / f"maskable_ppo_{checkpoint_steps}_steps.zip"
    )


def run_selection_experiment(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_run).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads((source_dir / "run_manifest.json").read_text())
    if source_manifest.get("status") != "COMPLETED":
        raise ValueError("Source replication run is not marked COMPLETED.")
    available_train_seeds = tuple(int(seed) for seed in source_manifest["train_seeds"])
    train_seeds = tuple(
        parse_seeds(args.train_seeds) if args.train_seeds else available_train_seeds
    )
    if not train_seeds or not set(train_seeds).issubset(available_train_seeds):
        raise ValueError("Requested training seeds are not all present in source run.")
    common = source_manifest["common_settings"]
    validation_seeds = tuple(int(seed) for seed in common["validation_seeds"])
    original_test_seeds = tuple(int(seed) for seed in common["test_seeds"])
    test_seeds = tuple(
        parse_seeds(args.test_seeds) if args.test_seeds else original_test_seeds
    )
    if not test_seeds or not set(test_seeds).issubset(original_test_seeds):
        raise ValueError("Requested test seeds are not all present in source run.")
    if set(validation_seeds) & set(test_seeds):
        raise ValueError("Validation and test seed panels must be disjoint.")

    resolved_device = resolve_device(args.device)
    config = BenchmarkConfig.from_json(args.config)
    validation_rows = _read_csv(source_dir / "checkpoint_validation.csv")
    selections = select_checkpoints(
        validation_rows,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
    )
    for selection in selections:
        checkpoint_path = _selected_checkpoint_path(
            source_dir,
            train_seed=int(selection["train_seed"]),
            checkpoint_steps=int(selection["checkpoint_steps"]),
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        selection["checkpoint_path"] = str(checkpoint_path.relative_to(source_dir))

    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "selection_rule": (
            "minimum mean objective on validation seeds; ties choose earlier step"
        ),
        "source_run": str(source_dir),
        "source_manifest_status": source_manifest["status"],
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "test_seeds": test_seeds,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "stable_baselines3": version("stable-baselines3"),
            "sb3_contrib": version("sb3-contrib"),
            "torch": version("torch"),
        },
    }
    manifest_path = output_dir / "selection_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(selections, output_dir / "selected_checkpoints.csv")

    selected_rows: list[dict[str, Any]] = []
    progress = tqdm(
        selections,
        desc="Validation-selected PPO",
        unit="model",
        disable=args.no_progress,
    )
    for selection in progress:
        train_seed = int(selection["train_seed"])
        progress.set_postfix(train_seed=train_seed)
        checkpoint_path = source_dir / str(selection["checkpoint_path"])
        model = MaskablePPO.load(checkpoint_path, device=resolved_device)
        rows = evaluate_ppo(
            model,
            config,
            test_seeds,
            split="test",
            show_progress=False,
        )
        for row in rows:
            row["policy"] = "validation_selected_ppo"
            row["train_seed"] = train_seed
            row["checkpoint_steps"] = int(selection["checkpoint_steps"])
        selected_rows.extend(rows)
        _write_csv(selected_rows, output_dir / "selected_ppo_episodes.partial.csv")
        del model
    progress.close()

    source_final_rows = _read_csv(source_dir / "ppo_episodes.csv")
    final_rows = [
        row
        for row in source_final_rows
        if row["split"] == "test"
        and int(row["train_seed"]) in train_seeds
        and int(row["seed"]) in test_seeds
    ]
    source_baseline_rows = _read_csv(source_dir / "baseline_episodes.csv")
    baseline_rows = [
        row
        for row in source_baseline_rows
        if row["split"] == "test" and int(row["seed"]) in test_seeds
    ]
    expected_ppo_rows = len(train_seeds) * len(test_seeds)
    if len(selected_rows) != expected_ppo_rows or len(final_rows) != expected_ppo_rows:
        raise ValueError("Selected or final PPO test panel is incomplete.")
    baseline_policies = {row["policy"] for row in baseline_rows}
    if any(
        sum(
            row["policy"] == policy and int(row["seed"]) in test_seeds
            for row in baseline_rows
        )
        != len(test_seeds)
        for policy in baseline_policies
    ):
        raise ValueError("Baseline test panel is incomplete.")

    summary = {
        "selection_rule": manifest["selection_rule"],
        "selected_checkpoints": selections,
        "selected_ppo": aggregate_results(selected_rows, baseline_rows),
        "final_ppo": aggregate_results(final_rows, baseline_rows),
        "selected_vs_final": compare_selected_with_final(selected_rows, final_rows),
    }
    _write_csv(selected_rows, output_dir / "selected_ppo_episodes.csv")
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["status"] = "COMPLETED"
    manifest["selected_checkpoint_count"] = len(selections)
    manifest["selected_test_episode_count"] = len(selected_rows)
    manifest["output_files"] = sorted(path.name for path in output_dir.iterdir())
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/minimal_benchmark.json")
    parser.add_argument("--train-seeds")
    parser.add_argument("--test-seeds")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run_selection_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
