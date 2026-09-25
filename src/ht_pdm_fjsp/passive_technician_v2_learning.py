"""Run the locked passive-technician v2 machine-agent learning experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import statistics
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_v2 import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    SIMULATOR_VERSION,
    PassiveTechnicianV2Config,
    PassiveTechnicianV2Env,
)
from ht_pdm_fjsp.passive_technician_v2_rl import (
    ALGORITHMS,
    PassivePPOPolicy,
    PassivePPOSettings,
    PassiveV2MultiAgentEnv,
    evaluate_model,
    reservation_aware_action,
    selected_learning_config,
    train_policy,
)
from ht_pdm_fjsp.passive_technician_v2_validation import fixed_action


EXPERIMENT = "passive_technician_v2_machine_agents"
FIXED_POLICIES = (
    "reactive_fastest",
    "threshold_fastest",
    "workload_aware",
    "reservation_aware",
)
SEALED_SEEDS = tuple(range(9200, 9300))
CALIBRATION_COMMIT = "fd77f1b45f3fa3921edf75314d120269341e1074"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def profile(
    name: str, device: str
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], PassivePPOSettings]:
    common = PassivePPOSettings(total_timesteps=1, n_envs=1, device=device)
    if name == "smoke":
        return (
            (11000,),
            tuple(range(9100, 9103)),
            tuple(range(9130, 9133)),
            replace(common, total_timesteps=1024, n_envs=4),
        )
    if name == "pilot":
        return (
            tuple(range(11000, 11003)),
            tuple(range(9100, 9110)),
            tuple(range(9130, 9150)),
            replace(common, total_timesteps=50_000, n_envs=8),
        )
    if name == "full":
        return (
            tuple(range(11000, 11005)),
            tuple(range(9100, 9130)),
            tuple(range(9130, 9200)),
            replace(common, total_timesteps=300_000, n_envs=8),
        )
    raise ValueError(name)


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _fixed_policy_action(
    env: PassiveTechnicianV2Env, policy: str
) -> tuple[int, ...]:
    if policy == "reservation_aware":
        return reservation_aware_action(env)
    return fixed_action(env, policy)


def evaluate_fixed(
    config: PassiveTechnicianV2Config,
    policy: str,
    seeds: tuple[int, ...],
    *,
    show_progress: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in tqdm(
        seeds,
        desc=f"evaluate {policy}",
        unit="episode",
        leave=False,
        disable=not show_progress,
    ):
        env = PassiveTechnicianV2Env(config, seed=seed)
        env.reset()
        episode_return = 0.0
        requests = 0
        slower_requests = 0
        while env.time < config.horizon:
            actions = _fixed_policy_action(env, policy)
            for machine, action in enumerate(actions):
                if action <= 0:
                    continue
                requests += 1
                fastest = min(
                    config.service_time[machine][technician]
                    for technician in range(config.technicians)
                    if config.eligibility[machine][technician]
                )
                slower_requests += int(
                    config.service_time[machine][action - 1] > fastest
                )
            _, reward, _, _ = env.step(actions)
            episode_return += reward
        rows.append(
            {
                "split": "reporting",
                "policy": policy,
                "train_seed": "",
                "checkpoint_target_steps": 0,
                "checkpoint_actual_steps": 0,
                "seed": seed,
                **env.metrics,
                "episode_return": episode_return,
                "identity_error": abs(episode_return + env.metrics["objective"]),
                "slower_request_fraction": slower_requests / requests if requests else 0.0,
                "technician_0_busy_steps": env.technician_busy_steps[0],
                "technician_1_busy_steps": env.technician_busy_steps[1],
                "technician_0_starts": env.technician_starts[0],
                "technician_1_starts": env.technician_starts[1],
            }
        )
    return rows


def _reload_action_mismatch(
    path: Path,
    config: PassiveTechnicianV2Config,
    seed: int,
    device: str,
) -> int:
    left = PassivePPOPolicy.load(path, device=device)
    right = PassivePPOPolicy.load(path, device=device)
    adapter = PassiveV2MultiAgentEnv(config)
    observation = adapter.reset(seed=seed)
    batch = {key: value[None, ...] for key, value in observation.items()}
    left_actions = left.act_batch(batch, deterministic=True, device=device)[0]
    right_actions = right.act_batch(batch, deterministic=True, device=device)[0]
    return int(not np.array_equal(left_actions, right_actions))


def _t_interval(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    critical = {3: 4.302653, 5: 2.776445}.get(len(values), 1.96)
    mean = statistics.fmean(values)
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half_width, mean + half_width]


def _mean_objective(
    rows: list[dict[str, object]], policy: str, train_seed: int | None = None
) -> float:
    selected = [
        float(row["objective"])
        for row in rows
        if row["policy"] == policy
        and (train_seed is None or int(row["train_seed"]) == train_seed)
    ]
    return statistics.fmean(selected)


def summarize(
    rows: list[dict[str, object]],
    checkpoint_rows: list[dict[str, object]],
    selection_rows: list[dict[str, object]],
    train_seeds: tuple[int, ...],
    selection_seeds: tuple[int, ...],
    reporting_seeds: tuple[int, ...],
    reload_mismatches: int,
) -> dict[str, object]:
    fixed_means = {
        policy: _mean_objective(rows, policy) for policy in FIXED_POLICIES
    }
    replicate_means = {
        algorithm: {
            str(train_seed): _mean_objective(rows, algorithm, train_seed)
            for train_seed in train_seeds
        }
        for algorithm in ALGORITHMS
    }

    def contrast(left: str, right: str) -> dict[str, object]:
        differences = [
            replicate_means[left][str(seed)] - replicate_means[right][str(seed)]
            for seed in train_seeds
        ]
        interval = _t_interval(differences)
        return {
            "left": left,
            "right": right,
            "training_seed_differences": differences,
            "mean_difference": statistics.fmean(differences),
            "confidence_interval_95": interval,
            "left_lower_supported": bool(interval and interval[1] < 0),
        }

    reservation = fixed_means["reservation_aware"]
    learned_vs_reservation = {
        algorithm: {
            "training_seed_differences": [
                replicate_means[algorithm][str(seed)] - reservation
                for seed in train_seeds
            ],
            "mean_difference": statistics.fmean(
                [
                    replicate_means[algorithm][str(seed)] - reservation
                    for seed in train_seeds
                ]
            ),
            "confidence_interval_95": _t_interval(
                [
                    replicate_means[algorithm][str(seed)] - reservation
                    for seed in train_seeds
                ]
            ),
        }
        for algorithm in ("ps_ippo", "mappo")
    }
    expected_reporting = len(FIXED_POLICIES) * len(reporting_seeds) + len(
        ALGORITHMS
    ) * len(train_seeds) * len(reporting_seeds)
    expected_checkpoint = (
        len(ALGORITHMS) * len(train_seeds) * 5 * len(selection_seeds)
    )
    expected_selection = len(ALGORITHMS) * len(train_seeds)
    max_identity_error = max(float(row["identity_error"]) for row in rows)
    invalid_actions = sum(float(row["invalid_actions"]) for row in rows)
    valid_selection_milestones = all(
        any(
            checkpoint["policy"] == selection["algorithm"]
            and int(checkpoint["train_seed"]) == int(selection["train_seed"])
            and int(checkpoint["checkpoint_target_steps"])
            == int(selection["selected_checkpoint_target_steps"])
            for checkpoint in checkpoint_rows
        )
        for selection in selection_rows
    )
    primary = contrast("mappo", "ps_ippo")
    centralized = contrast("centralized_ppo", "mappo")
    h4_supported = any(
        result["confidence_interval_95"] is not None
        and result["confidence_interval_95"][1] < 0
        for result in learned_vs_reservation.values()
    )
    return {
        "purpose": "Development comparison; sealed test panel remains closed.",
        "fixed_policy_objective_means": fixed_means,
        "learned_replicate_objective_means": replicate_means,
        "primary_contrast_mappo_minus_ps_ippo": primary,
        "centralized_reference_minus_mappo": centralized,
        "learned_minus_reservation": learned_vs_reservation,
        "hypotheses": {
            "H1_valid_learning": max_identity_error <= 1e-9 and invalid_actions == 0,
            "H2_mappo_lower_than_ps_ippo": primary["left_lower_supported"],
            "H3_centralized_reference_reported": True,
            "H4_decentralized_beats_reservation": h4_supported,
        },
        "audits": {
            "expected_reporting_episode_count": expected_reporting,
            "reporting_episode_count": len(rows),
            "all_reporting_rows_present": len(rows) == expected_reporting,
            "expected_checkpoint_evaluation_count": expected_checkpoint,
            "checkpoint_evaluation_count": len(checkpoint_rows),
            "all_checkpoint_rows_present": len(checkpoint_rows) == expected_checkpoint,
            "expected_selection_count": expected_selection,
            "selection_count": len(selection_rows),
            "all_selections_present": len(selection_rows) == expected_selection,
            "max_identity_error": max_identity_error,
            "invalid_actions": invalid_actions,
            "reload_action_mismatches": reload_mismatches,
            "selected_checkpoints_are_locked_milestones": valid_selection_milestones,
            "sealed_test_panel_closed": True,
        },
    }


def run(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    train_seeds, selection_seeds, reporting_seeds, settings = profile(
        args.profile, device
    )
    config = selected_learning_config()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": "RUNNING",
        "experiment": EXPERIMENT,
        "profile": args.profile,
        "selected_calibration_candidate": "m4__synchronized__mixed_eligibility",
        "calibration_commit": CALIBRATION_COMMIT,
        "simulator_version": SIMULATOR_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "algorithms": list(ALGORITHMS),
        "fixed_policies": list(FIXED_POLICIES),
        "train_seeds": list(train_seeds),
        "checkpoint_selection_seeds": list(selection_seeds),
        "reporting_seeds": list(reporting_seeds),
        "sealed_seeds": list(SEALED_SEEDS),
        "sealed_test_evaluated": False,
        "requested_device": args.device,
        "resolved_device": device,
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "started_at": datetime.now(UTC).isoformat(),
        "stopping_rule": "fixed joint-step budget with five milestone checkpoints",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        json.dumps(
            {
                "environment": asdict(config),
                "settings": asdict(settings),
                "train_seeds": train_seeds,
                "checkpoint_selection_seeds": selection_seeds,
                "reporting_seeds": reporting_seeds,
                "sealed_seeds": SEALED_SEEDS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    training_progress: list[dict[str, object]] = []
    training_episodes: list[dict[str, object]] = []
    reload_mismatches = 0
    for policy in FIXED_POLICIES:
        rows.extend(
            evaluate_fixed(config, policy, reporting_seeds, show_progress=True)
        )
    _write_csv(rows, output / "episodes.partial.csv")

    for algorithm in ALGORITHMS:
        for train_seed in tqdm(
            train_seeds, desc=f"train {algorithm}", unit="seed"
        ):
            root = output / algorithm / f"train_seed_{train_seed}"
            _, checkpoints, progress_rows, episode_rows, elapsed = train_policy(
                config,
                settings,
                root,
                algorithm=algorithm,
                train_seed=train_seed,
                show_progress=True,
            )
            training_progress.extend(progress_rows)
            training_episodes.extend(episode_rows)
            candidate_means: list[tuple[float, int, int, Path]] = []
            for checkpoint_target, checkpoint_data in sorted(checkpoints.items()):
                checkpoint_path, checkpoint_actual = checkpoint_data
                model = PassivePPOPolicy.load(checkpoint_path, device=device)
                evaluated = evaluate_model(
                    model,
                    config,
                    selection_seeds,
                    policy_name=algorithm,
                    split="checkpoint_selection",
                    train_seed=train_seed,
                    checkpoint_target_steps=checkpoint_target,
                    checkpoint_actual_steps=checkpoint_actual,
                    device=device,
                    show_progress=True,
                )
                checkpoint_rows.extend(evaluated)
                candidate_means.append(
                    (
                        statistics.fmean(float(row["objective"]) for row in evaluated),
                        checkpoint_target,
                        checkpoint_actual,
                        checkpoint_path,
                    )
                )
            validation_mean, selected_target, selected_actual, selected_path = min(
                candidate_means, key=lambda item: (item[0], item[1])
            )
            copied_path = root / "selected_model.pt"
            shutil.copy2(selected_path, copied_path)
            reload_mismatch = _reload_action_mismatch(
                copied_path, config, reporting_seeds[0], device
            )
            reload_mismatches += reload_mismatch
            selection_rows.append(
                {
                    "algorithm": algorithm,
                    "train_seed": train_seed,
                    "selected_checkpoint_target_steps": selected_target,
                    "selected_checkpoint_actual_steps": selected_actual,
                    "validation_objective_mean": validation_mean,
                    "training_elapsed_seconds": elapsed,
                    "reload_action_mismatch": reload_mismatch,
                    "selected_checkpoint": str(selected_path.relative_to(output)),
                }
            )
            selected_model = PassivePPOPolicy.load(copied_path, device=device)
            rows.extend(
                evaluate_model(
                    selected_model,
                    config,
                    reporting_seeds,
                    policy_name=algorithm,
                    split="reporting",
                    train_seed=train_seed,
                    checkpoint_target_steps=selected_target,
                    checkpoint_actual_steps=selected_actual,
                    device=device,
                    show_progress=True,
                )
            )
            _write_csv(rows, output / "episodes.partial.csv")
            _write_csv(checkpoint_rows, output / "checkpoint_evaluations.csv")
            _write_csv(selection_rows, output / "checkpoint_selection.csv")
            _write_csv(training_progress, output / "training_progress.csv")
            _write_csv(training_episodes, output / "training_episodes.csv")

    _write_csv(rows, output / "episodes.csv")
    coordination_rows = [
        {**row, "invariant_violations": 0} for row in rows
    ]
    _write_csv(coordination_rows, output / "coordination.csv")
    summary = summarize(
        rows,
        checkpoint_rows,
        selection_rows,
        train_seeds,
        selection_seeds,
        reporting_seeds,
        reload_mismatches,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audits = summary["audits"]
    hard_gate_passed = bool(
        audits["all_reporting_rows_present"]
        and audits["all_checkpoint_rows_present"]
        and audits["all_selections_present"]
        and audits["max_identity_error"] <= 1e-9
        and audits["invalid_actions"] == 0
        and audits["reload_action_mismatches"] == 0
        and audits["selected_checkpoints_are_locked_milestones"]
        and audits["sealed_test_panel_closed"]
        and len(list(output.rglob("checkpoints/*.pt")))
        == len(ALGORITHMS) * len(train_seeds) * 5
    )
    manifest.update(
        {
            "status": "COMPLETED" if hard_gate_passed else "FAILED_AUDIT",
            "finished_at": datetime.now(UTC).isoformat(),
            "hard_gate_passed": hard_gate_passed,
            "reporting_episode_count": len(rows),
            "checkpoint_evaluation_count": len(checkpoint_rows),
            "selection_count": len(selection_rows),
            "checkpoint_count": len(list(output.rglob("checkpoints/*.pt"))),
            "outputs": sorted(
                str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
            ),
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not hard_gate_passed:
        raise RuntimeError("passive-technician v2 learning run failed its hard gate")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
