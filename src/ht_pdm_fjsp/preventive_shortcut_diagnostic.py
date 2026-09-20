"""Diagnose preventive-maintenance preferences on identical policy states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.multi_seed_experiment import _stats
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.rl_experiment import resolve_device


PAIRS = {
    "plain": ("shared_scorer", "entity_scorer"),
    "entropy": ("shared_scorer_entropy", "entity_scorer_entropy"),
}
KINDS = ("advance", "production", "preventive", "corrective")
EPISODE_METRICS = (
    "objective",
    "makespan",
    "total_tardiness",
    "failures",
    "preventive_maintenance",
    "corrective_maintenance",
    "total_cost",
)
RISK_BINS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0 + 1e-9)


def diagnostic_defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "train_seeds": (10_000, 11_000),
            "diagnostic_seeds": tuple(range(43_000, 43_003)),
        }
    if profile == "full":
        return {
            "train_seeds": (10_000, 11_000, 12_000, 13_000, 14_000),
            "diagnostic_seeds": tuple(range(43_000, 43_100)),
        }
    raise ValueError(f"Unknown profile: {profile}")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _policy_probabilities(
    model: MaskablePPO,
    observation: dict[str, np.ndarray],
    action_mask: np.ndarray,
) -> np.ndarray:
    filtered = {
        key: observation[key] for key in model.observation_space.spaces
    }
    obs_tensor, _ = model.policy.obs_to_tensor(filtered)
    distribution = model.policy.get_distribution(
        obs_tensor,
        action_masks=action_mask,
    )
    probabilities = (
        distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
    )
    if probabilities.shape != action_mask.shape:
        raise ValueError("Policy probability vector does not match the action mask.")
    if not np.isclose(probabilities.sum(), 1.0):
        raise AssertionError("Masked policy probabilities do not sum to one.")
    if np.any(probabilities[action_mask == 0] > 1e-7):
        raise AssertionError("An infeasible action retained nonzero probability.")
    return probabilities


def _kind_probability(
    env: HTPdmFjspEnv, probabilities: np.ndarray, kind: str
) -> float:
    return float(
        sum(
            probabilities[index]
            for index, descriptor in enumerate(env.actions)
            if descriptor.kind == kind
        )
    )


def _machine_preventive_probability(
    env: HTPdmFjspEnv,
    probabilities: np.ndarray,
    machine_id: str,
) -> float:
    return float(
        sum(
            probabilities[index]
            for index, descriptor in enumerate(env.actions)
            if descriptor.kind == "preventive"
            and descriptor.machine_id == machine_id
        )
    )


def _state_features(env: HTPdmFjspEnv, mask: np.ndarray) -> dict[str, Any]:
    feasible_by_kind = {kind: 0 for kind in KINDS}
    feasible_risks: list[float] = []
    values: dict[str, Any] = {}
    for index in np.flatnonzero(mask):
        descriptor = env.actions[int(index)]
        feasible_by_kind[descriptor.kind] += 1
        if descriptor.kind == "production":
            feasible_risks.append(env.failure_probability(descriptor))
    values.update(
        {f"feasible_{kind}": count for kind, count in feasible_by_kind.items()}
    )
    values["preventive_opportunity"] = feasible_by_kind["preventive"] > 0
    values["max_feasible_production_risk"] = (
        max(feasible_risks) if feasible_risks else math.nan
    )
    values["mean_feasible_production_risk"] = (
        fmean(feasible_risks) if feasible_risks else math.nan
    )
    for machine in env.config.machines:
        machine_id = machine.machine_id
        values[f"{machine_id}_age_ratio"] = (
            env.machines[machine_id].effective_age / machine.weibull_eta
        )
        values[f"{machine_id}_preventive_feasible"] = any(
            mask[index]
            and descriptor.kind == "preventive"
            and descriptor.machine_id == machine_id
            for index, descriptor in enumerate(env.actions)
        )
    return values


def collect_episode(
    *,
    control_model: MaskablePPO,
    treatment_model: MaskablePPO,
    config: BenchmarkConfig,
    pair: str,
    behavior_role: str,
    train_seed: int,
    environment_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Roll out one behavior while scoring both policies on every visited state."""

    if behavior_role not in {"control", "treatment"}:
        raise ValueError(f"Unknown behavior role: {behavior_role}")
    env = HTPdmFjspEnv(config=config, include_action_context=True)
    observation, _ = env.reset(seed=environment_seed)
    state_rows: list[dict[str, Any]] = []
    action_counts = {kind: 0 for kind in KINDS}
    episode_return = 0.0
    while not env._done:
        mask = env.action_masks()
        control_prob = _policy_probabilities(control_model, observation, mask)
        treatment_prob = _policy_probabilities(treatment_model, observation, mask)
        control_action = int(np.argmax(control_prob))
        treatment_action = int(np.argmax(treatment_prob))
        behavior_action = (
            control_action if behavior_role == "control" else treatment_action
        )
        control_kind = env.actions[control_action].kind
        treatment_kind = env.actions[treatment_action].kind
        behavior_kind = env.actions[behavior_action].kind
        state_row: dict[str, Any] = {
            "pair": pair,
            "behavior_role": behavior_role,
            "train_seed": train_seed,
            "seed": environment_seed,
            "decision_index": env.decision_count + 1,
            "time": env.now,
            "feasible_actions": int(mask.sum()),
            "control_action": control_action,
            "treatment_action": treatment_action,
            "control_action_kind": control_kind,
            "treatment_action_kind": treatment_kind,
            "behavior_action": behavior_action,
            "behavior_action_kind": behavior_kind,
            "policies_agree": control_action == treatment_action,
            **_state_features(env, mask),
        }
        for kind in KINDS:
            control_mass = _kind_probability(env, control_prob, kind)
            treatment_mass = _kind_probability(env, treatment_prob, kind)
            state_row[f"control_probability_{kind}"] = control_mass
            state_row[f"treatment_probability_{kind}"] = treatment_mass
            state_row[f"delta_probability_{kind}"] = treatment_mass - control_mass
        for machine in config.machines:
            machine_id = machine.machine_id
            control_mass = _machine_preventive_probability(
                env, control_prob, machine_id
            )
            treatment_mass = _machine_preventive_probability(
                env, treatment_prob, machine_id
            )
            state_row[f"control_probability_preventive_{machine_id}"] = control_mass
            state_row[
                f"treatment_probability_preventive_{machine_id}"
            ] = treatment_mass
            state_row[f"delta_probability_preventive_{machine_id}"] = (
                treatment_mass - control_mass
            )
        state_rows.append(state_row)
        action_counts[behavior_kind] += 1
        observation, reward, _, _, _ = env.step(behavior_action)
        episode_return += reward
    result = env.result(f"{pair}_{behavior_role}")
    if not np.isclose(episode_return, -float(result.metrics["objective"])):
        raise AssertionError("Episode reward does not match the negative objective.")
    episode_row = {
        "pair": pair,
        "behavior_role": behavior_role,
        "train_seed": train_seed,
        "seed": environment_seed,
        "episode_return": episode_return,
        **result.metrics,
        **{f"actions_{kind}": count for kind, count in action_counts.items()},
        "decision_count": len(state_rows),
    }
    env.close()
    return episode_row, state_rows


def _bin_label(value: float) -> str | None:
    if not math.isfinite(value):
        return None
    for low, high in zip(RISK_BINS[:-1], RISK_BINS[1:], strict=True):
        if low <= value < high:
            return f"[{low:.1f},{min(high, 1.0):.1f})"
    return None


def summarize_shortcut(
    state_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train_seeds = sorted({int(row["train_seed"]) for row in state_rows})
    summary: dict[str, Any] = {
        "primary_endpoint": (
            "entity-minus-shared preventive probability on identical states"
        ),
        "replication_unit": "independent PPO training seed",
        "pairs": {},
    }
    for pair in PAIRS:
        pair_summary: dict[str, Any] = {}
        for behavior_role in ("control", "treatment"):
            selected = [
                row
                for row in state_rows
                if row["pair"] == pair
                and row["behavior_role"] == behavior_role
                and bool(row["preventive_opportunity"])
            ]
            if not selected:
                raise ValueError(f"No preventive opportunities for {pair}/{behavior_role}.")
            per_seed: dict[str, Any] = {}
            seed_probability_deltas: list[float] = []
            seed_take_rate_deltas: list[float] = []
            for train_seed in train_seeds:
                seed_rows = [
                    row for row in selected if int(row["train_seed"]) == train_seed
                ]
                probability_delta = fmean(
                    float(row["delta_probability_preventive"]) for row in seed_rows
                )
                control_take_rate = fmean(
                    row["control_action_kind"] == "preventive" for row in seed_rows
                )
                treatment_take_rate = fmean(
                    row["treatment_action_kind"] == "preventive"
                    for row in seed_rows
                )
                take_rate_delta = treatment_take_rate - control_take_rate
                seed_probability_deltas.append(probability_delta)
                seed_take_rate_deltas.append(take_rate_delta)
                per_seed[str(train_seed)] = {
                    "state_count": len(seed_rows),
                    "control_preventive_probability": fmean(
                        float(row["control_probability_preventive"])
                        for row in seed_rows
                    ),
                    "treatment_preventive_probability": fmean(
                        float(row["treatment_probability_preventive"])
                        for row in seed_rows
                    ),
                    "delta_preventive_probability": probability_delta,
                    "control_preventive_take_rate": control_take_rate,
                    "treatment_preventive_take_rate": treatment_take_rate,
                    "delta_preventive_take_rate": take_rate_delta,
                    "action_agreement_rate": fmean(
                        bool(row["policies_agree"]) for row in seed_rows
                    ),
                }
            machine_summary: dict[str, Any] = {}
            machine_ids = sorted(
                key.removesuffix("_preventive_feasible")
                for key in selected[0]
                if key.endswith("_preventive_feasible")
            )
            for machine_id in machine_ids:
                machine_rows = [
                    row
                    for row in selected
                    if bool(row[f"{machine_id}_preventive_feasible"])
                ]
                machine_summary[machine_id] = {
                    "state_count": len(machine_rows),
                    "mean_age_ratio": fmean(
                        float(row[f"{machine_id}_age_ratio"])
                        for row in machine_rows
                    ),
                    "control_preventive_probability": fmean(
                        float(row[f"control_probability_preventive_{machine_id}"])
                        for row in machine_rows
                    ),
                    "treatment_preventive_probability": fmean(
                        float(row[f"treatment_probability_preventive_{machine_id}"])
                        for row in machine_rows
                    ),
                    "delta_preventive_probability": fmean(
                        float(row[f"delta_probability_preventive_{machine_id}"])
                        for row in machine_rows
                    ),
                }
            risk_bins: dict[str, Any] = {}
            labels = sorted(
                {
                    label
                    for row in selected
                    if (label := _bin_label(float(row["max_feasible_production_risk"])))
                    is not None
                }
            )
            for label in labels:
                bin_rows = [
                    row
                    for row in selected
                    if _bin_label(float(row["max_feasible_production_risk"]))
                    == label
                ]
                risk_bins[label] = {
                    "state_count": len(bin_rows),
                    "delta_preventive_probability": fmean(
                        float(row["delta_probability_preventive"])
                        for row in bin_rows
                    ),
                    "control_preventive_take_rate": fmean(
                        row["control_action_kind"] == "preventive"
                        for row in bin_rows
                    ),
                    "treatment_preventive_take_rate": fmean(
                        row["treatment_action_kind"] == "preventive"
                        for row in bin_rows
                    ),
                }
            pair_summary[behavior_role] = {
                "preventive_opportunity_states": len(selected),
                "per_training_seed": per_seed,
                "delta_preventive_probability_across_training_seeds": _stats(
                    seed_probability_deltas
                ),
                "delta_preventive_take_rate_across_training_seeds": _stats(
                    seed_take_rate_deltas
                ),
                "machine_summary": machine_summary,
                "risk_bins": risk_bins,
            }
        pair_episodes = [row for row in episode_rows if row["pair"] == pair]
        behavior_outcomes: dict[str, Any] = {}
        for role in ("control", "treatment"):
            role_rows = [
                row for row in pair_episodes if row["behavior_role"] == role
            ]
            per_seed = {
                str(train_seed): {
                    metric: fmean(
                        float(row[metric])
                        for row in role_rows
                        if int(row["train_seed"]) == train_seed
                    )
                    for metric in EPISODE_METRICS
                }
                for train_seed in train_seeds
            }
            behavior_outcomes[role] = {
                "per_training_seed": per_seed,
                "across_training_seeds": {
                    metric: _stats(
                        per_seed[str(train_seed)][metric]
                        for train_seed in train_seeds
                    )
                    for metric in EPISODE_METRICS
                },
            }
        pair_summary["behavior_outcomes"] = behavior_outcomes
        summary["pairs"][pair] = pair_summary
    return summary


def run_diagnostic(args: argparse.Namespace) -> Path:
    shared_dir = Path(args.shared_source_run).expanduser().resolve()
    entity_dir = Path(args.entity_source_run).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_manifest = json.loads(
        (shared_dir / "architecture_manifest.json").read_text()
    )
    entity_manifest = json.loads((entity_dir / "entity_manifest.json").read_text())
    if shared_manifest.get("status") != "COMPLETED" or entity_manifest.get(
        "status"
    ) != "COMPLETED":
        raise ValueError("Both source experiments must be marked COMPLETED.")
    entity_shared_source = Path(str(entity_manifest["shared_source_run"]))
    if entity_shared_source.name != shared_dir.name:
        raise ValueError("The entity run was not built from the supplied shared run.")
    defaults = diagnostic_defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds)
        if args.train_seeds
        else defaults["train_seeds"]
    )
    available = set(int(seed) for seed in shared_manifest["train_seeds"]) & set(
        int(seed) for seed in entity_manifest["train_seeds"]
    )
    if not train_seeds or not set(train_seeds).issubset(available):
        raise ValueError("Requested training seeds are missing from a source run.")
    diagnostic_seeds = tuple(
        parse_seeds(args.diagnostic_seeds)
        if args.diagnostic_seeds
        else defaults["diagnostic_seeds"]
    )
    forbidden = (
        set(shared_manifest["validation_seeds"])
        | set(entity_manifest["validation_seeds"])
        | set(range(40_000, 43_000))
        | set(range(50_000, 50_100))
    )
    if not diagnostic_seeds or set(diagnostic_seeds) & forbidden:
        raise ValueError("Diagnostic seeds overlap an existing panel.")
    config_path = Path(args.config).expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if entity_manifest.get("config_sha256") != config_hash:
        raise ValueError("Current benchmark config does not match the entity run.")
    config = BenchmarkConfig.from_json(config_path)
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)
    resolved_device = resolve_device(args.device)
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "git_commit": _git_commit(),
        "config_sha256": config_hash,
        "shared_source_run": str(shared_dir),
        "entity_source_run": str(entity_dir),
        "pairs": PAIRS,
        "behavior_roles": ["control", "treatment"],
        "train_seeds": train_seeds,
        "diagnostic_seeds": diagnostic_seeds,
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "stable_baselines3": version("stable-baselines3"),
            "sb3_contrib": version("sb3-contrib"),
            "torch": version("torch"),
        },
    }
    manifest_path = output_dir / "shortcut_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    episode_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    total = len(PAIRS) * 2 * len(train_seeds) * len(diagnostic_seeds)
    progress = tqdm(
        total=total,
        desc="Preventive shortcut diagnostic",
        unit="episode",
        disable=args.no_progress,
    )
    for pair, (control_name, treatment_name) in PAIRS.items():
        for train_seed in train_seeds:
            control_path = (
                shared_dir / control_name / f"train_seed_{train_seed}" / "maskable_ppo.zip"
            )
            treatment_path = (
                entity_dir
                / treatment_name
                / f"train_seed_{train_seed}"
                / "maskable_ppo.zip"
            )
            control_model = MaskablePPO.load(control_path, device=resolved_device)
            treatment_model = MaskablePPO.load(treatment_path, device=resolved_device)
            if control_model.action_space != treatment_model.action_space:
                raise ValueError("Source policies use different action spaces.")
            for behavior_role in ("control", "treatment"):
                progress.set_postfix(
                    pair=pair,
                    train_seed=str(train_seed),
                    behavior=behavior_role,
                )
                for environment_seed in diagnostic_seeds:
                    episode, states = collect_episode(
                        control_model=control_model,
                        treatment_model=treatment_model,
                        config=config,
                        pair=pair,
                        behavior_role=behavior_role,
                        train_seed=train_seed,
                        environment_seed=environment_seed,
                    )
                    episode_rows.append(episode)
                    state_rows.extend(states)
                    progress.update(1)
            _write_csv(episode_rows, output_dir / "shortcut_episodes.partial.csv")
            _write_csv(state_rows, output_dir / "shortcut_states.partial.csv")
            del control_model, treatment_model
    progress.close()
    unique_episodes = {
        (row["pair"], row["behavior_role"], row["train_seed"], row["seed"])
        for row in episode_rows
    }
    if len(episode_rows) != total or len(unique_episodes) != total:
        raise ValueError("Shortcut diagnostic episode panel is incomplete.")
    summary = summarize_shortcut(state_rows, episode_rows)
    _write_csv(episode_rows, output_dir / "shortcut_episodes.csv")
    _write_csv(state_rows, output_dir / "shortcut_states.csv")
    (output_dir / "shortcut_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["status"] = "COMPLETED"
    manifest["episode_count"] = len(episode_rows)
    manifest["state_count"] = len(state_rows)
    manifest["output_files"] = sorted(path.name for path in output_dir.iterdir())
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-source-run", required=True)
    parser.add_argument("--entity-source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/minimal_benchmark.json")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--train-seeds")
    parser.add_argument("--diagnostic-seeds")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run_diagnostic(build_parser().parse_args())


if __name__ == "__main__":
    main()
