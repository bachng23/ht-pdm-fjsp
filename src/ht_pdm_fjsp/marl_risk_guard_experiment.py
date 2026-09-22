"""Evaluate a preventive-aware failure-risk guard on completed MARL policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.marl_diagnostic import DiagnosticPolicy
from ht_pdm_fjsp.marl_diagnostic_experiment import (
    INDEPENDENT_MAPPO,
    _combine_audits,
    _git_revision,
    _read_completed_manifest,
    _validate_source_config,
)
from ht_pdm_fjsp.marl_factorial_experiment import COMBINED_MAPPO, _paired_contrast
from ht_pdm_fjsp.marl_replication_experiment import one_sided_upper_95
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.shared_policy_experiment import METRICS, summarize_architectures


GUARDED_COMBINED = "independent_actor_mappo_broadcast_context_risk_guard"
CONDITIONS = (INDEPENDENT_MAPPO, COMBINED_MAPPO, GUARDED_COMBINED)
EXPECTED_TRAIN_SEEDS = 10


def defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "train_seeds": (15_000, 16_000),
            "diagnostic_seeds": tuple(range(56_900, 56_905)),
        }
    if profile == "full":
        return {
            "train_seeds": tuple(range(15_000, 25_000, 1_000)),
            "diagnostic_seeds": tuple(range(56_000, 56_200)),
        }
    raise ValueError(f"Unknown profile: {profile}")


def apply_failure_risk_guard(
    env: MachineAgentsCTDEEnv,
    observation: dict[str, np.ndarray],
    *,
    threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Mask high-risk production only when valid preventive action remains."""

    guarded = {key: value.copy() for key, value in observation.items()}
    original_masks = observation["action_masks"].astype(bool)
    guarded_masks = guarded["action_masks"].astype(bool)
    suppressed: list[dict[str, Any]] = []
    opportunity_agents = 0
    for agent_index, catalog in enumerate(env.local_action_catalogs):
        preventive_actions = []
        risky_actions = []
        for local_action, global_action in enumerate(catalog):
            if global_action is None or not original_masks[agent_index, local_action]:
                continue
            descriptor = env.core.actions[global_action]
            if descriptor.kind == "preventive":
                preventive_actions.append(local_action)
            elif descriptor.kind == "production":
                risk = env.core.failure_probability(descriptor)
                if risk >= threshold:
                    risky_actions.append((local_action, global_action, risk))
        if not preventive_actions or not risky_actions:
            continue
        opportunity_agents += 1
        for local_action, global_action, risk in risky_actions:
            guarded_masks[agent_index, local_action] = False
            descriptor = env.core.actions[global_action]
            suppressed.append(
                {
                    "agent_index": agent_index,
                    "machine_id": descriptor.machine_id,
                    "local_action": local_action,
                    "global_action": global_action,
                    "failure_probability": risk,
                }
            )
        if not guarded_masks[agent_index].any():
            raise AssertionError("Risk guard removed every local action")
    guarded["action_masks"] = guarded_masks
    return guarded, {
        "opportunity_agents": opportunity_agents,
        "suppressed_actions": suppressed,
    }


def _selected_action_stats(
    env: MachineAgentsCTDEEnv,
    actions: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    risks: list[float] = []
    preventive = 0
    for agent_index, action in enumerate(actions):
        global_action = env.local_to_global(agent_index, int(action))
        if global_action is None:
            continue
        descriptor = env.core.actions[global_action]
        if descriptor.kind == "production":
            risks.append(env.core.failure_probability(descriptor))
        elif descriptor.kind == "preventive":
            preventive += 1
    return {
        "selected_productions": len(risks),
        "selected_risky_productions": sum(risk >= threshold for risk in risks),
        "selected_preventive": preventive,
        "mean_selected_failure_probability": float(np.mean(risks)) if risks else 0.0,
        "max_selected_failure_probability": max(risks, default=0.0),
    }


def evaluate_condition(
    model: DiagnosticPolicy,
    config: BenchmarkConfig,
    seeds: Iterable[int],
    *,
    condition: str,
    train_seed: int,
    device: str,
    threshold: float,
    progress: tqdm,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    audit: dict[str, int] = {}
    guarded_condition = condition == GUARDED_COMBINED
    for seed in seeds:
        progress.set_postfix(
            condition=condition, train_seed=str(train_seed), seed=str(seed)
        )
        env = MachineAgentsCTDEEnv(
            config, include_broadcast_context=model.include_broadcast_context
        )
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        joint_step = 0
        totals = {
            "guard_opportunity_agents": 0,
            "suppressed_risky_actions": 0,
            "altered_agent_actions": 0,
            "selected_risky_productions": 0,
            "selected_preventive": 0,
        }
        while not env._done:
            original_actions, _, _, _ = model.act(
                observation, deterministic=True, device=device
            )
            guard_info: dict[str, Any] = {
                "opportunity_agents": 0,
                "suppressed_actions": [],
            }
            policy_observation = observation
            if guarded_condition:
                policy_observation, guard_info = apply_failure_risk_guard(
                    env, observation, threshold=threshold
                )
            actions, _, _, _ = model.act(
                policy_observation, deterministic=True, device=device
            )
            selected = _selected_action_stats(env, actions, threshold=threshold)
            altered = int(np.count_nonzero(actions != original_actions))
            suppressed_count = len(guard_info["suppressed_actions"])
            totals["guard_opportunity_agents"] += int(
                guard_info["opportunity_agents"]
            )
            totals["suppressed_risky_actions"] += suppressed_count
            totals["altered_agent_actions"] += altered
            totals["selected_risky_productions"] += int(
                selected["selected_risky_productions"]
            )
            totals["selected_preventive"] += int(selected["selected_preventive"])
            decision_rows.append(
                {
                    "condition": condition,
                    "train_seed": train_seed,
                    "seed": seed,
                    "joint_step": joint_step,
                    "simulation_time": env.core.now,
                    "guard_opportunity_agents": guard_info["opportunity_agents"],
                    "suppressed_risky_actions": suppressed_count,
                    "altered_agent_actions": altered,
                    **selected,
                }
            )
            observation, reward, _, _, _ = env.step(actions)
            episode_return += reward
            joint_step += 1
        result = env.result(condition)
        if not np.isclose(episode_return, -float(result.metrics["objective"])):
            raise AssertionError("Risk-guard reward/objective identity failed")
        episode_rows.append(
            {
                "split": "marl_risk_guard_diagnostic",
                "policy": condition,
                "condition": condition,
                "seed": seed,
                "train_seed": train_seed,
                "episode_return": episode_return,
                **result.metrics,
                **totals,
            }
        )
        for key, value in env.coordination_totals.items():
            audit[key] = audit.get(key, 0) + int(value)
        env.close()
        progress.update(1)
    return episode_rows, decision_rows, audit


def mechanism_decision(
    guard_contrast: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    coordination_audits: dict[str, Any],
) -> dict[str, Any]:
    per_seed = guard_contrast["per_training_seed"]
    objective = one_sided_upper_95(
        values["objective"] for values in per_seed.values()
    )
    failures = one_sided_upper_95(values["failures"] for values in per_seed.values())
    changed_by_seed = {
        seed: sum(
            int(row["altered_agent_actions"])
            for row in episode_rows
            if row["condition"] == GUARDED_COMBINED
            and int(row["train_seed"]) == int(seed)
        )
        for seed in per_seed
    }
    checks = {
        "exactly_ten_training_seeds": len(per_seed) == EXPECTED_TRAIN_SEEDS,
        "objective_nonworsening_upper_95_at_most_zero": objective["upper_95"] <= 0,
        "failure_reduction_upper_95_below_zero": failures["upper_95"] < 0,
        "at_least_seven_objective_deltas_nonpositive": sum(
            values["objective"] <= 0 for values in per_seed.values()
        )
        >= 7,
        "at_least_seven_failure_deltas_nonpositive": sum(
            values["failures"] <= 0 for values in per_seed.values()
        )
        >= 7,
        "guard_changed_actions_in_every_training_seed": all(
            count > 0 for count in changed_by_seed.values()
        ),
        "all_coordination_audits_passed": all(
            item["status"] == "PASS" for item in coordination_audits.values()
        ),
    }
    return {
        "supports_risk_aware_learning_followup": all(checks.values()),
        "checks": checks,
        "guard_minus_unguarded_objective_inference": objective,
        "guard_minus_unguarded_failure_inference": failures,
        "changed_agent_actions_by_training_seed": changed_by_seed,
        "note": (
            "Diagnostic mechanism gate only; passing does not authorize final-test "
            "evaluation or deployment of the hard guard."
        ),
    }


def run(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_run).resolve()
    source_manifest = _read_completed_manifest(
        source_dir, "marl_replication_manifest.json"
    )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds) if args.train_seeds else profile["train_seeds"]
    )
    available = {
        int(cell.rsplit(":", 1)[1])
        for cell in source_manifest["completed_training_cells"]
        if str(cell).startswith(f"{COMBINED_MAPPO}:")
    }
    available &= {
        int(cell.rsplit(":", 1)[1])
        for cell in source_manifest["completed_training_cells"]
        if str(cell).startswith(f"{INDEPENDENT_MAPPO}:")
    }
    if not train_seeds or not set(train_seeds) <= available:
        raise ValueError("Requested training seeds are missing from the source run")
    diagnostic_seeds = tuple(
        parse_seeds(args.diagnostic_seeds)
        if args.diagnostic_seeds
        else profile["diagnostic_seeds"]
    )
    forbidden = set(range(20_000, 56_000)) | set(range(50_000, 50_100))
    forbidden |= set(map(int, source_manifest["validation_seeds"]))
    if (
        not diagnostic_seeds
        or len(set(diagnostic_seeds)) != len(diagnostic_seeds)
        or set(diagnostic_seeds) & forbidden
    ):
        raise ValueError("Diagnostic panel overlaps prior or reserved data")

    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    _validate_source_config(source_dir, config_sha256, "replication")
    config = BenchmarkConfig.from_json(config_path)
    threshold = float(config.preventive_probability_threshold)
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)
    device = resolve_device(args.device)
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "profile": args.profile,
        "git_commit": _git_revision(),
        "config_sha256": config_sha256,
        "source_run": str(source_dir),
        "source_git_commit": source_manifest.get("git_commit"),
        "conditions": CONDITIONS,
        "train_seeds": train_seeds,
        "diagnostic_seeds": diagnostic_seeds,
        "failure_probability_threshold": threshold,
        "intervention": (
            "mask feasible production actions at or above the benchmark failure-"
            "probability threshold only when same-machine preventive maintenance "
            "is feasible"
        ),
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "requested_device": args.device,
        "resolved_device": device,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            **{
                name: version(name)
                for name in ("numpy", "stable-baselines3", "sb3-contrib", "torch")
            },
        },
    }
    manifest_path = output_dir / "marl_risk_guard_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(CONDITIONS) * len(train_seeds) * len(diagnostic_seeds),
        desc="MARL risk-guard diagnostic",
        unit="episode",
        disable=args.no_progress,
    )
    for train_seed in train_seeds:
        independent = DiagnosticPolicy.load(
            source_dir / INDEPENDENT_MAPPO / f"train_seed_{train_seed}" / "policy.pt",
            device=device,
        )
        combined = DiagnosticPolicy.load(
            source_dir / COMBINED_MAPPO / f"train_seed_{train_seed}" / "policy.pt",
            device=device,
        )
        for condition, model in (
            (INDEPENDENT_MAPPO, independent),
            (COMBINED_MAPPO, combined),
            (GUARDED_COMBINED, combined),
        ):
            episodes, decisions, audit = evaluate_condition(
                model,
                config,
                diagnostic_seeds,
                condition=condition,
                train_seed=train_seed,
                device=device,
                threshold=threshold,
                progress=progress,
            )
            episode_rows.extend(episodes)
            decision_rows.extend(decisions)
            audit_rows.append({"condition": condition, "train_seed": train_seed, **audit})
            _write_csv(
                episode_rows, output_dir / "marl_risk_guard_episodes.partial.csv"
            )
            _write_csv(
                decision_rows, output_dir / "marl_risk_guard_decisions.partial.csv"
            )
            _write_csv(
                audit_rows, output_dir / "marl_risk_guard_coordination.partial.csv"
            )
        del independent, combined
    progress.close()

    expected = len(CONDITIONS) * len(train_seeds) * len(diagnostic_seeds)
    unique = {
        (row["condition"], int(row["train_seed"]), int(row["seed"]))
        for row in episode_rows
    }
    if len(episode_rows) != expected or len(unique) != expected:
        raise ValueError("Incomplete MARL risk-guard diagnostic panel")
    if any(abs(float(row["episode_return"]) + float(row["objective"])) > 1e-8 for row in episode_rows):
        raise AssertionError("Reward/objective identity failed")
    coordination_audits = _combine_audits(audit_rows)
    if any(item["status"] != "PASS" for item in coordination_audits.values()):
        raise AssertionError("A MARL coordination audit failed")

    summary = summarize_architectures(
        episode_rows,
        conditions=CONDITIONS,
        baseline_condition=INDEPENDENT_MAPPO,
    )
    guard_contrast = _paired_contrast(
        summary,
        treatment=GUARDED_COMBINED,
        control=COMBINED_MAPPO,
        train_seeds=train_seeds,
    )
    guard_vs_independent = _paired_contrast(
        summary,
        treatment=GUARDED_COMBINED,
        control=INDEPENDENT_MAPPO,
        train_seeds=train_seeds,
    )
    summary.update(
        co_primary_endpoints=(
            "guard-minus-unguarded-combined objective and failure deltas by "
            "independent training seed"
        ),
        guard_minus_unguarded_combined=guard_contrast,
        guarded_combined_minus_independent=guard_vs_independent,
        coordination_audits=coordination_audits,
        mechanism_decision=mechanism_decision(
            guard_contrast, episode_rows, coordination_audits
        ),
        future_test_panel_opened=False,
    )
    _write_csv(episode_rows, output_dir / "marl_risk_guard_episodes.csv")
    _write_csv(decision_rows, output_dir / "marl_risk_guard_decisions.csv")
    _write_csv(audit_rows, output_dir / "marl_risk_guard_coordination.csv")
    (output_dir / "marl_risk_guard_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        status="COMPLETED",
        episode_count=len(episode_rows),
        decision_count=len(decision_rows),
        coordination_audits=coordination_audits,
        mechanism_decision=summary["mechanism_decision"],
        output_files=sorted(path.name for path in output_dir.iterdir()),
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed. Artifacts: {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
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
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
