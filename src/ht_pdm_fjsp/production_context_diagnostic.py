"""Diagnose seed instability introduced by production-only action context."""

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
import torch as th
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.multi_seed_experiment import _stats
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.rl_experiment import resolve_device


KINDS = ("advance", "production", "preventive", "corrective")
STATE_METRICS = (
    "delta_normalized_entropy",
    "delta_max_probability",
    "delta_effective_action_count",
    "delta_logit_span",
    "policies_agree",
    "zero_context_action_flip",
    "normal_zero_js_divergence",
    "context_base_input_norm_ratio",
    "context_base_preactivation_ratio",
    *(f"delta_probability_{kind}" for kind in KINDS),
)
EPISODE_METRICS = (
    "objective",
    "makespan",
    "total_tardiness",
    "failures",
    "preventive_maintenance",
    "corrective_maintenance",
    "total_cost",
)


def diagnostic_defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "train_seeds": (10_000, 11_000),
            "diagnostic_seeds": tuple(range(45_000, 45_003)),
        }
    if profile == "full":
        return {
            "train_seeds": (10_000, 11_000, 12_000, 13_000, 14_000),
            "diagnostic_seeds": tuple(range(45_000, 45_100)),
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


def _load_manifest(path: Path, filename: str) -> dict[str, Any]:
    manifest_path = path / filename
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"Source run is not marked COMPLETED: {path}")
    if manifest.get("future_test_panel_opened"):
        raise ValueError(f"Source run unexpectedly opened the test panel: {path}")
    return manifest


def _policy_state(
    model: MaskablePPO,
    observation: dict[str, np.ndarray],
    action_mask: np.ndarray,
    *,
    zero_production_context: bool = False,
) -> dict[str, Any]:
    filtered = {key: observation[key] for key in model.observation_space.spaces}
    obs_tensor, _ = model.policy.obs_to_tensor(filtered)
    if zero_production_context:
        if "production_context" not in obs_tensor:
            raise ValueError("Cannot zero missing production_context.")
        obs_tensor = {key: value.clone() for key, value in obs_tensor.items()}
        obs_tensor["production_context"].zero_()
    with th.no_grad():
        logits_tensor = model.policy.action_logits(obs_tensor).reshape(-1)
        distribution = model.policy.get_distribution(
            obs_tensor, action_masks=action_mask
        )
        probabilities = (
            distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
        )
    logits = logits_tensor.detach().cpu().numpy()
    feasible_logits = logits[action_mask.astype(bool)]
    feasible_probabilities = probabilities[action_mask.astype(bool)]
    if not np.isclose(probabilities.sum(), 1.0):
        raise AssertionError("Masked probabilities do not sum to one.")
    if np.any(probabilities[action_mask == 0] > 1e-7):
        raise AssertionError("An infeasible action retained nonzero probability.")
    entropy = float(
        -np.sum(
            feasible_probabilities
            * np.log(np.clip(feasible_probabilities, 1e-12, 1.0))
        )
    )
    feasible_count = int(action_mask.sum())
    raw_normalized_entropy = (
        entropy / math.log(feasible_count) if feasible_count > 1 else 1.0
    )
    normalized_entropy = min(1.0, max(0.0, raw_normalized_entropy))
    return {
        "probabilities": probabilities,
        "logits": logits,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "max_probability": float(feasible_probabilities.max()),
        "effective_action_count": math.exp(entropy),
        "logit_span": float(feasible_logits.max() - feasible_logits.min()),
        "logit_std": float(np.std(feasible_logits)),
        "action": int(np.argmax(probabilities)),
    }


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


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def kl_divergence(values: np.ndarray) -> float:
        selected = values > 0
        return float(
            np.sum(values[selected] * np.log(values[selected] / midpoint[selected]))
        )

    return max(0.0, 0.5 * (kl_divergence(left) + kl_divergence(right)))


def _context_contributions(
    model: MaskablePPO,
    observation: dict[str, np.ndarray],
    action_mask: np.ndarray,
    env: HTPdmFjspEnv,
) -> dict[str, float]:
    filtered = {key: observation[key] for key in model.observation_space.spaces}
    obs_tensor, _ = model.policy.obs_to_tensor(filtered)
    with th.no_grad():
        per_action = model.policy._per_action_input(obs_tensor)[0]
        base_dim = int(model.observation_space["action_features"].shape[1]) - 1
        linear = model.policy.action_encoder[0]
        weights = linear.weight
        base_values = per_action[:, :base_dim]
        context_values = per_action[:, base_dim:]
        base_contribution = th.nn.functional.linear(
            base_values, weights[:, :base_dim], bias=None
        )
        context_contribution = th.nn.functional.linear(
            context_values, weights[:, base_dim:], bias=None
        )
    production_feasible = np.asarray(
        [
            bool(action_mask[index]) and descriptor.kind == "production"
            for index, descriptor in enumerate(env.actions)
        ]
    )
    if not production_feasible.any():
        return {
            "base_input_norm": math.nan,
            "context_input_norm": math.nan,
            "context_base_input_norm_ratio": math.nan,
            "base_preactivation_norm": math.nan,
            "context_preactivation_norm": math.nan,
            "context_base_preactivation_ratio": math.nan,
        }
    selected = th.as_tensor(production_feasible, device=per_action.device)
    base_input_norm = float(th.linalg.vector_norm(base_values[selected], dim=1).mean())
    context_input_norm = float(
        th.linalg.vector_norm(context_values[selected], dim=1).mean()
    )
    base_pre_norm = float(
        th.linalg.vector_norm(base_contribution[selected], dim=1).mean()
    )
    context_pre_norm = float(
        th.linalg.vector_norm(context_contribution[selected], dim=1).mean()
    )
    return {
        "base_input_norm": base_input_norm,
        "context_input_norm": context_input_norm,
        "context_base_input_norm_ratio": context_input_norm
        / max(base_input_norm, 1e-12),
        "base_preactivation_norm": base_pre_norm,
        "context_preactivation_norm": context_pre_norm,
        "context_base_preactivation_ratio": context_pre_norm
        / max(base_pre_norm, 1e-12),
    }


def model_weight_diagnostics(model: MaskablePPO) -> dict[str, float]:
    base_dim = int(model.observation_space["action_features"].shape[1]) - 1
    weights = model.policy.action_encoder[0].weight.detach().cpu().numpy()
    base_rms = float(np.sqrt(np.mean(np.square(weights[:, :base_dim]))))
    context_rms = float(np.sqrt(np.mean(np.square(weights[:, base_dim:]))))
    return {
        "base_weight_rms": base_rms,
        "context_weight_rms": context_rms,
        "context_base_weight_rms_ratio": context_rms / max(base_rms, 1e-12),
    }


def collect_episode(
    *,
    control_model: MaskablePPO,
    treatment_model: MaskablePPO,
    config: BenchmarkConfig,
    behavior_role: str,
    train_seed: int,
    environment_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if behavior_role not in {"control", "treatment"}:
        raise ValueError(f"Unknown behavior role: {behavior_role}")
    env = HTPdmFjspEnv(config=config, include_production_context=True)
    observation, _ = env.reset(seed=environment_seed)
    state_rows: list[dict[str, Any]] = []
    action_counts = {kind: 0 for kind in KINDS}
    episode_return = 0.0
    while not env._done:
        mask = env.action_masks()
        control = _policy_state(control_model, observation, mask)
        treatment = _policy_state(treatment_model, observation, mask)
        zero_context = _policy_state(
            treatment_model,
            observation,
            mask,
            zero_production_context=True,
        )
        behavior_action = int(
            control["action"] if behavior_role == "control" else treatment["action"]
        )
        row: dict[str, Any] = {
            "behavior_role": behavior_role,
            "train_seed": train_seed,
            "seed": environment_seed,
            "decision_index": env.decision_count + 1,
            "time": env.now,
            "feasible_actions": int(mask.sum()),
            "control_action": control["action"],
            "treatment_action": treatment["action"],
            "zero_context_action": zero_context["action"],
            "control_action_kind": env.actions[int(control["action"])].kind,
            "treatment_action_kind": env.actions[int(treatment["action"])].kind,
            "behavior_action": behavior_action,
            "behavior_action_kind": env.actions[behavior_action].kind,
            "policies_agree": control["action"] == treatment["action"],
            "zero_context_action_flip": (
                treatment["action"] != zero_context["action"]
            ),
            "normal_zero_js_divergence": _jensen_shannon(
                treatment["probabilities"], zero_context["probabilities"]
            ),
            **_context_contributions(
                treatment_model, observation, mask, env
            ),
        }
        for metric in (
            "entropy",
            "normalized_entropy",
            "max_probability",
            "effective_action_count",
            "logit_span",
            "logit_std",
        ):
            row[f"control_{metric}"] = control[metric]
            row[f"treatment_{metric}"] = treatment[metric]
            row[f"zero_context_{metric}"] = zero_context[metric]
            row[f"delta_{metric}"] = treatment[metric] - control[metric]
        for kind in KINDS:
            control_mass = _kind_probability(env, control["probabilities"], kind)
            treatment_mass = _kind_probability(
                env, treatment["probabilities"], kind
            )
            zero_mass = _kind_probability(
                env, zero_context["probabilities"], kind
            )
            row[f"control_probability_{kind}"] = control_mass
            row[f"treatment_probability_{kind}"] = treatment_mass
            row[f"zero_context_probability_{kind}"] = zero_mass
            row[f"delta_probability_{kind}"] = treatment_mass - control_mass
        state_rows.append(row)
        behavior_kind = env.actions[behavior_action].kind
        action_counts[behavior_kind] += 1
        observation, reward, _, _, _ = env.step(behavior_action)
        episode_return += reward
    result = env.result(f"production_context_diagnostic_{behavior_role}")
    if not np.isclose(episode_return, -float(result.metrics["objective"])):
        raise AssertionError("Episode reward does not match the negative objective.")
    episode_row = {
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


def summarize_diagnostic(
    state_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def finite_mean(rows: list[dict[str, Any]], metric: str) -> float:
        values = [
            float(row[metric])
            for row in rows
            if math.isfinite(float(row[metric]))
        ]
        if not values:
            raise ValueError(f"No finite values available for {metric}.")
        return fmean(values)

    train_seeds = sorted({int(row["train_seed"]) for row in state_rows})
    summary: dict[str, Any] = {
        "primary_question": (
            "whether production context changes policy concentration and dominates "
            "the shared action representation"
        ),
        "replication_unit": "independent PPO training seed",
        "behavior_distributions": {},
    }
    for behavior_role in ("control", "treatment"):
        role_rows = [
            row for row in state_rows if row["behavior_role"] == behavior_role
        ]
        per_seed: dict[str, Any] = {}
        for train_seed in train_seeds:
            selected = [
                row
                for row in role_rows
                if int(row["train_seed"]) == train_seed
            ]
            per_seed[str(train_seed)] = {
                "state_count": len(selected),
                **{
                    metric: finite_mean(selected, metric)
                    for metric in STATE_METRICS
                },
            }
        summary["behavior_distributions"][behavior_role] = {
            "state_count": len(role_rows),
            "per_training_seed": per_seed,
            "across_training_seeds": {
                metric: _stats(
                    per_seed[str(train_seed)][metric]
                    for train_seed in train_seeds
                )
                for metric in STATE_METRICS
            },
        }
    summary["model_weights"] = {
        "per_training_seed": {
            str(int(row["train_seed"])): {
                key: float(value)
                for key, value in row.items()
                if key != "train_seed"
            }
            for row in model_rows
        },
        "across_training_seeds": {
            metric: _stats(float(row[metric]) for row in model_rows)
            for metric in (
                "base_weight_rms",
                "context_weight_rms",
                "context_base_weight_rms_ratio",
            )
        },
    }
    behavior_outcomes: dict[str, Any] = {}
    for behavior_role in ("control", "treatment"):
        per_seed = {
            str(train_seed): {
                metric: fmean(
                    float(row[metric])
                    for row in episode_rows
                    if row["behavior_role"] == behavior_role
                    and int(row["train_seed"]) == train_seed
                )
                for metric in EPISODE_METRICS
            }
            for train_seed in train_seeds
        }
        behavior_outcomes[behavior_role] = {
            "per_training_seed": per_seed,
            "across_training_seeds": {
                metric: _stats(
                    per_seed[str(train_seed)][metric]
                    for train_seed in train_seeds
                )
                for metric in EPISODE_METRICS
            },
        }
    summary["behavior_outcomes"] = behavior_outcomes
    return summary


def run_diagnostic(args: argparse.Namespace) -> Path:
    shared_dir = Path(args.shared_source_run).expanduser().resolve()
    production_dir = Path(args.production_source_run).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_manifest = _load_manifest(shared_dir, "architecture_manifest.json")
    production_manifest = _load_manifest(
        production_dir, "production_context_manifest.json"
    )
    if Path(production_manifest["shared_source_run"]).name != shared_dir.name:
        raise ValueError("Production run was not derived from the supplied shared run.")

    defaults = diagnostic_defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds)
        if args.train_seeds
        else defaults["train_seeds"]
    )
    available = set(map(int, shared_manifest["train_seeds"])) & set(
        map(int, production_manifest["train_seeds"])
    )
    if not train_seeds or not set(train_seeds).issubset(available):
        raise ValueError("Requested training seeds are missing from a source run.")
    diagnostic_seeds = tuple(
        parse_seeds(args.diagnostic_seeds)
        if args.diagnostic_seeds
        else defaults["diagnostic_seeds"]
    )
    forbidden = (
        set(map(int, shared_manifest["validation_seeds"]))
        | set(map(int, production_manifest["validation_seeds"]))
        | set(range(40_000, 45_000))
        | set(range(50_000, 50_100))
    )
    if not diagnostic_seeds or set(diagnostic_seeds) & forbidden:
        raise ValueError("Diagnostic seeds overlap an existing panel.")

    config_path = Path(args.config).expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    if production_manifest.get("config_sha256") != config_hash:
        raise ValueError("Current benchmark config does not match the source run.")
    config = BenchmarkConfig.from_json(config_path)
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)
    resolved_device = resolve_device(args.device)
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "git_commit": _git_commit(),
        "config_sha256": config_hash,
        "shared_source_run": str(shared_dir),
        "production_source_run": str(production_dir),
        "train_seeds": train_seeds,
        "diagnostic_seeds": diagnostic_seeds,
        "behavior_roles": ["control", "treatment"],
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
    manifest_path = output_dir / "context_diagnostic_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    episode_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    total = 2 * len(train_seeds) * len(diagnostic_seeds)
    progress = tqdm(
        total=total,
        desc="Production-context diagnostic",
        unit="episode",
        disable=args.no_progress,
    )
    for train_seed in train_seeds:
        control_path = (
            shared_dir
            / "shared_scorer_entropy"
            / f"train_seed_{train_seed}"
            / "maskable_ppo.zip"
        )
        treatment_path = (
            production_dir
            / "production_context_entropy"
            / f"train_seed_{train_seed}"
            / "maskable_ppo.zip"
        )
        if not control_path.is_file() or not treatment_path.is_file():
            raise FileNotFoundError(f"Missing model for training seed {train_seed}.")
        control_model = MaskablePPO.load(control_path, device=resolved_device)
        treatment_model = MaskablePPO.load(treatment_path, device=resolved_device)
        model_rows.append(
            {"train_seed": train_seed, **model_weight_diagnostics(treatment_model)}
        )
        for behavior_role in ("control", "treatment"):
            progress.set_postfix(
                train_seed=str(train_seed), behavior=behavior_role
            )
            for environment_seed in diagnostic_seeds:
                episode, states = collect_episode(
                    control_model=control_model,
                    treatment_model=treatment_model,
                    config=config,
                    behavior_role=behavior_role,
                    train_seed=train_seed,
                    environment_seed=environment_seed,
                )
                episode_rows.append(episode)
                state_rows.extend(states)
                progress.update(1)
        _write_csv(
            episode_rows, output_dir / "context_diagnostic_episodes.partial.csv"
        )
        _write_csv(state_rows, output_dir / "context_diagnostic_states.partial.csv")
        _write_csv(model_rows, output_dir / "context_diagnostic_models.partial.csv")
        del control_model, treatment_model
    progress.close()

    unique_episodes = {
        (row["behavior_role"], row["train_seed"], row["seed"])
        for row in episode_rows
    }
    if len(episode_rows) != total or len(unique_episodes) != total:
        raise ValueError("Production-context diagnostic panel is incomplete.")
    summary = summarize_diagnostic(state_rows, episode_rows, model_rows)
    _write_csv(episode_rows, output_dir / "context_diagnostic_episodes.csv")
    _write_csv(state_rows, output_dir / "context_diagnostic_states.csv")
    _write_csv(model_rows, output_dir / "context_diagnostic_models.csv")
    (output_dir / "context_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["status"] = "COMPLETED"
    manifest["episode_count"] = len(episode_rows)
    manifest["state_count"] = len(state_rows)
    manifest["model_count"] = len(model_rows)
    manifest["output_files"] = sorted(path.name for path in output_dir.iterdir())
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-source-run", required=True)
    parser.add_argument("--production-source-run", required=True)
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
