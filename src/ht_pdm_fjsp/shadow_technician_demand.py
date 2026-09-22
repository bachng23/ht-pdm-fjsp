"""Measure technician demand hidden by availability masks without changing runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.simulator import audit_result
from ht_pdm_fjsp.technician_conflict_experiment import (
    _describe,
    _student_t_critical_975,
    choose_joint_actions,
)
from ht_pdm_fjsp.technician_noop_experiment import (
    _conflict_sources,
    _proposal_stats,
    _zero_totals,
)


SHADOW_REACTIVE = "shadow_reactive_safe_noop"
SHADOW_018 = "shadow_threshold_018_safe_noop"
SHADOW_005 = "shadow_threshold_005_safe_noop"
CONDITIONS: dict[str, float | None] = {
    SHADOW_REACTIVE: None,
    SHADOW_018: 0.18,
    SHADOW_005: 0.05,
}

SHADOW_TOTAL_KEYS = (
    "shadow_request_snapshots",
    "shadow_preventive_request_snapshots",
    "shadow_corrective_request_snapshots",
    "shadow_busy_suppressed_snapshots",
    "shadow_preventive_busy_suppressed_snapshots",
    "shadow_corrective_busy_suppressed_snapshots",
    "shadow_t1_request_snapshots",
    "shadow_t2_request_snapshots",
    "shadow_contested_technician_snapshots",
    "shadow_contention_units",
    "shadow_request_onsets",
    "shadow_preventive_request_onsets",
    "shadow_corrective_request_onsets",
    "shadow_busy_suppression_onsets",
    "shadow_preventive_busy_suppression_onsets",
    "shadow_corrective_busy_suppression_onsets",
    "shadow_contention_spell_onsets",
)


@dataclass(frozen=True)
class ShadowRequest:
    machine_id: str
    kind: str
    technician_id: str
    all_qualified_technicians_busy: bool

    @property
    def key(self) -> tuple[str, str]:
        return self.machine_id, self.kind


def default_seeds(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(61_420, 61_425))
    if profile == "full":
        return tuple(range(61_500, 61_700))
    raise ValueError(f"Unknown profile: {profile}")


def _production_candidate(
    env: MachineAgentsCTDEEnv,
    observation: dict[str, np.ndarray],
    agent_index: int,
) -> Any | None:
    candidates = []
    for local_action, global_action in enumerate(
        env.local_action_catalogs[agent_index]
    ):
        if global_action is None or not observation["action_masks"][
            agent_index, local_action
        ]:
            continue
        descriptor = env.core.actions[global_action]
        if descriptor.kind == "production":
            candidates.append(descriptor)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda descriptor: (
            env.core._alternative(descriptor).processing_time,
            str(descriptor.job_id),
            int(descriptor.operation_index),
        ),
    )


def _qualified_maintenance_actions(
    env: MachineAgentsCTDEEnv, agent_index: int, kind: str
) -> list[Any]:
    output = []
    for global_action in env.local_action_catalogs[agent_index]:
        if global_action is None:
            continue
        descriptor = env.core.actions[global_action]
        if descriptor.kind == kind:
            output.append(descriptor)
    return output


def reconstruct_shadow_requests(
    env: MachineAgentsCTDEEnv,
    observation: dict[str, np.ndarray],
    *,
    preventive_threshold: float | None,
) -> list[ShadowRequest]:
    """Reconstruct maintenance requests before technician-idle masking."""

    requests: list[ShadowRequest] = []
    jobs_complete = env.core._all_jobs_complete()
    for agent_index, machine_id in enumerate(env.machine_ids):
        machine_state = env.core.machines[machine_id]
        if machine_state.status == "waiting_corrective":
            kind = "corrective"
        else:
            production = _production_candidate(env, observation, agent_index)
            preventive_needed = (
                preventive_threshold is not None
                and production is not None
                and not jobs_complete
                and machine_state.status == "idle"
                and machine_state.effective_age > 0.0
                and machine_state.preventive_armed
                and env.core.failure_probability(production)
                >= preventive_threshold
            )
            if not preventive_needed:
                continue
            kind = "preventive"

        qualified = _qualified_maintenance_actions(env, agent_index, kind)
        if not qualified:
            raise AssertionError("Maintenance need has no qualified technician")
        idle = [
            descriptor
            for descriptor in qualified
            if env.core.technicians[str(descriptor.technician_id)].status == "idle"
        ]
        candidates = idle or qualified
        selected = min(
            candidates,
            key=lambda descriptor: (
                env.technician_duration(descriptor),
                str(descriptor.technician_id),
            ),
        )
        requests.append(
            ShadowRequest(
                machine_id=str(machine_id),
                kind=kind,
                technician_id=str(selected.technician_id),
                all_qualified_technicians_busy=not idle,
            )
        )
    return requests


def _shadow_snapshot(requests: list[ShadowRequest]) -> dict[str, int]:
    technicians = Counter(request.technician_id for request in requests)
    return {
        "shadow_request_snapshots": len(requests),
        "shadow_preventive_request_snapshots": sum(
            request.kind == "preventive" for request in requests
        ),
        "shadow_corrective_request_snapshots": sum(
            request.kind == "corrective" for request in requests
        ),
        "shadow_busy_suppressed_snapshots": sum(
            request.all_qualified_technicians_busy for request in requests
        ),
        "shadow_preventive_busy_suppressed_snapshots": sum(
            request.kind == "preventive"
            and request.all_qualified_technicians_busy
            for request in requests
        ),
        "shadow_corrective_busy_suppressed_snapshots": sum(
            request.kind == "corrective"
            and request.all_qualified_technicians_busy
            for request in requests
        ),
        "shadow_t1_request_snapshots": technicians["T1"],
        "shadow_t2_request_snapshots": technicians["T2"],
        "shadow_contested_technician_snapshots": sum(
            count > 1 for count in technicians.values()
        ),
        "shadow_contention_units": sum(
            max(0, count - 1) for count in technicians.values()
        ),
    }


def _shadow_onsets(
    requests: list[ShadowRequest],
    *,
    previous_active: set[tuple[str, str]],
    previous_suppressed: set[tuple[str, str]],
    previous_contested: set[str],
) -> tuple[dict[str, int], set[tuple[str, str]], set[tuple[str, str]], set[str]]:
    by_key = {request.key: request for request in requests}
    active = set(by_key)
    suppressed = {
        request.key
        for request in requests
        if request.all_qualified_technicians_busy
    }
    technician_counts = Counter(request.technician_id for request in requests)
    contested = {
        technician_id
        for technician_id, count in technician_counts.items()
        if count > 1
    }
    request_onsets = active - previous_active
    suppression_onsets = suppressed - previous_suppressed
    values = {
        "shadow_request_onsets": len(request_onsets),
        "shadow_preventive_request_onsets": sum(
            by_key[key].kind == "preventive" for key in request_onsets
        ),
        "shadow_corrective_request_onsets": sum(
            by_key[key].kind == "corrective" for key in request_onsets
        ),
        "shadow_busy_suppression_onsets": len(suppression_onsets),
        "shadow_preventive_busy_suppression_onsets": sum(
            by_key[key].kind == "preventive" for key in suppression_onsets
        ),
        "shadow_corrective_busy_suppression_onsets": sum(
            by_key[key].kind == "corrective" for key in suppression_onsets
        ),
        "shadow_contention_spell_onsets": len(contested - previous_contested),
    }
    return values, active, suppressed, contested


def _all_totals() -> dict[str, int]:
    return {**_zero_totals(), **{key: 0 for key in SHADOW_TOTAL_KEYS}}


def evaluate_episode(
    config: BenchmarkConfig,
    *,
    condition: str,
    seed: int,
    threshold: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    env = MachineAgentsCTDEEnv(config, wait_policy="safe_noop")
    observation, _ = env.reset(seed=seed)
    episode_return = 0.0
    decision_rows: list[dict[str, Any]] = []
    totals = _all_totals()
    previous_active: set[tuple[str, str]] = set()
    previous_suppressed: set[tuple[str, str]] = set()
    previous_contested: set[str] = set()
    joint_step = 0
    while not env._done:
        shadow_requests = reconstruct_shadow_requests(
            env, observation, preventive_threshold=threshold
        )
        snapshot = _shadow_snapshot(shadow_requests)
        onsets, previous_active, previous_suppressed, previous_contested = (
            _shadow_onsets(
                shadow_requests,
                previous_active=previous_active,
                previous_suppressed=previous_suppressed,
                previous_contested=previous_contested,
            )
        )
        reasons: list[str] = []
        actions = choose_joint_actions(
            env,
            observation,
            preventive_threshold=threshold,
            reasons=reasons,
        )
        proposed = _proposal_stats(env, actions, reasons)
        sources = _conflict_sources(env, actions, reasons)
        simulation_time = env.core.now
        observation, reward, _, _, info = env.step(actions)
        resolution = info["coordination"]
        if proposed["duplicate_technician_proposals"] != resolution[
            "technician_conflicts"
        ]:
            raise AssertionError("Technician proposal accounting mismatch")
        if sum(sources.values()) != resolution["technician_conflicts"]:
            raise AssertionError("Technician conflict-source accounting mismatch")
        for key, value in {
            **resolution,
            **proposed,
            **sources,
            **snapshot,
            **onsets,
        }.items():
            totals[key] += int(value)
        decision_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "joint_step": joint_step,
                "simulation_time": simulation_time,
                **snapshot,
                **onsets,
                **proposed,
                **sources,
                **resolution,
            }
        )
        episode_return += float(reward)
        joint_step += 1

    result = env.result(condition)
    audit_result(config, result)
    if not np.isclose(episode_return, -float(result.metrics["objective"])):
        raise AssertionError("Episode return does not equal negative objective")
    if totals["shadow_request_onsets"] > totals["shadow_request_snapshots"]:
        raise AssertionError("Shadow request onsets exceed snapshots")
    shared = {
        "condition": condition,
        "seed": seed,
        **totals,
        "maintenance_proposals": (
            totals["proposed_preventive"] + totals["proposed_corrective"]
        ),
        "shadow_busy_suppression_rate": (
            totals["shadow_busy_suppression_onsets"]
            / totals["shadow_request_onsets"]
            if totals["shadow_request_onsets"]
            else 0.0
        ),
        "shadow_contention_spell_rate": (
            totals["shadow_contention_spell_onsets"]
            / totals["shadow_request_onsets"]
            if totals["shadow_request_onsets"]
            else 0.0
        ),
    }
    episode_row = {
        **shared,
        "threshold": "" if threshold is None else threshold,
        "episode_return": episode_return,
        **result.metrics,
        "rejection_rate": (
            totals["rejected"] / totals["proposals"]
            if totals["proposals"]
            else 0.0
        ),
    }
    env.close()
    return episode_row, decision_rows, shared


def _paired_contrast(
    episode_rows: list[dict[str, Any]], treatment: str, control: str
) -> dict[str, Any]:
    metrics = (
        "objective",
        "makespan",
        "failures",
        "preventive_maintenance",
        "shadow_request_onsets",
        "shadow_busy_suppression_onsets",
        "shadow_preventive_busy_suppression_onsets",
        "shadow_contention_spell_onsets",
        "shadow_busy_suppression_rate",
        "shadow_contention_spell_rate",
    )
    treatment_rows = {
        int(row["seed"]): row
        for row in episode_rows
        if row["condition"] == treatment
    }
    control_rows = {
        int(row["seed"]): row
        for row in episode_rows
        if row["condition"] == control
    }
    if treatment_rows.keys() != control_rows.keys():
        raise AssertionError("Paired conditions do not have identical seed panels")
    seeds = sorted(treatment_rows)
    critical = _student_t_critical_975(len(seeds) - 1)
    output: dict[str, Any] = {}
    for metric in metrics:
        differences = [
            float(treatment_rows[seed][metric])
            - float(control_rows[seed][metric])
            for seed in seeds
        ]
        mean = fmean(differences)
        standard_deviation = stdev(differences)
        half_width = critical * standard_deviation / math.sqrt(len(differences))
        output[metric] = {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "lower_95": mean - half_width,
            "upper_95": mean + half_width,
            "negative_seeds": sum(value < 0 for value in differences),
            "zero_seeds": sum(value == 0 for value in differences),
            "positive_seeds": sum(value > 0 for value in differences),
        }
    return {
        "treatment": treatment,
        "control": control,
        "seed_count": len(seeds),
        "degrees_of_freedom": len(seeds) - 1,
        "critical_value": critical,
        "metrics": output,
    }


def summarize(
    episode_rows: list[dict[str, Any]], coordination_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    per_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        episodes = [row for row in episode_rows if row["condition"] == condition]
        coordination = [
            row for row in coordination_rows if row["condition"] == condition
        ]
        totals = {
            key: sum(int(row[key]) for row in coordination)
            for key in _all_totals()
        }
        request_onsets = totals["shadow_request_onsets"]
        audit = {
            "zero_invalid_executions": totals["invalid_executions"] == 0,
            "zero_duplicate_operation_executions": totals[
                "duplicate_operation_executions"
            ]
            == 0,
            "zero_duplicate_technician_executions": totals[
                "duplicate_technician_executions"
            ]
            == 0,
            "request_kinds_sum_to_request_onsets": totals[
                "shadow_preventive_request_onsets"
            ]
            + totals["shadow_corrective_request_onsets"]
            == request_onsets,
            "suppression_kinds_sum_to_suppression_onsets": totals[
                "shadow_preventive_busy_suppression_onsets"
            ]
            + totals["shadow_corrective_busy_suppression_onsets"]
            == totals["shadow_busy_suppression_onsets"],
        }
        per_condition[condition] = {
            "episodes": len(episodes),
            "episodes_with_preventive_busy_suppression": sum(
                int(row["shadow_preventive_busy_suppression_onsets"]) > 0
                for row in episodes
            ),
            "episodes_with_any_busy_suppression": sum(
                int(row["shadow_busy_suppression_onsets"]) > 0
                for row in episodes
            ),
            "episodes_with_shadow_contention": sum(
                int(row["shadow_contention_spell_onsets"]) > 0
                for row in episodes
            ),
            "totals": totals,
            "shadow_busy_suppression_rate": (
                totals["shadow_busy_suppression_onsets"] / request_onsets
                if request_onsets
                else 0.0
            ),
            "shadow_contention_spell_rate": (
                totals["shadow_contention_spell_onsets"] / request_onsets
                if request_onsets
                else 0.0
            ),
            "metrics": {
                metric: _describe(float(row[metric]) for row in episodes)
                for metric in (
                    "objective",
                    "makespan",
                    "failures",
                    "preventive_maintenance",
                    "corrective_maintenance",
                    "shadow_request_onsets",
                    "shadow_busy_suppression_onsets",
                    "shadow_preventive_busy_suppression_onsets",
                    "shadow_contention_spell_onsets",
                )
            },
            "audit": audit,
        }

    intended = per_condition[SHADOW_018]
    seed_count = intended["episodes"]
    gate_checks = {
        "all_expected_episodes_completed": len(episode_rows) == 3 * seed_count,
        "all_feasibility_and_shadow_audits_passed": all(
            all(item["audit"].values()) for item in per_condition.values()
        ),
        "safe_noop_has_zero_forced_preventive_proposals": all(
            item["totals"]["forced_preventive_proposals"] == 0
            for item in per_condition.values()
        ),
        "intended_preventive_suppression_in_at_least_ten_percent_of_seeds": intended[
            "episodes_with_preventive_busy_suppression"
        ]
        >= math.ceil(0.10 * seed_count),
        "intended_contention_in_at_least_ten_percent_of_seeds": intended[
            "episodes_with_shadow_contention"
        ]
        >= math.ceil(0.10 * seed_count),
        "intended_suppression_rate_at_least_one_percent": intended[
            "shadow_busy_suppression_rate"
        ]
        >= 0.01,
        "intended_contention_rate_at_least_one_percent": intended[
            "shadow_contention_spell_rate"
        ]
        >= 0.01,
    }
    return {
        "primary_endpoint": (
            "shadow technician demand hidden by availability masks at threshold 0.18"
        ),
        "per_condition": per_condition,
        "paired_contrasts": {
            "shadow_018_minus_reactive": _paired_contrast(
                episode_rows, SHADOW_018, SHADOW_REACTIVE
            ),
            "shadow_005_minus_shadow_018": _paired_contrast(
                episode_rows, SHADOW_005, SHADOW_018
            ),
        },
        "gate": {
            "checks": gate_checks,
            "supports_queue_aware_technician_experiment": all(
                gate_checks.values()
            ),
            "note": "Shadow diagnostic only; trajectories are unchanged.",
        },
    }


def _write_partial(
    output_dir: Path,
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    coordination_rows: list[dict[str, Any]],
) -> None:
    _write_csv(episode_rows, output_dir / "shadow_demand_episodes.partial.csv")
    _write_csv(decision_rows, output_dir / "shadow_demand_decisions.partial.csv")
    _write_csv(
        coordination_rows, output_dir / "shadow_demand_coordination.partial.csv"
    )


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    raw_config = config_path.read_bytes()
    config = BenchmarkConfig.from_json(config_path)
    if config.instance_name != "ht_pdm_fjsp_brandimarte_mk01_v1":
        raise ValueError(
            "This diagnostic requires the frozen Brandimarte MK01 extension"
        )
    seeds = (
        tuple(parse_seeds(args.seeds))
        if args.seeds
        else default_seeds(args.profile)
    )
    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    manifest_path = output_dir / "shadow_demand_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "profile": args.profile,
        "git_commit": _git_revision(),
        "instance_name": config.instance_name,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "conditions": CONDITIONS,
        "seeds": list(seeds),
        "reserved_future_seeds": list(range(62_000, 62_100)),
        "future_test_panel_opened": False,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    coordination_rows: list[dict[str, Any]] = []
    total = len(CONDITIONS) * len(seeds)
    with tqdm(total=total, desc="Shadow technician demand", unit="episode") as progress:
        for condition, threshold in CONDITIONS.items():
            for seed in seeds:
                progress.set_postfix(condition=condition, seed=seed)
                episode, decisions, coordination = evaluate_episode(
                    config,
                    condition=condition,
                    seed=seed,
                    threshold=threshold,
                )
                episode_rows.append(episode)
                decision_rows.extend(decisions)
                coordination_rows.append(coordination)
                if len(episode_rows) % 10 == 0 or len(episode_rows) == total:
                    _write_partial(
                        output_dir, episode_rows, decision_rows, coordination_rows
                    )
                progress.update(1)

    summary = summarize(episode_rows, coordination_rows)
    _write_csv(episode_rows, output_dir / "shadow_demand_episodes.csv")
    _write_csv(decision_rows, output_dir / "shadow_demand_decisions.csv")
    _write_csv(coordination_rows, output_dir / "shadow_demand_coordination.csv")
    (output_dir / "shadow_demand_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        {
            "status": "COMPLETED",
            "episode_count": len(episode_rows),
            "decision_count": len(decision_rows),
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
    summary = json.loads((output_dir / "shadow_demand_summary.json").read_text())
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
