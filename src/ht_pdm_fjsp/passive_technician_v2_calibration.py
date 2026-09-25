"""Calibrate passive-technician v2 scenarios before learning experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_v2 import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    SIMULATOR_VERSION,
    PassiveTechnicianV2Config,
)
from ht_pdm_fjsp.passive_technician_v2_validation import run_episode


POLICIES = ("reactive_fastest", "threshold_fastest", "workload_aware")
PREVENTIVE_POLICIES = ("threshold_fastest", "workload_aware")
SEALED_SEEDS = tuple(range(9200, 9300))
CALIBRATION_VERSION = "passive_technician_v2_1_factorial"


def _profile(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(9100, 9103))
    if profile == "pilot":
        return tuple(range(9100, 9130))
    if profile == "full":
        return tuple(range(9100, 9200))
    raise ValueError(profile)


def _skill_structure(
    machines: int, skill: str
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[bool, bool], ...]]:
    service: list[tuple[int, int]] = []
    eligibility: list[tuple[bool, bool]] = []
    if skill == "alternating_specialists":
        for machine in range(machines):
            service.append((2, 5) if machine % 2 == 0 else (5, 2))
            eligibility.append((True, True))
    elif skill == "shared_bottleneck":
        patterns = ((2, 4), (2, 5), (3, 4))
        for machine in range(machines):
            service.append(patterns[machine % len(patterns)])
            eligibility.append((True, True))
    elif skill == "mixed_eligibility":
        patterns = (
            ((2, 5), (True, False)),
            ((2, 3), (True, True)),
            ((3, 2), (True, True)),
            ((5, 2), (False, True)),
        )
        for machine in range(machines):
            durations, allowed = patterns[machine % len(patterns)]
            service.append(durations)
            eligibility.append(allowed)
    else:
        raise ValueError(skill)
    return tuple(service), tuple(eligibility)


def candidate_configs() -> dict[str, PassiveTechnicianV2Config]:
    output: dict[str, PassiveTechnicianV2Config] = {}
    for machines in (4, 6):
        for timing in ("synchronized", "staggered"):
            ages = (
                (0,) * machines
                if timing == "synchronized"
                else tuple(machine * 6 // machines for machine in range(machines))
            )
            for skill in (
                "alternating_specialists",
                "shared_bottleneck",
                "mixed_eligibility",
            ):
                service, eligibility = _skill_structure(machines, skill)
                candidate_id = f"m{machines}__{timing}__{skill}"
                output[candidate_id] = PassiveTechnicianV2Config(
                    machines=machines,
                    technicians=2,
                    horizon=36,
                    failure_age=6,
                    failure_probability=0.30,
                    max_age=10,
                    service_time=service,
                    eligibility=eligibility,
                    initial_ages=ages,
                )
    return output


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
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


def _candidate_fingerprint(candidates: dict[str, PassiveTechnicianV2Config]) -> str:
    payload = json.dumps(
        {candidate_id: asdict(config) for candidate_id, config in candidates.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _request_metrics(
    config: PassiveTechnicianV2Config,
    decisions: list[dict[str, object]],
) -> tuple[int, int, int, int, Counter[int], int]:
    requests = 0
    slower = 0
    queued_machine_steps = 0
    max_queue = 0
    failures: Counter[int] = Counter()
    decision_rows = 0
    for row in decisions:
        actions = json.loads(str(row["raw_actions"]))
        for machine, action in enumerate(actions):
            if action <= 0:
                continue
            requests += 1
            selected = action - 1
            fastest = min(
                config.service_time[machine][technician]
                for technician in range(config.technicians)
                if config.eligibility[machine][technician]
            )
            if config.service_time[machine][selected] > fastest:
                slower += 1
        state = json.loads(str(row["post_state"]))
        queue_lengths = [len(queue) for queue in state["queues"]]
        queued_machine_steps += sum(queue_lengths)
        max_queue = max(max_queue, *queue_lengths)
        failures.update(json.loads(str(row["failure_machines"])))
        decision_rows += 1
    return requests, slower, queued_machine_steps, max_queue, failures, decision_rows


def _paired_action_counts(
    left: list[dict[str, object]], right: list[dict[str, object]]
) -> tuple[int, int]:
    matched = 0
    different = 0
    if len(left) != len(right):
        raise AssertionError("paired decision traces have different lengths")
    for left_row, right_row in zip(left, right):
        if left_row["time"] != right_row["time"]:
            raise AssertionError("paired decision traces have different time indices")
        left_actions = json.loads(str(left_row["raw_actions"]))
        right_actions = json.loads(str(right_row["raw_actions"]))
        for left_action, right_action in zip(left_actions, right_actions):
            if left_action > 0 and right_action > 0:
                matched += 1
                different += int(left_action != right_action)
    return matched, different


def _candidate_summary(
    candidate_id: str,
    config: PassiveTechnicianV2Config,
    seeds: tuple[int, ...],
    policy_episodes: dict[str, list[dict[str, object]]],
    trace_divergences: int,
    matched_requests: int,
    different_requests: int,
    workload_requests: int,
    workload_slower: int,
    queued_machine_steps: int,
    preventive_decision_rows: int,
    max_queue: int,
    failure_counts: Counter[int],
) -> dict[str, object]:
    threshold = {int(row["seed"]): row for row in policy_episodes["threshold_fastest"]}
    workload = {int(row["seed"]): row for row in policy_episodes["workload_aware"]}
    paired_differences = [
        float(threshold[seed]["objective"]) - float(workload[seed]["objective"])
        for seed in seeds
    ]
    trace_fraction = trace_divergences / len(seeds)
    action_fraction = different_requests / matched_requests if matched_requests else 0.0
    slower_fraction = workload_slower / workload_requests if workload_requests else 0.0
    queue_fraction = queued_machine_steps / (preventive_decision_rows * config.machines)
    utilizations = []
    preventive_episode_count = len(seeds) * len(PREVENTIVE_POLICIES)
    for technician in range(config.technicians):
        busy = sum(
            float(row[f"technician_{technician}_busy_steps"])
            for policy in PREVENTIVE_POLICIES
            for row in policy_episodes[policy]
        )
        utilizations.append(busy / (preventive_episode_count * config.horizon))
    utilization_imbalance = abs(utilizations[0] - utilizations[1])
    total_failures = sum(failure_counts.values())
    active_failure_machines = sum(count > 0 for count in failure_counts.values())
    max_failure_share = max(failure_counts.values(), default=0) / total_failures if total_failures else 1.0
    nonzero_paired_fraction = sum(abs(value) > 1e-12 for value in paired_differences) / len(seeds)
    max_identity_error = max(
        float(row["identity_error"])
        for policy in POLICIES
        for row in policy_episodes[policy]
    )
    replay_mismatches = sum(
        int(row["replay_mismatch"])
        for policy in POLICIES
        for row in policy_episodes[policy]
    )
    gates = {
        "mechanics_gate": max_identity_error <= 1e-9 and replay_mismatches == 0,
        "trace_divergence_gate": trace_fraction >= 0.50,
        "action_divergence_gate": action_fraction >= 0.05,
        "slower_assignment_gate": slower_fraction >= 0.02,
        "queue_fraction_gate": 0.01 <= queue_fraction <= 0.35,
        "queue_length_gate": 1 <= max_queue <= 3,
        "utilization_gate": all(0.10 <= utilization <= 0.90 for utilization in utilizations),
        "failure_spread_gate": active_failure_machines >= 2 and max_failure_share <= 0.70,
        "objective_divergence_gate": nonzero_paired_fraction >= 0.50,
    }
    qualifies = all(gates.values())
    score = (
        0.35 * trace_fraction
        + 0.30 * action_fraction
        + 0.20 * nonzero_paired_fraction
        + 0.15 * (1.0 - utilization_imbalance)
    )
    output: dict[str, object] = {
        "candidate_id": candidate_id,
        "machines": config.machines,
        "initial_ages": json.dumps(config.initial_ages),
        "trace_divergent_seed_fraction": trace_fraction,
        "matched_request_action_divergence_fraction": action_fraction,
        "workload_slower_request_fraction": slower_fraction,
        "nonzero_paired_objective_fraction": nonzero_paired_fraction,
        "mean_threshold_minus_workload_objective": sum(paired_differences) / len(seeds),
        "queue_positive_machine_step_fraction": queue_fraction,
        "max_queue_length": max_queue,
        "technician_0_utilization": utilizations[0],
        "technician_1_utilization": utilizations[1],
        "utilization_imbalance": utilization_imbalance,
        "active_failure_machines": active_failure_machines,
        "max_failure_share": max_failure_share,
        "max_identity_error": max_identity_error,
        "replay_mismatches": replay_mismatches,
        "qualification_score": score,
        "qualifies": qualifies,
        **gates,
    }
    for policy in POLICIES:
        rows = policy_episodes[policy]
        for metric in ("objective", "failures", "waiting", "downtime", "proposal_contention"):
            output[f"{policy}_{metric}_mean"] = sum(float(row[metric]) for row in rows) / len(rows)
    return output


def run(args: argparse.Namespace) -> Path:
    seeds = _profile(args.profile)
    candidates = candidate_configs()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": "RUNNING",
        "experiment": CALIBRATION_VERSION,
        "profile": args.profile,
        "device": args.device,
        "simulator_version": SIMULATOR_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "development_seeds": list(seeds),
        "sealed_seeds": list(SEALED_SEEDS),
        "sealed_test_evaluated": False,
        "policies": list(POLICIES),
        "candidate_count": len(candidates),
        "stopping_rule": "every candidate-policy-development-seed cell once plus deterministic replay",
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "resolved_candidates.json").write_text(
        json.dumps(
            {
                "calibration_version": CALIBRATION_VERSION,
                "fingerprint": _candidate_fingerprint(candidates),
                "candidates": {
                    candidate_id: asdict(config)
                    for candidate_id, config in candidates.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    episodes: list[dict[str, object]] = []
    coordination: list[dict[str, object]] = []
    pairwise: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    decisions_path = output / "decisions.csv"
    decision_handle = decisions_path.open("w", newline="", encoding="utf-8")
    decision_writer: csv.DictWriter | None = None
    progress = tqdm(
        total=len(candidates) * len(seeds) * len(POLICIES),
        desc="calibrate passive v2.1",
        unit="episode",
    )
    try:
        for candidate_id, config in candidates.items():
            policy_episodes: dict[str, list[dict[str, object]]] = defaultdict(list)
            trace_divergences = 0
            matched_requests = 0
            different_requests = 0
            workload_requests = 0
            workload_slower = 0
            queued_machine_steps = 0
            preventive_decision_rows = 0
            max_queue = 0
            failure_counts: Counter[int] = Counter()
            for seed_index, seed in enumerate(seeds):
                seed_results: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}
                for policy in POLICIES:
                    episode, decision_rows, digest = run_episode(config, policy, seed)
                    replay_episode, _, replay_digest = run_episode(
                        config, policy, seed, keep_decisions=False
                    )
                    mismatch = int(
                        digest != replay_digest
                        or float(episode["objective"]) != float(replay_episode["objective"])
                    )
                    episode_row = {
                        "candidate_id": candidate_id,
                        **episode,
                        "replay_mismatch": mismatch,
                    }
                    episodes.append(episode_row)
                    policy_episodes[policy].append(episode_row)
                    coordination.append(
                        {
                            "candidate_id": candidate_id,
                            "policy": policy,
                            "seed": seed,
                            "invariant_violations": 0,
                            "identity_error": episode["identity_error"],
                            "replay_mismatch": mismatch,
                            "trace_digest": digest,
                        }
                    )
                    output_rows = [
                        {"candidate_id": candidate_id, **row} for row in decision_rows
                    ]
                    if decision_writer is None:
                        decision_writer = csv.DictWriter(
                            decision_handle, fieldnames=list(output_rows[0])
                        )
                        decision_writer.writeheader()
                    decision_writer.writerows(output_rows)
                    decision_handle.flush()
                    seed_results[policy] = (episode_row, decision_rows)
                    if policy in PREVENTIVE_POLICIES:
                        requests, slower, queued, observed_max, failures, row_count = (
                            _request_metrics(config, decision_rows)
                        )
                        if policy == "workload_aware":
                            workload_requests += requests
                            workload_slower += slower
                        queued_machine_steps += queued
                        max_queue = max(max_queue, observed_max)
                        failure_counts.update(failures)
                        preventive_decision_rows += row_count
                    progress.update(1)
                threshold_episode, threshold_decisions = seed_results["threshold_fastest"]
                workload_episode, workload_decisions = seed_results["workload_aware"]
                trace_different = int(
                    threshold_episode["trace_digest"] != workload_episode["trace_digest"]
                )
                trace_divergences += trace_different
                matched, different = _paired_action_counts(
                    threshold_decisions, workload_decisions
                )
                matched_requests += matched
                different_requests += different
                pairwise.append(
                    {
                        "candidate_id": candidate_id,
                        "seed": seed,
                        "threshold_objective": threshold_episode["objective"],
                        "workload_objective": workload_episode["objective"],
                        "threshold_minus_workload_objective": float(
                            threshold_episode["objective"]
                        )
                        - float(workload_episode["objective"]),
                        "trace_different": trace_different,
                        "matched_requests": matched,
                        "different_technician_actions": different,
                    }
                )
                if (seed_index + 1) % 25 == 0 or seed_index + 1 == len(seeds):
                    _write_csv(episodes, output / "episodes.partial.csv")
            summaries.append(
                _candidate_summary(
                    candidate_id,
                    config,
                    seeds,
                    policy_episodes,
                    trace_divergences,
                    matched_requests,
                    different_requests,
                    workload_requests,
                    workload_slower,
                    queued_machine_steps,
                    preventive_decision_rows,
                    max_queue,
                    failure_counts,
                )
            )
    finally:
        progress.close()
        decision_handle.close()

    _write_csv(episodes, output / "episodes.csv")
    _write_csv(coordination, output / "coordination.csv")
    _write_csv(pairwise, output / "pairwise.csv")
    _write_csv(summaries, output / "candidate_summary.csv")
    qualifying = [row for row in summaries if row["qualifies"]]
    selected = min(
        qualifying,
        key=lambda row: (
            -float(row["qualification_score"]),
            (
                float(row["threshold_fastest_objective_mean"])
                + float(row["workload_aware_objective_mean"])
            )
            / 2.0,
            str(row["candidate_id"]),
        ),
        default=None,
    )
    mechanics_passed = bool(
        len(episodes) == len(candidates) * len(POLICIES) * len(seeds)
        and all(float(row["identity_error"]) <= 1e-9 for row in episodes)
        and all(int(row["replay_mismatch"]) == 0 for row in episodes)
    )
    choice_activated = any(
        float(row["matched_request_action_divergence_fraction"]) > 0.0
        for row in summaries
    )
    hard_gate_passed = mechanics_passed and (args.profile != "smoke" or choice_activated)
    summary_payload = {
        "purpose": "Exploratory environment calibration before MARL training.",
        "mechanics_passed": mechanics_passed,
        "choice_activated": choice_activated,
        "calibration_success": selected is not None,
        "authoritative_selection": args.profile == "full",
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "qualifying_candidate_ids": [row["candidate_id"] for row in qualifying],
        "candidates": summaries,
    }
    (output / "candidate_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "selected_candidate.json").write_text(
        json.dumps(
            {
                "candidate_id": selected["candidate_id"] if selected else None,
                "authoritative": args.profile == "full",
                "qualification_rule": "locked in docs/passive_technician_v2_calibration_plan.md",
                "metrics": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.update(
        {
            "status": "COMPLETED" if hard_gate_passed else "FAILED_AUDIT",
            "finished_at": datetime.now(UTC).isoformat(),
            "episode_count": len(episodes),
            "hard_gate_passed": hard_gate_passed,
            "calibration_success": selected is not None,
            "selected_candidate_id": selected["candidate_id"] if selected else None,
            "outputs": sorted(
                str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
            ),
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2, sort_keys=True))
    if not hard_gate_passed:
        raise RuntimeError("passive v2.1 calibration failed its hard audit gate")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
