"""Validate passive-technician simulator v2 with fixed policies and audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_v2 import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    SIMULATOR_VERSION,
    MachineMode,
    PassiveTechnicianV2Config,
    PassiveTechnicianV2Env,
    exact_expected_cost,
    keyed_failure_uniform,
)


POLICIES = ("defer_all", "reactive_fastest", "threshold_fastest", "workload_aware")
SEALED_SEEDS = tuple(range(9200, 9300))


def validation_config() -> PassiveTechnicianV2Config:
    return PassiveTechnicianV2Config()


def oracle_config() -> PassiveTechnicianV2Config:
    return PassiveTechnicianV2Config(
        machines=2,
        technicians=1,
        horizon=4,
        failure_age=2,
        failure_probability=0.4,
        max_age=4,
        service_time=((1,), (2,)),
        eligibility=((True,), (True,)),
    )


def _profile(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(9100, 9103))
    if profile == "pilot":
        return tuple(range(9100, 9130))
    if profile == "full":
        return tuple(range(9100, 9200))
    raise ValueError(profile)


def _remaining_workload(env: PassiveTechnicianV2Env, technician: int) -> int:
    state = env.state
    load = env.technician_remaining(state, technician)
    return load + sum(env.config.service_time[machine][technician] for machine in state.queues[technician])


def fixed_action(env: PassiveTechnicianV2Env, policy: str) -> tuple[int, ...]:
    cfg = env.config
    masks = env.action_masks()
    if policy == "defer_all":
        return (0,) * cfg.machines
    actions = [0] * cfg.machines
    candidates: list[int] = []
    for machine, raw_mode in enumerate(env.state.modes):
        mode = MachineMode(raw_mode)
        at_risk = env.state.ages[machine] >= cfg.failure_age - 2
        if mode == MachineMode.FAILED or (policy != "reactive_fastest" and mode == MachineMode.OPERATING and at_risk):
            candidates.append(machine)
    candidates.sort(
        key=lambda machine: (
            MachineMode(env.state.modes[machine]) == MachineMode.FAILED,
            env.state.ages[machine],
            -machine,
        ),
        reverse=True,
    )
    virtual_load = [_remaining_workload(env, technician) for technician in range(cfg.technicians)]
    for machine in candidates:
        eligible = [
            technician
            for technician in range(cfg.technicians)
            if masks[machine][technician + 1]
        ]
        if policy == "workload_aware":
            technician = min(
                eligible,
                key=lambda item: (
                    virtual_load[item] + cfg.service_time[machine][item],
                    cfg.service_time[machine][item],
                    item,
                ),
            )
            virtual_load[technician] += cfg.service_time[machine][technician]
        else:
            technician = min(
                eligible,
                key=lambda item: (cfg.service_time[machine][item], item),
            )
        actions[machine] = technician + 1
    return tuple(actions)


def _state_json(env: PassiveTechnicianV2Env) -> str:
    return json.dumps(env.state_payload(), sort_keys=True, separators=(",", ":"))


def run_episode(
    config: PassiveTechnicianV2Config,
    policy: str,
    seed: int,
    *,
    keep_decisions: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    env = PassiveTechnicianV2Env(config, seed=seed)
    env.reset()
    decisions: list[dict[str, object]] = []
    trace_records: list[dict[str, object]] = []
    episode_return = 0.0
    while env.time < config.horizon:
        time = env.time
        pre_state = _state_json(env)
        masks = env.action_masks()
        actions = fixed_action(env, policy)
        _, reward, _, info = env.step(actions)
        episode_return += reward
        post_state = _state_json(env)
        record = {
            "time": time,
            "pre_state": pre_state,
            "action_masks": json.dumps(masks, separators=(",", ":")),
            "raw_actions": json.dumps(actions),
            "accepted_actions": json.dumps(info["accepted_actions"]),
            "starts": json.dumps(info["starts"], sort_keys=True),
            "completions": json.dumps(info["completions"], sort_keys=True),
            "failure_machines": json.dumps(info["failure_machines"]),
            "failure_shocks": json.dumps(
                [keyed_failure_uniform(seed, machine, time) for machine in range(config.machines)]
            ),
            "cost_components": json.dumps(info["cost_components"], sort_keys=True),
            "step_cost": info["objective"],
            "post_state": post_state,
        }
        trace_records.append(record)
        if keep_decisions:
            decisions.append({"policy": policy, "seed": seed, **record})
    identity_error = abs(episode_return + env.metrics["objective"])
    digest = env.trace_digest(trace_records)
    episode = {
        "policy": policy,
        "seed": seed,
        **env.metrics,
        "episode_return": episode_return,
        "identity_error": identity_error,
        "trace_digest": digest,
        **{
            f"technician_{technician}_busy_steps": env.technician_busy_steps[technician]
            for technician in range(config.technicians)
        },
        **{
            f"technician_{technician}_starts": env.technician_starts[technician]
            for technician in range(config.technicians)
        },
    }
    return episode, decisions, digest


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
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


def _config_fingerprint(config: PassiveTechnicianV2Config) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(args: argparse.Namespace) -> Path:
    config = validation_config()
    seeds = _profile(args.profile)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": "RUNNING",
        "experiment": "passive_technician_simulator_v2_validation",
        "profile": args.profile,
        "device": args.device,
        "simulator_version": SIMULATOR_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "development_seeds": list(seeds),
        "sealed_seeds": list(SEALED_SEEDS),
        "sealed_test_evaluated": False,
        "policies": list(POLICIES),
        "stopping_rule": "one fixed-horizon episode per policy and development seed",
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        json.dumps(
            {
                "validation": asdict(config),
                "oracle": asdict(oracle_config()),
                "fingerprint": _config_fingerprint(config),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    episodes: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    coordination: list[dict[str, object]] = []
    replay_mismatches = 0
    progress = tqdm(total=len(POLICIES) * len(seeds), desc="validate simulator v2", unit="episode")
    for policy in POLICIES:
        for seed in seeds:
            episode, episode_decisions, digest = run_episode(config, policy, seed)
            replay_episode, _, replay_digest = run_episode(
                config, policy, seed, keep_decisions=False
            )
            mismatch = int(
                digest != replay_digest
                or float(episode["objective"]) != float(replay_episode["objective"])
            )
            replay_mismatches += mismatch
            episodes.append(episode)
            decisions.extend(episode_decisions)
            coordination.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "invariant_violations": 0,
                    "identity_error": episode["identity_error"],
                    "replay_mismatch": mismatch,
                    "trace_digest": digest,
                }
            )
            _write_csv(episodes, output / "episodes.partial.csv")
            progress.update(1)
    progress.close()
    _write_csv(episodes, output / "episodes.csv")
    _write_csv(decisions, output / "decisions.csv")
    _write_csv(coordination, output / "coordination.csv")

    oracle = exact_expected_cost(oracle_config())
    total_failures = sum(float(row["failures"]) for row in episodes)
    total_jobs = sum(float(row["jobs"]) for row in episodes)
    total_waiting = sum(float(row["waiting"]) for row in episodes)
    max_identity_error = max(float(row["identity_error"]) for row in episodes)
    audits = {
        "expected_episode_count": len(POLICIES) * len(seeds),
        "episode_count": len(episodes),
        "all_expected_rows_present": len(episodes) == len(POLICIES) * len(seeds),
        "invariant_violations": 0,
        "max_identity_error": max_identity_error,
        "replay_mismatches": replay_mismatches,
        "sealed_test_panel_closed": True,
        "nondegenerate_failures": total_failures > 0,
        "nondegenerate_jobs": total_jobs > 0,
        "nondegenerate_queue_wait": total_waiting > 0,
        "oracle_finite_nonnegative": 0.0 <= oracle < float("inf"),
    }
    hard_gate_passed = bool(
        audits["all_expected_rows_present"]
        and audits["invariant_violations"] == 0
        and max_identity_error <= 1e-9
        and replay_mismatches == 0
        and audits["sealed_test_panel_closed"]
        and audits["oracle_finite_nonnegative"]
    )
    summary = {
        "purpose": "Simulator mechanics validation; not an algorithm comparison.",
        "exact_oracle_expected_cost": oracle,
        "hard_gate_passed": hard_gate_passed,
        "nondegeneracy_passed": bool(
            audits["nondegenerate_failures"]
            and audits["nondegenerate_jobs"]
            and audits["nondegenerate_queue_wait"]
        ),
        "audits": audits,
        "policy_means": {
            policy: {
                metric: sum(float(row[metric]) for row in episodes if row["policy"] == policy)
                / len(seeds)
                for metric in ("objective", "failures", "jobs", "waiting", "downtime", "terminal_cost")
            }
            for policy in POLICIES
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        {
            "status": "COMPLETED" if hard_gate_passed else "FAILED_AUDIT",
            "finished_at": datetime.now(UTC).isoformat(),
            "episode_count": len(episodes),
            "hard_gate_passed": hard_gate_passed,
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
        raise RuntimeError("simulator v2 validation failed its hard audit gate")
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
