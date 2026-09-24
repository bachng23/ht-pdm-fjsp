"""Compare collision semantics for the passive shared-technician benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_marl import PassiveConfig, PassiveTechnicianEnv


SEMANTICS = ("priority_resolver", "fifo_queue", "invalid_collision")
POLICIES = ("random_independent", "greedy_independent")


def proposals(env: PassiveTechnicianEnv, policy: str, proposal_rng: random.Random) -> tuple[int, ...]:
    if policy == "random_independent":
        return tuple(proposal_rng.randrange(env.config.technicians + 1) for _ in range(env.config.machines))
    if policy == "greedy_independent":
        return tuple(
            min(range(env.config.technicians), key=lambda t: env.config.service_time[machine][t]) + 1
            if (env.state.failed[machine] or env.state.ages[machine] >= env.config.failure_age - 1)
            else 0
            for machine in range(env.config.machines)
        )
    raise ValueError(policy)


def duplicate_count(actions: Iterable[int]) -> int:
    counts: dict[int, int] = defaultdict(int)
    for action in actions:
        if action:
            counts[action] += 1
    return sum(max(0, count - 1) for count in counts.values())


def run_episode(config: PassiveConfig, semantics: str, policy: str, seed: int) -> dict[str, float | int | str]:
    config = replace(config, semantics=semantics)
    env = PassiveTechnicianEnv(config, seed=seed)
    env.reset()
    queues: dict[int, list[int]] = defaultdict(list)
    pending: set[int] = set()
    proposal_conflicts = 0
    invalid_proposals = 0
    queue_waiting = 0
    accepted_by_machine = [0] * config.machines
    proposal_rng = random.Random(seed + 91_417)
    done = False
    while not done:
        raw = proposals(env, policy, proposal_rng)
        proposal_conflicts += duplicate_count(raw)
        available_before = {technician for technician in range(config.technicians) if env.available(technician)}
        actions = list(raw)
        if semantics == "invalid_collision":
            conflicted = {a for a in raw if a and raw.count(a) > 1}
            invalid_proposals += sum(1 for a in raw if a in conflicted)
            actions = [0 if action in conflicted else action for action in raw]
        elif semantics == "fifo_queue":
            for machine, action in enumerate(raw):
                if action and machine not in pending:
                    queues[action - 1].append(machine)
                    pending.add(machine)
            actions = [0] * config.machines
            for technician in range(config.technicians):
                if env.available(technician) and queues[technician]:
                    machine = queues[technician].pop(0)
                    pending.remove(machine)
                    actions[machine] = technician + 1
            queue_waiting += len(pending)
        elif semantics != "priority_resolver":
            raise ValueError(semantics)
        accepted_now: set[int] = set()
        for technician in available_before:
            machines = [machine for machine, action in enumerate(actions) if action == technician + 1]
            if machines:
                accepted_now.add(min(machines))
        _, _, done, info = env.step(actions)
        for machine in accepted_now:
            accepted_by_machine[machine] += 1
    return {
        "policy": policy,
        "semantics": semantics,
        "seed": seed,
        "objective": env.metrics["objective"] + invalid_proposals * config.collision_cost,
        "failures": env.metrics["failures"],
        "jobs": env.metrics["jobs"],
        "collision_events": proposal_conflicts,
        "invalid_proposals": invalid_proposals,
        "waiting": env.metrics["waiting"] + queue_waiting,
        "machine_0_jobs": accepted_by_machine[0],
        "machine_1_jobs": accepted_by_machine[1],
        "machine_2_jobs": accepted_by_machine[2],
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    config = PassiveConfig()
    if args.profile == "smoke":
        seeds = tuple(range(7001, 7004))
    elif args.profile == "pilot":
        seeds = tuple(range(7001, 7031))
    else:
        seeds = tuple(range(7001, 7101))
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for semantics in tqdm(SEMANTICS, desc="semantics", unit="mode"):
        for policy in POLICIES:
            for seed in tqdm(seeds, desc=f"{semantics}/{policy}", unit="episode", leave=False):
                rows.append(run_episode(config, semantics, policy, seed))
    _write_csv(rows, output / "semantics_episodes.csv")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['semantics']}:{row['policy']}"].append(row)
    summary = {}
    for key, group in grouped.items():
        summary[key] = {
            metric: statistics.fmean(float(row[metric]) for row in group)
            for metric in ("objective", "failures", "jobs", "collision_events", "invalid_proposals", "waiting", "machine_0_jobs", "machine_1_jobs", "machine_2_jobs")
        }
    manifest = {
        "status": "COMPLETED",
        "experiment": "passive_technician_semantics",
        "profile": args.profile,
        "config": asdict(config),
        "seeds": list(seeds),
        "semantics": list(SEMANTICS),
        "policies": list(POLICIES),
        "selected_main_semantics": "fifo_queue",
        "reproduction_semantics": "invalid_collision",
        "diagnostic_only_semantics": "priority_resolver",
        "started_at": datetime.now(UTC).isoformat(),
        "sealed_test_evaluated": False,
        "outputs": ["semantics_episodes.csv", "semantics_summary.json", "manifest.json"],
    }
    (output / "semantics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
