"""Small, self-contained MARL benchmark for passive shared technicians.

The simulator deliberately has no technician learning, absence, substitution,
experience, maintenance windows, or hidden side effects. A machine chooses
defer or a technician; simultaneous technician collisions are resolved by the
same deterministic rule for every policy, while failure events are stochastic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm


OBJECTIVE_VERSION = "fifo_queue_waiting_v1"
OBSERVATION_VERSION = "resource_aware_v1"


@dataclass(frozen=True)
class PassiveConfig:
    machines: int = 3
    technicians: int = 2
    horizon: int = 12
    failure_age: int = 5
    failure_probability: float = 0.45
    max_age: int = 8
    service_time: tuple[tuple[int, ...], ...] = ((2, 4), (4, 2), (3, 3))
    maintenance_cost: float = 1.0
    downtime_cost: float = 8.0
    failure_cost: float = 15.0
    queue_waiting_cost: float = 0.5
    collision_cost: float = 4.0
    semantics: str = "fifo_queue"

    @property
    def observation_dim(self) -> int:
        return 3 + 6 * self.technicians

    def validate(self) -> None:
        if self.machines < 1 or self.technicians < 1 or self.horizon < 1:
            raise ValueError("machines, technicians, and horizon must be positive")
        if len(self.service_time) != self.machines:
            raise ValueError("service_time must have one row per machine")
        if any(len(row) != self.technicians for row in self.service_time):
            raise ValueError("service_time must have one entry per technician")
        if self.queue_waiting_cost < 0:
            raise ValueError("queue_waiting_cost must be non-negative")
        if self.semantics not in {"fifo_queue", "priority_resolver", "invalid_collision"}:
            raise ValueError("unsupported technician conflict semantics")


@dataclass(frozen=True)
class PassiveState:
    ages: tuple[int, ...]
    failed: tuple[bool, ...]
    busy_until: tuple[int, ...]
    assigned_machine: tuple[int, ...]
    queues: tuple[tuple[int, ...], ...]


class PassiveTechnicianEnv:
    """Finite-horizon maintenance environment with seeded failure events."""

    def __init__(self, config: PassiveConfig, *, seed: int = 0):
        config.validate()
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.time = 0
        self.state = self.initial_state()
        self.metrics: dict[str, float] = {}

    def initial_state(self) -> PassiveState:
        return PassiveState(
            ages=tuple(0 for _ in range(self.config.machines)),
            failed=tuple(False for _ in range(self.config.machines)),
            busy_until=tuple(0 for _ in range(self.config.technicians)),
            assigned_machine=tuple(-1 for _ in range(self.config.technicians)),
            queues=tuple(tuple() for _ in range(self.config.technicians)),
        )

    def reset(self) -> tuple[tuple[int, ...], ...]:
        self.time = 0
        self.state = self.initial_state()
        self.metrics = {
            "objective": 0.0,
            "failures": 0,
            "jobs": 0,
            "collisions": 0,
            "waiting": 0,
            "invalid_requests": 0,
            "busy_requests": 0,
        }
        return self.observations()

    def action_mask(self, machine: int) -> tuple[bool, ...]:
        """Return actions that do not create a duplicate machine request.

        FIFO queueing intentionally permits a request for a busy technician.
        A machine that is already queued or being serviced has no useful
        nonzero action, however, so only defer remains valid in that state.
        """
        if not 0 <= machine < self.config.machines:
            raise IndexError(f"unknown machine {machine}")
        queued = any(machine in queue for queue in self.state.queues)
        assigned = machine in self.state.assigned_machine
        if queued or assigned:
            return (True,) + (False,) * self.config.technicians
        return (True,) + tuple(
            math.isfinite(self.config.service_time[machine][technician])
            for technician in range(self.config.technicians)
        )

    def action_masks(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(self.action_mask(machine) for machine in range(self.config.machines))

    @property
    def observation_dim(self) -> int:
        """Dimension of one machine's local resource-aware observation."""
        return self.config.observation_dim

    def observations(self) -> tuple[tuple[int, ...], ...]:
        """Return structured local observations for all machine agents.

        Each machine sees its time, age, failure state, and per-technician
        availability, remaining busy time, self-assignment, queue length,
        queue position, and fixed service time. The service-time row is part of
        the observation so technician selection is an observable decision.
        """
        observations: list[tuple[int, ...]] = []
        for machine in range(self.config.machines):
            features = [self.time, self.state.ages[machine], int(self.state.failed[machine])]
            for technician in range(self.config.technicians):
                queue = self.state.queues[technician]
                position = queue.index(machine) + 1 if machine in queue else 0
                service_time = self.config.service_time[machine][technician]
                service_value = -1 if not math.isfinite(service_time) else int(service_time)
                features.extend(
                    [
                        int(self.available(technician)),
                        max(0, self.state.busy_until[technician] - self.time),
                        int(self.state.assigned_machine[technician] == machine),
                        len(queue),
                        position,
                        service_value,
                    ]
                )
            observations.append(tuple(features))
        return tuple(observations)

    def available(self, technician: int) -> bool:
        return self.state.busy_until[technician] <= self.time

    def transition(
        self,
        state: PassiveState,
        actions: tuple[int, ...],
        time: int,
        failure_events: tuple[bool, ...] | None = None,
    ) -> tuple[PassiveState, float, dict[str, float]]:
        cfg = self.config
        if len(actions) != cfg.machines or any(a < 0 or a > cfg.technicians for a in actions):
            raise ValueError("one action in {0..technicians} is required per machine")
        selected: dict[int, list[int]] = {}
        for machine, action in enumerate(actions):
            if action:
                selected.setdefault(action - 1, []).append(machine)
        accepted: dict[int, int] = {}
        collisions = sum(max(0, len(machines) - 1) for machines in selected.values())
        queues = [list(queue) for queue in state.queues]
        queued = {machine for queue in queues for machine in queue}
        assigned_machines = {machine for machine in state.assigned_machine if machine >= 0}
        invalid_requests = 0
        busy_requests = 0
        if cfg.semantics == "fifo_queue":
            for technician, machines in selected.items():
                for machine in machines:
                    busy_requests += int(state.busy_until[technician] > time)
                    if machine in queued or machine in assigned_machines:
                        invalid_requests += 1
                        continue
                    if machine not in queued:
                        queues[technician].append(machine)
                        queued.add(machine)
            for technician in range(cfg.technicians):
                if state.busy_until[technician] <= time and queues[technician]:
                    accepted[technician] = queues[technician].pop(0)
            waiting = sum(len(queue) for queue in queues)
        else:
            waiting = 0
            for technician, machines in selected.items():
                busy_requests += sum(state.busy_until[technician] > time for _ in machines)
                eligible = [
                    machine
                    for machine in machines
                    if machine not in queued and machine not in assigned_machines
                ]
                invalid_requests += len(machines) - len(eligible)
                if cfg.semantics == "invalid_collision" and len(machines) > 1:
                    continue
                if state.busy_until[technician] > time:
                    waiting += len(eligible)
                else:
                    if eligible:
                        accepted[technician] = min(eligible)
                        waiting += max(0, len(eligible) - 1)

        ages = list(state.ages)
        failed = list(state.failed)
        busy_until = list(state.busy_until)
        assigned = list(state.assigned_machine)
        objective = 0.0
        failures = 0
        jobs = 0
        objective += cfg.queue_waiting_cost * waiting
        # Cost for the state before service starts.
        for machine in range(cfg.machines):
            if failed[machine]:
                objective += cfg.downtime_cost
        for technician, machine in accepted.items():
            duration = cfg.service_time[machine][technician]
            busy_until[technician] = time + duration
            assigned[technician] = machine
            objective += cfg.maintenance_cost
            jobs += 1
            ages[machine] = 0
            failed[machine] = False
        for machine in range(cfg.machines):
            if machine not in accepted.values():
                ages[machine] = min(cfg.max_age, ages[machine] + 1)
                deterministic_failure = ages[machine] >= cfg.failure_age
                sampled_failure = failure_events is None or failure_events[machine]
                if deterministic_failure and sampled_failure and not failed[machine]:
                    failed[machine] = True
                    failures += 1
                    objective += cfg.failure_cost
        for technician in range(cfg.technicians):
            if busy_until[technician] <= time + 1:
                assigned[technician] = -1
        if cfg.semantics in {"priority_resolver", "invalid_collision"}:
            objective += cfg.collision_cost * collisions
        next_state = PassiveState(
            tuple(ages), tuple(failed), tuple(busy_until), tuple(assigned),
            tuple(tuple(queue) for queue in queues),
        )
        info = {
            "objective": objective,
            "failures": failures,
            "jobs": jobs,
            "collisions": collisions,
            "waiting": waiting,
            "invalid_requests": invalid_requests,
            "busy_requests": busy_requests,
        }
        return next_state, objective, info

    def step(self, actions: Iterable[int]) -> tuple[tuple[int, ...], float, bool, dict[str, float]]:
        events = tuple(
            self.rng.random() < self.config.failure_probability
            if (not failed and age + 1 >= self.config.failure_age)
            else False
            for age, failed in zip(self.state.ages, self.state.failed)
        )
        next_state, cost, info = self.transition(self.state, tuple(actions), self.time, events)
        self.state = next_state
        self.time += 1
        for key, value in info.items():
            self.metrics[key] = self.metrics.get(key, 0.0) + value
        done = self.time >= self.config.horizon
        return self.observations(), -cost, done, info


def dispatcher_action(env: PassiveTechnicianEnv) -> tuple[int, ...]:
    """Risk-first feasible dispatcher with skill-aware technician choice."""
    ranked = sorted(
        range(env.config.machines),
        key=lambda m: (env.state.failed[m], env.state.ages[m]), reverse=True,
    )
    available = {t for t in range(env.config.technicians) if env.available(t)}
    actions = [0] * env.config.machines
    for machine in ranked:
        if not available or (not env.state.failed[machine] and env.state.ages[machine] < env.config.failure_age - 1):
            continue
        technician = min(available, key=lambda t: env.config.service_time[machine][t])
        actions[machine] = technician + 1
        available.remove(technician)
    return tuple(actions)


def evaluate_policy(config: PassiveConfig, policy, seed: int) -> dict[str, float | int | str]:
    env = PassiveTechnicianEnv(config, seed=seed)
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(policy(env))
    return {"policy": getattr(policy, "__name__", "policy"), "seed": seed, "train_seed": "", **env.metrics}


def exact_optimum(config: PassiveConfig) -> float:
    """Return the optimal expected cost for the stochastic finite-horizon instance."""
    config.validate()

    @lru_cache(maxsize=None)
    def solve(time: int, state: PassiveState) -> float:
        if time >= config.horizon:
            return 0.0
        best = math.inf
        # Enumerate the joint action space; this is intentionally only for tiny instances.
        for actions in np.ndindex(*(config.technicians + 1 for _ in range(config.machines))):
            eligible = tuple(
                not failed and age + 1 >= config.failure_age
                for age, failed in zip(state.ages, state.failed)
            )
            expected = 0.0
            for events in itertools.product((False, True), repeat=config.machines):
                probability = 1.0
                for is_eligible, event in zip(eligible, events):
                    if not is_eligible:
                        probability *= float(not event)
                    else:
                        p = config.failure_probability
                        probability *= p if event else 1.0 - p
                if probability == 0.0:
                    continue
                next_state, cost, _ = PassiveTechnicianEnv(config).transition(
                    state, actions, time, events
                )
                expected += probability * (cost + solve(time + 1, next_state))
            best = min(best, expected)
        return best

    return solve(0, PassiveTechnicianEnv(config).initial_state())


class IndependentQ:
    def __init__(self, config: PassiveConfig, seed: int, alpha: float = 0.2, gamma: float = 0.95):
        self.config, self.rng = config, random.Random(seed)
        self.alpha, self.gamma = alpha, gamma
        self.q: list[dict[tuple[tuple[int, ...], int], float]] = [dict() for _ in range(config.machines)]

    def _key(self, obs: tuple[int, ...], action: int) -> tuple[tuple[int, ...], int]:
        return obs, action

    def action(
        self,
        machine: int,
        obs: tuple[int, ...],
        epsilon: float,
        mask: tuple[bool, ...] | None = None,
    ) -> int:
        valid_actions = [
            action
            for action in range(self.config.technicians + 1)
            if mask is None or mask[action]
        ]
        if not valid_actions:
            raise ValueError("action mask must leave at least one valid action")
        values = [
            self.q[machine].get(self._key(obs, action), 0.0)
            for action in valid_actions
        ]
        if self.rng.random() < epsilon:
            return self.rng.choice(valid_actions)
        return valid_actions[min(range(len(values)), key=lambda index: values[index])]

    def update(
        self,
        machine: int,
        obs: tuple[int, ...],
        action: int,
        cost: float,
        next_obs: tuple[int, ...],
        next_mask: tuple[bool, ...] | None = None,
    ) -> None:
        old = self.q[machine].get(self._key(obs, action), 0.0)
        valid_actions = [
            a
            for a in range(self.config.technicians + 1)
            if next_mask is None or next_mask[a]
        ]
        if not valid_actions:
            raise ValueError("next action mask must leave at least one valid action")
        future = min(self.q[machine].get(self._key(next_obs, a), 0.0) for a in valid_actions)
        self.q[machine][self._key(obs, action)] = old + self.alpha * (cost + self.gamma * future - old)

    def save(self, path: Path) -> None:
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "machines": self.config.machines,
            "technicians": self.config.technicians,
            "q": [
                {json.dumps([list(obs), action]): value for (obs, action), value in table.items()}
                for table in self.q
            ],
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path, config: PassiveConfig, seed: int = 0) -> "IndependentQ":
        payload = json.loads(path.read_text())
        learner = cls(config, seed, alpha=float(payload["alpha"]), gamma=float(payload["gamma"]))
        learner.q = [
            {
                (tuple(item[0]), int(item[1])): float(value)
                for key, value in table.items()
                for item in [json.loads(key)]
            }
            for table in payload["q"]
        ]
        return learner


def train_independent(config: PassiveConfig, seed: int, episodes: int) -> IndependentQ:
    learner = IndependentQ(config, seed)
    for episode in range(episodes):
        env = PassiveTechnicianEnv(config, seed=seed + episode)
        obs = env.reset()
        done = False
        epsilon = max(0.05, 1.0 - episode / max(1, episodes * 0.8))
        while not done:
            masks = env.action_masks()
            actions = tuple(
                learner.action(m, obs[m], epsilon, masks[m]) for m in range(config.machines)
            )
            next_obs, reward, done, info = env.step(actions)
            next_masks = env.action_masks()
            for machine in range(config.machines):
                learner.update(
                    machine,
                    obs[machine],
                    actions[machine],
                    -reward,
                    next_obs[machine],
                    next_masks[machine],
                )
            obs = next_obs
    return learner


def evaluate_marl(config: PassiveConfig, learner: IndependentQ, seed: int, train_seed: int) -> dict[str, float | int | str]:
    env = PassiveTechnicianEnv(config, seed=seed)
    obs = env.reset()
    done = False
    while not done:
        masks = env.action_masks()
        actions = tuple(
            learner.action(m, obs[m], 0.0, masks[m]) for m in range(config.machines)
        )
        obs, _, done, _ = env.step(actions)
    return {"policy": "independent_marl", "seed": seed, "train_seed": train_seed, **env.metrics}


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    # Keep the main comparison small enough that exact expected DP remains exact.
    config = PassiveConfig(
        machines=2,
        technicians=2,
        horizon=6,
        failure_age=3,
        max_age=5,
        service_time=((2, 4), (4, 2)),
    )
    if args.profile == "smoke":
        train_seeds, eval_seeds, episodes = (11,), (101, 102, 103), 40
    elif args.profile == "pilot":
        train_seeds, eval_seeds, episodes = (11, 12, 13), tuple(range(101, 111)), 1500
    else:
        train_seeds, eval_seeds, episodes = tuple(range(11, 21)), tuple(range(101, 201)), 10_000
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "RUNNING", "experiment": "passive_technician_marl", "profile": args.profile, "config": asdict(config), "train_seeds": list(train_seeds), "evaluation_seeds": list(eval_seeds), "episodes_per_train_seed": episodes, "sealed_test_evaluated": False, "git_revision": "uncommitted", "started_at": datetime.now(UTC).isoformat()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows: list[dict] = []
    progress_rows: list[dict] = []
    exact = exact_optimum(config)
    dispatcher_rows = [evaluate_policy(config, dispatcher_action, seed) for seed in tqdm(eval_seeds, desc="dispatcher", unit="episode")]
    rows.extend(dispatcher_rows)
    _write_csv(rows, output / "episodes.partial.csv")
    _write_csv(dispatcher_rows, output / "coordination.csv")
    for seed in tqdm(train_seeds, desc="independent MARL", unit="seed"):
        learner = train_independent(config, seed, episodes)
        model_dir = output / "independent_marl" / f"train_seed_{seed}" / "checkpoints"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.json"
        learner.save(model_path)
        learner = IndependentQ.load(model_path, config, seed=seed)
        progress_rows.append({"algorithm": "independent_marl", "train_seed": seed, "episodes": episodes, "checkpoint": str(model_path.relative_to(output))})
        for eval_seed in tqdm(eval_seeds, desc=f"evaluate seed {seed}", unit="episode", leave=False):
            rows.append(evaluate_marl(config, learner, eval_seed, seed))
        _write_csv(rows, output / "episodes.partial.csv")
    _write_csv(progress_rows, output / "training_progress.csv")
    _write_csv(rows, output / "episodes.csv")
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy"]), []).append(float(row["objective"]))
    summary = {"exact_optimum_cost": exact, "mean_cost": {policy: statistics.fmean(values) for policy, values in grouped.items()}, "gap_to_optimum": {policy: statistics.fmean(values) - exact for policy, values in grouped.items()}, "audits": {"episode_count": len(rows), "nonnegative_costs": all(float(row["objective"]) >= 0 for row in rows), "sealed_panel_closed": not set(train_seeds + eval_seeds).intersection(range(62000, 62100))}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest.update({"status": "COMPLETED", "finished_at": datetime.now(UTC).isoformat(), "exact_optimum_cost": exact, "outputs": sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
