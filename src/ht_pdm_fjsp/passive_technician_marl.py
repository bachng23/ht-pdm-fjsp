"""Small, self-contained MARL benchmark for passive shared technicians.

The simulator deliberately has no technician learning, absence, substitution,
experience, maintenance windows, or hidden side effects.  A machine chooses
defer or a technician; simultaneous technician collisions are resolved by the
same deterministic rule for every policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


@dataclass(frozen=True)
class PassiveConfig:
    machines: int = 3
    technicians: int = 2
    horizon: int = 12
    failure_age: int = 5
    max_age: int = 8
    service_time: tuple[tuple[int, ...], ...] = ((2, 4), (4, 2), (3, 3))
    maintenance_cost: float = 1.0
    downtime_cost: float = 8.0
    failure_cost: float = 15.0
    collision_cost: float = 4.0

    def validate(self) -> None:
        if self.machines < 1 or self.technicians < 1 or self.horizon < 1:
            raise ValueError("machines, technicians, and horizon must be positive")
        if len(self.service_time) != self.machines:
            raise ValueError("service_time must have one row per machine")
        if any(len(row) != self.technicians for row in self.service_time):
            raise ValueError("service_time must have one entry per technician")


@dataclass(frozen=True)
class PassiveState:
    ages: tuple[int, ...]
    failed: tuple[bool, ...]
    busy_until: tuple[int, ...]
    assigned_machine: tuple[int, ...]


class PassiveTechnicianEnv:
    """Finite-horizon deterministic maintenance environment."""

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
        )

    def reset(self) -> tuple[int, ...]:
        self.time = 0
        self.state = self.initial_state()
        self.metrics = {"objective": 0.0, "failures": 0, "jobs": 0, "collisions": 0, "waiting": 0}
        return self.observations()

    def observations(self) -> tuple[int, ...]:
        # Local observation: age, failure flag, and technician availability mask.
        available = sum(
            int(until <= self.time) << technician
            for technician, until in enumerate(self.state.busy_until)
        )
        return tuple(
            int(self.state.ages[machine])
            + self.config.max_age * int(self.state.failed[machine])
            + (self.config.max_age + 1) * 2 * available
            for machine in range(self.config.machines)
        )

    def available(self, technician: int) -> bool:
        return self.state.busy_until[technician] <= self.time

    def transition(self, state: PassiveState, actions: tuple[int, ...], time: int) -> tuple[PassiveState, float, dict[str, float]]:
        cfg = self.config
        if len(actions) != cfg.machines or any(a < 0 or a > cfg.technicians for a in actions):
            raise ValueError("one action in {0..technicians} is required per machine")
        selected: dict[int, list[int]] = {}
        for machine, action in enumerate(actions):
            if action:
                selected.setdefault(action - 1, []).append(machine)
        accepted: dict[int, int] = {}
        collisions = 0
        waiting = 0
        for technician, machines in selected.items():
            if len(machines) > 1:
                collisions += len(machines) - 1
            if state.busy_until[technician] > time:
                waiting += len(machines)
            else:
                accepted[technician] = min(machines)
                waiting += max(0, len(machines) - 1)

        ages = list(state.ages)
        failed = list(state.failed)
        busy_until = list(state.busy_until)
        assigned = list(state.assigned_machine)
        objective = 0.0
        failures = 0
        jobs = 0
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
                if ages[machine] >= cfg.failure_age and not failed[machine]:
                    failed[machine] = True
                    failures += 1
                    objective += cfg.failure_cost
        for technician in range(cfg.technicians):
            if busy_until[technician] <= time + 1:
                assigned[technician] = -1
        objective += cfg.collision_cost * collisions
        next_state = PassiveState(tuple(ages), tuple(failed), tuple(busy_until), tuple(assigned))
        info = {"objective": objective, "failures": failures, "jobs": jobs, "collisions": collisions, "waiting": waiting}
        return next_state, objective, info

    def step(self, actions: Iterable[int]) -> tuple[tuple[int, ...], float, bool, dict[str, float]]:
        next_state, cost, info = self.transition(self.state, tuple(actions), self.time)
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
    return {"policy": getattr(policy, "__name__", "policy"), "seed": seed, **env.metrics}


def exact_optimum(config: PassiveConfig) -> float:
    """Return the optimal cost for the deterministic finite-horizon instance."""
    config.validate()

    @lru_cache(maxsize=None)
    def solve(time: int, state: PassiveState) -> float:
        if time >= config.horizon:
            return 0.0
        best = math.inf
        # Enumerate the joint action space; this is intentionally only for tiny instances.
        for actions in np.ndindex(*(config.technicians + 1 for _ in range(config.machines))):
            next_state, cost, _ = PassiveTechnicianEnv(config).transition(state, actions, time)
            best = min(best, cost + solve(time + 1, next_state))
        return best

    return solve(0, PassiveTechnicianEnv(config).initial_state())


class IndependentQ:
    def __init__(self, config: PassiveConfig, seed: int, alpha: float = 0.2, gamma: float = 0.95):
        self.config, self.rng = config, random.Random(seed)
        self.alpha, self.gamma = alpha, gamma
        self.q: list[dict[tuple[int, int], float]] = [dict() for _ in range(config.machines)]

    def _key(self, obs: int, action: int) -> tuple[int, int]:
        return obs, action

    def action(self, machine: int, obs: int, epsilon: float) -> int:
        values = [self.q[machine].get(self._key(obs, action), 0.0) for action in range(self.config.technicians + 1)]
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.config.technicians + 1)
        return min(range(len(values)), key=lambda action: values[action])

    def update(self, machine: int, obs: int, action: int, cost: float, next_obs: int) -> None:
        old = self.q[machine].get(self._key(obs, action), 0.0)
        future = min(self.q[machine].get(self._key(next_obs, a), 0.0) for a in range(self.config.technicians + 1))
        self.q[machine][self._key(obs, action)] = old + self.alpha * (cost + self.gamma * future - old)

    def save(self, path: Path) -> None:
        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "machines": self.config.machines,
            "technicians": self.config.technicians,
            "q": [{f"{obs}:{action}": value for (obs, action), value in table.items()} for table in self.q],
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path, config: PassiveConfig, seed: int = 0) -> "IndependentQ":
        payload = json.loads(path.read_text())
        learner = cls(config, seed, alpha=float(payload["alpha"]), gamma=float(payload["gamma"]))
        learner.q = [
            {tuple(map(int, key.split(":"))): float(value) for key, value in table.items()}
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
            actions = tuple(learner.action(m, obs[m], epsilon) for m in range(config.machines))
            next_obs, reward, done, info = env.step(actions)
            for machine in range(config.machines):
                learner.update(machine, obs[machine], actions[machine], -reward + info["collisions"] * config.collision_cost, next_obs[machine])
            obs = next_obs
    return learner


def evaluate_marl(config: PassiveConfig, learner: IndependentQ, seed: int) -> dict[str, float | int | str]:
    env = PassiveTechnicianEnv(config, seed=seed)
    obs = env.reset()
    done = False
    while not done:
        actions = tuple(learner.action(m, obs[m], 0.0) for m in range(config.machines))
        obs, _, done, _ = env.step(actions)
    return {"policy": "independent_marl", "seed": seed, **env.metrics}


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    config = PassiveConfig()
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
            rows.append(evaluate_marl(config, learner, eval_seed))
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
