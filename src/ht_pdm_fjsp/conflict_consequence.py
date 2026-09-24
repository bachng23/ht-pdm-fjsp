"""Factorial diagnostic for material technician-conflict consequences."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.parallel_maintenance import (
    FAILED,
    MAINTENANCE,
    WORKING,
    ParallelMaintenanceConfig,
)


WAITING = 3
POLICIES = (
    "reactive_fibt",
    "coordinated_preventive",
    "independent_preventive",
    "coordinated_matching",
)
SEALED_TEST_SEEDS = tuple(range(63_200, 63_300))


@dataclass(frozen=True)
class ConsequenceCell:
    committed_request: bool
    maintenance_window: bool
    strong_substitution: bool

    @property
    def cell_id(self) -> str:
        return (
            f"commit{int(self.committed_request)}_"
            f"window{int(self.maintenance_window)}_"
            f"sub{int(self.strong_substitution)}"
        )


def cells_for_profile(profile: str) -> tuple[ConsequenceCell, ...]:
    if profile not in {"smoke", "full"}:
        raise ValueError(f"Unknown profile: {profile}")
    return tuple(
        ConsequenceCell(commitment, window, substitution)
        for commitment in (False, True)
        for window in (False, True)
        for substitution in (False, True)
    )


def seeds_for_profile(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(63_590, 63_593))
    if profile == "full":
        return tuple(range(63_500, 63_550))
    raise ValueError(f"Unknown profile: {profile}")


def selected_parent_config(base: ParallelMaintenanceConfig) -> ParallelMaintenanceConfig:
    return replace(
        base,
        maintenance=replace(
            base.maintenance,
            pm_duration_factor=0.35,
            pm_risk_threshold=0.005,
        ),
        costs=replace(base.costs, failure=36.0),
    )


def build_cell_config(
    base: ParallelMaintenanceConfig, cell: ConsequenceCell
) -> ParallelMaintenanceConfig:
    config = selected_parent_config(base)
    if not cell.strong_substitution:
        return config
    technicians = list(config.technicians)
    durations = [list(technician.base_duration) for technician in technicians]
    skills = [list(technician.base_skill) for technician in technicians]
    type_count = len(durations[0])
    for failure_type in range(type_count):
        specialist = min(
            range(len(technicians)),
            key=lambda index: durations[index][failure_type],
        )
        specialist_duration = durations[specialist][failure_type]
        for index in range(len(technicians)):
            if index == specialist:
                continue
            durations[index][failure_type] = max(
                durations[index][failure_type], 2.5 * specialist_duration
            )
            skills[index][failure_type] = min(skills[index][failure_type], 0.25)
    return replace(
        config,
        technicians=tuple(
            replace(
                technician,
                base_duration=tuple(durations[index]),
                base_skill=tuple(skills[index]),
            )
            for index, technician in enumerate(technicians)
        ),
    )


class ConflictConsequenceEnv:
    """Parallel-maintenance kernel with persistent requests and PM windows."""

    def __init__(
        self, config: ParallelMaintenanceConfig, cell: ConsequenceCell
    ) -> None:
        self.config = config
        self.cell = cell
        self.num_agents = len(config.machines)
        self.technician_count = len(config.technicians)
        self.action_count = self.technician_count + 1

    @property
    def failure_type_count(self) -> int:
        return max(machine.failure_type for machine in self.config.machines) + 1

    def reset(self, *, seed: int) -> dict[str, Any]:
        self.seed = int(seed)
        self.step_index = 0
        self.mode = np.full(self.num_agents, WORKING, dtype=np.int8)
        self.age = np.zeros(self.num_agents, dtype=np.float64)
        self.wait = np.zeros(self.num_agents, dtype=np.float64)
        self.pending_kind = np.zeros(self.num_agents, dtype=np.int8)
        self.pending_preferred = np.full(self.num_agents, -1, dtype=np.int16)
        self.window_remaining = np.zeros(self.num_agents, dtype=np.float64)
        self.overdue = np.zeros(self.num_agents, dtype=np.bool_)
        self.machine_technician = np.full(self.num_agents, -1, dtype=np.int16)
        self.active_kind = np.zeros(self.num_agents, dtype=np.int8)
        self.repair_will_succeed = np.zeros(self.num_agents, dtype=np.bool_)
        self.technician_machine = np.full(self.technician_count, -1, dtype=np.int16)
        self.technician_remaining = np.zeros(self.technician_count, dtype=np.float64)
        self.technician_absent = np.zeros(self.technician_count, dtype=np.bool_)
        self.experience = np.zeros(
            (self.technician_count, self.failure_type_count), dtype=np.float64
        )
        self.technician_busy_time = np.zeros(self.technician_count, dtype=np.float64)
        self.maintenance_attempts = np.zeros(
            (self.num_agents, self.technician_count), dtype=np.int32
        )
        self.metrics: dict[str, float | int] = {
            "objective": 0.0,
            "episode_return": 0.0,
            "production": 0.0,
            "downtime": 0.0,
            "failures": 0,
            "preventive": 0,
            "corrective": 0,
            "waiting": 0.0,
            "proposal_conflicts": 0,
            "conflict_steps": 0,
            "rejected_requests": 0,
            "committed_wait_steps": 0,
            "missed_windows": 0,
            "overdue_steps": 0,
            "rework": 0,
            "early_pm": 0.0,
            "invalid_executions": 0,
            "duplicate_machine_assignments": 0,
            "duplicate_technician_assignments": 0,
        }
        self.done = False
        self.last_decision: dict[str, Any] = {}
        return self.info()

    def _rng(self, channel: int, *keys: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, channel, *map(int, keys)])
        )

    def technician_available(self, technician: int) -> bool:
        return bool(
            self.technician_machine[technician] < 0
            and not self.technician_absent[technician]
        )

    def action_masks(self) -> np.ndarray:
        masks = np.zeros((self.num_agents, self.action_count), dtype=np.bool_)
        masks[:, 0] = True
        for machine in range(self.num_agents):
            if self.mode[machine] == MAINTENANCE:
                continue
            for technician in range(self.technician_count):
                masks[machine, technician + 1] = self.technician_available(
                    technician
                )
        return masks

    def base_failure_probability(self, machine: int) -> float:
        spec = self.config.machines[machine]
        increment = spec.age_rate * spec.load * self.config.time_step
        age = self.age[machine]
        exponent = (
            ((age + increment) / spec.weibull_scale) ** spec.weibull_shape
            - (age / spec.weibull_scale) ** spec.weibull_shape
        )
        return float(1.0 - math.exp(-max(0.0, exponent)))

    def failure_probability(self, machine: int) -> float:
        probability = self.base_failure_probability(machine)
        multiplier = 3.0 if self.overdue[machine] else 1.0
        return float(1.0 - (1.0 - probability) ** multiplier)

    def request_kind(self, machine: int) -> int:
        if self.pending_kind[machine]:
            return int(self.pending_kind[machine])
        return 2 if self.mode[machine] == FAILED else 1

    def expected_duration(self, machine: int, technician: int) -> float:
        failure_type = self.config.machines[machine].failure_type
        spec = self.config.technicians[technician]
        duration = spec.base_duration[failure_type] / (
            1.0
            + self.config.maintenance.experience_duration_gain
            * self.experience[technician, failure_type]
        )
        if self.request_kind(machine) == 1:
            duration *= self.config.maintenance.pm_duration_factor
        return float(duration)

    def success_probability(self, machine: int, technician: int) -> float:
        failure_type = self.config.machines[machine].failure_type
        spec = self.config.technicians[technician]
        parameters = self.config.maintenance
        logit = (
            parameters.success_intercept
            + parameters.success_skill_weight * spec.base_skill[failure_type]
            + parameters.success_experience_weight
            * self.experience[technician, failure_type]
        )
        return float(1.0 / (1.0 + math.exp(-logit)))

    def candidates(self, *, preventive: bool) -> list[int]:
        candidates = []
        for machine in range(self.num_agents):
            if self.mode[machine] == MAINTENANCE:
                continue
            if self.pending_kind[machine] or self.mode[machine] == FAILED or (
                preventive
                and self.mode[machine] == WORKING
                and self.failure_probability(machine)
                >= self.config.maintenance.pm_risk_threshold
            ):
                candidates.append(machine)
        candidates.sort(key=self.priority_key)
        return candidates

    def priority_key(self, machine: int) -> tuple[float, ...]:
        kind = self.request_kind(machine)
        return (
            0.0 if kind == 2 else 1.0,
            0.0 if self.overdue[machine] else 1.0,
            -self.wait[machine],
            -self.failure_probability(machine),
            float(machine),
        )

    def _resolve(
        self, actions: Sequence[int]
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
        masks = self.action_masks()
        proposals: dict[int, list[int]] = {}
        for machine, raw_action in enumerate(actions):
            action = int(raw_action)
            if not 0 <= action < self.action_count or not masks[machine, action]:
                raise ValueError("Masked or out-of-range action was proposed")
            if action:
                proposals.setdefault(action - 1, []).append(machine)
        accepted: list[tuple[int, int]] = []
        rejected: list[tuple[int, int]] = []
        conflicts = 0
        for technician, machines in proposals.items():
            ordered = sorted(machines, key=self.priority_key)
            accepted.append((ordered[0], technician))
            rejected.extend((machine, technician) for machine in ordered[1:])
            conflicts += max(0, len(ordered) - 1)
        return accepted, rejected, conflicts

    def _register_rejected(self, machine: int, technician: int) -> None:
        if not self.pending_kind[machine]:
            kind = 2 if self.mode[machine] == FAILED else 1
            self.pending_kind[machine] = kind
            if kind == 1 and self.cell.maintenance_window:
                self.window_remaining[machine] = 2.0
        self.pending_preferred[machine] = technician
        if self.cell.committed_request and self.mode[machine] == WORKING:
            self.mode[machine] = WAITING

    def _start(self, machine: int, technician: int) -> dict[str, float]:
        if not self.technician_available(technician) or self.mode[machine] == MAINTENANCE:
            self.metrics["invalid_executions"] += 1
            raise AssertionError("Infeasible maintenance start")
        kind = self.request_kind(machine)
        expected = self.expected_duration(machine, technician)
        failure_type = self.config.machines[machine].failure_type
        sigma = self.config.technicians[technician].duration_sigma[failure_type]
        attempt = int(self.maintenance_attempts[machine, technician])
        rng = self._rng(1, machine, technician, attempt)
        sampled = expected if sigma == 0 else float(
            rng.lognormal(math.log(expected) - 0.5 * sigma * sigma, sigma)
        )
        duration = max(self.config.time_step, sampled)
        success = bool(rng.random() < self.success_probability(machine, technician))
        self.maintenance_attempts[machine, technician] += 1
        self.mode[machine] = MAINTENANCE
        self.machine_technician[machine] = technician
        self.active_kind[machine] = kind
        self.repair_will_succeed[machine] = success
        self.technician_machine[technician] = machine
        self.technician_remaining[technician] = duration
        self.pending_kind[machine] = 0
        self.pending_preferred[machine] = -1
        self.window_remaining[machine] = 0.0
        self.overdue[machine] = False
        early_pm = 0.0
        if kind == 1:
            self.metrics["preventive"] += 1
            early_pm = max(
                0.0,
                self.config.maintenance.pm_risk_threshold
                - self.failure_probability(machine),
            )
            self.metrics["early_pm"] += early_pm
        else:
            self.metrics["corrective"] += 1
        return {"kind": float(kind), "duration": duration, "early_pm": early_pm}

    def step(self, actions: Sequence[int]) -> tuple[float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done; call reset")
        if len(actions) != self.num_agents:
            raise ValueError("Joint action requires one action per machine")
        accepted, rejected, conflicts = self._resolve(actions)
        incremental = self.config.costs.conflict * conflicts
        for machine, technician in rejected:
            self._register_rejected(machine, technician)
        starts: list[dict[str, Any]] = []
        for machine, technician in accepted:
            details = self._start(machine, technician)
            starts.append({"machine": machine, "technician": technician, **details})
            if int(details["kind"]) == 1:
                incremental += self.config.costs.preventive
                incremental += self.config.costs.early_pm * details["early_pm"]
            else:
                incremental += self.config.costs.corrective

        production = 0.0
        downtime = 0.0
        waiting = 0.0
        failures = 0
        reworks = 0
        for machine, spec in enumerate(self.config.machines):
            if self.mode[machine] == WORKING:
                production += spec.production_rate * spec.load * self.config.time_step
                probability = self.failure_probability(machine)
                self.age[machine] += spec.age_rate * spec.load * self.config.time_step
                if self._rng(2, self.step_index, machine).random() < probability:
                    self.mode[machine] = FAILED
                    self.pending_kind[machine] = 2
                    self.window_remaining[machine] = 0.0
                    self.overdue[machine] = False
                    self.wait[machine] = 0.0
                    failures += 1
            else:
                downtime += self.config.time_step
                if self.mode[machine] in {FAILED, WAITING}:
                    self.wait[machine] += self.config.time_step
                    waiting += self.config.time_step
                    if self.mode[machine] == WAITING:
                        self.metrics["committed_wait_steps"] += 1

        for machine in range(self.num_agents):
            if (
                self.pending_kind[machine] == 1
                and self.cell.maintenance_window
                and not self.overdue[machine]
            ):
                self.window_remaining[machine] -= self.config.time_step
                if self.window_remaining[machine] <= 0:
                    self.overdue[machine] = True
                    self.metrics["missed_windows"] += 1
            if self.overdue[machine]:
                self.metrics["overdue_steps"] += 1

        for technician in range(self.technician_count):
            machine = int(self.technician_machine[technician])
            if machine < 0:
                continue
            worked = min(self.config.time_step, self.technician_remaining[technician])
            self.technician_busy_time[technician] += worked
            self.technician_remaining[technician] -= self.config.time_step
            if self.technician_remaining[technician] > 1e-12:
                continue
            failure_type = self.config.machines[machine].failure_type
            kind = int(self.active_kind[machine])
            if self.repair_will_succeed[machine]:
                restoration = (
                    self.config.maintenance.pm_restoration
                    if kind == 1
                    else self.config.maintenance.cm_restoration
                )
                self.age[machine] *= max(0.0, 1.0 - restoration)
                self.mode[machine] = WORKING
                current = self.experience[technician, failure_type]
                rate = self.config.technicians[technician].learning_rate
                self.experience[technician, failure_type] = current + rate * (1 - current)
                self.wait[machine] = 0.0
            else:
                self.mode[machine] = FAILED
                self.pending_kind[machine] = 2
                reworks += 1
            self.machine_technician[machine] = -1
            self.active_kind[machine] = 0
            self.technician_machine[technician] = -1
            self.technician_remaining[technician] = 0.0

        for technician, spec in enumerate(self.config.technicians):
            if self.technician_machine[technician] < 0:
                self.technician_absent[technician] = bool(
                    self._rng(3, self.step_index + 1, technician).random()
                    < spec.absence_probability
                )
            else:
                self.technician_absent[technician] = False

        incremental += self.config.costs.downtime * downtime
        incremental += self.config.costs.failure * failures
        incremental += self.config.costs.waiting * waiting
        incremental += self.config.costs.rework * reworks
        incremental -= self.config.costs.production_credit * production
        reward = -float(incremental)
        updates = {
            "objective": incremental,
            "episode_return": reward,
            "production": production,
            "downtime": downtime,
            "failures": failures,
            "waiting": waiting,
            "proposal_conflicts": conflicts,
            "conflict_steps": int(conflicts > 0),
            "rejected_requests": len(rejected),
            "rework": reworks,
        }
        for key, value in updates.items():
            self.metrics[key] += value
        self.last_decision = {
            "step": self.step_index,
            "actions": list(map(int, actions)),
            "accepted": starts,
            "rejected": [list(pair) for pair in rejected],
            "proposal_conflicts": conflicts,
            "incremental_cost": incremental,
            "reward": reward,
        }
        self.step_index += 1
        self.done = self.step_index >= self.config.horizon
        self._audit()
        return reward, self.done, self.info()

    def _audit(self) -> None:
        machines = [int(value) for value in self.technician_machine if value >= 0]
        technicians = [int(value) for value in self.machine_technician if value >= 0]
        if len(machines) != len(set(machines)):
            self.metrics["duplicate_machine_assignments"] += 1
            raise AssertionError("Duplicate machine assignment")
        if len(technicians) != len(set(technicians)):
            self.metrics["duplicate_technician_assignments"] += 1
            raise AssertionError("Duplicate technician assignment")
        for machine, technician in enumerate(self.machine_technician):
            if technician >= 0 and self.technician_machine[technician] != machine:
                raise AssertionError("Non-reciprocal assignment")

    def info(self) -> dict[str, Any]:
        elapsed = max(self.config.time_step, self.step_index * self.config.time_step)
        utilization = self.technician_busy_time / elapsed
        return {
            **self.metrics,
            "technician_utilization": utilization.tolist(),
            "workload_imbalance": float(np.var(utilization)),
            "return_objective_error": abs(
                float(self.metrics["episode_return"] + self.metrics["objective"])
            ),
        }


def policy_actions(name: str, env: ConflictConsequenceEnv) -> np.ndarray:
    preventive = name != "reactive_fibt"
    candidates = env.candidates(preventive=preventive)
    available = [
        technician
        for technician in range(env.technician_count)
        if env.technician_available(technician)
    ]
    actions = np.zeros(env.num_agents, dtype=np.int64)
    if name == "independent_preventive":
        for machine in candidates:
            if available:
                actions[machine] = min(
                    available,
                    key=lambda technician: (
                        env.expected_duration(machine, technician),
                        technician,
                    ),
                ) + 1
        return actions
    if name == "coordinated_matching":
        best_score = -math.inf
        best: tuple[int, ...] = tuple(0 for _ in candidates)
        for assignment in itertools.product((0, *[item + 1 for item in available]), repeat=len(candidates)):
            used = [value for value in assignment if value]
            if len(used) != len(set(used)):
                continue
            score = 0.0
            for machine, action in zip(candidates, assignment, strict=True):
                if not action:
                    continue
                technician = action - 1
                kind = env.request_kind(machine)
                specialist_duration = min(
                    env.expected_duration(machine, item)
                    for item in range(env.technician_count)
                )
                substitution_ratio = (
                    env.expected_duration(machine, technician)
                    / specialist_duration
                )
                if kind == 1 and substitution_ratio > 1.75:
                    score -= 200.0
                    continue
                score += 1000.0 if kind == 2 else 100.0
                score += 50.0 if env.overdue[machine] else 0.0
                score += 10.0 * env.wait[machine]
                score -= env.expected_duration(machine, technician)
            if score > best_score:
                best_score, best = score, assignment
        for machine, action in zip(candidates, best, strict=True):
            actions[machine] = action
        return actions
    if name not in {"reactive_fibt", "coordinated_preventive"}:
        raise ValueError(f"Unknown policy: {name}")
    for machine in candidates:
        if not available:
            break
        technician = min(
            available,
            key=lambda item: (env.expected_duration(machine, item), item),
        )
        actions[machine] = technician + 1
        available.remove(technician)
    return actions


def evaluate_episode(
    config: ParallelMaintenanceConfig,
    cell: ConsequenceCell,
    policy: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = ConflictConsequenceEnv(config, cell)
    env.reset(seed=seed)
    decisions: list[dict[str, Any]] = []
    done = False
    while not done:
        actions = policy_actions(policy, env)
        _, done, info = env.step(actions)
        decisions.append(
            {
                "cell_id": cell.cell_id,
                "policy": policy,
                "seed": seed,
                **env.last_decision,
                "actions": json.dumps(env.last_decision["actions"]),
                "accepted": json.dumps(env.last_decision["accepted"]),
                "rejected": json.dumps(env.last_decision["rejected"]),
            }
        )
    utilization = info["technician_utilization"]
    episode = {
        "cell_id": cell.cell_id,
        **asdict(cell),
        "policy": policy,
        "seed": seed,
        **{
            key: info[key]
            for key in (
                "objective",
                "episode_return",
                "production",
                "downtime",
                "failures",
                "preventive",
                "corrective",
                "waiting",
                "proposal_conflicts",
                "conflict_steps",
                "rejected_requests",
                "committed_wait_steps",
                "missed_windows",
                "overdue_steps",
                "rework",
                "early_pm",
                "workload_imbalance",
                "return_objective_error",
                "invalid_executions",
                "duplicate_machine_assignments",
                "duplicate_technician_assignments",
            )
        },
        "technician_0_utilization": utilization[0],
        "technician_1_utilization": utilization[1],
    }
    return episode, decisions


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _critical_value(count: int) -> float:
    if count == 3:
        return 4.303
    if count == 50:
        return 2.009
    return 1.96


def _paired_interval(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        half = 0.0
    else:
        half = _critical_value(len(values)) * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def summarize(
    episodes: list[dict[str, Any]],
    *,
    cells: tuple[ConsequenceCell, ...],
    seeds: tuple[int, ...],
    profile: str,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    qualifying: list[ConsequenceCell] = []
    per_seed_gap: dict[tuple[str, int], float] = {}
    for cell in cells:
        selected = [row for row in episodes if row["cell_id"] == cell.cell_id]
        groups = {
            policy: [row for row in selected if row["policy"] == policy]
            for policy in POLICIES
        }
        coordinated = {int(row["seed"]): row for row in groups["coordinated_matching"]}
        independent = {int(row["seed"]): row for row in groups["independent_preventive"]}
        reactive = {int(row["seed"]): row for row in groups["reactive_fibt"]}
        gaps = [
            float(independent[seed]["objective"])
            - float(coordinated[seed]["objective"])
            for seed in seeds
        ]
        for seed, gap in zip(seeds, gaps, strict=True):
            per_seed_gap[(cell.cell_id, seed)] = gap
        interval = _paired_interval(gaps)
        coordinated_objective = _mean(groups["coordinated_matching"], "objective")
        relative_gap = interval["mean"] / max(abs(coordinated_objective), 1e-9)
        preventive_delta = statistics.fmean(
            float(coordinated[seed]["objective"])
            - float(reactive[seed]["objective"])
            for seed in seeds
        )
        reactive_objective = _mean(groups["reactive_fibt"], "objective")
        preventive_improvement = -preventive_delta / max(abs(reactive_objective), 1e-9)
        conflict_incidence = _mean(groups["independent_preventive"], "conflict_steps") / 168.0
        coordinated_conflicts = sum(
            float(row["proposal_conflicts"])
            for policy in ("coordinated_preventive", "coordinated_matching")
            for row in groups[policy]
        )
        checks = {
            "conflict_incidence_in_3_to_10_percent": 0.03 <= conflict_incidence <= 0.10,
            "relative_objective_gap_at_least_5_percent": relative_gap >= 0.05,
            "paired_95_percent_lower_bound_above_zero": interval["lower"] > 0,
            "independent_worse_on_70_percent_of_seeds": sum(gap > 0 for gap in gaps)
            / len(gaps)
            >= 0.70,
            "preventive_improves_reactive_by_5_percent": preventive_improvement >= 0.05,
            "coordinated_policies_have_zero_conflicts": coordinated_conflicts == 0,
        }
        if all(checks.values()):
            qualifying.append(cell)
        metrics = (
            "objective",
            "production",
            "downtime",
            "failures",
            "preventive",
            "corrective",
            "waiting",
            "proposal_conflicts",
            "conflict_steps",
            "rejected_requests",
            "committed_wait_steps",
            "missed_windows",
            "overdue_steps",
            "rework",
            "workload_imbalance",
        )
        summaries[cell.cell_id] = {
            "factors": asdict(cell),
            "independent_minus_coordinated": {
                **interval,
                "median": statistics.median(gaps),
                "independent_worse_rate": sum(gap > 0 for gap in gaps) / len(gaps),
                "relative_gap": relative_gap,
            },
            "coordinated_matching_minus_reactive_mean": preventive_delta,
            "preventive_relative_improvement": preventive_improvement,
            "independent_conflict_step_incidence": conflict_incidence,
            "policy_metrics": {
                policy: {metric: _mean(rows, metric) for metric in metrics}
                for policy, rows in groups.items()
            },
            "promotion_checks": checks,
            "qualifies": all(checks.values()),
        }

    interactions: dict[str, Any] = {}
    for factor in (
        "committed_request",
        "maintenance_window",
        "strong_substitution",
    ):
        values: list[float] = []
        other_factors = [
            name
            for name in (
                "committed_request",
                "maintenance_window",
                "strong_substitution",
            )
            if name != factor
        ]
        for seed in seeds:
            seed_differences: list[float] = []
            for combination in itertools.product((False, True), repeat=2):
                low = next(
                    cell
                    for cell in cells
                    if not getattr(cell, factor)
                    and all(
                        getattr(cell, name) == value
                        for name, value in zip(
                            other_factors, combination, strict=True
                        )
                    )
                )
                high = replace(low, **{factor: True})
                seed_differences.append(
                    per_seed_gap[(high.cell_id, seed)]
                    - per_seed_gap[(low.cell_id, seed)]
                )
            values.append(statistics.fmean(seed_differences))
        interactions[factor] = _paired_interval(values)

    selected_cell = None
    if qualifying:
        chosen = min(
            qualifying,
            key=lambda cell: (
                sum(asdict(cell).values()),
                -summaries[cell.cell_id]["independent_minus_coordinated"]["lower"],
                cell.cell_id,
            ),
        )
        selected_cell = chosen.cell_id
    expected = len(cells) * len(POLICIES) * len(seeds)
    audits = {
        "expected_episodes_completed": len(episodes) == expected,
        "reward_objective_identity": all(
            float(row["return_objective_error"]) <= 1e-6 for row in episodes
        ),
        "no_invalid_executions": all(int(row["invalid_executions"]) == 0 for row in episodes),
        "no_duplicate_machine_assignments": all(
            int(row["duplicate_machine_assignments"]) == 0 for row in episodes
        ),
        "no_duplicate_technician_assignments": all(
            int(row["duplicate_technician_assignments"]) == 0 for row in episodes
        ),
        "sealed_test_panel_remains_closed": not any(
            int(row["seed"]) in SEALED_TEST_SEEDS for row in episodes
        ),
    }
    return {
        "purpose": "Technician-conflict consequence diagnostic, not an algorithm claim.",
        "profile": profile,
        "cell_summaries": summaries,
        "factor_interactions": interactions,
        "qualifying_cells": sorted(cell.cell_id for cell in qualifying),
        "selected_candidate_cell": selected_cell,
        "hypotheses": {
            "H1_commitment_increases_conflict_gap": interactions["committed_request"]["lower"] > 0,
            "H2_window_increases_conflict_gap": interactions["maintenance_window"]["lower"] > 0,
            "H3_substitution_increases_conflict_gap": interactions["strong_substitution"]["lower"] > 0,
            "H4_at_least_one_cell_has_material_conflict_loss": bool(qualifying),
        },
        "audits": audits,
        "gate": {"passed": all(audits.values())},
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = ParallelMaintenanceConfig.from_json(args.config)
    cells = cells_for_profile(args.profile)
    seeds = tuple(parse_seeds(args.seeds)) if args.seeds else seeds_for_profile(args.profile)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "conflict_consequence_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "purpose": "technician_conflict_consequence_diagnostic",
        "profile": args.profile,
        "git_revision": _git_revision(),
        "started_at": datetime.now(UTC).isoformat(),
        "config": str(Path(args.config).resolve()),
        "cells": [asdict(cell) | {"cell_id": cell.cell_id} for cell in cells],
        "policies": list(POLICIES),
        "evaluation_seeds": list(seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    parent = selected_parent_config(base)
    (output_dir / "resolved_base_config.json").write_text(
        json.dumps(parent.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "cells.json").write_text(
        json.dumps(manifest["cells"], indent=2, sort_keys=True) + "\n"
    )
    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(cells) * len(POLICIES) * len(seeds),
        desc="Conflict consequence evaluation",
        unit="episode",
    )
    try:
        for cell in cells:
            config = build_cell_config(base, cell)
            for policy in POLICIES:
                for seed in seeds:
                    episode, trace = evaluate_episode(config, cell, policy, seed)
                    episodes.append(episode)
                    decisions.extend(trace)
                    progress.update(1)
                _write_csv(episodes, output_dir / "episodes.partial.csv")
        progress.close()
        summary = summarize(episodes, cells=cells, seeds=seeds, profile=args.profile)
        _write_csv(episodes, output_dir / "episodes.csv")
        _write_csv(decisions, output_dir / "decisions.csv")
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        manifest.update(
            {
                "status": "COMPLETED" if summary["gate"]["passed"] else "FAILED",
                "finished_at": datetime.now(UTC).isoformat(),
                "episode_count": len(episodes),
                "decision_count": len(decisions),
                "audit_gate": summary["gate"],
                "selected_candidate_cell": summary["selected_candidate_cell"],
                "outputs": sorted(
                    str(path.relative_to(output_dir))
                    for path in output_dir.rglob("*")
                    if path.is_file()
                ),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if not summary["gate"]["passed"]:
            raise RuntimeError("Conflict consequence audit gate failed")
        return summary
    except Exception:
        progress.close()
        manifest.update({"status": "FAILED", "finished_at": datetime.now(UTC).isoformat()})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/parallel_maintenance.json")
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
