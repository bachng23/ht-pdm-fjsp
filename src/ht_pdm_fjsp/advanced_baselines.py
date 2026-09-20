"""Masked heuristic, CP-SAT, and rolling-horizon Gymnasium baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from ortools.sat.python import cp_model

from ht_pdm_fjsp.gym_env import ActionDescriptor, HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.simulator import SimulationResult


class GymPolicy(Protocol):
    name: str

    def reset(self, env: HTPdmFjspEnv) -> None: ...

    def action(self, env: HTPdmFjspEnv) -> int: ...


def _feasible(env: HTPdmFjspEnv, kind: str | None = None) -> list[int]:
    mask = env.action_mask()
    return [
        index
        for index, descriptor in enumerate(env.actions)
        if mask[index] and (kind is None or descriptor.kind == kind)
    ]


def _fastest_maintenance(env: HTPdmFjspEnv, actions: list[int]) -> int:
    return min(
        actions,
        key=lambda index: (
            env.technician_specs[str(env.actions[index].technician_id)].duration(
                str(env.actions[index].machine_id), env.actions[index].kind
            ),
            -env.technician_specs[str(env.actions[index].technician_id)].restoration(
                str(env.actions[index].machine_id), env.actions[index].kind
            ),
            index,
        ),
    )


@dataclass
class MaskedSPTPolicy:
    name: str = field(default="masked_spt", init=False)

    def reset(self, env: HTPdmFjspEnv) -> None:
        del env

    def action(self, env: HTPdmFjspEnv) -> int:
        corrective = _feasible(env, "corrective")
        if corrective:
            return _fastest_maintenance(env, corrective)
        production = _feasible(env, "production")
        if production:
            return min(
                production,
                key=lambda index: (
                    env._alternative(env.actions[index]).processing_time,
                    env.job_specs[str(env.actions[index].job_id)].due_date,
                    index,
                ),
            )
        advance = _feasible(env, "advance")
        if advance:
            return advance[0]
        raise RuntimeError("No feasible action in a nonterminal state.")


@dataclass
class HealthThresholdPolicy(MaskedSPTPolicy):
    probability_threshold: float = 0.18
    name: str = field(default="health_threshold", init=False)

    def action(self, env: HTPdmFjspEnv) -> int:
        corrective = _feasible(env, "corrective")
        if corrective:
            return _fastest_maintenance(env, corrective)
        production = _feasible(env, "production")
        best_by_machine: dict[str, int] = {}
        for index in production:
            machine_id = str(env.actions[index].machine_id)
            incumbent = best_by_machine.get(machine_id)
            if incumbent is None or env._alternative(
                env.actions[index]
            ).processing_time < env._alternative(env.actions[incumbent]).processing_time:
                best_by_machine[machine_id] = index
        preventive = _feasible(env, "preventive")
        candidates: list[tuple[float, int]] = []
        for machine_id, production_index in best_by_machine.items():
            risk = env.failure_probability(env.actions[production_index])
            machine_pm = [
                index
                for index in preventive
                if env.actions[index].machine_id == machine_id
            ]
            if risk >= self.probability_threshold and machine_pm:
                candidates.append((risk, _fastest_maintenance(env, machine_pm)))
        if candidates:
            return max(candidates, key=lambda item: (item[0], -item[1]))[1]
        return super().action(env)


@dataclass
class JointRiskGreedyPolicy(MaskedSPTPolicy):
    """Compare technician-specific PM burden against expected failure loss."""

    name: str = field(default="joint_risk_greedy", init=False)

    def action(self, env: HTPdmFjspEnv) -> int:
        corrective = _feasible(env, "corrective")
        if corrective:
            return _fastest_maintenance(env, corrective)
        production = _feasible(env, "production")
        preventive = _feasible(env, "preventive")
        best_production_by_machine: dict[str, int] = {}
        for index in production:
            machine_id = str(env.actions[index].machine_id)
            incumbent = best_production_by_machine.get(machine_id)
            if incumbent is None:
                best_production_by_machine[machine_id] = index
                continue
            current = env._alternative(env.actions[index])
            previous = env._alternative(env.actions[incumbent])
            if current.processing_time < previous.processing_time:
                best_production_by_machine[machine_id] = index

        pm_scores: list[tuple[float, int]] = []
        for index in preventive:
            descriptor = env.actions[index]
            machine_id = str(descriptor.machine_id)
            if machine_id not in best_production_by_machine:
                continue
            production_index = best_production_by_machine[machine_id]
            alternative = env._alternative(env.actions[production_index])
            risk = env.failure_probability(env.actions[production_index])
            technician = env.technician_specs[str(descriptor.technician_id)]
            duration = technician.duration(machine_id, "preventive")
            restoration = technician.restoration(machine_id, "preventive")
            corrective_durations = [
                tech.duration(machine_id, "corrective")
                for tech in env.config.technicians
                if machine_id in tech.eligible_machines
            ]
            consequence = (
                env.config.costs.corrective
                + env.config.costs.downtime * min(corrective_durations)
                + alternative.processing_time
            )
            burden = (
                env.config.costs.preventive
                + env.config.costs.downtime * duration
            )
            benefit = risk * consequence * restoration - burden
            pm_scores.append((benefit, index))
        if pm_scores and max(pm_scores)[0] > 0.0:
            return max(pm_scores, key=lambda item: (item[0], -item[1]))[1]

        if production:
            return min(
                production,
                key=lambda index: (
                    env._alternative(env.actions[index]).processing_time
                    + env.failure_probability(env.actions[index])
                    * env.config.costs.corrective,
                    env.job_specs[str(env.actions[index].job_id)].due_date,
                    index,
                ),
            )
        advance = _feasible(env, "advance")
        if advance:
            return advance[0]
        raise RuntimeError("No feasible action in a nonterminal state.")


def solve_deterministic_cp_sat(
    config: BenchmarkConfig, *, time_limit_seconds: float = 5.0
) -> dict[tuple[str, int], tuple[str, float]]:
    """Solve deterministic FJSP makespan+tardiness and return machine/start plan."""

    scale = 100
    horizon = int(
        scale
        * sum(
            max(alt.processing_time for alt in operation.alternatives)
            for job in config.jobs
            for operation in job.operations
        )
    )
    model = cp_model.CpModel()
    operation_vars: dict[tuple[str, int], tuple[Any, Any]] = {}
    presence: dict[tuple[str, int, str], Any] = {}
    machine_intervals: dict[str, list[Any]] = {
        machine.machine_id: [] for machine in config.machines
    }
    for job in config.jobs:
        for operation_index, operation in enumerate(job.operations):
            key = (job.job_id, operation_index)
            start = model.new_int_var(0, horizon, f"s_{job.job_id}_{operation_index}")
            end = model.new_int_var(0, horizon, f"e_{job.job_id}_{operation_index}")
            operation_vars[key] = (start, end)
            selected = []
            for alternative in operation.alternatives:
                chosen = model.new_bool_var(
                    f"x_{job.job_id}_{operation_index}_{alternative.machine_id}"
                )
                duration = int(round(scale * alternative.processing_time))
                interval = model.new_optional_interval_var(
                    start,
                    duration,
                    end,
                    chosen,
                    f"i_{job.job_id}_{operation_index}_{alternative.machine_id}",
                )
                presence[(job.job_id, operation_index, alternative.machine_id)] = chosen
                machine_intervals[alternative.machine_id].append(interval)
                selected.append(chosen)
            model.add_exactly_one(selected)
        for operation_index in range(len(job.operations) - 1):
            model.add(
                operation_vars[(job.job_id, operation_index)][1]
                <= operation_vars[(job.job_id, operation_index + 1)][0]
            )
    for intervals in machine_intervals.values():
        model.add_no_overlap(intervals)
    final_ends = [
        operation_vars[(job.job_id, len(job.operations) - 1)][1]
        for job in config.jobs
    ]
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, final_ends)
    tardiness_vars = []
    for job, final_end in zip(config.jobs, final_ends):
        tardiness = model.new_int_var(0, horizon, f"tard_{job.job_id}")
        model.add(tardiness >= final_end - int(round(scale * job.due_date)))
        tardiness_vars.append(tardiness)
    model.minimize(makespan + sum(tardiness_vars))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise RuntimeError("CP-SAT did not find a feasible deterministic FJSP plan.")
    plan: dict[tuple[str, int], tuple[str, float]] = {}
    for job in config.jobs:
        for operation_index, operation in enumerate(job.operations):
            machine_id = next(
                alternative.machine_id
                for alternative in operation.alternatives
                if solver.value(
                    presence[(job.job_id, operation_index, alternative.machine_id)]
                )
            )
            start = solver.value(operation_vars[(job.job_id, operation_index)][0])
            plan[(job.job_id, operation_index)] = (machine_id, start / scale)
    return plan


@dataclass
class CPSATReactivePolicy(HealthThresholdPolicy):
    time_limit_seconds: float = 5.0
    name: str = field(default="cp_sat_reactive", init=False)
    plan: dict[tuple[str, int], tuple[str, float]] = field(default_factory=dict)

    def reset(self, env: HTPdmFjspEnv) -> None:
        self.plan = solve_deterministic_cp_sat(
            env.config, time_limit_seconds=self.time_limit_seconds
        )

    def action(self, env: HTPdmFjspEnv) -> int:
        corrective = _feasible(env, "corrective")
        if corrective:
            return _fastest_maintenance(env, corrective)
        production = _feasible(env, "production")
        planned = [
            index
            for index in production
            if self.plan[
                (str(env.actions[index].job_id), int(env.actions[index].operation_index))
            ][0]
            == env.actions[index].machine_id
        ]
        candidates = planned or production
        if candidates:
            selected = min(
                candidates,
                key=lambda index: (
                    self.plan[
                        (
                            str(env.actions[index].job_id),
                            int(env.actions[index].operation_index),
                        )
                    ][1],
                    index,
                ),
            )
            machine_id = str(env.actions[selected].machine_id)
            risk = env.failure_probability(env.actions[selected])
            machine_pm = [
                index
                for index in _feasible(env, "preventive")
                if env.actions[index].machine_id == machine_id
            ]
            if risk >= self.probability_threshold and machine_pm:
                return _fastest_maintenance(env, machine_pm)
            return selected
        advance = _feasible(env, "advance")
        if advance:
            return advance[0]
        raise RuntimeError("No feasible action in a nonterminal state.")


@dataclass
class RollingHorizonPolicy:
    depth: int = 2
    branch_width: int = 7
    scenario_count: int = 4
    planning_seed_base: int = 1_000_003
    improvement_margin: float = 20.0
    name: str = field(default="rolling_horizon", init=False)

    def __post_init__(self) -> None:
        if self.depth < 1 or self.branch_width < 1 or self.scenario_count < 1:
            raise ValueError("depth, branch_width, and scenario_count must be positive.")
        if self.improvement_margin < 0:
            raise ValueError("improvement_margin must be nonnegative.")

    def reset(self, env: HTPdmFjspEnv) -> None:
        del env

    def _leaf_value(self, env: HTPdmFjspEnv) -> float:
        # Roll out a competent base policy instead of relying on a myopic
        # handcrafted terminal score. This is a standard policy-improvement
        # construction: search a few decisions, then estimate the tail with the
        # joint risk-aware heuristic under the same planning scenario.
        clone = env.clone()
        base = JointRiskGreedyPolicy()
        value = 0.0
        while not clone._done:
            _, reward, _, _, _ = clone.step(base.action(clone))
            value += reward
        return value

    def _branches(self, env: HTPdmFjspEnv) -> list[int]:
        actions = _feasible(env)
        corrective = [i for i in actions if env.actions[i].kind == "corrective"]
        if corrective:
            # A failed machine accrues downtime until repair starts. Branching on
            # production/advance here creates dominated intentional waiting.
            return corrective[: self.branch_width]
        production = sorted(
            [i for i in actions if env.actions[i].kind == "production"],
            key=lambda i: env._alternative(env.actions[i]).processing_time,
        )
        preventive = [i for i in actions if env.actions[i].kind == "preventive"]
        risk_policy = JointRiskGreedyPolicy()
        recommended = risk_policy.action(env)
        preventive = sorted(
            [i for i in preventive if i == recommended],
            key=lambda i: env.technician_specs[
                str(env.actions[i].technician_id)
            ].duration(str(env.actions[i].machine_id), "preventive"),
        )
        advance = [i for i in actions if env.actions[i].kind == "advance"]
        # Enforce a non-delay schedule: do not advance time while a production
        # assignment is possible. A risk-justified PM remains an alternative.
        if production:
            return (production + preventive)[: self.branch_width]
        if preventive:
            return (preventive + advance)[: self.branch_width]
        return advance[:1]

    def _search(self, env: HTPdmFjspEnv, depth: int) -> float:
        if env._done:
            return 0.0
        if depth <= 0:
            return self._leaf_value(env)
        values = []
        for action in self._branches(env):
            clone = env.clone()
            _, reward, terminated, truncated, _ = clone.step(action)
            continuation = 0.0 if terminated or truncated else self._search(clone, depth - 1)
            values.append(reward + continuation)
        return max(values) if values else self._leaf_value(env)

    def _scenario_value(
        self, env: HTPdmFjspEnv, action: int, scenario_index: int
    ) -> float:
        clone = env.clone()
        # Deliberately decouple planning shocks from the evaluated episode seed.
        # All candidate actions share the same scenario seeds (CRN), avoiding
        # access to the episode's unrevealed future failures.
        clone.root_seed = self.planning_seed_base + scenario_index
        _, reward, terminated, truncated, _ = clone.step(action)
        continuation = (
            0.0
            if terminated or truncated
            else self._search(clone, self.depth - 1)
        )
        return reward + continuation

    def action(self, env: HTPdmFjspEnv) -> int:
        base_action = JointRiskGreedyPolicy().action(env)
        scored: list[tuple[float, int, int]] = []
        for action in self._branches(env):
            scenario_values = [
                self._scenario_value(env, action, scenario_index)
                for scenario_index in range(self.scenario_count)
            ]
            scored.append(
                (float(np.mean(scenario_values)), -action, action)
            )
        if not scored:
            raise RuntimeError("No feasible action in a nonterminal state.")
        best = max(scored)
        base = next((item for item in scored if item[2] == base_action), None)
        if base is None or best[0] >= base[0] + self.improvement_margin:
            return best[2]
        return base_action


def rollout_policy(
    env: HTPdmFjspEnv, policy: GymPolicy, *, seed: int
) -> tuple[SimulationResult, float]:
    env.reset(seed=seed)
    policy.reset(env)
    episode_return = 0.0
    while not env._done:
        action = policy.action(env)
        _, reward, _, _, _ = env.step(action)
        episode_return += reward
    result = env.result(policy.name)
    if not np.isclose(episode_return, -float(result.metrics["objective"])):
        raise AssertionError(
            f"Reward identity failed: return={episode_return}, "
            f"objective={result.metrics['objective']}"
        )
    return result, episode_return
