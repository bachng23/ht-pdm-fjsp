"""Validated parallel-machine maintenance and technician-allocation simulator."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


WORKING = 0
FAILED = 1
MAINTENANCE = 2


@dataclass(frozen=True)
class ParallelMachineSpec:
    machine_id: str
    failure_type: int
    production_rate: float
    load: float
    age_rate: float
    weibull_shape: float
    weibull_scale: float


@dataclass(frozen=True)
class ParallelTechnicianSpec:
    technician_id: str
    base_skill: tuple[float, ...]
    base_duration: tuple[float, ...]
    duration_sigma: tuple[float, ...]
    learning_rate: float
    absence_probability: float


@dataclass(frozen=True)
class MaintenanceParameters:
    pm_duration_factor: float
    pm_restoration: float
    cm_restoration: float
    experience_duration_gain: float
    success_intercept: float
    success_skill_weight: float
    success_experience_weight: float
    pm_risk_threshold: float


@dataclass(frozen=True)
class ParallelCostParameters:
    downtime: float
    failure: float
    preventive: float
    corrective: float
    waiting: float
    conflict: float
    rework: float
    early_pm: float
    production_credit: float


@dataclass(frozen=True)
class ParallelMaintenanceConfig:
    horizon: int
    time_step: float
    machines: tuple[ParallelMachineSpec, ...]
    technicians: tuple[ParallelTechnicianSpec, ...]
    maintenance: MaintenanceParameters
    costs: ParallelCostParameters

    @classmethod
    def from_json(cls, path: str | Path) -> "ParallelMaintenanceConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(
            horizon=int(payload["horizon"]),
            time_step=float(payload["time_step"]),
            machines=tuple(ParallelMachineSpec(**row) for row in payload["machines"]),
            technicians=tuple(
                ParallelTechnicianSpec(
                    **{
                        **row,
                        "base_skill": tuple(map(float, row["base_skill"])),
                        "base_duration": tuple(map(float, row["base_duration"])),
                        "duration_sigma": tuple(map(float, row["duration_sigma"])),
                    }
                )
                for row in payload["technicians"]
            ),
            maintenance=MaintenanceParameters(**payload["maintenance"]),
            costs=ParallelCostParameters(**payload["costs"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.horizon <= 0 or self.time_step <= 0:
            raise ValueError("Horizon and time step must be positive")
        if not self.machines or not self.technicians:
            raise ValueError("At least one machine and technician are required")
        failure_types = {machine.failure_type for machine in self.machines}
        type_count = max(failure_types) + 1
        if failure_types != set(range(type_count)):
            raise ValueError("Failure types must be contiguous and zero based")
        for machine in self.machines:
            if not (machine.weibull_shape > 0 and machine.weibull_scale > 0):
                raise ValueError("Weibull parameters must be positive")
        for technician in self.technicians:
            if not (
                len(technician.base_skill)
                == len(technician.base_duration)
                == len(technician.duration_sigma)
                == type_count
            ):
                raise ValueError("Technician vectors must cover every failure type")
            if any(duration <= 0 for duration in technician.base_duration):
                raise ValueError("Service durations must be positive")
            if not 0 <= technician.absence_probability <= 1:
                raise ValueError("Absence probability must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParallelMaintenanceEnv:
    """One machine agent per parallel machine with a deterministic resolver."""

    LOCAL_FEATURE_DIM = 15

    def __init__(self, config: ParallelMaintenanceConfig) -> None:
        self.config = config
        self.num_agents = len(config.machines)
        self.technician_count = len(config.technicians)
        self.action_count = self.technician_count + 1
        self.max_local_actions = self.action_count
        self.local_feature_dim = self.LOCAL_FEATURE_DIM
        self.global_state_dim = 1 + self.num_agents * 5 + self.technician_count * (
            3 + self.failure_type_count
        )
        self._done = False

    @property
    def failure_type_count(self) -> int:
        return max(machine.failure_type for machine in self.config.machines) + 1

    def reset(self, *, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.seed = int(seed)
        self.step_index = 0
        self.mode = np.full(self.num_agents, WORKING, dtype=np.int8)
        self.age = np.zeros(self.num_agents, dtype=np.float64)
        self.wait = np.zeros(self.num_agents, dtype=np.float64)
        self.machine_technician = np.full(self.num_agents, -1, dtype=np.int16)
        self.maintenance_kind = np.zeros(self.num_agents, dtype=np.int8)
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
        self.metrics = {
            "objective": 0.0,
            "production": 0.0,
            "downtime": 0.0,
            "failures": 0,
            "preventive": 0,
            "corrective": 0,
            "waiting": 0.0,
            "proposal_conflicts": 0,
            "conflict_steps": 0,
            "rework": 0,
            "early_pm": 0.0,
            "episode_return": 0.0,
            "invalid_executions": 0,
            "duplicate_machine_assignments": 0,
            "duplicate_technician_assignments": 0,
        }
        self._done = False
        self.last_decision: dict[str, Any] = {}
        return self._observation(), self._info()

    def _keyed_rng(self, channel: int, *keys: int) -> np.random.Generator:
        """Return a policy-order-independent random stream for one named shock."""

        sequence = np.random.SeedSequence([self.seed, channel, *map(int, keys)])
        return np.random.default_rng(sequence)

    def conditional_failure_probability(self, machine_index: int) -> float:
        machine = self.config.machines[machine_index]
        increment = machine.age_rate * machine.load * self.config.time_step
        current = self.age[machine_index]
        exponent = (
            ((current + increment) / machine.weibull_scale) ** machine.weibull_shape
            - (current / machine.weibull_scale) ** machine.weibull_shape
        )
        return float(1.0 - math.exp(-max(0.0, exponent)))

    def technician_available(self, technician_index: int) -> bool:
        return bool(
            self.technician_machine[technician_index] < 0
            and not self.technician_absent[technician_index]
        )

    def expected_duration(self, machine_index: int, technician_index: int) -> float:
        machine = self.config.machines[machine_index]
        technician = self.config.technicians[technician_index]
        failure_type = machine.failure_type
        gain = self.config.maintenance.experience_duration_gain
        duration = technician.base_duration[failure_type] / (
            1.0 + gain * self.experience[technician_index, failure_type]
        )
        if self.mode[machine_index] == WORKING:
            duration *= self.config.maintenance.pm_duration_factor
        return float(duration)

    def success_probability(self, machine_index: int, technician_index: int) -> float:
        failure_type = self.config.machines[machine_index].failure_type
        technician = self.config.technicians[technician_index]
        parameters = self.config.maintenance
        logit = (
            parameters.success_intercept
            + parameters.success_skill_weight * technician.base_skill[failure_type]
            + parameters.success_experience_weight
            * self.experience[technician_index, failure_type]
        )
        return float(1.0 / (1.0 + math.exp(-logit)))

    def _action_masks(self) -> np.ndarray:
        masks = np.zeros((self.num_agents, self.action_count), dtype=np.bool_)
        masks[:, 0] = True
        for machine_index in range(self.num_agents):
            if self.mode[machine_index] == MAINTENANCE:
                continue
            for technician_index in range(self.technician_count):
                masks[machine_index, technician_index + 1] = self.technician_available(
                    technician_index
                )
        return masks

    def _global_state(self) -> np.ndarray:
        machine_rows: list[float] = []
        for index, machine in enumerate(self.config.machines):
            machine_rows.extend(
                [
                    self.mode[index] / 2.0,
                    min(2.0, self.age[index] / machine.weibull_scale),
                    min(1.0, self.wait[index] / self.config.horizon),
                    self.conditional_failure_probability(index),
                    (self.machine_technician[index] + 1)
                    / (self.technician_count + 1),
                ]
            )
        technician_rows: list[float] = []
        for index in range(self.technician_count):
            technician_rows.extend(
                [
                    float(self.technician_available(index)),
                    float(self.technician_absent[index]),
                    min(
                        1.0,
                        self.technician_remaining[index] / self.config.horizon,
                    ),
                    *self.experience[index].tolist(),
                ]
            )
        return np.asarray(
            [
                self.step_index / self.config.horizon,
                *machine_rows,
                *technician_rows,
            ],
            dtype=np.float32,
        )

    def _observation(self) -> dict[str, np.ndarray]:
        masks = self._action_masks()
        local = np.zeros(
            (self.num_agents, self.action_count, self.local_feature_dim),
            dtype=np.float32,
        )
        for machine_index, machine in enumerate(self.config.machines):
            failure_type = machine.failure_type
            for action in range(self.action_count):
                technician_index = action - 1
                tech_features = [0.0] * 7
                if technician_index >= 0:
                    technician = self.config.technicians[technician_index]
                    tech_features = [
                        1.0,
                        float(self.technician_available(technician_index)),
                        float(self.technician_absent[technician_index]),
                        technician.base_skill[failure_type],
                        self.experience[technician_index, failure_type],
                        min(
                            1.0,
                            self.expected_duration(machine_index, technician_index)
                            / self.config.horizon,
                        ),
                        self.success_probability(machine_index, technician_index),
                    ]
                local[machine_index, action] = np.asarray(
                    [
                        self.step_index / self.config.horizon,
                        float(self.mode[machine_index] == WORKING),
                        float(self.mode[machine_index] == FAILED),
                        float(self.mode[machine_index] == MAINTENANCE),
                        min(2.0, self.age[machine_index] / machine.weibull_scale),
                        min(1.0, self.wait[machine_index] / self.config.horizon),
                        self.conditional_failure_probability(machine_index),
                        machine.production_rate / 2.0,
                        *tech_features,
                    ],
                    dtype=np.float32,
                )
        return {
            "local_observations": local,
            "action_masks": masks,
            "global_state": self._global_state(),
        }

    def _start_maintenance(self, machine_index: int, technician_index: int) -> dict[str, float]:
        if self.mode[machine_index] == MAINTENANCE or not self.technician_available(
            technician_index
        ):
            self.metrics["invalid_executions"] += 1
            raise AssertionError("Resolver attempted an infeasible maintenance start")
        kind = 1 if self.mode[machine_index] == WORKING else 2
        expected = self.expected_duration(machine_index, technician_index)
        failure_type = self.config.machines[machine_index].failure_type
        sigma = self.config.technicians[technician_index].duration_sigma[failure_type]
        attempt = int(self.maintenance_attempts[machine_index, technician_index])
        service_rng = self._keyed_rng(1, machine_index, technician_index, attempt)
        sampled = expected if sigma == 0 else float(
            service_rng.lognormal(math.log(expected) - 0.5 * sigma * sigma, sigma)
        )
        duration = max(self.config.time_step, sampled)
        success = bool(
            service_rng.random()
            < self.success_probability(machine_index, technician_index)
        )
        self.maintenance_attempts[machine_index, technician_index] += 1
        self.mode[machine_index] = MAINTENANCE
        self.machine_technician[machine_index] = technician_index
        self.maintenance_kind[machine_index] = kind
        self.repair_will_succeed[machine_index] = success
        self.technician_machine[technician_index] = machine_index
        self.technician_remaining[technician_index] = duration
        early_pm = 0.0
        if kind == 1:
            self.metrics["preventive"] += 1
            risk = self.conditional_failure_probability(machine_index)
            early_pm = max(0.0, self.config.maintenance.pm_risk_threshold - risk)
            self.metrics["early_pm"] += early_pm
        else:
            self.metrics["corrective"] += 1
        return {"kind": float(kind), "duration": duration, "early_pm": early_pm}

    def _resolve(self, actions: Sequence[int]) -> tuple[list[tuple[int, int]], int]:
        masks = self._action_masks()
        proposals: dict[int, list[int]] = {}
        for machine_index, raw_action in enumerate(actions):
            action = int(raw_action)
            if not 0 <= action < self.action_count or not masks[machine_index, action]:
                raise ValueError("Masked or out-of-range local action was proposed")
            if action:
                proposals.setdefault(action - 1, []).append(machine_index)
        accepted: list[tuple[int, int]] = []
        conflicts = 0
        for technician_index, machine_indices in proposals.items():
            conflicts += max(0, len(machine_indices) - 1)
            winner = min(
                machine_indices,
                key=lambda index: (
                    0 if self.mode[index] == FAILED else 1,
                    -self.wait[index],
                    -self.conditional_failure_probability(index),
                    index,
                ),
            )
            accepted.append((winner, technician_index))
        return accepted, conflicts

    def step(
        self, actions: Sequence[int]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is done; call reset()")
        if len(actions) != self.num_agents:
            raise ValueError("Joint action must contain one action per machine")
        accepted, conflicts = self._resolve(actions)
        incremental_cost = self.config.costs.conflict * conflicts
        starts: list[dict[str, Any]] = []
        for machine_index, technician_index in accepted:
            details = self._start_maintenance(machine_index, technician_index)
            starts.append(
                {
                    "machine": machine_index,
                    "technician": technician_index,
                    **details,
                }
            )
            if int(details["kind"]) == 1:
                incremental_cost += self.config.costs.preventive
                incremental_cost += self.config.costs.early_pm * details["early_pm"]
            else:
                incremental_cost += self.config.costs.corrective

        interval_production = 0.0
        interval_downtime = 0.0
        interval_waiting = 0.0
        failures = 0
        reworks = 0
        for machine_index, machine in enumerate(self.config.machines):
            if self.mode[machine_index] == WORKING:
                production = machine.production_rate * machine.load * self.config.time_step
                interval_production += production
                increment = machine.age_rate * machine.load * self.config.time_step
                failure_probability = self.conditional_failure_probability(machine_index)
                self.age[machine_index] += increment
                failure_rng = self._keyed_rng(2, self.step_index, machine_index)
                if failure_rng.random() < failure_probability:
                    self.mode[machine_index] = FAILED
                    self.wait[machine_index] = 0.0
                    failures += 1
            else:
                interval_downtime += self.config.time_step
                if self.mode[machine_index] == FAILED:
                    self.wait[machine_index] += self.config.time_step
                    interval_waiting += self.config.time_step

        for technician_index in range(self.technician_count):
            machine_index = int(self.technician_machine[technician_index])
            if machine_index < 0:
                continue
            worked = min(self.config.time_step, self.technician_remaining[technician_index])
            self.technician_busy_time[technician_index] += worked
            self.technician_remaining[technician_index] -= self.config.time_step
            if self.technician_remaining[technician_index] > 1e-12:
                continue
            failure_type = self.config.machines[machine_index].failure_type
            success = bool(self.repair_will_succeed[machine_index])
            kind = int(self.maintenance_kind[machine_index])
            if success:
                restoration = (
                    self.config.maintenance.pm_restoration
                    if kind == 1
                    else self.config.maintenance.cm_restoration
                )
                self.age[machine_index] *= max(0.0, 1.0 - restoration)
                self.mode[machine_index] = WORKING
                learning_rate = self.config.technicians[technician_index].learning_rate
                current = self.experience[technician_index, failure_type]
                self.experience[technician_index, failure_type] = current + learning_rate * (
                    1.0 - current
                )
                self.wait[machine_index] = 0.0
            else:
                self.mode[machine_index] = FAILED
                reworks += 1
            self.machine_technician[machine_index] = -1
            self.maintenance_kind[machine_index] = 0
            self.technician_machine[technician_index] = -1
            self.technician_remaining[technician_index] = 0.0

        for technician_index, technician in enumerate(self.config.technicians):
            if self.technician_machine[technician_index] < 0:
                absence_rng = self._keyed_rng(3, self.step_index + 1, technician_index)
                self.technician_absent[technician_index] = bool(
                    absence_rng.random() < technician.absence_probability
                )
            else:
                self.technician_absent[technician_index] = False

        incremental_cost += self.config.costs.downtime * interval_downtime
        incremental_cost += self.config.costs.failure * failures
        incremental_cost += self.config.costs.waiting * interval_waiting
        incremental_cost += self.config.costs.rework * reworks
        incremental_cost -= self.config.costs.production_credit * interval_production
        reward = -float(incremental_cost)
        self.metrics["objective"] += incremental_cost
        self.metrics["production"] += interval_production
        self.metrics["downtime"] += interval_downtime
        self.metrics["failures"] += failures
        self.metrics["waiting"] += interval_waiting
        self.metrics["proposal_conflicts"] += conflicts
        self.metrics["conflict_steps"] += int(conflicts > 0)
        self.metrics["rework"] += reworks
        self.metrics["episode_return"] += reward
        self.step_index += 1
        self._done = self.step_index >= self.config.horizon
        self.last_decision = {
            "step": self.step_index - 1,
            "actions": [int(value) for value in actions],
            "accepted": starts,
            "proposal_conflicts": conflicts,
            "failures": failures,
            "reworks": reworks,
            "incremental_cost": incremental_cost,
            "reward": reward,
        }
        self._audit_resources()
        return self._observation(), reward, self._done, False, self._info()

    def _audit_resources(self) -> None:
        assigned_machines = [int(value) for value in self.technician_machine if value >= 0]
        assigned_technicians = [
            int(value) for value in self.machine_technician if value >= 0
        ]
        if len(assigned_machines) != len(set(assigned_machines)):
            self.metrics["duplicate_machine_assignments"] += 1
            raise AssertionError("A machine was assigned to multiple technicians")
        if len(assigned_technicians) != len(set(assigned_technicians)):
            self.metrics["duplicate_technician_assignments"] += 1
            raise AssertionError("A technician was assigned to multiple machines")
        for machine_index, technician_index in enumerate(self.machine_technician):
            if technician_index >= 0 and self.technician_machine[technician_index] != machine_index:
                raise AssertionError("Machine/technician assignment is not reciprocal")

    def _info(self) -> dict[str, Any]:
        utilization = (
            self.technician_busy_time
            / max(self.config.time_step, self.step_index * self.config.time_step)
        )
        return {
            **self.metrics,
            "technician_utilization": utilization.tolist(),
            "workload_imbalance": float(np.var(utilization)),
            "return_objective_error": abs(
                float(self.metrics["episode_return"] + self.metrics["objective"])
            ),
            "last_decision": self.last_decision,
        }

    def close(self) -> None:
        return None


class CentralizedParallelMaintenanceEnv(gym.Env[np.ndarray, int]):
    """Gym wrapper enumerating all simultaneous machine proposals."""

    metadata = {"render_modes": []}

    def __init__(self, config: ParallelMaintenanceConfig) -> None:
        super().__init__()
        self.core = ParallelMaintenanceEnv(config)
        self.joint_actions = tuple(
            itertools.product(
                range(self.core.action_count), repeat=self.core.num_agents
            )
        )
        self.action_space = spaces.Discrete(len(self.joint_actions))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.core.global_state_dim,),
            dtype=np.float32,
        )
        self._last_observation: dict[str, np.ndarray] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observation, info = self.core.reset(seed=0 if seed is None else seed)
        self._last_observation = observation
        return observation["global_state"], info

    def action_masks(self) -> np.ndarray:
        if self._last_observation is None:
            raise RuntimeError("Call reset before requesting masks")
        local_masks = self._last_observation["action_masks"]
        valid = np.zeros(len(self.joint_actions), dtype=np.bool_)
        for index, joint_action in enumerate(self.joint_actions):
            chosen_technicians = [value for value in joint_action if value > 0]
            valid[index] = (
                len(chosen_technicians) == len(set(chosen_technicians))
                and all(local_masks[machine, action] for machine, action in enumerate(joint_action))
            )
        return valid

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("Centralized action is outside the catalog")
        if not self.action_masks()[int(action)]:
            raise ValueError("Masked centralized action was proposed")
        observation, reward, terminated, truncated, info = self.core.step(
            self.joint_actions[int(action)]
        )
        self._last_observation = observation
        return observation["global_state"], reward, terminated, truncated, info

    def close(self) -> None:
        self.core.close()
