"""RL interface for the validated committed technician-conflict kernel."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ht_pdm_fjsp.conflict_consequence import (
    FAILED,
    MAINTENANCE,
    WAITING,
    WORKING,
    ConflictConsequenceEnv,
    ConsequenceCell,
    build_cell_config,
)
from ht_pdm_fjsp.parallel_maintenance import ParallelMaintenanceConfig


COMMITTED_CELL = ConsequenceCell(True, False, False)


class CommittedConflictCTDEEnv:
    """Machine-agent CTDE view without changing the validated transition kernel."""

    def __init__(self, base_config: ParallelMaintenanceConfig) -> None:
        self.config = build_cell_config(base_config, COMMITTED_CELL)
        self.core = ConflictConsequenceEnv(self.config, COMMITTED_CELL)
        self.num_agents = self.core.num_agents
        self.action_count = self.core.action_count
        self.max_local_actions = self.action_count
        self.local_feature_dim = 40
        self.global_state_dim = (
            self.num_agents * self.action_count * self.local_feature_dim
            + self.num_agents * self.action_count
            + 1
            + self.core.technician_count * 3
            + self.core.technician_count * self.core.failure_type_count
        )
        self.last_agent_outcomes = tuple(0 for _ in range(self.num_agents))

    def reset(self, *, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        info = self.core.reset(seed=seed)
        self.last_agent_outcomes = tuple(0 for _ in range(self.num_agents))
        return self._observation(), info

    def _machine_features(self, machine: int) -> list[float]:
        spec = self.config.machines[machine]
        mode = int(self.core.mode[machine])
        mode_one_hot = [float(mode == value) for value in (WORKING, FAILED, MAINTENANCE, WAITING)]
        pending = int(self.core.pending_kind[machine])
        pending_one_hot = [float(pending == value) for value in (0, 1, 2)]
        request = self.core.request_kind(machine) if self.core.mode[machine] != MAINTENANCE else 0
        request_one_hot = [float(request == value) for value in (0, 1, 2)]
        assigned = int(self.core.machine_technician[machine])
        assigned_one_hot = [
            float(assigned == value) for value in (-1, 0, 1)
        ]
        candidates = set(self.core.candidates(preventive=True))
        return [
            self.core.step_index / self.config.horizon,
            *mode_one_hot,
            min(self.core.age[machine] / (3.0 * spec.weibull_scale), 1.0),
            min(self.core.failure_probability(machine) / 0.1, 1.0),
            min(self.core.wait[machine] / self.config.horizon, 1.0),
            *pending_one_hot,
            float(self.core.overdue[machine]),
            min(self.core.window_remaining[machine] / 2.0, 1.0),
            *assigned_one_hot,
            spec.load,
            spec.production_rate / 1.5,
            spec.age_rate / 1.25,
            *[float(spec.failure_type == value) for value in range(3)],
            *request_one_hot,
            float(machine in candidates),
            1.0,
        ]

    def _action_features(self, machine: int, action: int) -> list[float]:
        one_hot = [float(action == value) for value in range(self.action_count)]
        if action == 0:
            technician = [0.0] * 8
        else:
            index = action - 1
            failure_type = self.config.machines[machine].failure_type
            spec = self.config.technicians[index]
            technician = [
                float(self.core.technician_available(index)),
                float(self.core.technician_absent[index]),
                min(self.core.technician_remaining[index] / self.config.horizon, 1.0),
                min(self.core.expected_duration(machine, index) / 10.0, 1.0),
                spec.base_skill[failure_type],
                self.core.success_probability(machine, index),
                self.core.experience[index, failure_type],
                float(self.core.pending_preferred[machine] == index),
            ]
        available_fraction = statistics_fraction(
            self.core.technician_available(index)
            for index in range(self.core.technician_count)
        )
        absent_fraction = float(np.mean(self.core.technician_absent))
        return [available_fraction, absent_fraction, *one_hot, *technician]

    def _observation(self) -> dict[str, np.ndarray]:
        masks = self.core.action_masks().astype(np.bool_)
        local = np.empty(
            (self.num_agents, self.action_count, self.local_feature_dim),
            dtype=np.float32,
        )
        for machine in range(self.num_agents):
            machine_features = self._machine_features(machine)
            for action in range(self.action_count):
                features = machine_features + self._action_features(machine, action)
                if len(features) != self.local_feature_dim:
                    raise AssertionError("Committed-conflict feature schema changed")
                local[machine, action] = features
        global_tail = np.concatenate(
            [
                np.asarray([self.core.step_index / self.config.horizon], dtype=np.float32),
                (self.core.technician_remaining / self.config.horizon).astype(np.float32),
                self.core.technician_absent.astype(np.float32),
                ((self.core.technician_machine + 1) / (self.num_agents + 1)).astype(np.float32),
                self.core.experience.astype(np.float32).reshape(-1),
            ]
        )
        global_state = np.concatenate(
            [local.reshape(-1), masks.astype(np.float32).reshape(-1), global_tail]
        ).astype(np.float32)
        if global_state.shape != (self.global_state_dim,):
            raise AssertionError("Committed-conflict global state schema changed")
        if not np.isfinite(local).all() or not np.isfinite(global_state).all():
            raise AssertionError("RL observation contains non-finite values")
        return {
            "local_observations": local,
            "action_masks": masks,
            "global_state": global_state,
        }

    def step(
        self, actions: Sequence[int]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (self.num_agents,):
            raise ValueError("Joint action requires one action per machine")
        masks = self.core.action_masks()
        if not all(masks[index, action] for index, action in enumerate(action_array)):
            raise ValueError("Masked action was proposed")
        reward, done, info = self.core.step(action_array)
        accepted = {
            int(item["machine"]) for item in self.core.last_decision["accepted"]
        }
        rejected = {
            int(item[0]) for item in self.core.last_decision["rejected"]
        }
        outcomes = []
        for machine, action in enumerate(action_array):
            if action == 0:
                outcomes.append(0)
            elif machine in accepted:
                outcomes.append(1)
            elif machine in rejected:
                outcomes.append(-1)
            else:
                raise AssertionError("Non-wait proposal was neither accepted nor rejected")
        self.last_agent_outcomes = tuple(outcomes)
        return self._observation(), float(reward), bool(done), False, {
            **info,
            "agent_outcomes": self.last_agent_outcomes,
        }

    def close(self) -> None:
        return None


def statistics_fraction(values: Any) -> float:
    items = tuple(bool(value) for value in values)
    return sum(items) / len(items) if items else 0.0
