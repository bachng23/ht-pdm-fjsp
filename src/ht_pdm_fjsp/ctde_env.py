"""Synchronous machine-agent environment for CTDE multi-agent learning."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.simulator import SimulationResult


class MachineAgentsCTDEEnv:
    """Expose one decentralized agent per machine over the benchmark core.

    Agents propose one local action at each dispatch epoch.  A deterministic
    environment resolver accepts a conflict-free subset, then the simulator
    advances to the next event.  The actor-facing observation contains only
    the agent's machine, its candidate actions, candidate jobs, and candidate
    technician availability.  The separate global state is for the training
    critic only.
    """

    LOCAL_FEATURE_DIM = 23
    BROADCAST_CONTEXT_DIM = 8

    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        include_broadcast_context: bool = False,
        wait_policy: Literal["legacy", "safe_noop"] = "legacy",
    ) -> None:
        if wait_policy not in {"legacy", "safe_noop"}:
            raise ValueError(f"Unknown wait policy: {wait_policy}")
        self.config = config
        self.core = HTPdmFjspEnv(config=config)
        self.include_broadcast_context = include_broadcast_context
        self.wait_policy = wait_policy
        self.local_feature_dim = self.LOCAL_FEATURE_DIM + (
            self.BROADCAST_CONTEXT_DIM if include_broadcast_context else 0
        )
        self.machine_ids = tuple(machine.machine_id for machine in config.machines)
        catalogs: list[tuple[int | None, ...]] = []
        for machine_id in self.machine_ids:
            actions = tuple(
                index
                for index, descriptor in enumerate(self.core.actions)
                if descriptor.machine_id == machine_id
            )
            catalogs.append((None, *actions))
        self.local_action_catalogs = tuple(catalogs)
        self.max_local_actions = max(map(len, self.local_action_catalogs))
        sample, _ = self.core.reset(seed=0)
        self.global_state_dim = int(self._global_state(sample).shape[0])
        self._done = False
        self.coordination_totals: dict[str, int] = {}
        self.last_resolution: dict[str, int] = {}

    @property
    def num_agents(self) -> int:
        return len(self.machine_ids)

    def reset(self, *, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        observation, _ = self.core.reset(seed=seed)
        self._done = False
        self.coordination_totals = {
            "joint_steps": 0,
            "proposals": 0,
            "accepted": 0,
            "waits": 0,
            "production_conflicts": 0,
            "technician_conflicts": 0,
            "rejected": 0,
            "invalid_executions": 0,
            "duplicate_operation_executions": 0,
            "duplicate_technician_executions": 0,
        }
        self.last_resolution = {}
        return self._ctde_observation(observation), self._info()

    def _global_state(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                observation["time"].reshape(-1),
                observation["jobs"].reshape(-1),
                observation["machines"].reshape(-1),
                observation["technicians"].reshape(-1),
                observation["action_features"].reshape(-1),
                observation["action_mask"].astype(np.float32).reshape(-1),
            ]
        ).astype(np.float32)

    def _ctde_observation(
        self, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        local = np.zeros(
            (self.num_agents, self.max_local_actions, self.local_feature_dim),
            dtype=np.float32,
        )
        masks = np.zeros(
            (self.num_agents, self.max_local_actions), dtype=np.bool_
        )
        core_mask = observation["action_mask"].astype(bool)
        required_nonwait_agent = self._required_nonwait_agent(core_mask)
        for agent_index, catalog in enumerate(self.local_action_catalogs):
            machine_features = observation["machines"][agent_index]
            broadcast_context = self._broadcast_context(observation, agent_index)
            feasible_nonwait = any(
                global_action is not None and core_mask[global_action]
                for global_action in catalog
            )
            if self.wait_policy == "safe_noop":
                wait_allowed = agent_index != required_nonwait_agent
            else:
                wait_allowed = bool(self.core.events) or not feasible_nonwait
            for local_action, global_action in enumerate(catalog):
                action_features = np.zeros(10, dtype=np.float32)
                job_features = np.zeros(4, dtype=np.float32)
                technician_idle = 0.0
                if global_action is None:
                    action_features[0] = 1.0
                    masks[agent_index, local_action] = wait_allowed
                else:
                    descriptor = self.core.actions[global_action]
                    action_features = observation["action_features"][global_action]
                    masks[agent_index, local_action] = core_mask[global_action]
                    if descriptor.job_id is not None:
                        job_index = self.core.job_indices[str(descriptor.job_id)]
                        job_features = observation["jobs"][job_index]
                    if descriptor.technician_id is not None:
                        technician_index = self.core.technician_indices[
                            str(descriptor.technician_id)
                        ]
                        technician_idle = float(
                            observation["technicians"][technician_index, 0]
                        )
                features = [
                    observation["time"],
                    machine_features,
                    action_features,
                    job_features,
                    np.asarray([technician_idle], dtype=np.float32),
                ]
                if self.include_broadcast_context:
                    features.append(broadcast_context)
                local[agent_index, local_action] = np.concatenate(features)
            if not masks[agent_index].any():
                raise AssertionError("Every machine agent needs a feasible action")
        return {
            "local_observations": local,
            "action_masks": masks,
            "global_state": self._global_state(observation),
        }

    def _required_nonwait_agent(self, core_mask: np.ndarray) -> int | None:
        """Choose one progress anchor when safe no-op masks are enabled.

        If no simulator event can advance time, exactly one agent with a feasible
        non-wait action must act.  Production or corrective work is preferred so
        idle peers are not forced into optional preventive maintenance.  The
        rotating resolver priority keeps the anchor selection fair across agents.
        """

        if self.wait_policy != "safe_noop" or self.core.events:
            return None
        start = self.coordination_totals.get("joint_steps", 0) % self.num_agents
        ordered_agents = tuple(
            (start + offset) % self.num_agents for offset in range(self.num_agents)
        )
        for preferred_kinds in (
            {"production", "corrective"},
            {"production", "corrective", "preventive"},
        ):
            for agent_index in ordered_agents:
                for global_action in self.local_action_catalogs[agent_index]:
                    if global_action is None or not core_mask[global_action]:
                        continue
                    if self.core.actions[global_action].kind in preferred_kinds:
                        return agent_index
        return None

    def _broadcast_context(
        self, observation: dict[str, np.ndarray], agent_index: int
    ) -> np.ndarray:
        """Return observable shop aggregates without exposing entity identities."""

        jobs = observation["jobs"]
        machines = observation["machines"]
        technicians = observation["technicians"]
        other_machine_indices = [
            index for index in range(self.num_agents) if index != agent_index
        ]
        other_machines = machines[other_machine_indices]
        if len(other_machines):
            other_idle = float(other_machines[:, 0].mean())
            other_mean_age = float(other_machines[:, 4].mean())
            other_max_age = float(other_machines[:, 4].max())
        else:
            other_idle = 0.0
            other_mean_age = 0.0
            other_max_age = 0.0
        return np.asarray(
            [
                jobs[:, 0].mean(),
                jobs[:, 1].mean(),
                jobs[:, 2].mean(),
                jobs[:, 3].min(),
                other_idle,
                other_mean_age,
                other_max_age,
                technicians[:, 0].mean(),
            ],
            dtype=np.float32,
        )

    def local_to_global(self, agent_index: int, local_action: int) -> int | None:
        catalog = self.local_action_catalogs[agent_index]
        if local_action < 0 or local_action >= len(catalog):
            raise ValueError("Local action is outside the agent catalog")
        return catalog[local_action]

    def technician_duration(self, descriptor: Any) -> float:
        """Return service duration for a maintenance action descriptor."""

        if descriptor.kind not in {"preventive", "corrective"}:
            raise ValueError("Technician duration requires a maintenance action")
        technician = self.core.technician_specs[str(descriptor.technician_id)]
        return technician.duration(str(descriptor.machine_id), descriptor.kind)

    def step(
        self, local_actions: Sequence[int]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is done; call reset().")
        if len(local_actions) != self.num_agents:
            raise ValueError("Joint action must contain one action per machine")
        before = self._ctde_observation(self.core._observation())
        proposed: list[tuple[int, int]] = []
        waits = 0
        for agent_index, raw_action in enumerate(local_actions):
            local_action = int(raw_action)
            if not 0 <= local_action < self.max_local_actions:
                raise ValueError("Local action index is outside the padded catalog")
            if not before["action_masks"][agent_index, local_action]:
                raise ValueError("Masked local action was proposed")
            global_action = self.local_to_global(agent_index, local_action)
            if global_action is None:
                waits += 1
            else:
                proposed.append((agent_index, global_action))

        accepted: list[tuple[int, int]] = []
        claimed_operations: set[tuple[str, int]] = set()
        claimed_technicians: set[str] = set()
        production_conflicts = 0
        technician_conflicts = 0
        priority_start = (
            self.coordination_totals["joint_steps"] % self.num_agents
        )
        ordered_proposals = sorted(
            proposed,
            key=lambda item: (item[0] - priority_start) % self.num_agents,
        )
        for agent_index, global_action in ordered_proposals:
            descriptor = self.core.actions[global_action]
            if descriptor.kind == "production":
                key = (str(descriptor.job_id), int(descriptor.operation_index))
                if key in claimed_operations:
                    production_conflicts += 1
                    continue
                claimed_operations.add(key)
            elif descriptor.kind in {"preventive", "corrective"}:
                technician_id = str(descriptor.technician_id)
                if technician_id in claimed_technicians:
                    technician_conflicts += 1
                    continue
                claimed_technicians.add(technician_id)
            accepted.append((agent_index, global_action))

        accepted_operations = [
            (
                str(self.core.actions[action].job_id),
                int(self.core.actions[action].operation_index),
            )
            for _, action in accepted
            if self.core.actions[action].kind == "production"
        ]
        accepted_technicians = [
            str(self.core.actions[action].technician_id)
            for _, action in accepted
            if self.core.actions[action].kind in {"preventive", "corrective"}
        ]
        duplicate_operations = len(accepted_operations) - len(
            set(accepted_operations)
        )
        duplicate_technicians = len(accepted_technicians) - len(
            set(accepted_technicians)
        )
        if duplicate_operations or duplicate_technicians:
            raise AssertionError("Resolver accepted a conflicting joint action")

        reward = 0.0
        terminated = False
        truncated = False
        for _, global_action in accepted:
            if not self.core.action_mask()[global_action]:
                self.coordination_totals["invalid_executions"] += 1
                raise AssertionError("Resolver produced an invalid simulator action")
            _, value, terminated, truncated, _ = self.core.step(global_action)
            reward += float(value)
            if terminated or truncated:
                break
        if not (terminated or truncated):
            if not self.core.events:
                raise RuntimeError("Joint action made no progress and no event can advance")
            _, value, terminated, truncated, _ = self.core.step(0)
            reward += float(value)

        self._done = terminated or truncated
        resolution = {
            "joint_steps": 1,
            "proposals": len(proposed),
            "accepted": len(accepted),
            "waits": waits,
            "production_conflicts": production_conflicts,
            "technician_conflicts": technician_conflicts,
            "rejected": len(proposed) - len(accepted),
            "invalid_executions": 0,
            "duplicate_operation_executions": duplicate_operations,
            "duplicate_technician_executions": duplicate_technicians,
        }
        self.last_resolution = resolution
        for key, value in resolution.items():
            self.coordination_totals[key] += value
        observation = self._ctde_observation(self.core._observation())
        return observation, reward, terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        return {
            "coordination": dict(self.last_resolution),
            "coordination_totals": dict(self.coordination_totals),
            "metrics": self.core.metrics(),
        }

    def result(self, policy_name: str) -> SimulationResult:
        return self.core.result(policy_name)

    def close(self) -> None:
        self.core.close()
