"""Gymnasium interface for the minimal HT-PdM-FJSP transition model."""

from __future__ import annotations

import copy
import heapq
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.simulator import (
    Event,
    JobState,
    MachineState,
    SimulationResult,
    TaskRecord,
    TechnicianState,
    TIME_TOLERANCE,
    audit_result,
    conditional_weibull_failure_probability,
    sample_conditional_weibull_failure_age,
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "minimal_benchmark.json"
)


@dataclass(frozen=True)
class GymEnvConfig:
    max_decisions: int = 1_000
    invalid_action_penalty: float = 100.0
    makespan_weight: float = 1.0

    def validate(self) -> None:
        if self.max_decisions < 1:
            raise ValueError("max_decisions must be positive.")
        if self.invalid_action_penalty <= 0 or self.makespan_weight < 0:
            raise ValueError("Reward weights must be nonnegative and penalty positive.")


@dataclass(frozen=True)
class ActionDescriptor:
    kind: str
    job_id: str | None = None
    operation_index: int | None = None
    operation_id: str | None = None
    machine_id: str | None = None
    technician_id: str | None = None


class HTPdmFjspEnv(gym.Env):
    """Event-driven environment with a fixed catalog and dynamic action mask."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        *,
        config_path: str | Path | None = None,
        env_config: GymEnvConfig | None = None,
        include_action_context: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if config is not None and config_path is not None:
            raise ValueError("Provide config or config_path, not both.")
        self.config = config or BenchmarkConfig.from_json(
            config_path or DEFAULT_CONFIG_PATH
        )
        self.config.validate()
        self.env_config = env_config or GymEnvConfig()
        self.env_config.validate()
        self.include_action_context = include_action_context
        if render_mode not in {None, "ansi"}:
            raise ValueError("Only render_mode=None or 'ansi' is supported.")
        self.render_mode = render_mode

        self.machine_specs = {item.machine_id: item for item in self.config.machines}
        self.technician_specs = {
            item.technician_id: item for item in self.config.technicians
        }
        self.job_specs = {item.job_id: item for item in self.config.jobs}
        self.machine_indices = {
            item.machine_id: index for index, item in enumerate(self.config.machines)
        }
        self.technician_indices = {
            item.technician_id: index
            for index, item in enumerate(self.config.technicians)
        }
        self.job_indices = {
            item.job_id: index for index, item in enumerate(self.config.jobs)
        }
        self.actions = self._build_action_catalog()
        self.action_space = spaces.Discrete(len(self.actions))
        job_count = len(self.config.jobs)
        machine_count = len(self.config.machines)
        technician_count = len(self.config.technicians)
        finite_low = np.finfo(np.float32).min
        finite_high = np.finfo(np.float32).max
        observation_spaces: dict[str, spaces.Space] = {
            "time": spaces.Box(0.0, finite_high, shape=(1,), dtype=np.float32),
            "jobs": spaces.Box(
                finite_low, finite_high, shape=(job_count, 4), dtype=np.float32
            ),
            "machines": spaces.Box(
                0.0, finite_high, shape=(machine_count, 7), dtype=np.float32
            ),
            "technicians": spaces.Box(
                0.0,
                1.0,
                shape=(technician_count, 1 + machine_count),
                dtype=np.float32,
            ),
            "action_features": spaces.Box(
                finite_low,
                finite_high,
                shape=(len(self.actions), 10),
                dtype=np.float32,
            ),
            "action_mask": spaces.MultiBinary(len(self.actions)),
        }
        if self.include_action_context:
            action_context_dim = 17 + machine_count
            observation_spaces["action_context"] = spaces.Box(
                finite_low,
                finite_high,
                shape=(len(self.actions), action_context_dim),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(observation_spaces)
        self._has_reset = False
        self._done = False

    def _build_action_catalog(self) -> tuple[ActionDescriptor, ...]:
        actions = [ActionDescriptor("advance")]
        for job in self.config.jobs:
            for operation_index, operation in enumerate(job.operations):
                for alternative in operation.alternatives:
                    actions.append(
                        ActionDescriptor(
                            "production",
                            job_id=job.job_id,
                            operation_index=operation_index,
                            operation_id=operation.operation_id,
                            machine_id=alternative.machine_id,
                        )
                    )
        for kind in ("preventive", "corrective"):
            for machine in self.config.machines:
                for technician in self.config.technicians:
                    if machine.machine_id in technician.eligible_machines:
                        actions.append(
                            ActionDescriptor(
                                kind,
                                machine_id=machine.machine_id,
                                technician_id=technician.technician_id,
                            )
                        )
        return tuple(actions)

    def clone(self) -> "HTPdmFjspEnv":
        if not self._has_reset:
            raise RuntimeError("Call reset() before clone().")
        return copy.deepcopy(self)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.root_seed = int(
            seed
            if seed is not None
            else self.np_random.integers(0, np.iinfo(np.int32).max)
        )
        self.now = 0.0
        self.decision_count = 0
        self.event_sequence = 0
        self.events: list[Event] = []
        self.tasks: list[TaskRecord] = []
        self.jobs = {job.job_id: JobState() for job in self.config.jobs}
        self.machines = {
            machine.machine_id: MachineState(
                status="idle", effective_age=machine.initial_effective_age
            )
            for machine in self.config.machines
        }
        self.technicians = {
            technician.technician_id: TechnicianState()
            for technician in self.config.technicians
        }
        self.attempt_counts: dict[tuple[str, int, str], int] = {}
        self.failures = 0
        self.preventive_count = 0
        self.corrective_count = 0
        self.failed_processing_time = 0.0
        self.maintenance_time = 0.0
        self.maintenance_wait_time = 0.0
        self.production_makespan = 0.0
        self.cumulative_reward = 0.0
        self.cumulative_components = {
            "makespan": 0.0,
            "tardiness": 0.0,
            "preventive": 0.0,
            "corrective": 0.0,
            "downtime": 0.0,
            "invalid": 0.0,
        }
        self._has_reset = True
        self._done = False
        observation = self._observation()
        return observation, self._info({}, invalid_action=False)

    def _push_event(self, time: float, kind: str, payload: dict[str, Any]) -> None:
        self.event_sequence += 1
        heapq.heappush(
            self.events, Event(float(time), self.event_sequence, kind, payload)
        )

    def _all_jobs_complete(self) -> bool:
        return all(
            state.next_operation == len(self.job_specs[job_id].operations)
            for job_id, state in self.jobs.items()
        )

    def _terminal_ready(self) -> bool:
        return (
            self._all_jobs_complete()
            and not self.events
            and all(state.status == "idle" for state in self.machines.values())
        )

    def _alternative(self, descriptor: ActionDescriptor):
        operation = self.job_specs[str(descriptor.job_id)].operations[
            int(descriptor.operation_index)
        ]
        return next(
            item
            for item in operation.alternatives
            if item.machine_id == descriptor.machine_id
        )

    def failure_probability(self, descriptor: ActionDescriptor) -> float:
        if descriptor.kind != "production":
            raise ValueError("failure_probability requires a production action.")
        alternative = self._alternative(descriptor)
        machine = self.machine_specs[str(descriptor.machine_id)]
        state = self.machines[machine.machine_id]
        increment = machine.alpha * alternative.processing_time * alternative.load_factor
        return conditional_weibull_failure_probability(
            state.effective_age,
            increment,
            machine.weibull_eta,
            machine.weibull_beta,
        )

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.actions), dtype=np.int8)
        mask[0] = int(bool(self.events))
        jobs_complete = self._all_jobs_complete()
        for index, descriptor in enumerate(self.actions[1:], start=1):
            machine_state = self.machines[str(descriptor.machine_id)]
            if descriptor.kind == "production":
                job_state = self.jobs[str(descriptor.job_id)]
                mask[index] = int(
                    machine_state.status == "idle"
                    and not job_state.in_process
                    and job_state.next_operation == descriptor.operation_index
                )
            elif descriptor.kind == "preventive":
                technician_state = self.technicians[str(descriptor.technician_id)]
                mask[index] = int(
                    not jobs_complete
                    and machine_state.status == "idle"
                    and machine_state.effective_age > 0.0
                    and machine_state.preventive_armed
                    and technician_state.status == "idle"
                )
            elif descriptor.kind == "corrective":
                technician_state = self.technicians[str(descriptor.technician_id)]
                mask[index] = int(
                    machine_state.status == "waiting_corrective"
                    and technician_state.status == "idle"
                )
        return mask

    def action_masks(self) -> np.ndarray:
        """Compatibility hook used by sb3-contrib's MaskablePPO."""

        return self.action_mask()

    def _observation(self) -> dict[str, np.ndarray]:
        max_due = max(job.due_date for job in self.config.jobs)
        jobs_array = np.zeros((len(self.config.jobs), 4), dtype=np.float32)
        for row, job in enumerate(self.config.jobs):
            state = self.jobs[job.job_id]
            jobs_array[row] = (
                float(state.next_operation == len(job.operations)),
                float(state.in_process),
                state.next_operation / len(job.operations),
                (job.due_date - self.now) / max_due,
            )
        machines_array = np.zeros((len(self.config.machines), 7), dtype=np.float32)
        statuses = ("idle", "processing", "waiting_corrective", "maintenance")
        for row, machine in enumerate(self.config.machines):
            state = self.machines[machine.machine_id]
            one_hot = [float(state.status == status) for status in statuses]
            machines_array[row] = (
                *one_hot,
                state.effective_age / machine.weibull_eta,
                machine.weibull_beta / 10.0,
                float(state.preventive_armed),
            )
        technicians_array = np.zeros(
            (len(self.config.technicians), 1 + len(self.config.machines)),
            dtype=np.float32,
        )
        for row, technician in enumerate(self.config.technicians):
            technicians_array[row, 0] = float(
                self.technicians[technician.technician_id].status == "idle"
            )
            technicians_array[row, 1:] = [
                float(machine.machine_id in technician.eligible_machines)
                for machine in self.config.machines
            ]
        mask = self.action_mask()
        action_features = np.zeros((len(self.actions), 10), dtype=np.float32)
        for index, descriptor in enumerate(self.actions):
            kind_index = {
                "advance": 0,
                "production": 1,
                "preventive": 2,
                "corrective": 3,
            }[descriptor.kind]
            action_features[index, kind_index] = 1.0
            if descriptor.kind == "production":
                alternative = self._alternative(descriptor)
                machine = self.machine_specs[str(descriptor.machine_id)]
                job = self.job_specs[str(descriptor.job_id)]
                action_features[index, 4:] = (
                    alternative.processing_time / max_due,
                    alternative.load_factor,
                    self.failure_probability(descriptor),
                    self.machines[machine.machine_id].effective_age
                    / machine.weibull_eta,
                    (job.due_date - self.now) / max_due,
                    mask[index],
                )
            elif descriptor.kind in {"preventive", "corrective"}:
                technician = self.technician_specs[str(descriptor.technician_id)]
                machine_id = str(descriptor.machine_id)
                machine = self.machine_specs[machine_id]
                action_features[index, 4:] = (
                    technician.duration(machine_id, descriptor.kind) / max_due,
                    technician.restoration(machine_id, descriptor.kind),
                    0.0,
                    self.machines[machine_id].effective_age / machine.weibull_eta,
                    0.0,
                    mask[index],
                )
            else:
                action_features[index, 9] = mask[index]
        observation = {
            "time": np.asarray([self.now / max_due], dtype=np.float32),
            "jobs": jobs_array,
            "machines": machines_array,
            "technicians": technicians_array,
            "action_features": action_features,
            "action_mask": mask,
        }
        if self.include_action_context:
            context = np.zeros(
                (
                    len(self.actions),
                    int(self.observation_space["action_context"].shape[1]),
                ),
                dtype=np.float32,
            )
            for index, descriptor in enumerate(self.actions):
                offset = 0
                if descriptor.job_id is not None:
                    job_index = self.job_indices[str(descriptor.job_id)]
                    context[index, offset] = 1.0
                    context[index, offset + 1 : offset + 5] = jobs_array[job_index]
                offset += 5
                if descriptor.machine_id is not None:
                    machine_index = self.machine_indices[str(descriptor.machine_id)]
                    context[index, offset] = 1.0
                    context[index, offset + 1 : offset + 8] = machines_array[
                        machine_index
                    ]
                offset += 8
                if descriptor.technician_id is not None:
                    technician_index = self.technician_indices[
                        str(descriptor.technician_id)
                    ]
                    technician_width = technicians_array.shape[1]
                    context[index, offset] = 1.0
                    context[
                        index, offset + 1 : offset + 1 + technician_width
                    ] = technicians_array[technician_index]
                offset += 1 + technicians_array.shape[1]
                if descriptor.operation_index is not None:
                    job = self.job_specs[str(descriptor.job_id)]
                    operation_count = len(job.operations)
                    context[index, offset] = (
                        int(descriptor.operation_index) / operation_count
                    )
                    context[index, offset + 1] = (
                        operation_count - int(descriptor.operation_index)
                    ) / operation_count
            observation["action_context"] = context
        return observation

    def _start_production(self, descriptor: ActionDescriptor) -> None:
        job_id = str(descriptor.job_id)
        machine_id = str(descriptor.machine_id)
        operation_index = int(descriptor.operation_index)
        alternative = self._alternative(descriptor)
        machine = self.machine_specs[machine_id]
        state = self.machines[machine_id]
        self.jobs[job_id].in_process = True
        state.status = "processing"
        rate = machine.alpha * alternative.load_factor
        increment = rate * alternative.processing_time
        attempt_key = (job_id, operation_index, machine_id)
        attempt_number = self.attempt_counts.get(attempt_key, 0)
        self.attempt_counts[attempt_key] = attempt_number + 1
        shock_rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.root_seed,
                    self.job_indices[job_id],
                    operation_index,
                    self.machine_indices[machine_id],
                    attempt_number,
                    0x48545044,
                ]
            )
        )
        failure_age = sample_conditional_weibull_failure_age(
            shock_rng,
            effective_age=state.effective_age,
            weibull_eta=machine.weibull_eta,
            weibull_beta=machine.weibull_beta,
        )
        payload = {
            "machine_id": machine_id,
            "job_id": job_id,
            "operation_index": operation_index,
            "operation_id": descriptor.operation_id,
            "start": self.now,
            "age_before": state.effective_age,
            "age_increment": increment,
        }
        if failure_age < state.effective_age + increment:
            payload["failure_age"] = failure_age
            self._push_event(
                self.now + (failure_age - state.effective_age) / rate,
                "machine_failure",
                payload,
            )
        else:
            self._push_event(
                self.now + alternative.processing_time,
                "operation_complete",
                payload,
            )

    def _start_maintenance(self, descriptor: ActionDescriptor) -> None:
        machine_id = str(descriptor.machine_id)
        technician_id = str(descriptor.technician_id)
        technician = self.technician_specs[technician_id]
        machine_state = self.machines[machine_id]
        self.technicians[technician_id].status = "busy"
        machine_state.status = "maintenance"
        duration = technician.duration(machine_id, descriptor.kind)
        self._push_event(
            self.now + duration,
            "maintenance_complete",
            {
                "machine_id": machine_id,
                "technician_id": technician_id,
                "maintenance_kind": descriptor.kind,
                "start": self.now,
                "age_before": machine_state.effective_age,
                "restoration": technician.restoration(machine_id, descriptor.kind),
            },
        )

    def _advance(self, components: dict[str, float]) -> None:
        next_time = self.events[0].time
        delta = next_time - self.now
        if not self._all_jobs_complete():
            components["makespan"] -= self.env_config.makespan_weight * delta
        blocked = sum(
            state.status in {"maintenance", "waiting_corrective"}
            for state in self.machines.values()
        )
        downtime_cost = self.config.costs.downtime * blocked * delta
        components["downtime"] -= downtime_cost
        self.maintenance_wait_time += delta * sum(
            state.status == "waiting_corrective" for state in self.machines.values()
        )
        self.now = next_time
        simultaneous: list[Event] = []
        while self.events and abs(self.events[0].time - self.now) <= TIME_TOLERANCE:
            simultaneous.append(heapq.heappop(self.events))
        for event in sorted(simultaneous, key=lambda item: item.sequence):
            payload = event.payload
            machine_id = payload["machine_id"]
            machine_state = self.machines[machine_id]
            if event.kind == "operation_complete":
                machine_state.effective_age = (
                    payload["age_before"] + payload["age_increment"]
                )
                machine_state.status = "idle"
                machine_state.preventive_armed = True
                job_state = self.jobs[payload["job_id"]]
                job_state.in_process = False
                job_state.next_operation += 1
                if job_state.next_operation == len(
                    self.job_specs[payload["job_id"]].operations
                ):
                    job_state.completion_time = self.now
                    self.production_makespan = max(self.production_makespan, self.now)
                    tardiness = max(
                        0.0,
                        self.now - self.job_specs[payload["job_id"]].due_date,
                    )
                    components["tardiness"] -= self.config.costs.tardiness * tardiness
                self.tasks.append(
                    TaskRecord(
                        kind="production",
                        machine_id=machine_id,
                        start=payload["start"],
                        end=self.now,
                        job_id=payload["job_id"],
                        operation_id=payload["operation_id"],
                        completed=True,
                        effective_age_before=payload["age_before"],
                        effective_age_after=machine_state.effective_age,
                    )
                )
            elif event.kind == "machine_failure":
                self.failures += 1
                self.failed_processing_time += self.now - payload["start"]
                machine_state.effective_age = payload["failure_age"]
                machine_state.status = "waiting_corrective"
                machine_state.preventive_armed = False
                self.jobs[payload["job_id"]].in_process = False
                self.tasks.append(
                    TaskRecord(
                        kind="production",
                        machine_id=machine_id,
                        start=payload["start"],
                        end=self.now,
                        job_id=payload["job_id"],
                        operation_id=payload["operation_id"],
                        completed=False,
                        effective_age_before=payload["age_before"],
                        effective_age_after=machine_state.effective_age,
                    )
                )
            elif event.kind == "maintenance_complete":
                duration = self.now - payload["start"]
                self.maintenance_time += duration
                age_after = max(
                    0.0,
                    payload["age_before"] * (1.0 - payload["restoration"]),
                )
                machine_state.effective_age = age_after
                machine_state.status = "idle"
                machine_state.preventive_armed = False
                self.technicians[payload["technician_id"]].status = "idle"
                self.tasks.append(
                    TaskRecord(
                        kind="maintenance",
                        machine_id=machine_id,
                        start=payload["start"],
                        end=self.now,
                        technician_id=payload["technician_id"],
                        maintenance_kind=payload["maintenance_kind"],
                        completed=True,
                        effective_age_before=payload["age_before"],
                        effective_age_after=age_after,
                    )
                )
            else:
                raise RuntimeError(f"Unknown event kind: {event.kind}")

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done; call reset().")
        self.decision_count += 1
        components = {key: 0.0 for key in self.cumulative_components}
        mask = self.action_mask()
        invalid = not self.action_space.contains(action) or not bool(mask[int(action)])
        if invalid:
            components["invalid"] = -self.env_config.invalid_action_penalty
        else:
            descriptor = self.actions[int(action)]
            if descriptor.kind == "advance":
                self._advance(components)
            elif descriptor.kind == "production":
                self._start_production(descriptor)
            elif descriptor.kind == "preventive":
                self.preventive_count += 1
                components["preventive"] -= self.config.costs.preventive
                self._start_maintenance(descriptor)
            elif descriptor.kind == "corrective":
                self.corrective_count += 1
                components["corrective"] -= self.config.costs.corrective
                self._start_maintenance(descriptor)
            else:
                raise RuntimeError(f"Unknown action kind: {descriptor.kind}")
        reward = float(sum(components.values()))
        self.cumulative_reward += reward
        for key, value in components.items():
            self.cumulative_components[key] += value
        terminated = self._terminal_ready()
        truncated = self.decision_count >= self.env_config.max_decisions and not terminated
        self._done = terminated or truncated
        observation = self._observation()
        info = self._info(components, invalid_action=invalid)
        return observation, reward, terminated, truncated, info

    def metrics(self) -> dict[str, float | int]:
        completion_times = {
            job_id: float(state.completion_time)
            for job_id, state in self.jobs.items()
            if state.completion_time is not None
        }
        total_tardiness = sum(
            max(0.0, completion_times[job.job_id] - job.due_date)
            for job in self.config.jobs
            if job.job_id in completion_times
        )
        downtime = self.maintenance_time + self.maintenance_wait_time
        total_cost = (
            self.config.costs.tardiness * total_tardiness
            + self.config.costs.preventive * self.preventive_count
            + self.config.costs.corrective * self.corrective_count
            + self.config.costs.downtime * downtime
        )
        objective = (
            self.env_config.makespan_weight * self.production_makespan
            + total_cost
            - self.cumulative_components["invalid"]
        )
        return {
            "makespan": self.production_makespan,
            "schedule_end": self.now,
            "total_tardiness": total_tardiness,
            "failures": self.failures,
            "preventive_maintenance": self.preventive_count,
            "corrective_maintenance": self.corrective_count,
            "maintenance_time": self.maintenance_time,
            "maintenance_wait_time": self.maintenance_wait_time,
            "failed_processing_time": self.failed_processing_time,
            "total_cost": total_cost,
            "objective": objective,
            "decision_count": self.decision_count,
        }

    def result(self, policy_name: str) -> SimulationResult:
        if not self._terminal_ready():
            raise RuntimeError("A complete result is available only after termination.")
        completion_times = {
            job_id: float(state.completion_time)
            for job_id, state in self.jobs.items()
            if state.completion_time is not None
        }
        result = SimulationResult(
            instance_name=self.config.instance_name,
            policy=policy_name,
            seed=self.root_seed,
            metrics=self.metrics(),
            tasks=tuple(sorted(self.tasks, key=lambda item: (item.start, item.end))),
            final_machine_ages={
                key: state.effective_age for key, state in self.machines.items()
            },
            job_completion_times=completion_times,
        )
        audit_result(self.config, result)
        return result

    def _info(
        self, reward_components: dict[str, float], *, invalid_action: bool
    ) -> dict[str, Any]:
        mask = self.action_mask()
        return {
            "time": self.now,
            "action_mask": mask.copy(),
            "feasible_action_count": int(mask.sum()),
            "reward_components": dict(reward_components),
            "cumulative_reward_components": dict(self.cumulative_components),
            "cumulative_reward": self.cumulative_reward,
            "invalid_action": invalid_action,
            "metrics": self.metrics(),
        }

    def render(self) -> str | None:
        text = (
            f"t={self.now:.3f} jobs="
            f"{sum(state.completion_time is not None for state in self.jobs.values())}/"
            f"{len(self.jobs)} failures={self.failures} "
            f"feasible={int(self.action_mask().sum())}"
        )
        return text if self.render_mode == "ansi" else None

    def close(self) -> None:
        return None
