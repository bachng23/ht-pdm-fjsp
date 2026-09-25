"""Physically consistent simulator for passive shared technicians.

Machines are decision makers. Technicians are fixed, passive resources whose
eligibility and service duration depend on the machine-technician pair.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Iterable


SIMULATOR_VERSION = "passive_technician_v2"
OBJECTIVE_VERSION = "downtime_queue_terminal_v2"
OBSERVATION_VERSION = "machine_mode_resource_v2"


class MachineMode(IntEnum):
    OPERATING = 0
    QUEUED_PM = 1
    QUEUED_CM = 2
    IN_SERVICE_PM = 3
    IN_SERVICE_CM = 4
    FAILED = 5


@dataclass(frozen=True)
class PassiveTechnicianV2Config:
    machines: int = 3
    technicians: int = 2
    horizon: int = 24
    failure_age: int = 6
    failure_probability: float = 0.35
    max_age: int = 10
    service_time: tuple[tuple[int, ...], ...] = ((2, 4), (4, 2), (3, 3))
    eligibility: tuple[tuple[bool, ...], ...] = (
        (True, True),
        (True, True),
        (True, True),
    )
    initial_ages: tuple[int, ...] = ()
    preventive_cost: float = 1.0
    corrective_cost: float = 3.0
    downtime_cost: float = 4.0
    failure_cost: float = 15.0
    queue_waiting_cost: float = 0.5
    terminal_age_cost: float = 0.5
    terminal_unfinished_cost: float = 12.0

    @property
    def action_count(self) -> int:
        return self.technicians + 1

    @property
    def observation_dim(self) -> int:
        return 12 + 7 * self.technicians

    def validate(self) -> None:
        if self.machines < 1 or self.technicians < 1 or self.horizon < 1:
            raise ValueError("machines, technicians, and horizon must be positive")
        if self.failure_age < 1 or self.max_age < self.failure_age:
            raise ValueError("max_age must be at least failure_age >= 1")
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError("failure_probability must be in [0, 1]")
        if len(self.service_time) != self.machines or len(self.eligibility) != self.machines:
            raise ValueError("service_time and eligibility need one row per machine")
        if self.initial_ages and len(self.initial_ages) != self.machines:
            raise ValueError("initial_ages needs one entry per machine")
        if any(not 0 <= age <= self.max_age for age in self.initial_ages):
            raise ValueError("initial ages must be between zero and max_age")
        for machine in range(self.machines):
            if len(self.service_time[machine]) != self.technicians:
                raise ValueError("service_time needs one entry per technician")
            if len(self.eligibility[machine]) != self.technicians:
                raise ValueError("eligibility needs one entry per technician")
            if not any(self.eligibility[machine]):
                raise ValueError("every machine needs at least one eligible technician")
            for technician in range(self.technicians):
                if self.eligibility[machine][technician] and self.service_time[machine][technician] < 1:
                    raise ValueError("eligible service durations must be positive integers")
        costs = (
            self.preventive_cost,
            self.corrective_cost,
            self.downtime_cost,
            self.failure_cost,
            self.queue_waiting_cost,
            self.terminal_age_cost,
            self.terminal_unfinished_cost,
        )
        if any(cost < 0 for cost in costs):
            raise ValueError("all costs must be nonnegative")


@dataclass(frozen=True)
class PassiveTechnicianV2State:
    ages: tuple[int, ...]
    modes: tuple[int, ...]
    machine_technician: tuple[int, ...]
    service_remaining: tuple[int, ...]
    technician_machine: tuple[int, ...]
    queues: tuple[tuple[int, ...], ...]
    request_times: tuple[int, ...]


def keyed_failure_uniform(seed: int, machine: int, time: int) -> float:
    """Return a policy-order-independent failure shock in [0, 1)."""
    payload = f"passive-v2:{seed}:{machine}:{time}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(1 << 64)


class PassiveTechnicianV2Env:
    """Finite-horizon environment with explicit queue and service states."""

    METRIC_KEYS = (
        "objective",
        "failures",
        "jobs",
        "pm_starts",
        "cm_starts",
        "proposal_contention",
        "invalid_actions",
        "waiting",
        "downtime",
        "operating",
        "service",
        "terminal_cost",
    )

    def __init__(self, config: PassiveTechnicianV2Config, *, seed: int = 0) -> None:
        config.validate()
        self.config = config
        self.seed = int(seed)
        self.time = 0
        self.state = self.initial_state()
        self.metrics: dict[str, float] = {key: 0.0 for key in self.METRIC_KEYS}
        self.technician_busy_steps = [0] * config.technicians
        self.technician_starts = [0] * config.technicians

    def initial_state(self) -> PassiveTechnicianV2State:
        cfg = self.config
        ages = cfg.initial_ages if cfg.initial_ages else (0,) * cfg.machines
        return PassiveTechnicianV2State(
            ages=ages,
            modes=(int(MachineMode.OPERATING),) * cfg.machines,
            machine_technician=(-1,) * cfg.machines,
            service_remaining=(0,) * cfg.machines,
            technician_machine=(-1,) * cfg.technicians,
            queues=tuple(() for _ in range(cfg.technicians)),
            request_times=(-1,) * cfg.machines,
        )

    def reset(self) -> tuple[tuple[int, ...], ...]:
        self.time = 0
        self.state = self.initial_state()
        self.metrics = {key: 0.0 for key in self.METRIC_KEYS}
        self.technician_busy_steps = [0] * self.config.technicians
        self.technician_starts = [0] * self.config.technicians
        self.audit_state(self.state)
        return self.observations()

    def queued_technician(self, state: PassiveTechnicianV2State, machine: int) -> int:
        for technician, queue in enumerate(state.queues):
            if machine in queue:
                return technician
        return -1

    def technician_remaining(self, state: PassiveTechnicianV2State, technician: int) -> int:
        machine = state.technician_machine[technician]
        return state.service_remaining[machine] if machine >= 0 else 0

    def available(self, technician: int) -> bool:
        return self.state.technician_machine[technician] < 0

    def action_masks(
        self, state: PassiveTechnicianV2State | None = None
    ) -> tuple[tuple[bool, ...], ...]:
        state = self.state if state is None else state
        masks: list[tuple[bool, ...]] = []
        for machine, raw_mode in enumerate(state.modes):
            mode = MachineMode(raw_mode)
            can_request = mode in {MachineMode.OPERATING, MachineMode.FAILED}
            masks.append(
                (True,)
                + tuple(
                    bool(can_request and self.config.eligibility[machine][technician])
                    for technician in range(self.config.technicians)
                )
            )
        return tuple(masks)

    def observations(self) -> tuple[tuple[int, ...], ...]:
        state = self.state
        output: list[tuple[int, ...]] = []
        for machine, raw_mode in enumerate(state.modes):
            mode = MachineMode(raw_mode)
            queued_technician = self.queued_technician(state, machine)
            features = [
                self.time,
                self.config.horizon - self.time,
                state.ages[machine],
                *(int(mode == candidate) for candidate in MachineMode),
                queued_technician,
                state.machine_technician[machine],
                state.service_remaining[machine],
            ]
            for technician in range(self.config.technicians):
                queue = state.queues[technician]
                position = queue.index(machine) + 1 if machine in queue else 0
                features.extend(
                    [
                        int(self.config.eligibility[machine][technician]),
                        self.config.service_time[machine][technician],
                        int(state.technician_machine[technician] < 0),
                        self.technician_remaining(state, technician),
                        len(queue),
                        position,
                        int(state.technician_machine[technician] == machine),
                    ]
                )
            output.append(tuple(features))
        return tuple(output)

    def audit_state(self, state: PassiveTechnicianV2State) -> None:
        cfg = self.config
        machine_lengths = (
            len(state.ages),
            len(state.modes),
            len(state.machine_technician),
            len(state.service_remaining),
            len(state.request_times),
        )
        if any(length != cfg.machines for length in machine_lengths):
            raise AssertionError("machine state vectors have inconsistent lengths")
        if len(state.technician_machine) != cfg.technicians or len(state.queues) != cfg.technicians:
            raise AssertionError("technician state vectors have inconsistent lengths")
        flat_queue = [machine for queue in state.queues for machine in queue]
        if len(flat_queue) != len(set(flat_queue)):
            raise AssertionError("a machine appears in more than one queue")
        assigned = [machine for machine in state.technician_machine if machine >= 0]
        if len(assigned) != len(set(assigned)):
            raise AssertionError("a machine has more than one technician")
        for technician, machine in enumerate(state.technician_machine):
            if machine < 0:
                continue
            if not 0 <= machine < cfg.machines:
                raise AssertionError("technician references an invalid machine")
            if state.machine_technician[machine] != technician:
                raise AssertionError("machine-technician assignment is inconsistent")
            if not cfg.eligibility[machine][technician]:
                raise AssertionError("an ineligible technician is servicing a machine")
        for machine, raw_mode in enumerate(state.modes):
            mode = MachineMode(raw_mode)
            queue_technician = self.queued_technician(state, machine)
            service_technician = state.machine_technician[machine]
            remaining = state.service_remaining[machine]
            if not 0 <= state.ages[machine] <= cfg.max_age:
                raise AssertionError("machine age is outside the configured range")
            if mode in {MachineMode.QUEUED_PM, MachineMode.QUEUED_CM}:
                if queue_technician < 0 or service_technician >= 0 or remaining != 0:
                    raise AssertionError("queued machine state is inconsistent")
                if state.request_times[machine] < 0:
                    raise AssertionError("queued machine has no request time")
            elif mode in {MachineMode.IN_SERVICE_PM, MachineMode.IN_SERVICE_CM}:
                if queue_technician >= 0 or service_technician < 0 or remaining <= 0:
                    raise AssertionError("in-service machine state is inconsistent")
                if service_technician >= cfg.technicians:
                    raise AssertionError("machine references an invalid technician")
                if state.technician_machine[service_technician] != machine:
                    raise AssertionError("technician-machine assignment is inconsistent")
                if state.request_times[machine] < 0:
                    raise AssertionError("in-service machine has no request time")
            else:
                if queue_technician >= 0 or service_technician >= 0 or remaining != 0:
                    raise AssertionError("idle machine state is inconsistent")
                if state.request_times[machine] != -1:
                    raise AssertionError("idle machine retains a request time")
        for technician, queue in enumerate(state.queues):
            if any(not cfg.eligibility[machine][technician] for machine in queue):
                raise AssertionError("queue contains an ineligible assignment")

    def terminal_cost(self, state: PassiveTechnicianV2State) -> float:
        cfg = self.config
        cost = 0.0
        for machine, raw_mode in enumerate(state.modes):
            mode = MachineMode(raw_mode)
            if mode in {MachineMode.OPERATING, MachineMode.QUEUED_PM}:
                cost += cfg.terminal_age_cost * state.ages[machine]
            if mode != MachineMode.OPERATING:
                cost += cfg.terminal_unfinished_cost
        return cost

    def transition(
        self,
        state: PassiveTechnicianV2State,
        actions: tuple[int, ...],
        time: int,
        *,
        failure_events: tuple[bool, ...] | None = None,
    ) -> tuple[PassiveTechnicianV2State, float, dict[str, object]]:
        cfg = self.config
        self.audit_state(state)
        if len(actions) != cfg.machines:
            raise ValueError("one action is required per machine")
        masks = self.action_masks(state)
        valid_requests: dict[int, list[int]] = {technician: [] for technician in range(cfg.technicians)}
        invalid_actions = 0
        accepted_actions = [0] * cfg.machines
        for machine, action in enumerate(actions):
            if not 0 <= action < cfg.action_count:
                invalid_actions += 1
                continue
            if action == 0:
                continue
            if not masks[machine][action]:
                invalid_actions += 1
                continue
            accepted_actions[machine] = action
            valid_requests[action - 1].append(machine)
        contention = sum(max(0, len(machines) - 1) for machines in valid_requests.values())

        ages = list(state.ages)
        modes = list(state.modes)
        machine_technician = list(state.machine_technician)
        service_remaining = list(state.service_remaining)
        technician_machine = list(state.technician_machine)
        queues = [list(queue) for queue in state.queues]
        request_times = list(state.request_times)

        for technician, machines in valid_requests.items():
            ordered = sorted(machines, key=lambda machine: (machine - time) % cfg.machines)
            for machine in ordered:
                request_times[machine] = time
                modes[machine] = int(
                    MachineMode.QUEUED_CM
                    if MachineMode(state.modes[machine]) == MachineMode.FAILED
                    else MachineMode.QUEUED_PM
                )
                queues[technician].append(machine)

        starts: list[dict[str, int | str]] = []
        for technician in range(cfg.technicians):
            if technician_machine[technician] >= 0 or not queues[technician]:
                continue
            machine = queues[technician].pop(0)
            queued_mode = MachineMode(modes[machine])
            kind = "CM" if queued_mode == MachineMode.QUEUED_CM else "PM"
            modes[machine] = int(
                MachineMode.IN_SERVICE_CM if kind == "CM" else MachineMode.IN_SERVICE_PM
            )
            machine_technician[machine] = technician
            technician_machine[technician] = machine
            service_remaining[machine] = cfg.service_time[machine][technician]
            starts.append(
                {
                    "machine": machine,
                    "technician": technician,
                    "kind": kind,
                    "duration": service_remaining[machine],
                }
            )

        components = {
            "preventive": cfg.preventive_cost * sum(start["kind"] == "PM" for start in starts),
            "corrective": cfg.corrective_cost * sum(start["kind"] == "CM" for start in starts),
            "downtime": 0.0,
            "failure": 0.0,
            "queue_waiting": 0.0,
            "terminal": 0.0,
        }
        waiting = sum(len(queue) for queue in queues)
        components["queue_waiting"] = cfg.queue_waiting_cost * waiting
        downtime = sum(
            MachineMode(mode)
            in {
                MachineMode.QUEUED_CM,
                MachineMode.IN_SERVICE_PM,
                MachineMode.IN_SERVICE_CM,
                MachineMode.FAILED,
            }
            for mode in modes
        )
        operating = sum(
            MachineMode(mode) in {MachineMode.OPERATING, MachineMode.QUEUED_PM}
            for mode in modes
        )
        in_service = sum(
            MachineMode(mode) in {MachineMode.IN_SERVICE_PM, MachineMode.IN_SERVICE_CM}
            for mode in modes
        )
        components["downtime"] = cfg.downtime_cost * downtime

        for machine, raw_mode in enumerate(modes):
            if MachineMode(raw_mode) in {MachineMode.OPERATING, MachineMode.QUEUED_PM}:
                ages[machine] = min(cfg.max_age, ages[machine] + 1)

        eligible_to_fail = tuple(
            MachineMode(modes[machine]) in {MachineMode.OPERATING, MachineMode.QUEUED_PM}
            and ages[machine] >= cfg.failure_age
            for machine in range(cfg.machines)
        )
        if failure_events is None:
            failure_events = tuple(
                keyed_failure_uniform(self.seed, machine, time) < cfg.failure_probability
                for machine in range(cfg.machines)
            )
        if len(failure_events) != cfg.machines:
            raise ValueError("failure_events must have one value per machine")
        failures: list[int] = []
        for machine, event in enumerate(failure_events):
            if not (eligible_to_fail[machine] and event):
                continue
            failures.append(machine)
            modes[machine] = int(
                MachineMode.QUEUED_CM
                if MachineMode(modes[machine]) == MachineMode.QUEUED_PM
                else MachineMode.FAILED
            )
        components["failure"] = cfg.failure_cost * len(failures)

        completions: list[dict[str, int | str]] = []
        busy_technicians = [
            technician
            for technician, machine in enumerate(technician_machine)
            if machine >= 0
        ]
        for technician, machine in enumerate(technician_machine):
            if machine < 0:
                continue
            service_remaining[machine] -= 1
            if service_remaining[machine] > 0:
                continue
            kind = "CM" if MachineMode(modes[machine]) == MachineMode.IN_SERVICE_CM else "PM"
            completions.append({"machine": machine, "technician": technician, "kind": kind})
            ages[machine] = 0
            modes[machine] = int(MachineMode.OPERATING)
            machine_technician[machine] = -1
            service_remaining[machine] = 0
            technician_machine[technician] = -1
            request_times[machine] = -1

        next_state = PassiveTechnicianV2State(
            ages=tuple(ages),
            modes=tuple(modes),
            machine_technician=tuple(machine_technician),
            service_remaining=tuple(service_remaining),
            technician_machine=tuple(technician_machine),
            queues=tuple(tuple(queue) for queue in queues),
            request_times=tuple(request_times),
        )
        self.audit_state(next_state)
        if time + 1 >= cfg.horizon:
            components["terminal"] = self.terminal_cost(next_state)
        cost = float(sum(components.values()))
        info: dict[str, object] = {
            "objective": cost,
            "failures": len(failures),
            "failure_machines": failures,
            "failure_eligible": list(eligible_to_fail),
            "jobs": len(starts),
            "pm_starts": sum(start["kind"] == "PM" for start in starts),
            "cm_starts": sum(start["kind"] == "CM" for start in starts),
            "starts": starts,
            "completions": completions,
            "proposal_contention": contention,
            "invalid_actions": invalid_actions,
            "accepted_actions": accepted_actions,
            "waiting": waiting,
            "downtime": downtime,
            "operating": operating,
            "service": in_service,
            "busy_technicians": busy_technicians,
            "terminal_cost": components["terminal"],
            "cost_components": components,
            "invariant_violations": 0,
        }
        return next_state, cost, info

    def step(
        self, actions: Iterable[int]
    ) -> tuple[tuple[tuple[int, ...], ...], float, bool, dict[str, object]]:
        if self.time >= self.config.horizon:
            raise RuntimeError("episode is complete; call reset before stepping again")
        action_tuple = tuple(int(action) for action in actions)
        next_state, cost, info = self.transition(self.state, action_tuple, self.time)
        self.state = next_state
        for key in self.METRIC_KEYS:
            self.metrics[key] += float(info[key])
        for start in info["starts"]:  # type: ignore[union-attr]
            self.technician_starts[int(start["technician"])] += 1
        for technician in info["busy_technicians"]:  # type: ignore[union-attr]
            self.technician_busy_steps[int(technician)] += 1
        self.time += 1
        done = self.time >= self.config.horizon
        return self.observations(), -cost, done, info

    def state_payload(self, state: PassiveTechnicianV2State | None = None) -> dict[str, object]:
        state = self.state if state is None else state
        return {
            "ages": list(state.ages),
            "modes": [MachineMode(mode).name for mode in state.modes],
            "machine_technician": list(state.machine_technician),
            "service_remaining": list(state.service_remaining),
            "technician_machine": list(state.technician_machine),
            "queues": [list(queue) for queue in state.queues],
            "request_times": list(state.request_times),
        }

    def trace_digest(self, records: list[dict[str, object]]) -> str:
        encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def exact_expected_cost(config: PassiveTechnicianV2Config) -> float:
    """Solve a tiny v2 cell by exact finite-horizon dynamic programming."""
    config.validate()
    if config.machines > 2 or config.technicians > 2 or config.horizon > 6:
        raise ValueError("exact oracle is restricted to at most 2x2 and six steps")
    env = PassiveTechnicianV2Env(config)

    @lru_cache(maxsize=None)
    def solve(time: int, state: PassiveTechnicianV2State) -> float:
        if time >= config.horizon:
            return 0.0
        masks = env.action_masks(state)
        action_sets = [tuple(index for index, allowed in enumerate(mask) if allowed) for mask in masks]
        best = math.inf
        for actions in itertools.product(*action_sets):
            no_failure, _, no_failure_info = env.transition(
                state, tuple(actions), time, failure_events=(False,) * config.machines
            )
            del no_failure
            candidates = [
                machine
                for machine, eligible in enumerate(no_failure_info["failure_eligible"])
                if eligible
            ]
            expected = 0.0
            for outcomes in itertools.product((False, True), repeat=len(candidates)):
                events = [False] * config.machines
                probability = 1.0
                for machine, event in zip(candidates, outcomes):
                    events[machine] = event
                    probability *= (
                        config.failure_probability if event else 1.0 - config.failure_probability
                    )
                next_state, cost, _ = env.transition(
                    state, tuple(actions), time, failure_events=tuple(events)
                )
                expected += probability * (cost + solve(time + 1, next_state))
            best = min(best, expected)
        return best

    return solve(0, env.initial_state())
