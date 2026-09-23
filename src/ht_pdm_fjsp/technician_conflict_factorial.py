"""Screen literature-derived mechanisms for learnable technician contention."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
import platform
import sys
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from statistics import NormalDist, fmean, pstdev, stdev
from typing import Any, Iterable, Mapping

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig, Technician
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
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


FACTORS = ("Q", "W", "O", "A", "D")
POLICIES = ("independent_greedy", "conflict_aware_matcher")
DURATION_MULTIPLIER = 2.0
WINDOW_RELEASES: Mapping[str, tuple[float, float]] = {
    "M1": (12.0, 36.0),
    "M2": (13.0, 37.0),
    "M3": (14.0, 38.0),
    "M4": (12.5, 36.5),
    "M5": (13.5, 37.5),
    "M6": (14.5, 38.5),
}
WINDOW_WIDTH = 8.0
UNAVAILABLE_INTERVALS: Mapping[str, tuple[tuple[float, float], ...]] = {
    "T1": ((18.0, 24.0), (42.0, 48.0)),
    "T2": ((26.0, 32.0), (50.0, 56.0)),
}


@dataclass(frozen=True)
class FactorSpec:
    request_persistence: bool
    maintenance_windows: bool
    skill_overlap: bool
    availability_calendar: bool
    duration_pressure: bool

    @property
    def bits(self) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in (
                self.request_persistence,
                self.maintenance_windows,
                self.skill_overlap,
                self.availability_calendar,
                self.duration_pressure,
            )
        )

    @property
    def name(self) -> str:
        return "_".join(
            f"{factor.lower()}{value}" for factor, value in zip(FACTORS, self.bits)
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "request_persistence": self.request_persistence,
            "maintenance_windows": self.maintenance_windows,
            "skill_overlap": self.skill_overlap,
            "availability_calendar": self.availability_calendar,
            "duration_pressure": self.duration_pressure,
        }


@dataclass
class MaintenanceWindow:
    machine_id: str
    index: int
    release: float
    deadline: float
    completed: bool = False
    missed: bool = False


@dataclass(frozen=True)
class MaintenanceDemand:
    machine_id: str
    kind: str
    created_at: float
    window_index: int | None = None

    @property
    def key(self) -> tuple[str, str, int | None]:
        return self.machine_id, self.kind, self.window_index


@dataclass(frozen=True)
class Proposal:
    machine_id: str
    kind: str
    job_id: str | None = None
    operation_index: int | None = None
    operation_id: str | None = None
    processing_time: float | None = None
    load_factor: float | None = None
    technician_id: str | None = None
    demand: MaintenanceDemand | None = None


def condition_specs() -> tuple[FactorSpec, ...]:
    return tuple(FactorSpec(*(bool(value) for value in bits)) for bits in itertools.product((0, 1), repeat=5))


def default_seeds(profile: str) -> tuple[int, ...]:
    if profile == "smoke":
        return tuple(range(61_700, 61_703))
    if profile == "full":
        return tuple(range(61_800, 61_900))
    raise ValueError(f"Unknown profile: {profile}")


def _fastest_base(base: BenchmarkConfig, machine_id: str, kind: str) -> tuple[float, float]:
    eligible = [item for item in base.technicians if machine_id in item.eligible_machines]
    chosen = min(
        eligible,
        key=lambda item: (item.duration(machine_id, kind), item.technician_id),
    )
    return chosen.duration(machine_id, kind), chosen.restoration(machine_id, kind)


def build_condition_config(base: BenchmarkConfig, spec: FactorSpec) -> BenchmarkConfig:
    multiplier = DURATION_MULTIPLIER if spec.duration_pressure else 1.0
    if not spec.skill_overlap:
        technicians = tuple(
            replace(
                technician,
                preventive_duration={
                    machine: value * multiplier
                    for machine, value in technician.preventive_duration.items()
                },
                corrective_duration={
                    machine: value * multiplier
                    for machine, value in technician.corrective_duration.items()
                },
            )
            for technician in base.technicians
        )
    else:
        machines = tuple(machine.machine_id for machine in base.machines)
        output: list[Technician] = []
        for technician_index, technician in enumerate(base.technicians):
            preferred = set(machines[:3] if technician_index == 0 else machines[3:])
            preventive_duration: dict[str, float] = {}
            corrective_duration: dict[str, float] = {}
            preventive_restoration: dict[str, float] = {}
            corrective_restoration: dict[str, float] = {}
            for machine_id in machines:
                pm_duration, pm_restoration = _fastest_base(base, machine_id, "preventive")
                cm_duration, cm_restoration = _fastest_base(base, machine_id, "corrective")
                penalty = 1.0 if machine_id in preferred else 1.25
                if machine_id in technician.eligible_machines:
                    pm_duration = technician.duration(machine_id, "preventive")
                    cm_duration = technician.duration(machine_id, "corrective")
                    pm_restoration = technician.restoration(machine_id, "preventive")
                    cm_restoration = technician.restoration(machine_id, "corrective")
                preventive_duration[machine_id] = pm_duration * penalty * multiplier
                corrective_duration[machine_id] = cm_duration * penalty * multiplier
                preventive_restoration[machine_id] = pm_restoration
                corrective_restoration[machine_id] = cm_restoration
            output.append(
                Technician(
                    technician_id=technician.technician_id,
                    eligible_machines=machines,
                    preventive_duration=preventive_duration,
                    corrective_duration=corrective_duration,
                    preventive_restoration=preventive_restoration,
                    corrective_restoration=corrective_restoration,
                )
            )
        technicians = tuple(output)
    config = replace(base, technicians=technicians)
    config.validate()
    return config


def _is_on_shift(technician_id: str, now: float, enabled: bool) -> bool:
    if not enabled:
        return True
    return not any(
        start - TIME_TOLERANCE <= now < end - TIME_TOLERANCE
        for start, end in UNAVAILABLE_INTERVALS.get(technician_id, ())
    )


def _t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("Paired intervals require at least two observations")
    z = NormalDist().inv_cdf(0.975)
    df = float(degrees_of_freedom)
    return z + (z**3 + z) / (4.0 * df) + (
        5.0 * z**5 + 16.0 * z**3 + 3.0 * z
    ) / (96.0 * df**2)


def _describe(values: Iterable[float]) -> dict[str, float]:
    data = list(values)
    return {
        "mean": fmean(data),
        "std": pstdev(data),
        "min": min(data),
        "max": max(data),
    }


class LiteratureFactorialSimulator:
    """Small event-driven simulator used only for the preregistered screen."""

    def __init__(self, config: BenchmarkConfig, spec: FactorSpec, policy: str):
        if policy not in POLICIES:
            raise ValueError(f"Unknown assignment policy: {policy}")
        self.config = config
        self.spec = spec
        self.policy = policy
        self.machine_specs = {item.machine_id: item for item in config.machines}
        self.technician_specs = {item.technician_id: item for item in config.technicians}
        self.job_specs = {item.job_id: item for item in config.jobs}
        self.machine_indices = {item.machine_id: index for index, item in enumerate(config.machines)}
        self.job_indices = {item.job_id: index for index, item in enumerate(config.jobs)}

    def _qualified(self, demand: MaintenanceDemand) -> tuple[str, ...]:
        return tuple(
            technician.technician_id
            for technician in self.config.technicians
            if demand.machine_id in technician.eligible_machines
        )

    def _optimal_assignment(
        self,
        demands: list[MaintenanceDemand],
        serviceable: Mapping[str, bool],
    ) -> dict[int, str]:
        best_key: tuple[Any, ...] | None = None
        best: dict[int, str] = {}

        def visit(index: int, used: set[str], chosen: dict[int, str], duration: float) -> None:
            nonlocal best_key, best
            if index == len(demands):
                signature = tuple(chosen.get(i, "~") for i in range(len(demands)))
                key = (-len(chosen), duration, signature)
                if best_key is None or key < best_key:
                    best_key = key
                    best = dict(chosen)
                return
            visit(index + 1, used, chosen, duration)
            demand = demands[index]
            for technician_id in self._qualified(demand):
                if technician_id in used or not serviceable[technician_id]:
                    continue
                technician = self.technician_specs[technician_id]
                chosen[index] = technician_id
                used.add(technician_id)
                visit(
                    index + 1,
                    used,
                    chosen,
                    duration + technician.duration(demand.machine_id, demand.kind),
                )
                used.remove(technician_id)
                del chosen[index]

        visit(0, set(), {}, 0.0)
        return best

    def run(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        jobs = {job.job_id: JobState() for job in self.config.jobs}
        machines = {
            machine.machine_id: MachineState(
                status="idle", effective_age=machine.initial_effective_age
            )
            for machine in self.config.machines
        }
        technicians = {
            technician.technician_id: TechnicianState()
            for technician in self.config.technicians
        }
        events: list[Event] = []
        tasks: list[TaskRecord] = []
        pending: dict[str, MaintenanceDemand] = {}
        windows = [
            MaintenanceWindow(machine_id, index, release, release + WINDOW_WIDTH)
            for machine_id, releases in WINDOW_RELEASES.items()
            for index, release in enumerate(releases)
        ] if self.spec.maintenance_windows else []
        now = 0.0
        event_sequence = 0
        processed_events = 0
        joint_step = 0
        failures = 0
        preventive_count = 0
        corrective_count = 0
        failed_processing_time = 0.0
        maintenance_time = 0.0
        preventive_wait_time = 0.0
        failure_wait_time = 0.0
        attempt_counts: dict[tuple[str, int, str], int] = {}
        previous_request_keys: set[tuple[str, str, int | None]] = set()
        totals = {
            "joint_steps": 0,
            "proposals": 0,
            "accepted": 0,
            "rejected": 0,
            "production_proposals": 0,
            "maintenance_proposals": 0,
            "production_conflicts": 0,
            "technician_conflicts": 0,
            "technician_conflict_epochs": 0,
            "maintenance_request_onsets": 0,
            "contention_opportunity_epochs": 0,
            "contention_opportunity_units": 0,
            "avoidable_conflict_units": 0,
            "capacity_unavoidable_wait_units": 0,
            "optimal_serviceable_assignments": 0,
            "busy_or_offshift_claims": 0,
            "masked_busy_or_offshift_requests": 0,
            "persistent_queued_request_epochs": 0,
            "window_deadline_misses": 0,
            "invalid_executions": 0,
            "duplicate_operation_executions": 0,
            "duplicate_technician_executions": 0,
        }

        def push_event(time: float, kind: str, payload: dict[str, Any]) -> None:
            nonlocal event_sequence
            event_sequence += 1
            heapq.heappush(events, Event(float(time), event_sequence, kind, payload))

        if self.spec.maintenance_windows:
            for window in windows:
                push_event(window.release, "window_boundary", {})
                push_event(window.deadline, "window_deadline", {"machine_id": window.machine_id, "index": window.index})
        if self.spec.availability_calendar:
            for intervals in UNAVAILABLE_INTERVALS.values():
                for start, end in intervals:
                    push_event(start, "calendar_boundary", {})
                    push_event(end, "calendar_boundary", {})

        def all_jobs_complete() -> bool:
            return all(
                state.next_operation == len(self.job_specs[job_id].operations)
                for job_id, state in jobs.items()
            )

        def terminal() -> bool:
            return all_jobs_complete() and all(
                state.status == "idle" for state in machines.values()
            )

        def ready_candidates(machine_id: str) -> list[Proposal]:
            output: list[Proposal] = []
            for job in self.config.jobs:
                state = jobs[job.job_id]
                if state.in_process or state.next_operation >= len(job.operations):
                    continue
                operation = job.operations[state.next_operation]
                for alternative in operation.alternatives:
                    if alternative.machine_id == machine_id:
                        output.append(
                            Proposal(
                                machine_id=machine_id,
                                kind="production",
                                job_id=job.job_id,
                                operation_index=state.next_operation,
                                operation_id=operation.operation_id,
                                processing_time=alternative.processing_time,
                                load_factor=alternative.load_factor,
                            )
                        )
            return output

        def production_choice(machine_id: str) -> Proposal | None:
            candidates = ready_candidates(machine_id)
            return min(
                candidates,
                key=lambda item: (
                    float(item.processing_time),
                    str(item.job_id),
                    int(item.operation_index),
                ),
                default=None,
            )

        def window_demand(machine_id: str) -> MaintenanceDemand | None:
            active = [
                window for window in windows
                if window.machine_id == machine_id
                and not window.completed
                and window.release <= now + TIME_TOLERANCE
            ]
            if not active:
                return None
            window = min(active, key=lambda item: (item.deadline, item.index))
            return MaintenanceDemand(machine_id, "preventive", window.release, window.index)

        def threshold_demand(machine_id: str) -> MaintenanceDemand | None:
            choice = production_choice(machine_id)
            state = machines[machine_id]
            if choice is None or state.effective_age <= 0.0 or not state.preventive_armed:
                return None
            machine = self.machine_specs[machine_id]
            increment = machine.alpha * float(choice.processing_time) * float(choice.load_factor)
            probability = conditional_weibull_failure_probability(
                state.effective_age, increment, machine.weibull_eta, machine.weibull_beta
            )
            if probability < self.config.preventive_probability_threshold:
                return None
            return MaintenanceDemand(machine_id, "preventive", now)

        def current_demands() -> list[MaintenanceDemand]:
            if all_jobs_complete():
                pending.clear()
                return []
            output: list[MaintenanceDemand] = []
            for machine_id in sorted(machines):
                state = machines[machine_id]
                if state.status == "waiting_corrective":
                    demand = pending.get(machine_id) or MaintenanceDemand(machine_id, "corrective", now)
                elif state.status == "idle" and state.effective_age > 0.0:
                    demand = pending.get(machine_id)
                    if demand is None:
                        demand = window_demand(machine_id) if self.spec.maintenance_windows else threshold_demand(machine_id)
                else:
                    demand = None
                if demand is not None:
                    if self.spec.request_persistence:
                        pending[machine_id] = demand
                    output.append(demand)
            return output

        def serviceable_map() -> dict[str, bool]:
            return {
                technician_id: state.status == "idle"
                and _is_on_shift(technician_id, now, self.spec.availability_calendar)
                for technician_id, state in technicians.items()
            }

        def start_production(proposal: Proposal) -> None:
            machine_id = proposal.machine_id
            job_id = str(proposal.job_id)
            operation_index = int(proposal.operation_index)
            state = machines[machine_id]
            job_state = jobs[job_id]
            job_state.in_process = True
            state.status = "processing"
            machine = self.machine_specs[machine_id]
            rate = machine.alpha * float(proposal.load_factor)
            increment = rate * float(proposal.processing_time)
            attempt_key = (job_id, operation_index, machine_id)
            attempt = attempt_counts.get(attempt_key, 0)
            attempt_counts[attempt_key] = attempt + 1
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, self.job_indices[job_id], operation_index, self.machine_indices[machine_id], attempt, 0x48545044]
                )
            )
            failure_age = sample_conditional_weibull_failure_age(
                rng,
                effective_age=state.effective_age,
                weibull_eta=machine.weibull_eta,
                weibull_beta=machine.weibull_beta,
            )
            payload = {
                "machine_id": machine_id,
                "job_id": job_id,
                "operation_index": operation_index,
                "operation_id": proposal.operation_id,
                "start": now,
                "age_before": state.effective_age,
                "age_increment": increment,
            }
            if failure_age < state.effective_age + increment:
                payload["failure_age"] = failure_age
                push_event(now + (failure_age - state.effective_age) / rate, "machine_failure", payload)
            else:
                push_event(now + float(proposal.processing_time), "operation_complete", payload)

        def start_maintenance(proposal: Proposal) -> None:
            nonlocal preventive_count, corrective_count
            demand = proposal.demand
            if demand is None or proposal.technician_id is None:
                raise AssertionError("Maintenance proposal lacks demand or technician")
            technician_id = proposal.technician_id
            if not serviceable_map()[technician_id]:
                totals["invalid_executions"] += 1
                raise AssertionError("Attempted to execute with unavailable technician")
            machine_id = proposal.machine_id
            technician = self.technician_specs[technician_id]
            machines[machine_id].status = "maintenance"
            technicians[technician_id].status = "busy"
            if demand.kind == "preventive":
                preventive_count += 1
                if demand.window_index is not None:
                    next(
                        window for window in windows
                        if window.machine_id == machine_id and window.index == demand.window_index
                    ).completed = True
            else:
                corrective_count += 1
            pending.pop(machine_id, None)
            push_event(
                now + technician.duration(machine_id, demand.kind),
                "maintenance_complete",
                {
                    "machine_id": machine_id,
                    "technician_id": technician_id,
                    "maintenance_kind": demand.kind,
                    "start": now,
                    "age_before": machines[machine_id].effective_age,
                    "restoration": technician.restoration(machine_id, demand.kind),
                },
            )

        while not terminal():
            if joint_step >= self.config.max_events:
                raise RuntimeError("max_events exceeded; possible unstable factorial condition")
            demands = current_demands()
            request_keys = {demand.key for demand in demands}
            totals["maintenance_request_onsets"] += len(request_keys - previous_request_keys)
            previous_request_keys = request_keys
            serviceable = serviceable_map()
            optimal = self._optimal_assignment(demands, serviceable)
            totals["optimal_serviceable_assignments"] += len(optimal)
            totals["capacity_unavoidable_wait_units"] += max(0, len(demands) - len(optimal))
            shared_pairs = sum(
                bool(set(self._qualified(left)) & set(self._qualified(right)))
                for index, left in enumerate(demands)
                for right in demands[index + 1 :]
            )
            if shared_pairs:
                totals["contention_opportunity_epochs"] += 1
                totals["contention_opportunity_units"] += shared_pairs

            maintenance_by_machine = {demand.machine_id: demand for demand in demands}
            proposals: list[Proposal] = []
            suppressed = 0
            if self.policy == "conflict_aware_matcher":
                for index, technician_id in optimal.items():
                    demand = demands[index]
                    proposals.append(
                        Proposal(demand.machine_id, demand.kind, technician_id=technician_id, demand=demand)
                    )
            else:
                for demand in demands:
                    candidates = list(self._qualified(demand))
                    if not self.spec.request_persistence:
                        candidates = [item for item in candidates if serviceable[item]]
                    if not candidates:
                        suppressed += 1
                        continue
                    technician_id = min(
                        candidates,
                        key=lambda item: (
                            self.technician_specs[item].duration(demand.machine_id, demand.kind),
                            item,
                        ),
                    )
                    proposals.append(
                        Proposal(demand.machine_id, demand.kind, technician_id=technician_id, demand=demand)
                    )

            maintenance_proposal_machines = {proposal.machine_id for proposal in proposals}
            for machine_id in sorted(machines):
                if machines[machine_id].status != "idle":
                    continue
                if machine_id in maintenance_proposal_machines:
                    continue
                if machine_id in maintenance_by_machine and self.spec.request_persistence:
                    continue
                choice = production_choice(machine_id)
                if choice is not None:
                    proposals.append(choice)

            totals["masked_busy_or_offshift_requests"] += suppressed
            totals["persistent_queued_request_epochs"] += sum(
                self.spec.request_persistence
                and not any(serviceable[item] for item in self._qualified(demand))
                for demand in demands
            )
            totals["joint_steps"] += 1
            totals["proposals"] += len(proposals)
            totals["production_proposals"] += sum(item.kind == "production" for item in proposals)
            totals["maintenance_proposals"] += sum(item.kind != "production" for item in proposals)

            priority = joint_step % len(self.config.machines)
            ordered = sorted(
                proposals,
                key=lambda item: (self.machine_indices[item.machine_id] - priority) % len(self.config.machines),
            )
            claimed_operations: set[tuple[str, int]] = set()
            claimed_technicians: set[str] = set()
            accepted: list[Proposal] = []
            technician_conflicts = 0
            for proposal in ordered:
                if proposal.kind == "production":
                    key = (str(proposal.job_id), int(proposal.operation_index))
                    if key in claimed_operations:
                        totals["production_conflicts"] += 1
                        continue
                    claimed_operations.add(key)
                    accepted.append(proposal)
                    continue
                technician_id = str(proposal.technician_id)
                if technician_id in claimed_technicians:
                    technician_conflicts += 1
                    continue
                claimed_technicians.add(technician_id)
                if not serviceable[technician_id]:
                    totals["busy_or_offshift_claims"] += 1
                    continue
                accepted.append(proposal)

            totals["technician_conflicts"] += technician_conflicts
            totals["technician_conflict_epochs"] += int(technician_conflicts > 0)
            maintenance_accepted = sum(item.kind != "production" for item in accepted)
            if self.policy == "independent_greedy":
                totals["avoidable_conflict_units"] += min(
                    technician_conflicts,
                    max(0, len(optimal) - maintenance_accepted),
                )
            totals["accepted"] += len(accepted)
            totals["rejected"] += len(proposals) - len(accepted)

            accepted_operations = [
                (str(item.job_id), int(item.operation_index))
                for item in accepted if item.kind == "production"
            ]
            accepted_technicians = [
                str(item.technician_id) for item in accepted if item.kind != "production"
            ]
            totals["duplicate_operation_executions"] += len(accepted_operations) - len(set(accepted_operations))
            totals["duplicate_technician_executions"] += len(accepted_technicians) - len(set(accepted_technicians))
            if totals["duplicate_operation_executions"] or totals["duplicate_technician_executions"]:
                raise AssertionError("Resolver accepted a conflicting action set")

            for proposal in accepted:
                if proposal.kind == "production":
                    start_production(proposal)
                else:
                    start_maintenance(proposal)

            joint_step += 1
            if terminal():
                break
            if not events:
                raise RuntimeError("Factorial simulator deadlock: no event can advance time")
            next_time = events[0].time
            delta = next_time - now
            if delta < -TIME_TOLERANCE:
                raise AssertionError("Event time moved backwards")
            waiting_demands = current_demands()
            preventive_wait_time += delta * sum(item.kind == "preventive" for item in waiting_demands)
            failure_wait_time += delta * sum(item.kind == "corrective" for item in waiting_demands)
            now = next_time
            simultaneous: list[Event] = []
            while events and abs(events[0].time - now) <= TIME_TOLERANCE:
                simultaneous.append(heapq.heappop(events))
            for event in sorted(simultaneous, key=lambda item: item.sequence):
                processed_events += 1
                payload = event.payload
                if event.kind in {"calendar_boundary", "window_boundary"}:
                    continue
                if event.kind == "window_deadline":
                    window = next(
                        item for item in windows
                        if item.machine_id == payload["machine_id"] and item.index == payload["index"]
                    )
                    if not window.completed and not window.missed:
                        window.missed = True
                        totals["window_deadline_misses"] += 1
                    continue
                machine_id = str(payload["machine_id"])
                machine_state = machines[machine_id]
                if event.kind == "operation_complete":
                    machine_state.effective_age = payload["age_before"] + payload["age_increment"]
                    machine_state.status = "idle"
                    machine_state.preventive_armed = True
                    job_state = jobs[str(payload["job_id"])]
                    job_state.in_process = False
                    job_state.next_operation += 1
                    if job_state.next_operation == len(self.job_specs[str(payload["job_id"])].operations):
                        job_state.completion_time = now
                    tasks.append(
                        TaskRecord(
                            kind="production", machine_id=machine_id,
                            start=payload["start"], end=now,
                            job_id=payload["job_id"], operation_id=payload["operation_id"],
                            completed=True, effective_age_before=payload["age_before"],
                            effective_age_after=machine_state.effective_age,
                        )
                    )
                elif event.kind == "machine_failure":
                    failures += 1
                    failed_processing_time += now - payload["start"]
                    machine_state.effective_age = payload["failure_age"]
                    machine_state.status = "waiting_corrective"
                    machine_state.preventive_armed = False
                    jobs[str(payload["job_id"])].in_process = False
                    demand = MaintenanceDemand(machine_id, "corrective", now)
                    if self.spec.request_persistence:
                        pending[machine_id] = demand
                    tasks.append(
                        TaskRecord(
                            kind="production", machine_id=machine_id,
                            start=payload["start"], end=now,
                            job_id=payload["job_id"], operation_id=payload["operation_id"],
                            completed=False, effective_age_before=payload["age_before"],
                            effective_age_after=machine_state.effective_age,
                        )
                    )
                elif event.kind == "maintenance_complete":
                    duration = now - payload["start"]
                    maintenance_time += duration
                    age_after = max(0.0, payload["age_before"] * (1.0 - payload["restoration"]))
                    machine_state.effective_age = age_after
                    machine_state.status = "idle"
                    machine_state.preventive_armed = False
                    technicians[str(payload["technician_id"])].status = "idle"
                    tasks.append(
                        TaskRecord(
                            kind="maintenance", machine_id=machine_id,
                            start=payload["start"], end=now,
                            technician_id=payload["technician_id"],
                            maintenance_kind=payload["maintenance_kind"], completed=True,
                            effective_age_before=payload["age_before"], effective_age_after=age_after,
                        )
                    )
                else:
                    raise RuntimeError(f"Unknown event kind: {event.kind}")

        completion_times = {
            job_id: float(state.completion_time)
            for job_id, state in jobs.items() if state.completion_time is not None
        }
        makespan = max(completion_times.values())
        schedule_end = max((task.end for task in tasks), default=makespan)
        total_tardiness = sum(
            max(0.0, completion_times[job.job_id] - job.due_date)
            for job in self.config.jobs
        )
        maintenance_wait_time = preventive_wait_time + failure_wait_time
        total_cost = (
            self.config.costs.tardiness * total_tardiness
            + self.config.costs.preventive * preventive_count
            + self.config.costs.corrective * corrective_count
            + self.config.costs.downtime * (maintenance_time + maintenance_wait_time)
        )
        objective = makespan + total_cost
        utilization = maintenance_time / (schedule_end * len(technicians)) if schedule_end else 0.0
        result = SimulationResult(
            instance_name=self.config.instance_name,
            policy=self.policy,
            seed=seed,
            metrics={
                "makespan": makespan,
                "schedule_end": schedule_end,
                "total_tardiness": total_tardiness,
                "failures": failures,
                "preventive_maintenance": preventive_count,
                "corrective_maintenance": corrective_count,
                "maintenance_time": maintenance_time,
                "failure_wait_time": failure_wait_time,
                "preventive_wait_time": preventive_wait_time,
                "maintenance_wait_time": maintenance_wait_time,
                "failed_processing_time": failed_processing_time,
                "technician_utilization": utilization,
                "total_cost": total_cost,
                "objective": objective,
                "processed_events": processed_events,
            },
            tasks=tuple(sorted(tasks, key=lambda item: (item.start, item.end, item.kind))),
            final_machine_ages={key: state.effective_age for key, state in machines.items()},
            job_completion_times=completion_times,
        )
        audit_result(self.config, result)
        totals["feasibility_audit_passed"] = 1
        episode = {
            "condition": self.spec.name,
            **{factor: value for factor, value in zip(FACTORS, self.spec.bits)},
            "policy": self.policy,
            "seed": seed,
            **result.metrics,
            **totals,
            "technician_conflict_rate": (
                totals["technician_conflicts"] / totals["maintenance_proposals"]
                if totals["maintenance_proposals"] else 0.0
            ),
            "avoidable_conflict_fraction": (
                min(1.0, totals["avoidable_conflict_units"] / totals["technician_conflicts"])
                if totals["technician_conflicts"] else 0.0
            ),
            "rejection_rate": totals["rejected"] / totals["proposals"] if totals["proposals"] else 0.0,
        }
        coordination = {
            key: episode[key]
            for key in (
                "condition", *FACTORS, "policy", "seed", "joint_steps", "proposals",
                "accepted", "rejected", "production_proposals", "maintenance_proposals",
                "production_conflicts", "technician_conflicts", "technician_conflict_epochs",
                "maintenance_request_onsets", "contention_opportunity_epochs",
                "contention_opportunity_units", "avoidable_conflict_units",
                "capacity_unavoidable_wait_units", "optimal_serviceable_assignments",
                "busy_or_offshift_claims", "masked_busy_or_offshift_requests",
                "persistent_queued_request_epochs", "window_deadline_misses",
                "invalid_executions", "duplicate_operation_executions",
                "duplicate_technician_executions", "feasibility_audit_passed",
            )
        }
        return episode, coordination


SUMMARY_METRICS = (
    "technician_conflicts", "technician_conflict_rate", "avoidable_conflict_fraction",
    "contention_opportunity_epochs", "capacity_unavoidable_wait_units",
    "maintenance_wait_time", "window_deadline_misses", "production_conflicts",
    "rejection_rate", "objective", "makespan", "failures",
    "preventive_maintenance", "corrective_maintenance", "technician_utilization",
)


def _paired_policy_contrast(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    greedy = {int(row["seed"]): row for row in rows if row["condition"] == condition and row["policy"] == POLICIES[0]}
    matcher = {int(row["seed"]): row for row in rows if row["condition"] == condition and row["policy"] == POLICIES[1]}
    if greedy.keys() != matcher.keys():
        raise AssertionError("Policy rows do not share identical seeds")
    output: dict[str, Any] = {}
    for metric in ("technician_conflicts", "maintenance_wait_time", "objective", "makespan"):
        deltas = [float(greedy[seed][metric]) - float(matcher[seed][metric]) for seed in sorted(greedy)]
        mean = fmean(deltas)
        half = _t_critical_975(len(deltas) - 1) * stdev(deltas) / math.sqrt(len(deltas))
        output[metric] = {"mean": mean, "lower_95": mean - half, "upper_95": mean + half}
    return output


def factorial_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    greedy = [row for row in rows if row["policy"] == POLICIES[0]]
    metrics = ("technician_conflicts", "technician_conflict_rate", "avoidable_conflict_units", "maintenance_wait_time")
    effects: list[dict[str, Any]] = []
    seeds = sorted({int(row["seed"]) for row in greedy})
    for factor in FACTORS:
        for metric in metrics:
            deltas = []
            for seed in seeds:
                seed_rows = [row for row in greedy if int(row["seed"]) == seed]
                on = [float(row[metric]) for row in seed_rows if int(row[factor]) == 1]
                off = [float(row[metric]) for row in seed_rows if int(row[factor]) == 0]
                deltas.append(fmean(on) - fmean(off))
            mean = fmean(deltas)
            half = _t_critical_975(len(deltas) - 1) * stdev(deltas) / math.sqrt(len(deltas))
            effects.append({"effect_type": "main", "factor_1": factor, "factor_2": "", "metric": metric, "mean": mean, "lower_95": mean - half, "upper_95": mean + half})
    for left_index, left in enumerate(FACTORS):
        for right in FACTORS[left_index + 1 :]:
            for metric in metrics:
                deltas = []
                for seed in seeds:
                    seed_rows = [row for row in greedy if int(row["seed"]) == seed]
                    cells = {
                        (a, b): fmean(
                            float(row[metric]) for row in seed_rows
                            if int(row[left]) == a and int(row[right]) == b
                        )
                        for a, b in itertools.product((0, 1), repeat=2)
                    }
                    deltas.append((cells[(1, 1)] - cells[(1, 0)]) - (cells[(0, 1)] - cells[(0, 0)]))
                mean = fmean(deltas)
                half = _t_critical_975(len(deltas) - 1) * stdev(deltas) / math.sqrt(len(deltas))
                effects.append({"effect_type": "pairwise_interaction", "factor_1": left, "factor_2": right, "metric": metric, "mean": mean, "lower_95": mean - half, "upper_95": mean + half})
    return effects


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_condition: dict[str, Any] = {}
    candidates: list[tuple[float, float, int, str]] = []
    for spec in condition_specs():
        condition = spec.name
        policy_summaries: dict[str, Any] = {}
        for policy in POLICIES:
            episodes = [row for row in rows if row["condition"] == condition and row["policy"] == policy]
            totals = {metric: sum(float(row[metric]) for row in episodes) for metric in ("technician_conflicts", "maintenance_proposals", "avoidable_conflict_units", "rejected", "proposals")}
            policy_summaries[policy] = {
                "episodes": len(episodes),
                "episode_conflict_incidence": fmean(float(row["technician_conflicts"] > 0) for row in episodes),
                "aggregate_conflict_rate": totals["technician_conflicts"] / totals["maintenance_proposals"] if totals["maintenance_proposals"] else 0.0,
                "aggregate_avoidable_fraction": min(1.0, totals["avoidable_conflict_units"] / totals["technician_conflicts"]) if totals["technician_conflicts"] else 0.0,
                "aggregate_rejection_rate": totals["rejected"] / totals["proposals"] if totals["proposals"] else 0.0,
                "metrics": {metric: _describe(float(row[metric]) for row in episodes) for metric in SUMMARY_METRICS},
                "audit": {
                    "all_feasibility_audits_passed": all(int(row["feasibility_audit_passed"]) == 1 for row in episodes),
                    "zero_invalid_executions": all(int(row["invalid_executions"]) == 0 for row in episodes),
                    "zero_duplicate_operation_executions": all(int(row["duplicate_operation_executions"]) == 0 for row in episodes),
                    "zero_duplicate_technician_executions": all(int(row["duplicate_technician_executions"]) == 0 for row in episodes),
                },
            }
        contrast = _paired_policy_contrast(rows, condition)
        greedy = policy_summaries[POLICIES[0]]
        matcher = policy_summaries[POLICIES[1]]
        conflict_reduction = (
            1.0 - matcher["metrics"]["technician_conflicts"]["mean"] / greedy["metrics"]["technician_conflicts"]["mean"]
            if greedy["metrics"]["technician_conflicts"]["mean"] > 0 else 0.0
        )
        checks = {
            "episode_conflict_incidence_between_10_and_50_percent": 0.10 <= greedy["episode_conflict_incidence"] <= 0.50,
            "conflict_rate_between_1_and_10_percent": 0.01 <= greedy["aggregate_conflict_rate"] <= 0.10,
            "avoidable_fraction_at_least_50_percent": greedy["aggregate_avoidable_fraction"] >= 0.50,
            "matcher_reduces_conflicts_by_at_least_30_percent": conflict_reduction >= 0.30,
            "greedy_rejection_rate_at_most_20_percent": greedy["aggregate_rejection_rate"] <= 0.20,
            "all_audits_pass": all(all(audit.values()) for audit in (greedy["audit"], matcher["audit"])),
        }
        qualifies = all(checks.values())
        if qualifies:
            candidates.append((-contrast["technician_conflicts"]["mean"], greedy["metrics"]["objective"]["mean"], sum(spec.bits), condition))
        per_condition[condition] = {
            "factors": spec.to_dict(),
            "policies": policy_summaries,
            "greedy_minus_matcher": contrast,
            "conflict_reduction_fraction": conflict_reduction,
            "selection_checks": checks,
            "qualifies": qualifies,
        }
    selected = min(candidates)[-1] if candidates else None
    expected = len(rows) // (len(condition_specs()) * len(POLICIES))
    global_checks = {
        "all_expected_episodes_completed": len(rows) == len(condition_specs()) * len(POLICIES) * expected,
        "all_feasibility_audits_passed": all(int(row["feasibility_audit_passed"]) == 1 for row in rows),
        "zero_invalid_executions": all(int(row["invalid_executions"]) == 0 for row in rows),
        "zero_duplicate_executions": all(int(row["duplicate_operation_executions"]) == 0 and int(row["duplicate_technician_executions"]) == 0 for row in rows),
    }
    return {
        "primary_endpoint": "avoidable technician-proposal conflict under independent greedy assignment",
        "selected_condition": selected,
        "selection_rule": "Apply all locked range/headroom/audit gates; maximize paired greedy-minus-matcher conflict reduction, then lower greedy objective, fewer enabled factors, and condition name.",
        "per_condition": per_condition,
        "factorial_effects": factorial_effects(rows),
        "gate": {
            "checks": global_checks,
            "scientific_selection_found": selected is not None,
            "run_valid": all(global_checks.values()),
            "note": "Development mechanism screen; not an RL algorithm-performance claim.",
        },
    }


def run(args: argparse.Namespace) -> Path:
    if args.device != "cpu":
        raise ValueError("This simulator-only factorial experiment is CPU-only")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    raw_config = config_path.read_bytes()
    base = BenchmarkConfig.from_json(config_path)
    if base.instance_name != "ht_pdm_fjsp_brandimarte_mk01_v1":
        raise ValueError("This experiment requires the frozen Brandimarte MK01 extension")
    if not math.isclose(base.preventive_probability_threshold, 0.18):
        raise ValueError("This experiment requires the locked PM threshold 0.18")
    seeds = tuple(parse_seeds(args.seeds)) if args.seeds else default_seeds(args.profile)
    specs = condition_specs()
    configs = {spec.name: build_condition_config(base, spec) for spec in specs}
    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    (output_dir / "resolved_conditions.json").write_text(
        json.dumps(
            {
                spec.name: {"factors": spec.to_dict(), "config": configs[spec.name].to_dict()}
                for spec in specs
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    manifest_path = output_dir / "technician_factorial_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "profile": args.profile,
        "device": args.device,
        "git_commit": _git_revision(),
        "instance_name": base.instance_name,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "factor_order": list(FACTORS),
        "factor_definitions": {
            "Q": "persistent requests may claim busy/off-shift technicians",
            "W": "overlapping release-deadline preventive-maintenance windows",
            "O": "full heterogeneous two-technician skill overlap",
            "A": "staggered deterministic technician unavailable intervals",
            "D": f"maintenance duration multiplier {DURATION_MULTIPLIER}",
        },
        "conditions": [spec.name for spec in specs],
        "policies": list(POLICIES),
        "seeds": list(seeds),
        "reserved_confirmation_seeds": list(range(62_100, 62_300)),
        "confirmation_panel_opened": False,
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": version("numpy")},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    episode_rows: list[dict[str, Any]] = []
    coordination_rows: list[dict[str, Any]] = []
    total = len(specs) * len(POLICIES) * len(seeds)
    with tqdm(total=total, desc="Technician-conflict factorial", unit="episode") as progress:
        for spec in specs:
            config = configs[spec.name]
            for policy in POLICIES:
                for seed in seeds:
                    progress.set_postfix(condition=spec.name, policy=policy, seed=seed)
                    episode, coordination = LiteratureFactorialSimulator(config, spec, policy).run(seed=seed)
                    episode_rows.append(episode)
                    coordination_rows.append(coordination)
                    if len(episode_rows) % 16 == 0 or len(episode_rows) == total:
                        _write_csv(episode_rows, output_dir / "technician_factorial_episodes.partial.csv")
                        _write_csv(coordination_rows, output_dir / "technician_factorial_coordination.partial.csv")
                    progress.update(1)
    summary = summarize(episode_rows)
    _write_csv(episode_rows, output_dir / "technician_factorial_episodes.csv")
    _write_csv(coordination_rows, output_dir / "technician_factorial_coordination.csv")
    _write_csv(summary["factorial_effects"], output_dir / "technician_factorial_effects.csv")
    (output_dir / "technician_factorial_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest.update(
        {
            "status": "COMPLETED",
            "episode_count": len(episode_rows),
            "selected_condition": summary["selected_condition"],
            "gate": summary["gate"],
            "output_files": sorted(path.name for path in output_dir.iterdir()),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--config", default="configs/brandimarte_mk01_ht_pdm.json")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    output_dir = run(build_parser().parse_args())
    summary = json.loads((output_dir / "technician_factorial_summary.json").read_text())
    print(json.dumps({"selected_condition": summary["selected_condition"], "gate": summary["gate"]}, indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
