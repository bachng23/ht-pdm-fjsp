"""Event-driven simulator for the minimal HT-PdM-FJSP benchmark."""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ht_pdm_fjsp.models import (
    BenchmarkConfig,
    Job,
    Machine,
    Operation,
    OperationAlternative,
    Technician,
)
from ht_pdm_fjsp.policies import SchedulingPolicy


TIME_TOLERANCE = 1e-9


def conditional_weibull_failure_probability(
    effective_age: float,
    age_increment: float,
    weibull_eta: float,
    weibull_beta: float,
) -> float:
    """Return failure probability over an effective-age interval."""

    if effective_age < 0 or age_increment < 0:
        raise ValueError("Age and age increment must be nonnegative.")
    if weibull_eta <= 0 or weibull_beta <= 0:
        raise ValueError("Weibull eta and beta must be positive.")
    h0 = (effective_age / weibull_eta) ** weibull_beta
    h1 = ((effective_age + age_increment) / weibull_eta) ** weibull_beta
    return float(-math.expm1(-(h1 - h0)))


def sample_conditional_weibull_failure_age(
    rng: np.random.Generator,
    *,
    effective_age: float,
    weibull_eta: float,
    weibull_beta: float,
) -> float:
    """Sample the next failure age conditional on survival to effective_age."""

    h0 = (effective_age / weibull_eta) ** weibull_beta
    exponential_increment = float(rng.exponential())
    return float(weibull_eta * (h0 + exponential_increment) ** (1.0 / weibull_beta))


@dataclass(frozen=True)
class Candidate:
    job: Job
    operation_index: int
    operation: Operation
    alternative: OperationAlternative


@dataclass
class JobState:
    next_operation: int = 0
    in_process: bool = False
    completion_time: float | None = None


@dataclass
class MachineState:
    status: str
    effective_age: float
    preventive_armed: bool = True


@dataclass
class TechnicianState:
    status: str = "idle"


@dataclass
class MaintenanceRequest:
    machine_id: str
    kind: str
    created_at: float


@dataclass(order=True)
class Event:
    time: float
    sequence: int
    kind: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False)


@dataclass(frozen=True)
class TaskRecord:
    kind: str
    machine_id: str
    start: float
    end: float
    job_id: str | None = None
    operation_id: str | None = None
    technician_id: str | None = None
    maintenance_kind: str | None = None
    completed: bool = True
    effective_age_before: float | None = None
    effective_age_after: float | None = None


@dataclass(frozen=True)
class SimulationResult:
    instance_name: str
    policy: str
    seed: int
    metrics: dict[str, float | int]
    tasks: tuple[TaskRecord, ...]
    final_machine_ages: dict[str, float]
    job_completion_times: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_name": self.instance_name,
            "policy": self.policy,
            "seed": self.seed,
            "metrics": dict(self.metrics),
            "tasks": [asdict(item) for item in self.tasks],
            "final_machine_ages": dict(self.final_machine_ages),
            "job_completion_times": dict(self.job_completion_times),
        }


class Simulator:
    """Run one fully specified policy/seed episode."""

    def __init__(self, config: BenchmarkConfig):
        config.validate()
        self.config = config
        self.machine_specs = {item.machine_id: item for item in config.machines}
        self.technician_specs = {
            item.technician_id: item for item in config.technicians
        }
        self.job_specs = {item.job_id: item for item in config.jobs}

    def run(self, policy: SchedulingPolicy, *, seed: int) -> SimulationResult:
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
        requests: list[MaintenanceRequest] = []
        tasks: list[TaskRecord] = []
        now = 0.0
        event_sequence = 0
        processed_events = 0
        failures = 0
        preventive_count = 0
        corrective_count = 0
        failure_wait_time = 0.0
        preventive_wait_time = 0.0
        maintenance_time = 0.0
        failed_processing_time = 0.0
        machine_indices = {
            machine.machine_id: index for index, machine in enumerate(self.config.machines)
        }
        job_indices = {job.job_id: index for index, job in enumerate(self.config.jobs)}
        attempt_counts: dict[tuple[str, int, str], int] = {}

        def push_event(time: float, kind: str, payload: dict[str, Any]) -> None:
            nonlocal event_sequence
            event_sequence += 1
            heapq.heappush(events, Event(time, event_sequence, kind, payload))

        def ready_candidates(machine_id: str) -> list[Candidate]:
            candidates: list[Candidate] = []
            for job in self.config.jobs:
                state = jobs[job.job_id]
                if state.in_process or state.next_operation >= len(job.operations):
                    continue
                operation = job.operations[state.next_operation]
                for alternative in operation.alternatives:
                    if alternative.machine_id == machine_id:
                        candidates.append(
                            Candidate(
                                job=job,
                                operation_index=state.next_operation,
                                operation=operation,
                                alternative=alternative,
                            )
                        )
            return candidates

        def assign_maintenance() -> bool:
            nonlocal corrective_count, preventive_count
            nonlocal failure_wait_time, preventive_wait_time
            changed = False
            requests.sort(
                key=lambda item: (
                    0 if item.kind == "corrective" else 1,
                    item.created_at,
                    item.machine_id,
                )
            )
            for request in list(requests):
                available = [
                    spec
                    for spec in self.config.technicians
                    if technicians[spec.technician_id].status == "idle"
                    and request.machine_id in spec.eligible_machines
                ]
                if not available:
                    continue
                technician = min(
                    available,
                    key=lambda item: (
                        item.duration(request.machine_id, request.kind),
                        item.technician_id,
                    ),
                )
                duration = technician.duration(request.machine_id, request.kind)
                restoration = technician.restoration(request.machine_id, request.kind)
                machine_state = machines[request.machine_id]
                machine_state.status = "maintenance"
                technicians[technician.technician_id].status = "busy"
                if request.kind == "corrective":
                    corrective_count += 1
                    failure_wait_time += now - request.created_at
                else:
                    preventive_count += 1
                    preventive_wait_time += now - request.created_at
                push_event(
                    now + duration,
                    "maintenance_complete",
                    {
                        "machine_id": request.machine_id,
                        "technician_id": technician.technician_id,
                        "maintenance_kind": request.kind,
                        "start": now,
                        "age_before": machine_state.effective_age,
                        "restoration": restoration,
                    },
                )
                requests.remove(request)
                changed = True
            return changed

        def create_preventive_requests() -> bool:
            changed = False
            for machine_id in sorted(machines):
                state = machines[machine_id]
                if state.status != "idle":
                    continue
                candidates = ready_candidates(machine_id)
                if not candidates:
                    continue
                candidate = policy.choose_candidate(candidates)
                spec = self.machine_specs[machine_id]
                increment = (
                    spec.alpha
                    * candidate.alternative.processing_time
                    * candidate.alternative.load_factor
                )
                probability = conditional_weibull_failure_probability(
                    state.effective_age,
                    increment,
                    spec.weibull_eta,
                    spec.weibull_beta,
                )
                if policy.wants_preventive(
                    machine=spec,
                    effective_age=state.effective_age,
                    alternative=candidate.alternative,
                    failure_probability=probability,
                    preventive_armed=state.preventive_armed,
                ):
                    state.status = "waiting_maintenance"
                    requests.append(
                        MaintenanceRequest(machine_id, "preventive", now)
                    )
                    changed = True
            return changed

        def start_production() -> bool:
            changed = False
            for machine_id in sorted(machines):
                state = machines[machine_id]
                if state.status != "idle":
                    continue
                candidates = ready_candidates(machine_id)
                if not candidates:
                    continue
                candidate = policy.choose_candidate(candidates)
                job_state = jobs[candidate.job.job_id]
                job_state.in_process = True
                state.status = "processing"
                spec = self.machine_specs[machine_id]
                rate = spec.alpha * candidate.alternative.load_factor
                increment = rate * candidate.alternative.processing_time
                attempt_key = (
                    candidate.job.job_id,
                    candidate.operation_index,
                    machine_id,
                )
                attempt_number = attempt_counts.get(attempt_key, 0)
                attempt_counts[attempt_key] = attempt_number + 1
                shock_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            int(seed),
                            job_indices[candidate.job.job_id],
                            candidate.operation_index,
                            machine_indices[machine_id],
                            attempt_number,
                            0x48545044,
                        ]
                    )
                )
                failure_age = sample_conditional_weibull_failure_age(
                    shock_rng,
                    effective_age=state.effective_age,
                    weibull_eta=spec.weibull_eta,
                    weibull_beta=spec.weibull_beta,
                )
                payload = {
                    "machine_id": machine_id,
                    "job_id": candidate.job.job_id,
                    "operation_index": candidate.operation_index,
                    "operation_id": candidate.operation.operation_id,
                    "start": now,
                    "age_before": state.effective_age,
                    "age_increment": increment,
                }
                if failure_age < state.effective_age + increment:
                    failure_offset = (failure_age - state.effective_age) / rate
                    payload["failure_age"] = failure_age
                    push_event(now + failure_offset, "machine_failure", payload)
                else:
                    push_event(
                        now + candidate.alternative.processing_time,
                        "operation_complete",
                        payload,
                    )
                changed = True
            return changed

        def dispatch() -> None:
            assign_maintenance()
            if create_preventive_requests():
                assign_maintenance()
            start_production()

        def all_jobs_complete() -> bool:
            return all(
                state.next_operation == len(self.job_specs[job_id].operations)
                for job_id, state in jobs.items()
            )

        dispatch()
        while not (all_jobs_complete() and not events and not requests):
            if not events:
                raise RuntimeError(
                    "Simulation deadlock: unfinished work exists but no event can advance time."
                )
            now = events[0].time
            simultaneous: list[Event] = []
            while events and abs(events[0].time - now) <= TIME_TOLERANCE:
                simultaneous.append(heapq.heappop(events))
            for event in sorted(simultaneous, key=lambda item: item.sequence):
                processed_events += 1
                if processed_events > self.config.max_events:
                    raise RuntimeError("max_events exceeded; possible unstable configuration.")
                payload = event.payload
                machine_id = payload["machine_id"]
                machine_state = machines[machine_id]
                if event.kind == "operation_complete":
                    machine_state.effective_age = (
                        payload["age_before"] + payload["age_increment"]
                    )
                    machine_state.status = "idle"
                    machine_state.preventive_armed = True
                    job_state = jobs[payload["job_id"]]
                    job_state.in_process = False
                    job_state.next_operation += 1
                    if job_state.next_operation == len(
                        self.job_specs[payload["job_id"]].operations
                    ):
                        job_state.completion_time = now
                    tasks.append(
                        TaskRecord(
                            kind="production",
                            machine_id=machine_id,
                            start=payload["start"],
                            end=now,
                            job_id=payload["job_id"],
                            operation_id=payload["operation_id"],
                            completed=True,
                            effective_age_before=payload["age_before"],
                            effective_age_after=machine_state.effective_age,
                        )
                    )
                elif event.kind == "machine_failure":
                    failures += 1
                    failed_processing_time += now - payload["start"]
                    machine_state.effective_age = payload["failure_age"]
                    machine_state.status = "waiting_maintenance"
                    machine_state.preventive_armed = False
                    jobs[payload["job_id"]].in_process = False
                    requests.append(MaintenanceRequest(machine_id, "corrective", now))
                    tasks.append(
                        TaskRecord(
                            kind="production",
                            machine_id=machine_id,
                            start=payload["start"],
                            end=now,
                            job_id=payload["job_id"],
                            operation_id=payload["operation_id"],
                            completed=False,
                            effective_age_before=payload["age_before"],
                            effective_age_after=machine_state.effective_age,
                        )
                    )
                elif event.kind == "maintenance_complete":
                    maintenance_time += now - payload["start"]
                    age_after = max(
                        0.0,
                        payload["age_before"] * (1.0 - payload["restoration"]),
                    )
                    machine_state.effective_age = age_after
                    machine_state.status = "idle"
                    machine_state.preventive_armed = False
                    technicians[payload["technician_id"]].status = "idle"
                    tasks.append(
                        TaskRecord(
                            kind="maintenance",
                            machine_id=machine_id,
                            start=payload["start"],
                            end=now,
                            technician_id=payload["technician_id"],
                            maintenance_kind=payload["maintenance_kind"],
                            completed=True,
                            effective_age_before=payload["age_before"],
                            effective_age_after=age_after,
                        )
                    )
                else:
                    raise RuntimeError(f"Unknown event kind: {event.kind}")
            dispatch()

        completion_times = {
            job_id: float(state.completion_time)
            for job_id, state in jobs.items()
            if state.completion_time is not None
        }
        makespan = max(completion_times.values())
        total_tardiness = sum(
            max(0.0, completion_times[job.job_id] - job.due_date)
            for job in self.config.jobs
        )
        maintenance_wait_time = failure_wait_time + preventive_wait_time
        downtime = maintenance_time + maintenance_wait_time
        total_cost = (
            self.config.costs.tardiness * total_tardiness
            + self.config.costs.preventive * preventive_count
            + self.config.costs.corrective * corrective_count
            + self.config.costs.downtime * downtime
        )
        schedule_end = max((task.end for task in tasks), default=makespan)
        technician_utilization = (
            maintenance_time / (schedule_end * len(self.config.technicians))
            if schedule_end > 0
            else 0.0
        )
        result = SimulationResult(
            instance_name=self.config.instance_name,
            policy=policy.name,
            seed=int(seed),
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
                "technician_utilization": technician_utilization,
                "total_cost": total_cost,
                "processed_events": processed_events,
            },
            tasks=tuple(sorted(tasks, key=lambda item: (item.start, item.end, item.kind))),
            final_machine_ages={
                machine_id: state.effective_age for machine_id, state in machines.items()
            },
            job_completion_times=completion_times,
        )
        audit_result(self.config, result)
        return result


def _assert_no_overlap(records: list[TaskRecord], resource_name: str) -> None:
    ordered = sorted(records, key=lambda item: (item.start, item.end))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.end - TIME_TOLERANCE:
            raise AssertionError(
                f"Overlapping tasks on {resource_name}: {previous} and {current}."
            )


def audit_result(config: BenchmarkConfig, result: SimulationResult) -> None:
    """Raise AssertionError if a result violates the frozen feasibility contract."""

    machine_ids = {item.machine_id for item in config.machines}
    technician_by_id = {item.technician_id: item for item in config.technicians}
    successful = [
        task for task in result.tasks if task.kind == "production" and task.completed
    ]
    for machine_id in machine_ids:
        _assert_no_overlap(
            [task for task in result.tasks if task.machine_id == machine_id],
            f"machine {machine_id}",
        )
    for technician_id in technician_by_id:
        _assert_no_overlap(
            [task for task in result.tasks if task.technician_id == technician_id],
            f"technician {technician_id}",
        )
    for task in result.tasks:
        if task.end <= task.start:
            raise AssertionError(f"Non-positive task duration: {task}")
        if task.kind == "maintenance":
            technician = technician_by_id[str(task.technician_id)]
            if task.machine_id not in technician.eligible_machines:
                raise AssertionError(f"Ineligible maintenance assignment: {task}")
            if task.effective_age_after is None or task.effective_age_before is None:
                raise AssertionError("Maintenance records must expose age transition.")
            if task.effective_age_after > task.effective_age_before + TIME_TOLERANCE:
                raise AssertionError(f"Maintenance increased effective age: {task}")
    for job in config.jobs:
        job_tasks = sorted(
            [task for task in successful if task.job_id == job.job_id],
            key=lambda item: item.start,
        )
        if [task.operation_id for task in job_tasks] != [
            operation.operation_id for operation in job.operations
        ]:
            raise AssertionError(f"Operation precedence/completeness failed for {job.job_id}.")
        for operation, task in zip(job.operations, job_tasks):
            eligible = {item.machine_id for item in operation.alternatives}
            if task.machine_id not in eligible:
                raise AssertionError(f"Ineligible machine assignment: {task}")
        for previous, current in zip(job_tasks, job_tasks[1:]):
            if current.start < previous.end - TIME_TOLERANCE:
                raise AssertionError(f"Job precedence overlap for {job.job_id}.")
    if len(result.job_completion_times) != len(config.jobs):
        raise AssertionError("Not all jobs have completion times.")
    observed_failures = sum(
        task.kind == "production" and not task.completed for task in result.tasks
    )
    if int(result.metrics["failures"]) != observed_failures:
        raise AssertionError("Failure metric does not match trace.")
