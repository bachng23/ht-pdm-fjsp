"""Validated data model for the minimal HT-PdM-FJSP benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}.")


def _unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}.")


@dataclass(frozen=True)
class OperationAlternative:
    machine_id: str
    processing_time: float
    load_factor: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OperationAlternative":
        item = cls(
            machine_id=str(values["machine_id"]),
            processing_time=float(values["processing_time"]),
            load_factor=float(values["load_factor"]),
        )
        _positive("processing_time", item.processing_time)
        _positive("load_factor", item.load_factor)
        return item


@dataclass(frozen=True)
class Operation:
    operation_id: str
    alternatives: tuple[OperationAlternative, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Operation":
        alternatives = tuple(
            OperationAlternative.from_mapping(item)
            for item in values.get("alternatives", [])
        )
        if not alternatives:
            raise ValueError("Every operation needs at least one machine alternative.")
        machine_ids = [item.machine_id for item in alternatives]
        if len(machine_ids) != len(set(machine_ids)):
            raise ValueError("An operation cannot repeat a machine alternative.")
        return cls(operation_id=str(values["operation_id"]), alternatives=alternatives)


@dataclass(frozen=True)
class Job:
    job_id: str
    due_date: float
    operations: tuple[Operation, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Job":
        operations = tuple(
            Operation.from_mapping(item) for item in values.get("operations", [])
        )
        if not operations:
            raise ValueError("Every job needs at least one operation.")
        operation_ids = [item.operation_id for item in operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("Operation IDs must be unique within a job.")
        due_date = float(values["due_date"])
        _positive("due_date", due_date)
        return cls(job_id=str(values["job_id"]), due_date=due_date, operations=operations)


@dataclass(frozen=True)
class Machine:
    machine_id: str
    alpha: float
    weibull_eta: float
    weibull_beta: float
    initial_effective_age: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Machine":
        item = cls(
            machine_id=str(values["machine_id"]),
            alpha=float(values["alpha"]),
            weibull_eta=float(values["weibull_eta"]),
            weibull_beta=float(values["weibull_beta"]),
            initial_effective_age=float(values.get("initial_effective_age", 0.0)),
        )
        _positive("alpha", item.alpha)
        _positive("weibull_eta", item.weibull_eta)
        _positive("weibull_beta", item.weibull_beta)
        if item.initial_effective_age < 0 or not math.isfinite(item.initial_effective_age):
            raise ValueError("initial_effective_age must be finite and nonnegative.")
        return item


@dataclass(frozen=True)
class Technician:
    technician_id: str
    eligible_machines: tuple[str, ...]
    preventive_duration: Mapping[str, float]
    corrective_duration: Mapping[str, float]
    preventive_restoration: Mapping[str, float]
    corrective_restoration: Mapping[str, float]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Technician":
        eligible = tuple(str(item) for item in values.get("eligible_machines", []))
        if not eligible:
            raise ValueError("Every technician needs at least one eligible machine.")
        item = cls(
            technician_id=str(values["technician_id"]),
            eligible_machines=eligible,
            preventive_duration={
                str(key): float(value)
                for key, value in values.get("preventive_duration", {}).items()
            },
            corrective_duration={
                str(key): float(value)
                for key, value in values.get("corrective_duration", {}).items()
            },
            preventive_restoration={
                str(key): float(value)
                for key, value in values.get("preventive_restoration", {}).items()
            },
            corrective_restoration={
                str(key): float(value)
                for key, value in values.get("corrective_restoration", {}).items()
            },
        )
        for machine_id in eligible:
            for field_name, mapping in (
                ("preventive_duration", item.preventive_duration),
                ("corrective_duration", item.corrective_duration),
            ):
                if machine_id not in mapping:
                    raise ValueError(
                        f"{item.technician_id} lacks {field_name} for {machine_id}."
                    )
                _positive(field_name, mapping[machine_id])
            for field_name, mapping in (
                ("preventive_restoration", item.preventive_restoration),
                ("corrective_restoration", item.corrective_restoration),
            ):
                if machine_id not in mapping:
                    raise ValueError(
                        f"{item.technician_id} lacks {field_name} for {machine_id}."
                    )
                _unit_interval(field_name, mapping[machine_id])
        return item

    def duration(self, machine_id: str, kind: str) -> float:
        mapping = (
            self.preventive_duration if kind == "preventive" else self.corrective_duration
        )
        return float(mapping[machine_id])

    def restoration(self, machine_id: str, kind: str) -> float:
        mapping = (
            self.preventive_restoration
            if kind == "preventive"
            else self.corrective_restoration
        )
        return float(mapping[machine_id])


@dataclass(frozen=True)
class CostConfig:
    tardiness: float = 1.0
    preventive: float = 5.0
    corrective: float = 25.0
    downtime: float = 1.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CostConfig":
        item = cls(**{key: float(value) for key, value in values.items()})
        for key, value in asdict(item).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Cost {key} must be finite and nonnegative.")
        return item


@dataclass(frozen=True)
class BenchmarkConfig:
    instance_name: str
    machines: tuple[Machine, ...]
    technicians: tuple[Technician, ...]
    jobs: tuple[Job, ...]
    preventive_probability_threshold: float
    costs: CostConfig
    max_events: int = 100_000

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BenchmarkConfig":
        item = cls(
            instance_name=str(values["instance_name"]),
            machines=tuple(Machine.from_mapping(v) for v in values.get("machines", [])),
            technicians=tuple(
                Technician.from_mapping(v) for v in values.get("technicians", [])
            ),
            jobs=tuple(Job.from_mapping(v) for v in values.get("jobs", [])),
            preventive_probability_threshold=float(
                values["preventive_probability_threshold"]
            ),
            costs=CostConfig.from_mapping(values.get("costs", {})),
            max_events=int(values.get("max_events", 100_000)),
        )
        item.validate()
        return item

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkConfig":
        with Path(path).expanduser().open(encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("Benchmark config must be a JSON object.")
        return cls.from_mapping(values)

    def validate(self) -> None:
        if not self.machines or not self.technicians or not self.jobs:
            raise ValueError("At least one machine, technician, and job are required.")
        for label, identifiers in (
            ("machine", [item.machine_id for item in self.machines]),
            ("technician", [item.technician_id for item in self.technicians]),
            ("job", [item.job_id for item in self.jobs]),
        ):
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"Duplicate {label} IDs are not allowed.")
        machine_ids = {item.machine_id for item in self.machines}
        for job in self.jobs:
            for operation in job.operations:
                unknown = {alt.machine_id for alt in operation.alternatives} - machine_ids
                if unknown:
                    raise ValueError(f"Unknown operation machines: {sorted(unknown)}")
        for technician in self.technicians:
            unknown = set(technician.eligible_machines) - machine_ids
            if unknown:
                raise ValueError(f"Unknown technician machines: {sorted(unknown)}")
        covered = {
            machine_id
            for technician in self.technicians
            for machine_id in technician.eligible_machines
        }
        if covered != machine_ids:
            raise ValueError(
                f"Every machine needs an eligible technician; missing {sorted(machine_ids-covered)}."
            )
        if not 0.0 < self.preventive_probability_threshold < 1.0:
            raise ValueError("preventive_probability_threshold must lie in (0, 1).")
        if self.max_events < 1:
            raise ValueError("max_events must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
