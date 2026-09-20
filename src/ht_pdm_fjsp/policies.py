"""Transparent scheduling and maintenance baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from ht_pdm_fjsp.models import Job, Machine, OperationAlternative


CandidateT = TypeVar("CandidateT")


class SchedulingPolicy(Protocol):
    name: str

    def choose_candidate(self, candidates: list[CandidateT]) -> CandidateT: ...

    def wants_preventive(
        self,
        *,
        machine: Machine,
        effective_age: float,
        alternative: OperationAlternative,
        failure_probability: float,
        preventive_armed: bool,
    ) -> bool: ...


@dataclass(frozen=True)
class ProductionFirstSPTPolicy:
    """Shortest-processing-time dispatching with corrective maintenance only."""

    name: str = field(default="production_first_spt", init=False)

    def choose_candidate(self, candidates: list[CandidateT]) -> CandidateT:
        return min(
            candidates,
            key=lambda item: (
                item.alternative.processing_time,
                item.job.due_date,
                item.job.job_id,
                item.operation.operation_id,
            ),
        )

    def wants_preventive(
        self,
        *,
        machine: Machine,
        effective_age: float,
        alternative: OperationAlternative,
        failure_probability: float,
        preventive_armed: bool,
    ) -> bool:
        del machine, effective_age, alternative, failure_probability, preventive_armed
        return False


@dataclass(frozen=True)
class HealthThresholdSPTPolicy(ProductionFirstSPTPolicy):
    """SPT with oracle effective-age threshold maintenance."""

    probability_threshold: float = 0.15
    name: str = field(default="health_threshold_spt", init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError("probability_threshold must lie in (0, 1).")

    def wants_preventive(
        self,
        *,
        machine: Machine,
        effective_age: float,
        alternative: OperationAlternative,
        failure_probability: float,
        preventive_armed: bool,
    ) -> bool:
        del machine, alternative
        return (
            preventive_armed
            and effective_age > 0.0
            and failure_probability >= self.probability_threshold
        )
