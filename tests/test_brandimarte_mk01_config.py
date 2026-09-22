from __future__ import annotations

import hashlib
from pathlib import Path

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.policies import HealthThresholdSPTPolicy, ProductionFirstSPTPolicy
from ht_pdm_fjsp.simulator import Simulator, audit_result


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"
SOURCE_PATH = ROOT / "configs" / "source" / "brandimarte_mk01.fjs"
SOURCE_SHA256 = "449ada8093e03a84bf2255fa2a1673ccbc80cce7b82e855c5216533a330751ff"


def _parse_fjs_source() -> tuple[int, int, list[list[list[tuple[int, int]]]]]:
    rows = [
        [int(token) for token in line.split()]
        for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    job_count, machine_count = rows[0]
    jobs: list[list[list[tuple[int, int]]]] = []
    for row in rows[1:]:
        cursor = 0
        operation_count = row[cursor]
        cursor += 1
        operations: list[list[tuple[int, int]]] = []
        for _ in range(operation_count):
            alternative_count = row[cursor]
            cursor += 1
            alternatives = []
            for _ in range(alternative_count):
                machine_index, processing_time = row[cursor : cursor + 2]
                cursor += 2
                alternatives.append((machine_index, processing_time))
            operations.append(alternatives)
        assert cursor == len(row)
        jobs.append(operations)
    assert len(jobs) == job_count
    return job_count, machine_count, jobs


def test_derived_config_preserves_brandimarte_mk01_production_data() -> None:
    assert hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() == SOURCE_SHA256
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    source_job_count, source_machine_count, source_jobs = _parse_fjs_source()

    assert len(config.jobs) == source_job_count == 10
    assert len(config.machines) == source_machine_count == 6
    assert sum(len(job.operations) for job in config.jobs) == 55

    for job, source_operations in zip(config.jobs, source_jobs, strict=True):
        assert len(job.operations) == len(source_operations)
        for operation, source_alternatives in zip(
            job.operations, source_operations, strict=True
        ):
            derived = [
                (
                    int(alternative.machine_id.removeprefix("M")) - 1,
                    int(alternative.processing_time),
                )
                for alternative in operation.alternatives
            ]
            assert derived == source_alternatives
            assert all(
                alternative.load_factor == 1.0
                for alternative in operation.alternatives
            )


def test_mk01_extension_has_six_agents_and_overlapping_technician_skills() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    env = MachineAgentsCTDEEnv(config)
    coverage = {
        technician.technician_id: set(technician.eligible_machines)
        for technician in config.technicians
    }

    assert env.num_agents == 6
    assert coverage == {
        "T1": {"M1", "M2", "M3", "M4"},
        "T2": {"M3", "M4", "M5", "M6"},
    }
    assert config.costs.tardiness == 0.0
    assert {job.due_date for job in config.jobs} == {40.0}


def test_mk01_extension_smoke_policies_complete_and_are_auditable() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    simulator = Simulator(config)
    results = [
        simulator.run(policy, seed=seed)
        for seed in (61000, 61001)
        for policy in (
            ProductionFirstSPTPolicy(),
            HealthThresholdSPTPolicy(config.preventive_probability_threshold),
        )
    ]

    for result in results:
        audit_result(config, result)
        assert len(result.job_completion_times) == 10
        assert result.metrics["processed_events"] >= 55

    assert any(result.metrics["failures"] > 0 for result in results)
    assert any(result.metrics["preventive_maintenance"] > 0 for result in results)
