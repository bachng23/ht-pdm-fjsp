from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ht_pdm_fjsp.benchmark import parse_seeds, run_panel, summarize, write_outputs
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.policies import HealthThresholdSPTPolicy, ProductionFirstSPTPolicy
from ht_pdm_fjsp.simulator import (
    Simulator,
    audit_result,
    conditional_weibull_failure_probability,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "minimal_benchmark.json"


@pytest.fixture(scope="module")
def config() -> BenchmarkConfig:
    return BenchmarkConfig.from_json(CONFIG_PATH)


def test_weibull_probability_is_bounded_and_monotone() -> None:
    low_age = conditional_weibull_failure_probability(2.0, 3.0, 20.0, 3.0)
    high_age = conditional_weibull_failure_probability(10.0, 3.0, 20.0, 3.0)
    high_exposure = conditional_weibull_failure_probability(2.0, 6.0, 20.0, 3.0)
    assert 0.0 < low_age < high_age < 1.0
    assert low_age < high_exposure < 1.0
    assert conditional_weibull_failure_probability(2.0, 0.0, 20.0, 3.0) == 0.0


def test_invalid_machine_reference_is_rejected() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["jobs"][0]["operations"][0]["alternatives"][0]["machine_id"] = "M404"
    with pytest.raises(ValueError, match="Unknown operation machines"):
        BenchmarkConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "policy_factory",
    [
        ProductionFirstSPTPolicy,
        lambda: HealthThresholdSPTPolicy(0.18),
    ],
)
def test_bundled_instance_is_feasible_and_complete(config, policy_factory) -> None:
    result = Simulator(config).run(policy_factory(), seed=11)
    audit_result(config, result)
    assert len(result.job_completion_times) == len(config.jobs)
    assert result.metrics["makespan"] > 0
    assert result.metrics["processed_events"] >= sum(
        len(job.operations) for job in config.jobs
    )


def test_same_seed_reproduces_exact_trace(config) -> None:
    simulator = Simulator(config)
    policy = HealthThresholdSPTPolicy(0.18)
    first = simulator.run(policy, seed=90210).to_dict()
    second = simulator.run(policy, seed=90210).to_dict()
    assert first == second


def test_maintenance_reduces_effective_age_and_respects_skill(config) -> None:
    result = Simulator(config).run(HealthThresholdSPTPolicy(0.05), seed=7)
    maintenance = [task for task in result.tasks if task.kind == "maintenance"]
    assert maintenance
    for task in maintenance:
        assert task.effective_age_after <= task.effective_age_before
        technician = next(
            item for item in config.technicians if item.technician_id == task.technician_id
        )
        assert task.machine_id in technician.eligible_machines


def test_failure_forces_corrective_repair_under_stress(config) -> None:
    stressed_machines = tuple(
        replace(machine, weibull_eta=8.0) for machine in config.machines
    )
    stressed = replace(config, machines=stressed_machines)
    result = Simulator(stressed).run(ProductionFirstSPTPolicy(), seed=3)
    assert result.metrics["failures"] > 0
    assert result.metrics["corrective_maintenance"] == result.metrics["failures"]
    assert any(
        task.maintenance_kind == "corrective"
        for task in result.tasks
        if task.kind == "maintenance"
    )


def test_multi_seed_panel_and_summary(config) -> None:
    results = run_panel(config, [0, 1, 2, 3])
    assert len(results) == 8
    report = summarize(results)
    assert set(report) == {"production_first_spt", "health_threshold_spt"}
    assert report["production_first_spt"]["episodes"] == 4


def test_skill_bottleneck_creates_maintenance_waiting(config) -> None:
    results = run_panel(config, range(40))
    assert any(float(item.metrics["maintenance_wait_time"]) > 0 for item in results)


def test_cli_artifact_writer_is_reproducible(config, tmp_path) -> None:
    results = run_panel(config, [4, 5])
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_outputs(config, results, first)
    write_outputs(config, results, second)
    expected = {
        "config_snapshot.json",
        "episodes.csv",
        "metadata.json",
        "summary.json",
        "traces.json",
    }
    assert {item.name for item in first.iterdir()} == expected
    for filename in expected:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_seed_parser() -> None:
    assert parse_seeds("0:5") == [0, 1, 2, 3, 4]
    assert parse_seeds("1,3,9") == [1, 3, 9]
    with pytest.raises(ValueError, match="unique"):
        parse_seeds("1,1")
