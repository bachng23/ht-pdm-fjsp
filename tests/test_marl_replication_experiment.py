from __future__ import annotations

import pytest

from ht_pdm_fjsp.marl_diagnostic_experiment import INDEPENDENT_MAPPO
from ht_pdm_fjsp.marl_factorial_experiment import COMBINED_MAPPO
from ht_pdm_fjsp.marl_replication_experiment import (
    FAILURE_NONINFERIORITY_MARGIN,
    confirmation_decision,
    defaults,
    one_sided_upper_95,
)


def _contrast(objective: tuple[float, ...], failures: tuple[float, ...]) -> dict:
    return {
        "per_training_seed": {
            str(15_000 + index * 1_000): {
                "objective": objective_delta,
                "failures": failure_delta,
            }
            for index, (objective_delta, failure_delta) in enumerate(
                zip(objective, failures, strict=True)
            )
        }
    }


def _audits(status: str = "PASS") -> dict:
    return {
        INDEPENDENT_MAPPO: {"status": status},
        COMBINED_MAPPO: {"status": status},
    }


def test_full_profile_uses_ten_fresh_seeds_and_keeps_test_closed() -> None:
    profile = defaults("full")
    assert profile["train_seeds"] == tuple(range(15_000, 25_000, 1_000))
    assert profile["validation_seeds"] == tuple(range(55_000, 55_200))
    assert set(profile["validation_seeds"]).isdisjoint(range(50_000, 50_100))
    assert set(profile["validation_seeds"]).isdisjoint(range(54_000, 54_200))


def test_smoke_panel_is_disjoint_from_full_panel() -> None:
    smoke = defaults("smoke")
    full = defaults("full")
    assert set(smoke["train_seeds"]).isdisjoint(full["train_seeds"])
    assert set(smoke["validation_seeds"]).isdisjoint(full["validation_seeds"])


def test_one_sided_upper_bound_uses_training_seed_replications() -> None:
    result = one_sided_upper_95((-2.0,) * 10)
    assert result["degrees_of_freedom"] == 9
    assert result["mean"] == -2.0
    assert result["upper_95"] == -2.0
    with pytest.raises(ValueError, match="At least two"):
        one_sided_upper_95((-1.0,))


def test_confirmation_gate_passes_only_complete_noninferior_result() -> None:
    objective = (-2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, 0.1, 0.2)
    failures = (0.01,) * 10
    decision = confirmation_decision(_contrast(objective, failures), _audits())
    assert decision["eligible_for_separately_authorized_held_out_test"]
    assert decision["failure_noninferiority_margin"] == FAILURE_NONINFERIORITY_MARGIN


@pytest.mark.parametrize(
    ("objective", "failures", "audit_status"),
    (
        ((-0.1,) * 6 + (0.5,) * 4, (0.01,) * 10, "PASS"),
        ((-2.0,) * 10, (0.06,) * 10, "PASS"),
        ((-2.0,) * 10, (0.01,) * 10, "FAIL"),
        ((-2.0,) * 9, (0.01,) * 9, "PASS"),
    ),
)
def test_confirmation_gate_rejects_failed_checks(
    objective: tuple[float, ...], failures: tuple[float, ...], audit_status: str
) -> None:
    decision = confirmation_decision(
        _contrast(objective, failures), _audits(audit_status)
    )
    assert not decision["eligible_for_separately_authorized_held_out_test"]
