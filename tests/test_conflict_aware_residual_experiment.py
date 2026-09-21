from __future__ import annotations

from ht_pdm_fjsp.conflict_aware_residual_experiment import (
    CURRENT,
    SHARED,
    TREATMENT,
    defaults,
    promotion_decision,
    summarize_audit,
)


def _audit(*, enforce: bool = True, nonconflicting: int = 0):
    return {
        "condition": TREATMENT if enforce else CURRENT,
        "enforce_conflicts": enforce,
        "episodes": 5,
        "decisions": 50,
        "production_baseline_decisions": 30,
        "production_reroutes": 10,
        "same_machine_reroutes": 8,
        "same_operation_reroutes": 2,
        "nonconflicting_production_reroutes": nonconflicting,
        "decision_kind_mismatches": 0,
        "nonproduction_action_mismatches": 0,
        "invalid_actions": 0,
    }


def test_full_profile_uses_fresh_panel_and_keeps_test_closed() -> None:
    full = defaults("full")
    assert full["train_seeds"] == (10_000, 11_000, 12_000, 13_000, 14_000)
    assert full["validation_seeds"] == tuple(range(48_000, 48_200))
    assert set(full["validation_seeds"]).isdisjoint(range(50_000, 50_100))


def test_treatment_audit_rejects_nonconflicting_reroutes() -> None:
    assert summarize_audit(_audit())["status"] == "PASS"
    failed = summarize_audit(_audit(nonconflicting=1))
    assert failed["status"] == "FAIL"
    assert not failed["checks"]["zero_nonconflicting_production_reroutes"]


def test_current_control_reports_but_does_not_fail_nonconflicting_reroutes() -> None:
    result = summarize_audit(_audit(enforce=False, nonconflicting=7))
    assert result["status"] == "PASS"
    assert "zero_nonconflicting_production_reroutes" not in result["checks"]


def test_promotion_requires_every_predeclared_gate() -> None:
    condition = {
        "across_training_seeds": {"objective": {"std": 1.0}}
    }
    summary = {
        "comparisons": {
            f"{TREATMENT}_minus_{SHARED}": {
                "per_training_seed_objective_delta": {
                    "10000": -1.0,
                    "11000": -0.5,
                    "12000": 0.0,
                    "13000": -0.2,
                    "14000": 0.1,
                },
                "delta_across_training_seeds": {
                    "objective": {"mean": -0.3},
                    "failures": {"mean": 0.0},
                },
            }
        },
        "per_condition": {
            TREATMENT: condition,
            CURRENT: {
                "across_training_seeds": {"objective": {"std": 1.2}}
            },
        },
    }
    passed_audit = summarize_audit(_audit())
    decision = promotion_decision(summary, passed_audit)
    assert decision["eligible_for_future_held_out_test"]
    summary["comparisons"][f"{TREATMENT}_minus_{SHARED}"][
        "delta_across_training_seeds"
    ]["failures"]["mean"] = 0.01
    assert not promotion_decision(summary, passed_audit)[
        "eligible_for_future_held_out_test"
    ]
