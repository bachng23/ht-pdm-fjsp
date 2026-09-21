from __future__ import annotations

from ht_pdm_fjsp.ctde_experiment import CENTRALIZED, CTDE, defaults, promotion_decision


def test_full_profile_keeps_final_test_panel_closed() -> None:
    profile = defaults("full")
    assert profile["train_seeds"] == (10_000, 11_000, 12_000, 13_000, 14_000)
    assert len(profile["validation_seeds"]) == 200
    assert set(profile["validation_seeds"]).isdisjoint(range(50_000, 50_100))


def test_promotion_requires_all_ctde_gates() -> None:
    summary = {
        "comparisons": {
            f"{CTDE}_minus_{CENTRALIZED}": {
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
        }
    }
    audit = {"status": "PASS"}
    assert promotion_decision(summary, audit)["eligible_for_future_held_out_test"]
    summary["comparisons"][f"{CTDE}_minus_{CENTRALIZED}"][
        "delta_across_training_seeds"
    ]["failures"]["mean"] = 0.1
    assert not promotion_decision(summary, audit)[
        "eligible_for_future_held_out_test"
    ]
