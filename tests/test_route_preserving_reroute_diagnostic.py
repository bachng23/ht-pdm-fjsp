from __future__ import annotations

import csv
from pathlib import Path

from ht_pdm_fjsp.route_preserving_reroute_diagnostic import (
    EVENT_FIELDS,
    _describe,
    _metrics_match,
    _write_csv,
    summarize,
)
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig


def _episode(train_seed: int, *, reroutes: int = 1) -> dict[str, object]:
    return {
        "train_seed": train_seed,
        "seed": 47_000,
        "decision_count": 20,
        "production_baseline_decisions": 10,
        "reroute_count": reroutes,
        "decision_kind_mismatches": 0,
        "nonproduction_action_mismatches": 0,
        "invalid_actions": 0,
        "internal_shared_action_mismatches": 0,
        "source_replay_match": True,
        "residual_branch_replay_mismatches": 0,
    }


def _event(train_seed: int, delta: float) -> dict[str, object]:
    row = {name: 0 for name in EVENT_FIELDS}
    row.update(
        train_seed=train_seed,
        seed=47_000,
        delta_objective_residual_minus_shared=delta,
    )
    return row


def test_describe_handles_no_reroutes_without_nan() -> None:
    assert _describe([]) == {
        "count": 0,
        "mean": None,
        "min": None,
        "max": None,
    }


def test_summary_keeps_training_seed_as_replication_unit() -> None:
    episodes = [_episode(10_000), _episode(11_000, reroutes=0)]
    events = [_event(10_000, -2.0), _event(10_000, 3.0)]
    seed_rows, summary = summarize(episodes, events)
    assert summary["status"] == "PASS"
    assert summary["replication_unit"] == "independent PPO training seed"
    assert summary["totals"]["beneficial_reroutes"] == 1
    assert summary["totals"]["harmful_reroutes"] == 1
    assert len(seed_rows) == 2
    assert seed_rows[1]["mean_delta_objective_residual_minus_shared"] is None


def test_summary_fails_any_route_or_replay_invariant() -> None:
    episode = _episode(10_000)
    episode["decision_kind_mismatches"] = 1
    _, summary = summarize([episode], [])
    assert summary["status"] == "FAIL"
    assert not summary["checks"]["zero_decision_kind_mismatches"]


def test_metrics_match_uses_numeric_tolerance() -> None:
    left = {
        "objective": 1.0,
        "makespan": 2.0,
        "total_tardiness": 3.0,
        "failures": 0.0,
        "preventive_maintenance": 1.0,
        "corrective_maintenance": 0.0,
        "total_cost": 4.0,
    }
    right = {key: str(value + 1e-10) for key, value in left.items()}
    assert _metrics_match(left, right)
    right["objective"] = "1.1"
    assert not _metrics_match(left, right)


def test_empty_event_table_still_has_full_schema(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    _write_csv([], path, fieldnames=EVENT_FIELDS)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == EVENT_FIELDS
        assert list(reader) == []


def test_clone_retains_prefix_reward_for_counterfactual_accounting() -> None:
    root = Path(__file__).resolve().parents[1]
    config = BenchmarkConfig.from_json(root / "configs/minimal_benchmark.json")
    env = HTPdmFjspEnv(config=config, include_production_context=True)
    _, _ = env.reset(seed=47_000)
    while env.cumulative_reward == 0.0 and not env._done:
        first = int(
            next(index for index, valid in enumerate(env.action_masks()) if valid)
        )
        env.step(first)
    clone = env.clone()
    assert clone.cumulative_reward == env.cumulative_reward
    assert clone.cumulative_reward != 0.0
