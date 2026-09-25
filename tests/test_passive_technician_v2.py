from __future__ import annotations

import json
import random
from argparse import Namespace
from dataclasses import replace

import pytest

from ht_pdm_fjsp.passive_technician_v2 import (
    MachineMode,
    PassiveTechnicianV2Config,
    PassiveTechnicianV2Env,
    exact_expected_cost,
    keyed_failure_uniform,
)
from ht_pdm_fjsp.passive_technician_v2_validation import run


def config(
    *,
    machines: int = 1,
    technicians: int = 1,
    horizon: int = 4,
    failure_age: int = 3,
    failure_probability: float = 0.0,
    service_time: tuple[tuple[int, ...], ...] = ((2,),),
    eligibility: tuple[tuple[bool, ...], ...] = ((True,),),
) -> PassiveTechnicianV2Config:
    return PassiveTechnicianV2Config(
        machines=machines,
        technicians=technicians,
        horizon=horizon,
        failure_age=failure_age,
        failure_probability=failure_probability,
        max_age=8,
        service_time=service_time,
        eligibility=eligibility,
    )


def test_service_occupies_full_duration_and_restores_only_on_completion() -> None:
    env = PassiveTechnicianV2Env(config(service_time=((2,),)))
    env.reset()

    _, _, done, first = env.step((1,))
    assert not done
    assert first["starts"] == [{"machine": 0, "technician": 0, "kind": "PM", "duration": 2}]
    assert first["completions"] == []
    assert env.state.modes == (MachineMode.IN_SERVICE_PM,)
    assert env.state.service_remaining == (1,)
    assert env.state.ages == (0,)
    assert env.state.technician_machine == (0,)

    _, _, _, second = env.step((1,))
    assert second["invalid_actions"] == 1
    assert second["accepted_actions"] == [0]
    assert second["completions"] == [{"machine": 0, "technician": 0, "kind": "PM"}]
    assert env.state.modes == (MachineMode.OPERATING,)
    assert env.state.ages == (0,)
    assert env.state.technician_machine == (-1,)
    assert env.technician_busy_steps == [2]


def test_in_service_machine_neither_ages_nor_fails() -> None:
    cfg = config(failure_age=1, failure_probability=1.0, service_time=((3,),))
    env = PassiveTechnicianV2Env(cfg)
    state = replace(env.initial_state(), ages=(7,))

    in_service, _, first = env.transition(state, (1,), 0, failure_events=(True,))
    assert first["failures"] == 0
    assert in_service.ages == (7,)
    assert in_service.modes == (MachineMode.IN_SERVICE_PM,)

    next_state, _, second = env.transition(in_service, (0,), 1, failure_events=(True,))
    assert second["failures"] == 0
    assert next_state.ages == (7,)
    assert next_state.modes == (MachineMode.IN_SERVICE_PM,)


def test_fifo_queue_and_rotating_simultaneous_priority() -> None:
    cfg = config(
        machines=2,
        horizon=5,
        service_time=((2,), (1,)),
        eligibility=((True,), (True,)),
    )
    env = PassiveTechnicianV2Env(cfg)
    state = env.initial_state()

    after_request, _, info = env.transition(state, (1, 1), 0, failure_events=(False, False))
    assert info["proposal_contention"] == 1
    assert after_request.technician_machine == (0,)
    assert after_request.queues == ((1,),)

    after_completion, _, _ = env.transition(
        after_request, (0, 0), 1, failure_events=(False, False)
    )
    assert after_completion.technician_machine == (-1,)
    assert after_completion.queues == ((1,),)

    after_fifo_start, _, info = env.transition(
        after_completion, (0, 0), 2, failure_events=(False, False)
    )
    assert info["starts"][0]["machine"] == 1
    assert after_fifo_start.queues == ((),)

    rotated, _, _ = env.transition(state, (1, 1), 1, failure_events=(False, False))
    assert rotated.technician_machine == (-1,)
    assert rotated.queues == ((0,),)


def test_queued_pm_failure_becomes_corrective_and_keeps_queue_place() -> None:
    cfg = config(
        machines=2,
        horizon=5,
        failure_age=1,
        service_time=((3,), (2,)),
        eligibility=((True,), (True,)),
    )
    env = PassiveTechnicianV2Env(cfg)

    occupied, _, _ = env.transition(
        env.initial_state(), (1, 0), 0, failure_events=(False, False)
    )
    queued, _, info = env.transition(occupied, (0, 1), 1, failure_events=(False, True))

    assert info["failure_machines"] == [1]
    assert queued.modes[1] == MachineMode.QUEUED_CM
    assert queued.queues == ((1,),)
    assert queued.request_times[1] == 1


def test_action_masks_encode_machine_state_and_technician_eligibility() -> None:
    cfg = config(
        machines=2,
        technicians=2,
        service_time=((2, 3), (4, 2)),
        eligibility=((True, False), (False, True)),
    )
    env = PassiveTechnicianV2Env(cfg)
    assert env.action_masks() == ((True, True, False), (True, False, True))

    queued, _, _ = env.transition(
        env.initial_state(), (1, 2), 0, failure_events=(False, False)
    )
    assert env.action_masks(queued) == ((True, False, False), (True, False, False))


def test_initial_age_profile_is_validated_and_restored_on_reset() -> None:
    cfg = config(
        machines=2,
        service_time=((2,), (2,)),
        eligibility=((True,), (True,)),
    )
    cfg = replace(cfg, initial_ages=(1, 4))
    env = PassiveTechnicianV2Env(cfg)
    assert env.state.ages == (1, 4)
    env.step((0, 0))
    assert env.state.ages == (2, 5)
    env.reset()
    assert env.state.ages == (1, 4)

    with pytest.raises(ValueError, match="initial_ages"):
        PassiveTechnicianV2Env(replace(cfg, initial_ages=(1,)))


def test_state_audit_rejects_duplicate_assignment() -> None:
    cfg = config(
        machines=1,
        technicians=2,
        service_time=((2, 2),),
        eligibility=((True, True),),
    )
    env = PassiveTechnicianV2Env(cfg)
    invalid = replace(
        env.initial_state(),
        modes=(MachineMode.IN_SERVICE_PM,),
        machine_technician=(0,),
        service_remaining=(1,),
        technician_machine=(0, 0),
        request_times=(0,),
    )
    with pytest.raises(AssertionError, match="more than one technician"):
        env.audit_state(invalid)


def test_state_audit_rejects_one_sided_assignment() -> None:
    env = PassiveTechnicianV2Env(config())
    invalid = replace(
        env.initial_state(),
        modes=(MachineMode.IN_SERVICE_PM,),
        machine_technician=(0,),
        service_remaining=(1,),
        request_times=(0,),
    )
    with pytest.raises(AssertionError, match="assignment is inconsistent"):
        env.audit_state(invalid)


def test_keyed_failure_shocks_and_episode_replay_are_deterministic() -> None:
    assert keyed_failure_uniform(7, 1, 2) == keyed_failure_uniform(7, 1, 2)
    assert keyed_failure_uniform(7, 1, 2) != keyed_failure_uniform(7, 1, 3)
    assert keyed_failure_uniform(7, 1, 2) != keyed_failure_uniform(7, 0, 2)

    cfg = config(failure_age=1, failure_probability=0.5)
    left = PassiveTechnicianV2Env(cfg, seed=99)
    right = PassiveTechnicianV2Env(cfg, seed=99)
    left.reset()
    right.reset()
    for _ in range(cfg.horizon):
        assert left.step((0,))[1:] == right.step((0,))[1:]
    assert left.state == right.state
    assert left.metrics == right.metrics


def test_terminal_cost_and_return_objective_identity() -> None:
    cfg = config(horizon=1, service_time=((2,),))
    env = PassiveTechnicianV2Env(cfg)
    env.reset()
    _, reward, done, info = env.step((1,))

    assert done
    assert info["terminal_cost"] == cfg.terminal_unfinished_cost
    assert reward == -env.metrics["objective"]
    with pytest.raises(RuntimeError, match="episode is complete"):
        env.step((0,))


def test_random_action_stress_preserves_all_state_invariants() -> None:
    cfg = PassiveTechnicianV2Config()
    for seed in range(25):
        action_rng = random.Random(seed)
        env = PassiveTechnicianV2Env(cfg, seed=seed)
        for _ in range(cfg.horizon):
            actions = tuple(
                action_rng.randrange(cfg.action_count + 1) for _ in range(cfg.machines)
            )
            env.step(actions)
            env.audit_state(env.state)


def test_exact_oracle_is_finite_and_repeatable() -> None:
    cfg = config(
        machines=2,
        horizon=3,
        failure_age=2,
        failure_probability=0.4,
        service_time=((1,), (2,)),
        eligibility=((True,), (True,)),
    )
    first = exact_expected_cost(cfg)
    assert first >= 0.0
    assert first == pytest.approx(exact_expected_cost(cfg), abs=1e-12)


def test_smoke_validation_writes_complete_auditable_artifacts(tmp_path) -> None:
    output = tmp_path / "passive_v2_smoke_20260925T000000Z"
    result = run(Namespace(profile="smoke", device="cpu", output_dir=str(output)))

    assert result == output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 12
    assert manifest["device"] == "cpu"
    assert isinstance(manifest["git_dirty"], bool)
    assert manifest["sealed_test_evaluated"] is False
    assert summary["hard_gate_passed"] is True
    assert summary["nondegeneracy_passed"] is True
    assert summary["audits"]["replay_mismatches"] == 0
    assert summary["audits"]["max_identity_error"] <= 1e-9
    assert (output / "resolved_config.json").is_file()
    assert (output / "episodes.partial.csv").is_file()
    assert (output / "episodes.csv").is_file()
    assert (output / "decisions.csv").is_file()
    assert (output / "coordination.csv").is_file()
