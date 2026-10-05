from dataclasses import replace
import itertools

import numpy as np
import pytest

from ht_pdm_fjsp.maintenance_dispatch import (
    DispatchConfig,
    DispatchEnv,
    DispatchState,
    eligible_edges,
    family_config,
    matching_actions,
    observation,
    training_config,
    transition,
)


def config(**kwargs):
    return replace(
        DispatchConfig(2, 2, 5, 3, 1.0, ((2, 1), (1, 3)), ((0.5, 1.0), (1.0, 0.75))),
        **kwargs,
    )


def state(
    ages=(4, 1),
    failed=(True, False),
    pending=(2, 0),
    remaining=(0, 0),
    assigned=(-1, -1),
):
    return DispatchState(ages, failed, pending, remaining, assigned)


def test_wait_then_service_restores_only_at_completion():
    cfg = config()
    waiting, cost, info = transition(cfg, state(), (), (True, True))
    assert waiting == state((4, 2), pending=(3, 0))
    assert cost == 6 and info["failed_waiting_ticks"] == 1
    servicing, cost, info = transition(cfg, waiting, ((0, 0),), (True, True))
    # Failed machine does not age, remains failed, and is charged once.
    assert servicing == state((4, 3), (True, True), (3, 0), (1, 0), (0, -1))
    assert cost == 2 + 6 + 15
    assert info["unavailable_ticks"] == 1 and info["failed_waiting_ticks"] == 0
    completed, cost, info = transition(cfg, servicing, (), (True, True))
    assert completed == state((2, 3), (False, True), (0, 1))
    assert cost == 12 and info["completed_jobs"] == 1
    assert info["service_ticks"] == 1 and info["failed_waiting_ticks"] == 1


def test_duration_one_releases_at_boundary_and_shocks_cannot_hit_service():
    cfg = config()
    repaired, cost, info = transition(
        cfg, state((2, 2), (False, False), (0, 0)), ((0, 1), (1, 0)), (True, True)
    )
    assert repaired == state((0, 0), (False, False), (0, 0))
    assert cost == 2 + 12 and info["failures"] == 0
    assert info["completed_jobs"] == 2 and info["preventive_jobs"] == 2


def test_busy_edge_is_masked_no_queued_reservations():
    cfg = config()
    busy = state(remaining=(1, 0), assigned=(0, -1))
    assert eligible_edges(cfg, busy) == ((False, False), (False, True))
    assert set(matching_actions(cfg, busy)) == {(), ((1, 1),)}
    for action in (((1, 0),), ((0, 1),), ((1, 1), (1, 0))):
        with pytest.raises(ValueError, match="infeasible"):
            transition(cfg, busy, action, (False, False))


@pytest.mark.parametrize(
    "action", [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((2, 0),), ((0, -1),)]
)
def test_invalid_matching_rejected_before_mutation(action):
    env = DispatchEnv(config())
    before = env.state
    with pytest.raises(ValueError):
        env.step(action)
    assert env.state == before and env.time == 0
    assert env.metrics["objective"] == 0


def test_terminal_unfinished_service_still_has_cost_and_no_completion():
    env = DispatchEnv(config(horizon=1))
    env.reset(1, state())
    _, reward, done, info = env.step(((0, 0),))
    assert done and reward == -8 and info["completed_jobs"] == 0
    assert env.state.failed[0] and env.state.ages[0] == 4
    assert env.state.remaining[0] == 1
    with pytest.raises(RuntimeError, match="finished"):
        env.step(())


def test_physical_action_order_is_irrelevant():
    cfg = config()
    pairs = ((0, 0), (1, 1))
    assert transition(cfg, state(), pairs, (True, True)) == transition(
        cfg, state(), pairs[::-1], (True, True)
    )


def test_entity_relabeling_preserves_transition_and_cost():
    cfg = config()
    initial = state(remaining=(1, 0), assigned=(0, -1))
    following, cost, info = transition(cfg, initial, ((1, 1),), (False, True))
    # Both machine and technician labels are swapped, including active links.
    shuffled = replace(
        cfg,
        service_time=tuple(tuple(r[::-1]) for r in cfg.service_time[::-1]),
        restoration=tuple(tuple(r[::-1]) for r in cfg.restoration[::-1]),
    )

    def swap(s):
        return DispatchState(
            s.ages[::-1],
            s.failed[::-1],
            s.pending_wait[::-1],
            s.remaining[::-1],
            tuple(1 - m if m >= 0 else -1 for m in s.assigned[::-1]),
        )

    actual, actual_cost, actual_info = transition(
        shuffled, swap(initial), ((0, 0),), (True, False)
    )
    assert actual == swap(following) and actual_cost == cost and actual_info == info


def test_common_random_numbers_are_policy_independent_and_hidden():
    cfg = config()
    a, b = DispatchEnv(cfg), DispatchEnv(cfg)
    obs_a, obs_b = a.reset(123), b.reset(123)
    assert a.events == b.events
    assert set(obs_a) == {"machines", "technicians", "edges", "global_features"}
    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key], obs_b[key])
    a.step(((0, 1),))
    b.step(())
    assert a.events == b.events
    obs = observation(cfg, state(), 1)
    assert obs["machines"][0, 0] == 0.5
    assert obs["machines"][0, 5] == 0  # failed machines cannot sample new failures
    assert obs["global_features"][0] == pytest.approx(4 / 12)


def test_sparse_configuration_and_all_feasible_actions_have_valid_timeline():
    for seed in range(20):
        cfg = training_config(seed, "smoke")
        cfg.validate()
        env = DispatchEnv(cfg)
        env.reset(seed)
        for action in matching_actions(cfg, env.state):
            for events in itertools.product((False, True), repeat=cfg.machines):
                following, cost, info = transition(cfg, env.state, action, events)
                assert cost == sum(
                    info[k]
                    for k in ("maintenance_cost", "unavailability_cost", "failure_cost")
                )
                assert len({m for m in following.assigned if m >= 0}) == sum(
                    r > 0 for r in following.remaining
                )
    sparse = family_config("n5_k3_sparse", 112000, "full")
    sparse.validate()
    cfg = config(service_time=((0, 1), (1, 0)))
    with pytest.raises(ValueError, match="infeasible"):
        transition(cfg, state(), ((0, 0),), (False, False))
