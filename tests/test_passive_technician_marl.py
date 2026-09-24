from pathlib import Path

from ht_pdm_fjsp.passive_technician_marl import (
    PassiveConfig,
    PassiveTechnicianEnv,
    dispatcher_action,
    exact_optimum,
)


def test_fixed_technician_service_times_and_collision_resolution() -> None:
    config = PassiveConfig(machines=2, technicians=1, horizon=2, service_time=((1,), (3,)))
    env = PassiveTechnicianEnv(config)
    env.reset()
    _, _, _, info = env.step((1, 1))
    assert info["collisions"] == 1
    assert env.state.busy_until == (1,)


def test_dispatcher_uses_fastest_feasible_technician() -> None:
    config = PassiveConfig()
    env = PassiveTechnicianEnv(config)
    env.reset()
    env.state = env.state.__class__((4, 4, 1), (True, False, False), (0, 0), (-1, -1))
    assert dispatcher_action(env)[0] in (1, 2)


def test_exact_solver_is_finite_and_nonnegative() -> None:
    config = PassiveConfig(machines=2, technicians=1, horizon=3, service_time=((1,), (1,)))
    assert exact_optimum(config) >= 0
