from ht_pdm_fjsp.passive_technician_marl import PassiveConfig
from ht_pdm_fjsp.passive_technician_semantics import run_episode


def test_invalid_semantics_counts_duplicate_proposals() -> None:
    config = PassiveConfig(machines=3, technicians=1, horizon=4, service_time=((2,), (2,), (2,)))
    row = run_episode(config, "invalid_collision", "greedy_independent", 7)
    assert row["invalid_proposals"] >= 0
    assert row["collision_events"] >= row["invalid_proposals"] / 2


def test_queue_semantics_records_waiting() -> None:
    config = PassiveConfig(machines=3, technicians=1, horizon=8, service_time=((2,), (2,), (2,)))
    row = run_episode(config, "fifo_queue", "greedy_independent", 7)
    assert row["waiting"] >= 0
    assert row["jobs"] >= 0
