"""Read-only replay/independent-cost audit of a completed context/waiting run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from ht_pdm_fjsp import maintenance_context_waiting as study
from ht_pdm_fjsp import maintenance_waiting as envmod


def rows(path):
    with path.open() as f:
        yield from csv.DictReader(f)


def key(row):
    return (
        row["algorithm"],
        int(row["train_seed"]),
        row["family"],
        int(row["eval_seed"]),
    )


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["protocol"] == study.PROTOCOL
    assert all(manifest["audits"].values())
    for name, digest in manifest["source_hashes"].items():
        assert (
            hashlib.sha256((run / "source_snapshot" / name).read_bytes()).hexdigest()
            == digest
        )
    assert (
        hashlib.sha256(
            json.dumps(manifest["source_hashes"], sort_keys=True).encode()
        ).hexdigest()
        == manifest["source_bundle_sha256"]
    )
    cfg = manifest["settings"]
    episodes = list(rows(run / "episodes.csv"))
    development = list(rows(run / "development_episodes.csv"))
    references = list(rows(run / "reference_episodes.csv"))
    learned_machines = list(rows(run / "machine_metrics.csv"))
    reference_machines = list(rows(run / "reference_machine_metrics.csv"))
    actual_keys = {key(r) for r in episodes + development + references}
    assert len(actual_keys) == len(episodes + development + references)
    assert len(episodes) == manifest["expected_test_episodes"]
    assert len(development) == manifest["expected_development_episodes"]
    assert len(references) == manifest["expected_reference_episodes"]
    episode_index = {
        key(r): r for r in episodes + references if r["family"] != "development"
    }
    machine_index = {
        (key(r), int(r["machine"])): r
        for r in learned_machines + reference_machines
        if r["family"] != "development"
    }
    tick_count = 0
    replayed = {}
    machine_jobs = {}
    machine_max_wait = {}
    for filename in ("decisions.csv", "reference_decisions.csv"):
        current = None
        simulator = None
        for row in rows(run / filename):
            k = key(row)
            arm, train_seed, family, ev = k
            if k != current:
                if simulator is not None:
                    assert simulator.time == simulator.config.horizon
                current = k
                assert k not in replayed
                price = 0 if arm == study.RULE else study.factors(arm)[1]
                config = envmod.family_config(family, ev, manifest["profile"], price)
                simulator = envmod.WaitingEnv(config)
                simulator.reset(ev)
                replayed[k] = dict(
                    base_cost=0.0,
                    waiting_cost=0.0,
                    overdue_waiting_ticks=0.0,
                    objective=0.0,
                    service_adjusted_cost=0.0,
                )
                for m in range(config.machines):
                    machine_jobs[k, m] = 0
                    machine_max_wait[k, m] = 0
            assert int(row["time"]) == simulator.time
            state = simulator.state
            for field in ("ages", "failed", "pending_wait", "remaining", "assigned"):
                assert tuple(json.loads(row[field])) == getattr(state, field)
            pairs = tuple(tuple(p) for p in json.loads(row["pairs"]))
            selected = {m for m, j in pairs}
            servicing = selected | {m for m in state.assigned if m >= 0}
            # Separate scalar calculation from the simulator's returned cost.
            economic = (
                sum(
                    config.corrective_cost
                    if state.failed[m]
                    else config.preventive_cost
                    for m, j in pairs
                )
                + config.unavailable_cost
                * sum(f or m in servicing for m, f in enumerate(state.failed))
                + config.failure_cost * float(row["failures"])
            )
            overdue = sum(
                f
                and m not in servicing
                and state.pending_wait[m] >= config.waiting_limit
                for m, f in enumerate(state.failed)
            )
            assert economic == float(row["base_cost"])
            assert overdue == float(row["overdue_waiting_ticks"])
            assert config.waiting_price * overdue == float(row["waiting_cost"])
            assert economic + config.waiting_price * overdue == float(row["objective"])
            assert economic + envmod.WAIT_PRICE * overdue == float(
                row["service_adjusted_cost"]
            )
            for m in range(config.machines):
                machine_jobs[k, m] += int(m in selected)
                machine_max_wait[k, m] = max(
                    machine_max_wait[k, m], state.pending_wait[m]
                )
            _, _, _, info = simulator.step(pairs)
            for name, value in info.items():
                assert abs(float(row[name]) - value) < 1e-8
            for name in replayed[k]:
                replayed[k][name] += info[name]
            tick_count += 1
            if simulator.time == config.horizon:
                for m in range(config.machines):
                    mr = machine_index[k, m]
                    maximum = max(
                        machine_max_wait[k, m], simulator.state.pending_wait[m]
                    )
                    assert maximum == int(mr["max_pending_wait"])
                    assert machine_jobs[k, m] == int(mr["service_starts"])
                    assert simulator.state.pending_wait[m] == int(
                        mr["pending_wait_final"]
                    )
                    assert simulator.state.failed[m] == (
                        mr["failed_at_terminal"] == "True"
                    )
        if simulator is not None:
            assert simulator.time == simulator.config.horizon
    assert set(replayed) == set(episode_index)
    for k, components in replayed.items():
        for name, value in components.items():
            assert abs(float(episode_index[k][name]) - value) < 1e-8
    for arm in study.ARMS:
        for seed in cfg["train_seeds"]:
            model, payload = study.load_checkpoint(
                run / arm / f"train_seed_{seed}" / "model.pt"
            )
            assert model.technician_context == study.factors(arm)[0]
            assert payload["source_bundle_sha256"] == manifest["source_bundle_sha256"]
    assert study.training_population_audit(run / "training_episodes.csv", cfg)
    exact = list(rows(run / "exact_small_metrics.csv"))
    assert len(exact) == manifest["expected_exact_rows"]
    for row in exact:
        assert row["exact_objective"] == "base_cost"
        gap = float(row["exact_policy_cost"]) - float(row["exact_optimum"])
        assert gap >= -1e-8 and abs(gap - float(row["optimality_gap"])) < 1e-8
        assert (
            abs(
                gap
                - float(row["exact_selection_regret"])
                - float(row["exact_allocation_regret"])
            )
            < 1e-8
        )
    train = list(rows(run / "training_episodes.csv"))
    assert len(train) == manifest["actual_training_episodes"]
    assert all(
        abs(float(r["objective"]) - float(r["base_cost"]) - float(r["waiting_cost"]))
        < 1e-8
        for r in train
    )
    console = run.with_suffix(".log").read_text()
    assert "train local_cost/" in console and "evaluate context_wait/" in console
    return dict(
        status="PASSED",
        run=str(run),
        replayed_test_episodes=len(replayed),
        replayed_physical_ticks=tick_count,
        training_episodes_checked=len(train),
        source_snapshot_verified=True,
        checkpoint_contracts_verified=True,
        independent_scalar_cost_and_waiting_boundary_verified=True,
        simulator_replay="same implementation; not independent simulator reproduction",
        training_wait_penalty_positive_rows=sum(
            float(r["waiting_cost"]) > 0 for r in train
        ),
        full_scientific_inference=manifest["profile"] == "full",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
