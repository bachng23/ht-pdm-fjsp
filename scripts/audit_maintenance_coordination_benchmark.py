"""Read-only source, checkpoint, arbitration and scalar-cost replay audit."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_coordination_benchmark as study
from ht_pdm_fjsp import maintenance_coordination_policy as agents
from ht_pdm_fjsp import maintenance_coordination_references as refs
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
    torch.set_num_threads(1)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["protocol"] == study.PROTOCOL and manifest["status"] == "COMPLETED"
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
    current = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in study.source_snapshot().items()
    }
    assert current == manifest["source_hashes"], (
        "Use the exact frozen commit for policy replay"
    )
    cfg = manifest["settings"]
    episodes = list(rows(run / "episodes.csv"))
    development = list(rows(run / "development_episodes.csv"))
    references = list(rows(run / "reference_episodes.csv"))
    message_off = list(rows(run / "message_off_episodes.csv"))
    for panel, field in [
        (episodes, "expected_test_episodes"),
        (development, "expected_development_episodes"),
        (references, "expected_reference_episodes"),
        (message_off, "expected_message_off_episodes"),
    ]:
        assert len(panel) == manifest[field] and len({key(r) for r in panel}) == len(
            panel
        )
    allrows = episodes + development + references + message_off
    assert len({key(r) for r in allrows}) == len(allrows)
    episode_index = {key(r): r for r in allrows if r["family"] != "development"}
    machines = list(rows(run / "machine_metrics.csv"))
    machine_index = {(key(r), int(r["machine"])): r for r in machines}
    assert len(machine_index) == len(machines)
    assert study.population_audit(
        run / "training_episodes.csv", cfg, manifest["profile"]
    )
    selection = json.loads((run / "reference_selection.json").read_text())
    # Convert only the development data required by the frozen selection rule.
    dev_refs = [
        {**r, "base_cost": float(r["base_cost"])}
        for r in references
        if r["family"] == "development"
    ]
    dev_machines = [
        {**r, "max_pending_wait": int(r["max_pending_wait"])}
        for r in machines
        if r["family"] == "development" and r["algorithm"] in refs.REFERENCES
    ]
    assert study.choose_reference(dev_refs, dev_machines) == selection
    configs = json.loads((run / "evaluation_configs.json").read_text())
    controllers = {}
    current_key = None
    seen = set()
    total_ticks = proposal_ticks = 0
    sim = None
    for r in tqdm(
        rows(run / "decisions.csv"), desc="replay policy + scalar costs", unit="tick"
    ):
        k = key(r)
        a, s, family, e = k
        if k != current_key:
            if sim is not None:
                assert sim.time == sim.config.horizon
            assert k not in seen
            seen.add(k)
            current_key = k
            sim = envmod.WaitingEnv(envmod.WaitingConfig(**configs[family][str(e)]))
            sim.reset(e)
            jobs, waits, unavailable, maxwait = (
                [0] * sim.config.machines,
                [0] * sim.config.machines,
                [0] * sim.config.machines,
                [0] * sim.config.machines,
            )
            accumulated = dict.fromkeys(study.DIAGNOSTICS, 0)
            name = "mappo_message" if a == "mappo_message_off" else a
            if (name, s) not in controllers:
                if name in refs.REFERENCES:
                    controllers[name, s] = refs.MatchingReference(name)
                else:
                    controllers[name, s], payload = study.load_checkpoint(
                        run / name / f"train_seed_{s}" / "model.pt"
                    )
                    assert payload["settings"] == cfg
            controller = controllers[name, s]
            if name == "mappo_message":
                controller.message_enabled = a != "mappo_message_off"
        assert int(r["time"]) == sim.time
        state, config = sim.state, sim.config
        for field in ("ages", "failed", "pending_wait", "remaining", "assigned"):
            assert tuple(json.loads(r[field])) == getattr(state, field)
        obs = envmod.observation(config, state, sim.time)
        pairs = tuple(tuple(p) for p in json.loads(r["pairs"]))
        action, tokens, _, _ = controller.act(obs, "learned_both", deterministic=True)
        assert tuple(action) == pairs and json.loads(r["sequence"]) == [
            list(p) for p in tokens
        ]
        if a in agents.PROPOSAL_ARMS or a == "mappo_message_off":
            proposals = json.loads(r["proposals"])
            assert proposals == json.loads(controller.last_diagnostics["proposals"])
            # Independent arbitration calculation, not a call to resolve().
            winners = []
            for j in range(config.technicians):
                candidates = [m for m, p in enumerate(proposals) if p == j]
                if candidates:
                    winner = min(
                        candidates,
                        key=lambda m: (
                            -(
                                100 * float(obs["machines"][m, 1])
                                + 10 * float(obs["machines"][m, 2])
                                + float(obs["machines"][m, 0])
                                + float(obs["machines"][m, 5])
                            ),
                            m,
                        ),
                    )
                    winners.append((winner, j))
            assert tuple(sorted(winners)) == pairs
            for field, value in controller.last_diagnostics.items():
                if field != "proposals":
                    assert float(r[field]) == value
            proposal_ticks += 1
        selected = {m for m, j in pairs}
        servicing = selected | {m for m in state.assigned if m >= 0}
        # Scalar economic calculation independent from env.step's cost result.
        economic = (
            sum(
                config.corrective_cost if state.failed[m] else config.preventive_cost
                for m, j in pairs
            )
            + config.unavailable_cost
            * sum(f or m in servicing for m, f in enumerate(state.failed))
            + config.failure_cost * float(r["failures"])
        )
        overdue = sum(
            f
            and m not in servicing
            and state.pending_wait[m] + 1 > config.waiting_limit
            for m, f in enumerate(state.failed)
        )
        assert economic == float(r["base_cost"])
        assert overdue == float(r["overdue_waiting_ticks"])
        assert config.waiting_price * overdue == float(r["waiting_cost"])
        for m in range(config.machines):
            jobs[m] += int(m in selected)
            waits[m] += int(state.failed[m] and m not in servicing)
            unavailable[m] += int(state.failed[m] or m in servicing)
            maxwait[m] = max(maxwait[m], state.pending_wait[m])
        for field in study.DIAGNOSTICS:
            accumulated[field] += float(r[field])
        _, _, _, info = sim.step(pairs)
        assert all(abs(float(r[field]) - value) < 1e-8 for field, value in info.items())
        total_ticks += 1
        if sim.time == config.horizon:
            episode = episode_index[k]
            assert all(
                abs(float(episode[field]) - value) < 1e-8
                for field, value in sim.metrics.items()
            )
            assert all(
                float(episode[field]) == value for field, value in accumulated.items()
            )
            for m in range(config.machines):
                mr = machine_index[k, m]
                assert int(mr["service_starts"]) == jobs[m]
                assert int(mr["failed_waiting_ticks"]) == waits[m]
                assert int(mr["unavailable_ticks"]) == unavailable[m]
                assert int(mr["max_pending_wait"]) == max(
                    maxwait[m], sim.state.pending_wait[m]
                )
    assert sim.time == sim.config.horizon
    assert seen == set(episode_index)
    return dict(
        status="PASS",
        replayed_episodes=len(seen),
        replayed_ticks=total_ticks,
        proposal_ticks=proposal_ticks,
        source_and_checkpoint_replay=True,
        independent_arbitration=True,
        independent_scalar_costs=True,
        per_machine_reconciliation=True,
        development_only_reference_reselection=True,
        physical_training_population_identical=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve()), indent=2))


if __name__ == "__main__":
    main()
