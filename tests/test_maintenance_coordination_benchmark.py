import argparse
import copy
import csv
import itertools
import json
import math
from functools import lru_cache

import pytest
import torch

from ht_pdm_fjsp import maintenance_dispatch as physics
from ht_pdm_fjsp import maintenance_dispatch_policy as bp
from ht_pdm_fjsp import maintenance_waiting as env
from ht_pdm_fjsp import maintenance_coordination_benchmark as study
from ht_pdm_fjsp import maintenance_coordination_policy as agents
from ht_pdm_fjsp import maintenance_coordination_references as refs
from ht_pdm_fjsp import maintenance_coordination_training as training


def config():
    return env.WaitingConfig(
        3,
        2,
        6,
        3,
        0.5,
        ((2, 2), (1, 2), (2, 0)),
        ((0.75, 0.75), (1.0, 0.5), (0.5, 0.5)),
        waiting_price=12,
    )


def observation():
    return env.observation(
        config(),
        physics.DispatchState(
            (2, 4, 0), (False, True, False), (0, 4, 0), (0, 0), (-1, -1)
        ),
        0,
    )


@pytest.mark.parametrize("arm", agents.PROPOSAL_ARMS)
def test_identical_modules_initialization(arm):
    cfg = study.settings("smoke")
    a = study.model_for(cfg, 33, "ippo_local")
    b = study.model_for(cfg, 33, arm)
    assert sum(p.numel() for p in a.parameters()) == sum(
        p.numel() for p in b.parameters()
    )
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


@pytest.mark.parametrize("arm", ["ippo_local", "mappo_local"])
def test_local_actor_has_no_peer_or_occupant_status_leak(arm):
    torch.manual_seed(42)
    model = agents.ProposalActorCritic(8, arm)
    obs = observation()
    peer = copy.deepcopy(obs)
    peer["machines"][1] = [1, 0, 0, 0, 0, 0.7]
    peer["edges"][1] = [[0, 0, 0, 0], [1, 0.3, 0.25, 0]]
    peer["technicians"][0, 2] = 1  # Hidden occupant failure status never public.
    a, av = model.distributions(bp.batch_observations([obs]))
    b, bv = model.distributions(bp.batch_observations([peer]))
    assert torch.equal(a.probs[0, 0], b.probs[0, 0])
    if arm == "ippo_local":
        assert torch.equal(av[0, 0], bv[0, 0])
    else:
        assert not torch.allclose(av[0, 0], bv[0, 0], atol=1e-8, rtol=0)


@pytest.mark.parametrize("arm", ["central_proposal", "mappo_message"])
def test_peer_information_changes_logits_and_message_off_removes_it(arm):
    torch.manual_seed(42)
    model = agents.ProposalActorCritic(8, arm)
    obs = observation()
    peer = copy.deepcopy(obs)
    peer["machines"][1, 0] = 1
    peer["edges"][1, :, 2] = 0.1
    a, _ = model.distributions(bp.batch_observations([obs]))
    b, _ = model.distributions(bp.batch_observations([peer]))
    assert not torch.allclose(a.probs[0, 0], b.probs[0, 0], atol=1e-8, rtol=0)
    if arm == "mappo_message":
        model.message_enabled = False
        a, _ = model.distributions(bp.batch_observations([obs]))
        b, _ = model.distributions(bp.batch_observations([peer]))
        assert torch.equal(a.probs[0, 0], b.probs[0, 0])


def test_arbitration_rejects_collisions_and_does_not_reassign():
    obs = observation()
    pairs, diag = agents.resolve(obs, [0, 0, 0])
    assert pairs == ((1, 0),)  # Failed/waiting machine wins; worker1 remains idle.
    assert diag["rejected_proposals"] == 2
    assert diag["contested_technicians"] == 1
    assert diag["avoidable_unassigned_requests"] == 1
    state = physics.DispatchState(
        (2, 4, 0), (False, True, False), (0, 4, 0), (0, 0), (-1, -1)
    )
    physics.validate_matching(config(), state, pairs)
    with pytest.raises(ValueError, match="infeasible"):
        agents.resolve(obs, [2, 2, 1])
    with pytest.raises(ValueError, match="one proposal"):
        agents.resolve(obs, [0])


def test_arbitration_stop_and_stable_identifier_tie():
    obs = observation()
    obs["machines"][:] = [0, 0, 0, 0, 0, 0]
    pairs, diag = agents.resolve(obs, [0, 0, 2])
    assert pairs == ((0, 0),) and diag["proposal_count"] == 2
    assert agents.resolve(obs, [2, 2, 2])[0] == ()


@pytest.mark.parametrize("arm", agents.PROPOSAL_ARMS)
def test_sample_replay_mixed_sizes_and_busy_forced_stop(arm):
    torch.manual_seed(27)
    model = agents.ProposalActorCritic(8, arm)
    small = env.WaitingEnv(
        env.WaitingConfig(
            2, 1, 4, 3, 0.5, ((1,), (2,)), ((1.0,), (0.5,)), waiting_price=12
        )
    ).reset(21)
    busy = env.observation(
        config(),
        physics.DispatchState(
            (2, 4, 0), (False, True, False), (0, 4, 0), (2, 0), (0, -1)
        ),
        0,
    )
    observations = [small, observation(), busy]
    samples = [
        model.propose(obs, generator=torch.Generator().manual_seed(31 + i))
        for i, obs in enumerate(observations)
    ]
    batch, tokens, oldp, oldv = training.proposal_batch(observations, *zip(*samples))
    p, entropy, v = model.evaluate(batch, tokens)
    mask = batch["machine_exists"]
    assert torch.allclose(p[mask], oldp[mask], atol=1e-6)
    assert torch.allclose(v[mask], oldv[mask], atol=1e-6)
    assert samples[2][0][0] == 2 and samples[2][1][0] == 0
    assert torch.isfinite(entropy).all()
    for obs, sample in zip(observations, samples):
        agents.resolve(obs, sample[0])
    # Illegal replayed edge must abort; don't renormalize a false action.
    tokens[2, 0] = 0
    with pytest.raises(RuntimeError, match="infeasible"):
        model.evaluate(batch, tokens)


def test_per_row_weighting_and_mc_team_returns():
    obs = [
        env.WaitingEnv(env.family_config("small", 11, "smoke", 12)).reset(2),
        observation(),
    ]
    weights = training.row_weights(bp.batch_observations(obs))
    assert torch.allclose(weights.sum(-1), torch.ones(2))
    assert weights[0, 2] == 0
    assert float(
        training.weighted_mean(
            torch.tensor([[2.0, 2.0, 999.0], [4.0, 4.0, 4.0]]), weights
        )
    ) == pytest.approx(3)
    assert torch.equal(
        study.core.returns_for_episodes([-1, -2, -3, -4], [False, True, False, True]),
        torch.tensor([-3.0, -2.0, -7.0, -4.0]),
    )


@pytest.mark.parametrize("seed", range(4))
def test_optimal_matching_against_exhaustive_subset_enumeration(seed):
    generator = torch.Generator().manual_seed(seed)
    weights = (torch.randint(-3, 8, (4, 3), generator=generator).double()).tolist()
    weights[0][1] = -math.inf
    result = refs.maximum_weight_matching(weights)
    candidates = []
    for actions in itertools.product(range(4), repeat=4):
        pairs = tuple((m, j) for m, j in enumerate(actions) if j < 3)
        if len({j for m, j in pairs}) < len(pairs):
            continue
        if any(
            weights[m][j] <= 0 or not math.isfinite(weights[m][j]) for m, j in pairs
        ):
            continue
        candidates.append((sum(weights[m][j] for m, j in pairs), pairs))
    expected = min(candidates, key=lambda r: (-r[0], r[1]))[1]
    assert result == expected
    assert refs.maximum_weight_matching([[-1, -2], [0, -math.inf]]) == ()


@pytest.mark.parametrize("duration", [1, 2, 3])
@pytest.mark.parametrize("failed,wait", [(False, 0), (True, 3), (True, 4)])
def test_isolated_lookahead_against_one_machine_physical_dp(duration, failed, wait):
    cfg = env.WaitingConfig(
        1, 1, 4, 3, 0.25, ((duration,),), ((0.75,),), waiting_price=12
    )
    state = physics.DispatchState((2,), (failed,), (wait,), (0,), (-1,))

    @lru_cache(None)
    def dp(t, state):
        if t == 4:
            return 0
        return min(q(t, state, a) for a in physics.matching_actions(cfg, state))

    def q(t, state, a):
        value = 0
        for event, prob in [(False, 0.75), (True, 0.25)]:
            following, cost, _ = env.transition(cfg, state, a, (event,))
            value += prob * (cost + dp(t + 1, following))
        return value

    isolated = refs.isolated_value(
        4, 2, failed, wait, 0, duration, 0.75, 3, 0.25, 1, 2, 6, 15, 4, 12
    )
    assert isolated == pytest.approx(dp(0, state))
    obs = env.observation(cfg, state, 0)
    assert refs.marginal_gain(obs, 0, 0) == pytest.approx(
        q(0, state, ()) - q(0, state, ((0, 0),))
    )


def test_reference_selection_uses_development_and_respects_feasibility():
    episodes, machines = [], []
    for name, cost, wait in zip(refs.REFERENCES, [20, 10, 15], [0, 7, 2]):
        episodes.extend(
            [
                dict(algorithm=name, family="development", base_cost=cost),
                dict(
                    algorithm=name,
                    family="n3_pressure",
                    base_cost=0 if name == "deadline_matching" else 999,
                ),
            ]
        )
        machines.extend(
            [
                dict(algorithm=name, family="development", max_pending_wait=wait),
                dict(algorithm=name, family="n3_pressure", max_pending_wait=0),
            ]
        )
    selection = study.choose_reference(episodes, machines)
    assert (
        selection["selected"] == "lookahead_matching"
        and selection["feasible_reference_exists"]
    )
    for r in machines:
        if r["family"] == "development":
            r["max_pending_wait"] = 5
    assert not study.choose_reference(episodes, machines)["feasible_reference_exists"]


def test_full_gates_require_all_conditions_and_smoke_never_claims_science():
    differences = [-10.0] * 10
    assert (
        study.hypothesis(differences, 0.1, True, True, True, True)["status"] == "PASS"
    )
    for kw in [
        dict(learner_sla=False),
        dict(base_guard=False),
        dict(nominal_guard=False),
        dict(eligible=False),
        dict(reductions=0.049),
        dict(differences=[-1] * 7 + [1] * 3),
    ]:
        args = dict(
            differences=differences,
            reductions=0.1,
            learner_sla=True,
            base_guard=True,
            nominal_guard=True,
            full=True,
        )
        args.update(kw)
        assert study.hypothesis(**args)["status"] == "FAIL"
    assert (
        study.hypothesis([-100], 0.5, True, True, True, False)["status"]
        == "ENGINEERING_ONLY"
    )
    with pytest.raises(RuntimeError, match="ten paired"):
        study.hypothesis([-100], 0.5, True, True, True, True)


def test_seed_budgets_and_nonempty_full_dirty_guards(tmp_path, monkeypatch):
    for profile in ("smoke", "full"):
        assert study.seed_audit(study.settings(profile))["prior_declared_disjoint"]
    cfg = study.settings("full")
    assert len(study.ARMS) * len(cfg["train_seeds"]) * cfg["env_steps"] == 6000000
    cfg["evaluation_seeds"] = [124000]
    with pytest.raises(ValueError, match="historical"):
        study.seed_audit(cfg)
    (tmp_path / "existing").write_text("preserve")
    with pytest.raises(FileExistsError):
        study.run(
            argparse.Namespace(profile="smoke", device="cpu", output_dir=str(tmp_path))
        )
    monkeypatch.setattr(study.helpers, "_git_state", lambda: ("abc", True))
    with pytest.raises(RuntimeError, match="clean"):
        study.run(
            argparse.Namespace(
                profile="full", device="cpu", output_dir=str(tmp_path / "new")
            )
        )
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("arm", ["ippo_local", "mappo_local", "mappo_message"])
def test_agent_training_smoke_return_audit_and_finite_gradients(tmp_path, arm):
    cfg = study.settings("smoke")
    torch.set_num_threads(1)
    model = study.model_for(cfg, 131000, arm)
    trained, coverage = training.fit(model, cfg, 131000, "smoke", tmp_path, [])
    assert (
        coverage["env_steps"] == 48
        and coverage["episodes"] == 12
        and coverage["optimizer_steps"] == 8
    )
    assert (
        coverage["complete_episode_return_audit"]
        and coverage["sampled_agent_probability_audit"]
    )
    assert all(torch.isfinite(p).all() for p in trained.parameters())
    records = list(csv.DictReader((tmp_path / "training_episodes.csv").open()))
    assert len(records) == 12 and all(
        float(r["objective"]) == float(r["base_cost"]) + float(r["waiting_cost"])
        for r in records
    )


def test_end_to_end_smoke_panels_checkpoint_and_trace_replay(tmp_path):
    output = study.run(
        argparse.Namespace(
            profile="smoke", device="cpu", output_dir=str(tmp_path / "smoke")
        )
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED" and all(manifest["audits"].values())
    assert manifest["actual_test_episodes"] == 120
    assert manifest["actual_reference_episodes"] == 75
    assert manifest["actual_message_off_episodes"] == 24
    assert (
        manifest["actual_training_steps"] == 240
        and manifest["actual_optimizer_steps"] == 40
    )
    summary = json.loads((output / "summary.json").read_text())
    assert not summary["marl_necessity_claim"]
    assert all(
        r["status"] == "ENGINEERING_ONLY" for r in summary["hypotheses"].values()
    )
    cfgs = json.loads((output / "evaluation_configs.json").read_text())
    models = {}
    previous = None
    with (output / "decisions.csv").open() as f:
        for r in csv.DictReader(f):
            key = r["algorithm"], int(r["train_seed"]), r["family"], int(r["eval_seed"])
            a, s, family, e = key
            if key != previous:
                sim = env.WaitingEnv(env.WaitingConfig(**cfgs[family][str(e)]))
                sim.reset(e)
                previous = key
            obs = env.observation(sim.config, sim.state, sim.time)
            pairs = tuple(tuple(p) for p in json.loads(r["pairs"]))
            if a in refs.REFERENCES:
                controller = refs.MatchingReference(a)
            else:
                name = "mappo_message" if a == "mappo_message_off" else a
                if (name, s) not in models:
                    models[name, s], _ = study.load_checkpoint(
                        output / name / f"train_seed_{s}" / "model.pt"
                    )
                controller = models[name, s]
                if name == "mappo_message":
                    controller.message_enabled = a != "mappo_message_off"
            assert (
                tuple(controller.act(obs, "learned_both", deterministic=True)[0])
                == pairs
            )
            if a in agents.PROPOSAL_ARMS or a == "mappo_message_off":
                assert json.loads(r["proposals"]) == json.loads(
                    controller.last_diagnostics["proposals"]
                )
                assert agents.resolve(obs, json.loads(r["proposals"]))[0] == pairs
            _, _, _, info = sim.step(pairs)
            assert all(float(r[k]) == value for k, value in info.items())


def test_agent_trainer_uses_positive_overdue_cost_in_team_mc_targets(
    tmp_path, monkeypatch
):
    cfg = study.settings("smoke")
    forced = env.WaitingConfig(
        2, 1, 4, 3, 0.5, ((1,), (2,)), ((1.0,), (0.5,)), waiting_price=12
    )
    original_environment = env.WaitingEnv

    class ForcedWaiting(original_environment):
        def reset(self, seed, initial=None):
            initial = physics.DispatchState((3, 3), (True, True), (4, 4), (0,), (-1,))
            return super().reset(seed, initial)

    monkeypatch.setattr(env, "WaitingEnv", ForcedWaiting)
    monkeypatch.setattr(env, "training_config", lambda *args: forced)
    model = study.model_for(cfg, 131000, "mappo_message")
    with torch.no_grad():
        model.stop_head[-1].bias.fill_(100)
        model.proposal_head[-1].bias.fill_(-100)
    _, coverage = training.fit(model, cfg, 131000, "smoke", tmp_path, [])
    assert coverage["complete_episode_return_audit"]
    records = list(csv.DictReader((tmp_path / "training_episodes.csv").open()))
    assert len(records) == 12
    assert all(float(r["overdue_waiting_ticks"]) == 8 for r in records)
    assert all(
        float(r["waiting_cost"]) == 96
        and float(r["base_cost"]) == 48
        and float(r["objective"]) == 144
        for r in records
    )
