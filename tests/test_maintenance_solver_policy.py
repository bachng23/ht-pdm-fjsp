"""Reservation likelihood, physical reward credit and actor information audits."""
from dataclasses import replace
import math
import pytest
import torch
from ht_pdm_fjsp import maintenance_solver_policy as policy
from ht_pdm_fjsp import maintenance_heterogeneous_policy as previous
from ht_pdm_fjsp.maintenance_heterogeneous_model import cohort_config, feasible, fixed_candidates
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def setup(condition="nominal"):
    c = cohort_config(165020, condition, 12)
    state = DispatchState((2,) * 4, (False,) * 4, (0,) * 4,
                          (0,) * c.technicians, (-1,) * c.technicians)
    return c, state


@pytest.mark.parametrize("arm", policy.ARMS)
def test_identical_initial_tensors_and_sample_teacher_forcing_gradients(arm):
    models = []
    for a in policy.ARMS:
        torch.manual_seed(91)
        models.append(policy.MatchingActorCritic(a, 16))
    for model in models[1:]:
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in models[0].state_dict().items())
    model = models[policy.ARMS.index(arm)]
    c, state = setup()
    batch = policy.feature_batch([c] * 8, [state] * 8, [12, 11, 10, 9] * 2)
    action, prob, values = model.sample(batch, torch.Generator().manual_seed(8))
    checked, entropy, checked_values = model.evaluate(batch, action)
    assert torch.allclose(prob, checked, atol=1e-6)
    assert torch.allclose(values, checked_values, atol=1e-6)
    assert prob.shape == ((8, 4) if arm == policy.INDEPENDENT else (8,))
    loss = -checked.mean() + checked_values.square().mean() - .01 * entropy.mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    for i in range(8):
        model.decode(c, state, 12 - i % 4, action[i])


@pytest.mark.parametrize("arm", policy.SERIAL)
def test_positive_service_logits_use_every_symmetric_worker(arm):
    c, state = setup()
    c = replace(c, service_time=((2, 2, 2, 2),) * 4, restoration=((.75,) * 4,) * 4)
    model = policy.MatchingActorCritic(arm, 16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Actor coordinate 6h distinguishes WAIT (0) from servicing (1/4).
        model.actor[0].weight[0, 6 * model.hidden] = 4
        model.actor[2].weight[0, 0] = 1
    for time in range(4):
        pairs, diagnostic = model.select(c, state, 12 - time)
        assert len(pairs) == 4
        assert len({j for _, j in pairs}) == 4
        assert diagnostic["collision_rejections"] == 0
        assert dict(pairs)[time] == 0


@pytest.mark.parametrize("arm", policy.SERIAL)
@pytest.mark.parametrize("condition", ["nominal", "skill_mask"])
def test_every_feasible_matching_has_positive_normalized_serial_probability(arm, condition):
    c, state = setup(condition)
    model = policy.MatchingActorCritic(arm, 16)
    matchings = feasible(c, state)
    raw = []
    for pairs in matchings:
        row = [0] * 4
        for m, j in pairs:
            row[m] = j + 1
        raw.append(row)
    for time in range(4):
        batch = policy.feature_batch([c] * len(raw), [state] * len(raw), [12 - time] * len(raw))
        probabilities, _, _ = model.evaluate(batch, torch.tensor(raw))
        if arm == policy.INDEPENDENT:
            probabilities = probabilities.sum(-1)
        assert torch.isfinite(probabilities).all()
        assert math.isclose(float(probabilities.exp().sum().detach()), 1.0, abs_tol=1e-6)
        for pairs, action in zip(matchings, raw, strict=True):
            assert model.decode(c, state, 12 - time, action)[0] == pairs
    bad = torch.ones((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="infeasible"):
        model.evaluate(policy.feature_batch([c], [state], [12]), bad)


@pytest.mark.parametrize("arm", [policy.INDEPENDENT, policy.LOCAL])
def test_local_logits_and_independent_critic_do_not_leak_other_machines(arm):
    c, state = setup()
    model = policy.MatchingActorCritic(arm, 16)
    batch = policy.feature_batch([c], [state], [12])
    altered = {k: v.clone() for k, v in batch.items()}
    altered["machines"][:, 1:, :6] = torch.randn(1, 3, 6)
    altered["edges"][:, 1:] = torch.randn(1, 3, 4, 4)
    altered["technicians"][:, :, 2] = 1
    altered["proposal_mask"][:, 1:] = False
    altered["matching_mask"][:] = False
    altered["fixed_mask"][:] = False
    for reserved in [torch.zeros((1, 4), dtype=torch.bool), torch.tensor([[False, True, False, True]])]:
        dist, mask = model.serial_distribution(batch, torch.tensor([0]), reserved)
        other, other_mask = model.serial_distribution(altered, torch.tensor([0]), reserved)
        assert torch.equal(mask, other_mask)
        assert torch.equal(dist.logits, other.logits)
    if arm == policy.INDEPENDENT:
        v1 = model._values(batch, model._encoded(batch, local=True))
        v2 = model._values(altered, model._encoded(altered, local=True))
        assert torch.equal(v1[:, 0], v2[:, 0])


@pytest.mark.parametrize("arm", [policy.CENTRAL, policy.FIXED])
def test_central_and_fixed_behavior_exactly_preserved(arm):
    torch.manual_seed(94)
    model = policy.MatchingActorCritic(arm, 16)
    torch.manual_seed(94)
    old = previous.MatchingActorCritic(arm, 16)
    c, state = setup("specialized")
    batch = policy.feature_batch([c], [state], [12])
    dist, values = model.distribution(batch)
    old_dist, old_values = old.distribution(batch)
    assert torch.equal(dist.logits, old_dist.logits)
    assert torch.equal(values, old_values)
    assert model.select(c, state, 12)[0] == old.select(c, state, 12)[0]
    if arm == policy.FIXED:
        assert model.select(c, state, 12)[0] in fixed_candidates(c, state)


def test_per_machine_costs_reconcile_physics_and_complete_episode_returns():
    from ht_pdm_fjsp.maintenance_solver_training import machine_costs, monte_carlo
    for condition in ("specialized", "nominal", "skill_mask"):
        c, _ = setup(condition)
        env = WaitingEnv(c)
        env.reset(165040, DispatchState((3,) * 4, (True,) * 4, (5,) * 4,
                                       (0,) * c.technicians, (-1,) * c.technicians))
        rewards = []
        generator = torch.Generator().manual_seed(4)
        for t in range(12):
            candidates = feasible(c, env.state)
            choice = candidates[int(torch.randint(len(candidates), (), generator=generator))]
            before = env.state
            _, reward, _, info = env.step(choice)
            costs = machine_costs(c, before, env.state, choice, info)
            assert sum(costs) == -reward
            rewards.append([[-v for v in costs]])
        tensor = torch.tensor(rewards)
        returns = monte_carlo(tensor)
        assert returns.shape == (12, 1, 4)
        assert torch.equal(returns[0, 0], tensor[:, 0].sum(0))
        assert float(returns[0].sum()) == -env.metrics["objective"]


@pytest.mark.parametrize("arm", policy.ARMS)
def test_small_complete_episode_training_audits(arm):
    from ht_pdm_fjsp.maintenance_solver_training import fit
    class Journal:
        def __init__(self):
            self.rows = []
        def add(self, row):
            self.rows.append(row)
    cfg = dict(profile="smoke", horizons=[4], vector_envs=2, env_steps=16,
               rollout=16, minibatch=8, epochs=1, learning_rate=.0003,
               clip=.2, value_weight=.5, entropy_weight=.01, gradient_clip=.5,
               checkpoint_updates=[1])
    model = policy.MatchingActorCritic(arm, 8)
    journals = {name: Journal() for name in ("training_episodes", "training_progress")}
    checkpoints = []
    coverage = fit(model, cfg, 165000, set(), journals, lambda _, u, s: checkpoints.append((u, s)))
    assert coverage["per_machine_reward_reconciliation"]
    assert coverage["maximum_reward_reconciliation_error"] == 0
    assert coverage["env_steps"] == 16 and coverage["optimizer_steps"] == 2
    assert coverage["training_wall_seconds"] > 0
    assert len(journals["training_episodes"].rows) == 4
    assert checkpoints == [(1, 16)]


@pytest.mark.parametrize("arm", policy.SERIAL)
def test_serial_execution_does_not_evaluate_any_critic(arm, monkeypatch):
    c, state = setup()
    model = policy.MatchingActorCritic(arm, 16)
    action, _, _ = model.sample(policy.feature_batch([c], [state], [12]), deterministic=True)
    expected, _ = model.decode(c, state, 12, action[0])
    def forbidden(*args, **kwargs):
        raise AssertionError("critic accessed during serial execution")
    monkeypatch.setattr(model, "_values", forbidden)
    pairs, diagnostic = model.select(c, state, 12)
    assert pairs == expected
    assert diagnostic["value"] is None
    assert not diagnostic["critic_evaluated"]
