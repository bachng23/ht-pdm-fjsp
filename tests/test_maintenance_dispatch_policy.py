import numpy as np
import pytest
import torch

from ht_pdm_fjsp.maintenance_dispatch import (
    DispatchEnv,
    DispatchState,
    family_config,
    training_config,
    validate_matching,
)
from ht_pdm_fjsp.maintenance_dispatch_policy import (
    MODES,
    DispatchActorCritic,
    batch_observations,
    batch_sequences,
    rule_action,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("mode", MODES)
def test_sample_replay_batched_padding_gradient_and_capacity(mode):
    torch.manual_seed(7)
    model = DispatchActorCritic(16)
    observations, sequences, probabilities, values = [], [], [], []
    generator = torch.Generator().manual_seed(17)
    for seed in range(12):
        cfg = training_config(seed, "smoke")
        env = DispatchEnv(cfg)
        obs = env.reset(seed)
        for _ in range(cfg.horizon):
            pairs, tokens, prob, value = model.act(obs, mode, generator)
            validate_matching(cfg, env.state, pairs)
            observations.append(obs)
            sequences.append(tokens)
            probabilities.append(prob)
            values.append(value)
            obs, _, _, _ = env.step(pairs)
    sequence, probs, entropy, value = model.decode(
        batch_observations(observations), mode, batch_sequences(sequences)
    )
    assert torch.allclose(probs, torch.tensor(probabilities), atol=5e-5)
    assert torch.allclose(value, torch.tensor(values), atol=1e-5)
    assert torch.isfinite(entropy).all()
    (-probs.mean() + value.square().mean() - 0.01 * entropy.mean()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    unused = (
        model.machine_head
        if mode == "fixed_priority"
        else model.technician_head
        if mode == "fixed_allocation"
        else None
    )
    if unused is not None:
        assert all(p.grad is None for p in unused.parameters())


@pytest.mark.parametrize("mode", MODES)
def test_all_busy_or_incompatible_candidates_produce_only_stop(mode):
    cfg = family_config("small", 0, "smoke")
    env = DispatchEnv(cfg)
    obs = env.reset(2, DispatchState((3, 3), (True, True), (2, 4), (1, 1), (0, 1)))
    model = DispatchActorCritic(16)
    pairs, tokens, prob, value = model.act(obs, mode)
    assert pairs == () and tokens == [(-1, -1)]
    _, p, entropy, v = model.decode(
        batch_observations([obs]), mode, batch_sequences([tokens])
    )
    assert p.item() == 0 and entropy.item() == 0
    (p.sum() + entropy.sum() + v.square().sum()).backward()
    assert all(
        torch.isfinite(param.grad).all()
        for param in model.parameters()
        if param.grad is not None
    )


def test_encoder_and_matching_probabilities_are_entity_equivariant():
    model = DispatchActorCritic(16)
    cfg = family_config("n5_k3_sparse", 114010, "smoke")
    obs = DispatchEnv(cfg).reset(114010)
    machines = np.array([2, 4, 0, 1, 3])
    technicians = np.array([1, 2, 0])
    shuffled = dict(
        machines=obs["machines"][machines],
        technicians=obs["technicians"][technicians],
        edges=obs["edges"][machines][:, technicians],
        global_features=obs["global_features"],
    )
    original = model.encode(batch_observations([obs]))
    permuted = model.encode(batch_observations([shuffled]))
    assert torch.allclose(original[0][:, machines], permuted[0], atol=1e-6)
    assert torch.allclose(original[1][:, technicians], permuted[1], atol=1e-6)
    assert torch.allclose(
        original[2][:, machines][:, :, technicians], permuted[2], atol=1e-6
    )
    assert torch.allclose(original[3], permuted[3], atol=1e-6)
    # Replay the same physical choices after relabeling, preserving latent order.
    m, j = next(
        (m, j)
        for m, row in enumerate(cfg.service_time)
        for j, duration in enumerate(row)
        if duration > 0
    )
    sequence = [(m, j), (-1, -1)]
    translated = [
        (int(np.where(machines == m)[0][0]), int(np.where(technicians == j)[0][0])),
        (-1, -1),
    ]
    a = model.decode(
        batch_observations([obs]), "learned_both", batch_sequences([sequence])
    )
    b = model.decode(
        batch_observations([shuffled]), "learned_both", batch_sequences([translated])
    )
    for x, y in zip(a[1:], b[1:]):
        assert torch.allclose(x, y, atol=1e-6)


def test_padded_entities_cannot_change_real_policy_values():
    model = DispatchActorCritic(16)
    small = DispatchEnv(family_config("small", 0, "smoke")).reset(1)
    large = DispatchEnv(family_config("n5_k3_sparse", 2, "smoke")).reset(2)
    sequence = [(-1, -1)]
    a = model.decode(
        batch_observations([small]), "learned_both", batch_sequences([sequence])
    )
    b = model.decode(
        batch_observations([small, large]),
        "learned_both",
        batch_sequences([sequence, sequence]),
    )
    for x, y in zip(a[1:], b[1:]):
        assert torch.allclose(x[0], y[0], atol=1e-6)


@pytest.mark.parametrize(
    "sequence",
    [
        [(0, 0)],
        [(0, 0), (1, 0), (-1, -1)],
        [(-2, -2)],
        [(-1, -1), (0, 0)],
        [(-3, -1)],
        [(-1, 0)],
    ],
)
def test_replay_rejects_incomplete_conflicting_or_malformed_sequences(sequence):
    cfg = family_config("small", 0, "smoke")
    batch = batch_observations([DispatchEnv(cfg).reset(9)])
    with pytest.raises(RuntimeError):
        DispatchActorCritic(16).decode(
            batch, "learned_both", batch_sequences([sequence])
        )


def test_rule_uses_risk_wait_and_compatible_free_skill_only():
    cfg = family_config("small", 0, "smoke")
    env = DispatchEnv(cfg)
    assert (
        rule_action(
            env.reset(
                0, DispatchState((0, 0), (False, False), (0, 0), (0, 0), (-1, -1))
            )
        )
        == ()
    )
    obs = env.reset(0, DispatchState((3, 3), (True, True), (4, 1), (0, 0), (-1, -1)))
    assert rule_action(obs) == ((0, 0), (1, 1))
