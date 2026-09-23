from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch as th

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.qplex import (
    ALL_FEASIBLE_GRAPH,
    GRAPH_VARIANTS,
    NULL_GRAPH,
    POLICY_INTENT_TOP2_GRAPH,
    QPLEXMixer,
    QPLEXPolicy,
    build_resource_conflict_tensor,
    train_qplex,
)
from ht_pdm_fjsp.technician_capacity_screening import build_condition_config
from ht_pdm_fjsp.value_decomposition import ValueLearningSettings


ROOT = Path(__file__).resolve().parents[1]


def _config() -> BenchmarkConfig:
    base = BenchmarkConfig.from_json(
        ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"
    )
    return build_condition_config(base, topology="two_specialists", multiplier=2.0)


def _policy(
    env: MachineAgentsCTDEEnv, graph_variant: str, *, hidden: int = 16
) -> QPLEXPolicy:
    return QPLEXPolicy(
        graph_variant=graph_variant,
        agent_count=env.num_agents,
        action_count=env.max_local_actions,
        local_feature_dim=env.local_feature_dim,
        global_state_dim=env.global_state_dim,
        resource_conflicts=build_resource_conflict_tensor(env),
        hidden_dim=hidden,
        mixer_hidden_dim=8,
        mixer_kernels=2,
    )


def test_resource_conflict_tensor_is_symmetric_and_zero_on_agent_diagonal() -> None:
    env = MachineAgentsCTDEEnv(_config(), wait_policy="safe_noop")
    tensor = build_resource_conflict_tensor(env)
    assert np.array_equal(tensor, tensor.transpose(1, 0, 3, 2))
    assert not any(tensor[index, index].any() for index in range(env.num_agents))
    assert tensor.any()
    env.close()


def test_graph_variants_are_symmetric_capacity_matched_and_nested() -> None:
    env = MachineAgentsCTDEEnv(_config(), wait_policy="safe_noop")
    observation, _ = env.reset(seed=61_990)
    local = th.as_tensor(observation["local_observations"]).unsqueeze(0)
    masks = th.as_tensor(observation["action_masks"]).unsqueeze(0)
    policies = {variant: _policy(env, variant) for variant in GRAPH_VARIANTS}
    parameter_counts = {
        variant: sum(parameter.numel() for parameter in policy.parameters())
        for variant, policy in policies.items()
    }
    assert len(set(parameter_counts.values())) == 1
    graphs = {}
    for variant, policy in policies.items():
        values = policy.q_values(local)
        graphs[variant] = policy.graph_adjacency(values, masks)
        assert th.equal(graphs[variant], graphs[variant].transpose(1, 2))
        assert not th.diagonal(graphs[variant], dim1=1, dim2=2).any()
    assert not graphs[NULL_GRAPH].any()
    assert th.all(graphs[POLICY_INTENT_TOP2_GRAPH] <= graphs[ALL_FEASIBLE_GRAPH])
    env.close()


def test_qplex_mixer_preserves_individual_global_max() -> None:
    th.manual_seed(3)
    agents, actions = 3, 3
    mixer = QPLEXMixer(
        agent_count=agents,
        action_count=actions,
        state_dim=5,
        hidden_dim=8,
        kernels=3,
    )
    q_values = th.tensor(
        [[[1.0, 4.0, -2.0], [3.0, 0.0, 2.0], [-1.0, 2.0, 5.0]]]
    )
    max_q = q_values.max(dim=-1).values
    greedy = q_values.argmax(dim=-1).squeeze(0).tolist()
    state = th.randn(1, 5)
    adjacency = th.tensor(
        [[[False, True, False], [True, False, True], [False, True, False]]]
    )
    totals: dict[tuple[int, ...], float] = {}
    for joint_action in itertools.product(range(actions), repeat=agents):
        action_tensor = th.tensor([joint_action])
        chosen = q_values.gather(-1, action_tensor.unsqueeze(-1)).squeeze(-1)
        onehot = th.nn.functional.one_hot(
            action_tensor, num_classes=actions
        ).float()
        total, lambdas = mixer(chosen, max_q, state, onehot, adjacency)
        assert th.all(lambdas > 0)
        totals[joint_action] = float(total.item())
    assert max(totals, key=totals.get) == tuple(greedy)


def test_qplex_save_load_and_masked_action_inference(tmp_path: Path) -> None:
    env = MachineAgentsCTDEEnv(_config(), wait_policy="safe_noop")
    observation, _ = env.reset(seed=7)
    policy = _policy(env, POLICY_INTENT_TOP2_GRAPH)
    actions = policy.act(observation, deterministic=True, device="cpu")
    assert all(
        observation["action_masks"][agent, action]
        for agent, action in enumerate(actions)
    )
    path = tmp_path / "qplex.pt"
    policy.save(path)
    loaded = QPLEXPolicy.load(path, device="cpu")
    loaded_actions = loaded.act(observation, deterministic=True, device="cpu")
    assert np.array_equal(actions, loaded_actions)
    diagnostics = loaded.mixer_diagnostics(
        observation, loaded_actions, device="cpu"
    )
    assert np.asarray(diagnostics["adjacency"]).shape == (
        env.num_agents,
        env.num_agents,
    )
    env.close()


def test_tiny_qplex_training_saves_all_checkpoints(tmp_path: Path) -> None:
    settings = ValueLearningSettings(
        total_timesteps=48,
        replay_capacity=48,
        learning_starts=8,
        batch_size=4,
        train_frequency=2,
        gradient_steps=1,
        learning_rate=3e-4,
        gamma=0.99,
        target_update_interval=8,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_fraction=0.5,
        hidden_dim=16,
        mixer_hidden_dim=8,
        device="cpu",
        n_envs=4,
    )
    targets = (16, 32, 48)
    output = tmp_path / "train"
    train_qplex(
        _config(),
        settings,
        output,
        graph_variant=NULL_GRAPH,
        train_seed=7,
        show_progress=False,
        checkpoint_targets=targets,
    )
    assert len(list((output / "checkpoints").glob("*.pt"))) == 3
    loaded = QPLEXPolicy.load(output / "model.pt", device="cpu")
    assert loaded.graph_variant == NULL_GRAPH
