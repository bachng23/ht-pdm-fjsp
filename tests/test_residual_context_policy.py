from pathlib import Path
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.residual_context_experiment import defaults
from ht_pdm_fjsp.residual_context_policy import ResidualContextMaskablePolicy
from ht_pdm_fjsp.shared_action_policy import SharedActionMaskablePolicy

ROOT=Path(__file__).resolve().parents[1]
CONFIG=BenchmarkConfig.from_json(ROOT/"configs/minimal_benchmark.json")

def _model(policy, context=False):
    return MaskablePPO(policy,Monitor(HTPdmFjspEnv(config=CONFIG,include_production_context=context)),n_steps=32,batch_size=16,n_epochs=1,seed=1,device="cpu",verbose=0)

def test_residual_starts_exactly_at_transferred_shared_policy():
    shared=_model(SharedActionMaskablePolicy)
    residual=_model(ResidualContextMaskablePolicy,True)
    residual.policy.load_shared_base(shared.policy)
    env=HTPdmFjspEnv(config=CONFIG,include_production_context=True); obs,_=env.reset(seed=7)
    rt,_=residual.policy.obs_to_tensor(obs)
    base={k:obs[k] for k in shared.observation_space.spaces}; st,_=shared.policy.obs_to_tensor(base)
    with th.no_grad():
        assert th.allclose(residual.policy.action_logits(rt),shared.policy.action_logits(st))
        assert th.count_nonzero(residual.policy.context_residual(rt))==0
    assert all(not p.requires_grad for m in (residual.policy.global_encoder,residual.policy.action_encoder,residual.policy.action_scorer) for p in m.parameters())

def test_residual_training_only_changes_production_logits():
    model=_model(ResidualContextMaskablePolicy,True)
    model.policy.context_scorer.bias.data.fill_(0.5)
    env=HTPdmFjspEnv(config=CONFIG,include_production_context=True); obs,_=env.reset(seed=8); t,_=model.policy.obs_to_tensor(obs)
    values=model.policy.context_residual(t)[0]
    for i,a in enumerate(env.actions):
        assert (values[i]!=0).item()==(a.kind=="production")
    model.learn(total_timesteps=64)

def test_residual_profile_keeps_test_panel_closed():
    full=defaults("full")
    assert len(full["train_seeds"])==5 and len(full["validation_seeds"])==200
    assert set(full["validation_seeds"]).isdisjoint(range(50000,50100))
