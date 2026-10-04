"""Train-only constraints, detached Double-Q, budgets and handoff artifacts."""
import copy,json
from argparse import Namespace
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import passive_technician_td_assignment as exp
from ht_pdm_fjsp import passive_technician_oracle_representation as base

def fixture():
    cfg=exp.settings('smoke');config=base.cells('smoke')['pressure'];oracle=base.ExactOracle(config,progress=False)
    data=base.dataset(oracle,98102);return cfg,config,oracle,data

def test_train_teacher_is_invariant_to_heldout_constraints_and_values():
    _,_,oracle,data=fixture();w,labels,pool=exp.train_teacher(oracle,data,98102)
    altered=copy.deepcopy(oracle)
    for k in altered.keys:
        if data['split'][k]=='heldout':
            altered.nodes[k]['actions']=[(0,0)];altered.nodes[k]['q95']=[-999999.];altered.nodes[k]['masks']=((False,)*3,)*2
    w2,labels2,pool2=exp.train_teacher(altered,data,98102)
    assert labels==labels2 and w['witness_policy']==w2['witness_policy']
    assert set(labels)=={k for k in oracle.keys if data['split'][k]=='train'}
    assert set(pool['ids'])=={oracle.key_to_id[k] for k in labels}
    assert torch.equal(pool['labels'],pool2['labels'])


def test_terminal_doubleq_targets_mask_and_detach():
    class Online:
        def agent_q(self,x):return torch.tensor([[[100.,2.,1.],[0.,1.,3.]]]).expand(len(x),-1,-1)
    class Target:
        def __init__(self):self.p=torch.tensor(3.,requires_grad=True)
        def total_q(self,x,a):return self.p*a.sum(-1)
    target=Target();batch=dict(nx=torch.zeros(2,2,15),nm=torch.tensor([[[False,True,True],[True,True,True]]]*2),
        rewards=torch.tensor([1.,2.]),done=torch.tensor([False,True]))
    y=exp.td_targets(Online(),target,batch)
    assert torch.allclose(y,torch.tensor([1.+.95*9.,2.]))
    assert not y.requires_grad and target.p.grad is None


def test_replay_stores_real_scaled_rewards_no_teacher_or_oracle_fields():
    r=exp.Replay(2);x=torch.zeros(2,15)
    for value in [1.,2.,3.]:r.add(x,[[True]*3]*2,(0,1),value,x,[[True]*3]*2,False)
    assert len(r)==2 and float(r.storage['rewards'][0])==pytest.approx(3/20)
    b=r.sample(8,np.random.default_rng(0));assert b['x'].dtype==torch.float32
    assert set(b)=={'x','masks','actions','rewards','nx','nm','done'}


def test_budget_epsilon_and_shared_initialization():
    full=exp.settings('full');smoke=exp.settings('smoke')
    assert full['updates']==14873 and smoke['updates']==27
    assert exp.epsilon(0,full)==1. and exp.epsilon(60000,full)==.05
    _,config,_,_=fixture();ref=exp.model_for(exp.ARMS[0],config,smoke,98400)
    for a in exp.ARMS:
        m=exp.model_for(a,config,smoke,98400)
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in ref.state_dict().items())
    assert exp.seed_audit(full)['prior_panels_disjoint']


def test_assignment_masks_gradient_and_pool_membership():
    cfg,config,oracle,data=fixture();_,_,pool=exp.train_teacher(oracle,data,98102)
    m=exp.model_for(exp.ARMS[1],config,cfg,98400);ids=torch.arange(len(pool['ids']))
    loss=exp.assignment_loss(m,pool,ids);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None for p in m.mixer.parameters())
    bad=dict(pool,masks=pool['masks'].clone());bad['masks'][0,0,pool['labels'][0,0]]=False
    with pytest.raises(RuntimeError,match='masked teacher'):exp.assignment_loss(m,bad,torch.tensor([0]))


def good_rows():
    cfg=exp.settings('full')
    return cfg,[dict(cell=c,algorithm=a,train_seed=s,undiscounted_policy_cost=4. if c=='nominal' else (20. if a=='monotonic_td' else 10.),
        train_regret=0.,train_oracle_optimal=1.) for c in ['nominal','shared_preference','pressure'] for a in exp.ARMS for s in cfg['train_seeds']]


def test_primary_cost_guard_and_control_gates():
    cfg,rows=good_rows();paired,s=exp.summarize(rows,cfg,True)
    assert len(paired)==10 and s['primary']['passed'] and s['empirical_attribution_supported']
    for r in rows:
        if r['algorithm']=='monotonic_policy_control' and r['cell']=='pressure':r['train_oracle_optimal']=.98
    s=exp.summarize(rows,cfg,True)[1];assert s['primary']['passed'] and not s['empirical_attribution_supported']
    cfg,rows=good_rows()
    for r in rows:
        if r['algorithm']=='monotonic_td_assignment' and r['cell']=='nominal':r['undiscounted_policy_cost']=5.
    assert not exp.summarize(rows,cfg,True)[1]['primary']['passed']


def test_real_smoke_runner_contract_counts_and_closed_teacher(tmp_path):
    out=tmp_path/'run';args=Namespace(profile='smoke',device='cpu',output_dir=str(out));exp.run(args)
    m=json.loads((out/'manifest.json').read_text());s=json.loads((out/'summary.json').read_text())
    assert m['status']=='COMPLETED' and m['actual_models']==9
    assert m['actual_training_env_steps']==720 and m['actual_optimizer_updates']==243
    assert m['actual_training_episodes']==240 and m['actual_evaluation_rows']==27
    assert m['audits']['no_heldout_supervision'] and m['audits']['matched_teacher_batches']
    assert s['primary']['passed'] is None and s['positive_control']['passed'] is None
    assert all((out/p).is_file() for p in m['outputs'])
    for cell in ['nominal','shared_preference','pressure']:
        st=json.loads((out/cell/'oracle_states.json').read_text());ids={r['state_id'] for r in st if r['split']=='train'}
        teacher=json.loads((out/cell/'train_teacher_actions.json').read_text());assert {r['state_id'] for r in teacher}==ids
    checkpoint=next(out.glob('*/monotonic_td/*/model.pt'));payload=torch.load(checkpoint,weights_only=True)
    payload['reward_scale']=1.;torch.save(payload,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='contract'):exp.load_checkpoint(tmp_path/'bad.pt')
    with pytest.raises(FileExistsError):exp.run(args)


def test_failure_retains_manifest_no_retry(monkeypatch,tmp_path):
    calls=[]
    def fail(*args):calls.append(True);raise RuntimeError('injected failure')
    monkeypatch.setattr(exp,'fit',fail);out=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='injected'):exp.run(Namespace(profile='smoke',device='cpu',output_dir=str(out)))
    assert len(calls)==1 and json.loads((out/'manifest.json').read_text())['status']=='FAILED'
    assert (out/'nominal/train_teacher_actions.json').is_file()


def test_full_train_only_teacher_and_control_budget():
    for (cell,config),seed in zip(base.cells('full').items(),exp.DATASET_SEEDS):
        oracle=base.ExactOracle(config,progress=False);data=base.dataset(oracle,seed)
        witness,labels,pool=exp.train_teacher(oracle,data,seed)
        assert witness['assignment_verified'] and witness['state_count']==len(labels)
        assert len(labels)<len(oracle.keys)
        assert all(data['split'][k]=='train' for k in labels)
        assert len(pool['ids'])==len(labels)


def test_td_only_never_computes_assignment_loss(monkeypatch,tmp_path):
    cfg,config,oracle,data=fixture();_,_,pool=exp.train_teacher(oracle,data,98102)
    cfg.update(env_steps=16,updates=1,learning_starts=16,log_interval=1)
    def forbidden(*args):raise AssertionError('TD-only must not compute CE')
    monkeypatch.setattr(exp,'assignment_loss',forbidden)
    _,coverage=exp.fit(exp.model_for('monotonic_td',config,cfg,98400),oracle,cfg,98400,'monotonic_td','pressure',pool,tmp_path,[],[])
    assert coverage['optimizer_updates']==1 and coverage['env_steps']==16
