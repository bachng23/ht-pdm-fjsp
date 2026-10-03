"""Consistent SAT-policy supervision with fixed ID+peer and monotonic mixer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from importlib.metadata import version
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from ht_pdm_fjsp import passive_technician_oracle_representation as base
from ht_pdm_fjsp import passive_technician_role_context as role
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig

PROTOCOL = "ra_qmix_policy_supervision_v1"
ARMS = ("monotonic_q", "monotonic_policy", "monotonic_hybrid", "joint_q")
REGIMES = ("ceiling", "split")
DATASET_SEEDS = (97100, 97101, 97102)
FEATURE_CONTRACT = dict(role.FEATURE_CONTRACT, identity_enabled=True, peer_enabled=True,
                        teacher="one full-population consistent SAT assignment",
                        split_interpretation="transductive fixed-teacher transfer; not independent oracle generalization")


def settings(profile):
    cfg = base.settings(profile)
    cfg.update(train_seeds=list(range(97000, 97010)) if profile == "full" else [97400],
               evaluation_seeds=list(range(97200, 97250)) if profile == "full" else [97410, 97411, 97412])
    return cfg


def seed_audit(cfg):
    registry = json.loads((Path(__file__).resolve().parents[2]/"configs/ra_qmix_policy_supervision_seed_registry.json").read_text())
    panels = [set(cfg["train_seeds"]), set(cfg["evaluation_seeds"]), set(DATASET_SEEDS), set(range(201, 301))]
    if any(a & b for i, a in enumerate(panels) for b in panels[i+1:]):
        raise ValueError("seed panels overlap")
    if set().union(*panels[:3]) & set(registry["declared_seed_values"]):
        raise ValueError("seed panels overlap prior local manifests")
    return dict(scope=registry["scope"], prior_manifest_count=registry["manifest_count"],
                prior_panels_disjoint=True, panels_disjoint=True, sealed_panel_closed=True,
                environment_population_previously_inspected=True)


def model_for(arm, config, cfg, seed):
    if arm not in ARMS:
        raise ValueError(arm)
    return role.model_for("joint_q" if arm == "joint_q" else "monotonic_role_context", config, cfg, seed)


def teacher_labels(oracle, witness):
    if witness.get("status") not in ("OPTIMAL", "FEASIBLE") or witness.get("assignment_verified") is not True:
        raise RuntimeError("teacher needs verified SAT; UNKNOWN/UNSAT is not usable")
    lookup = {}
    for item in witness["witness_policy"]:
        key = tuple(item["inputs"])
        if key in lookup and lookup[key] != item["action"]:
            raise RuntimeError("inconsistent teacher labels for identical input")
        lookup[key] = item["action"]
    actions = {}
    for key in oracle.keys:
        node = oracle.nodes[key]
        action = tuple(lookup[role.raw_feature_key(node["observations"], i, True, True)] for i in range(2))
        if action not in node["actions"] or any(not node["masks"][i][a] for i, a in enumerate(action)):
            raise RuntimeError("teacher selected invalid action")
        if max(node["q95"])-node["q95"][node["actions"].index(action)] > 1e-8:
            raise RuntimeError("teacher assignment not oracle-optimal")
        actions[key] = action
    return actions


def regime_data(original, oracle, teacher, regime):
    if regime not in REGIMES:
        raise ValueError(regime)
    data = dict(original)
    if regime == "ceiling":
        data["train_indices"] = torch.arange(len(data["raw_y"]))
    fit_y = data["raw_y"][data["train_indices"]]
    data["mean"] = fit_y.mean().item()
    data["std"] = max(fit_y.std(unbiased=False).item(), 1e-6)
    data["y"] = (data["raw_y"]-data["mean"])/data["std"]
    data["teacher"] = torch.tensor([teacher[oracle.keys[sid]] for sid in data["ids"]], dtype=torch.long)
    return data


def objective(model, data, selected, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    x, actions = data["x"][selected], data["actions"][selected]
    if arm == "monotonic_policy":
        # No full-Q computation/gradient in this arm; mixer stays at initialization.
        mse = x.new_zeros(())
    else:
        mse = (model.total_q(x, actions)-data["y"][selected]).square().mean()
    ce = x.new_zeros(())
    if arm in ("monotonic_policy", "monotonic_hybrid"):
        masks = data["masks"][selected]
        labels = data["teacher"][selected]
        if not masks.gather(-1, labels.unsqueeze(-1)).all():
            raise RuntimeError("masked-out teacher label")
        logits = model.agent_q(x).masked_fill(~masks, -torch.inf)
        ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    return mse+ce, mse, ce


def fit(model, data, cfg, seed, arm, cell, regime, output, progress):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed+17)
    indices = data["train_indices"]
    stream = hashlib.sha256()
    mixer_initial = {k: v.clone() for k, v in model.mixer.state_dict().items()} if arm != "joint_q" else None
    model.train()
    for update in tqdm(range(1, cfg["updates"]+1), desc=f"fit {cell}/{regime}/{arm}/{seed}", unit="update", leave=False):
        selected = indices[torch.randint(len(indices), (cfg["batch_size"],), generator=generator)]
        stream.update(selected.numpy().tobytes())
        loss, mse, ce = objective(model, data, selected, arm)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if update % cfg["log_interval"] == 0 or update == cfg["updates"]:
            progress.append(dict(cell=cell, regime=regime, algorithm=arm, train_seed=seed, update=update,
                                 loss=loss.item(), normalized_mse=mse.item(), policy_ce=ce.item(),
                                 gradient_norm=float(norm)))
            base._write_csv(output/"training_progress.csv", progress)
    if arm == "monotonic_policy" and not all(torch.equal(v, model.mixer.state_dict()[k]) for k, v in mixer_initial.items()):
        raise RuntimeError("policy-only mixer changed")
    return model.eval(), stream.hexdigest()


def load_checkpoint(path):
    payload = torch.load(path, weights_only=True)
    if payload["protocol_version"] != PROTOCOL or payload["feature_contract"] != FEATURE_CONTRACT:
        raise ValueError("checkpoint protocol/feature contract mismatch")
    if payload["regime"] not in REGIMES or payload["objective"] != payload["algorithm"]:
        raise ValueError("checkpoint regime/objective mismatch")
    model = model_for(payload["algorithm"], PassiveConfig(**payload["config"]), payload["settings"], payload["train_seed"])
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def evaluate(model, arm, oracle, data, teacher, cell, regime, seed):
    indices, individual, aggregate = base.evaluate_tables(model, arm, oracle, data, cell, seed)
    for row in individual:
        key = oracle.keys[row["state_id"]]
        node = oracle.nodes[key]
        action = node["actions"][indices[key]]
        diagonal = [q for a, q in zip(node["actions"], node["q95"]) if a[0] == a[1]]
        row.update(regime=regime, selected_actions=json.dumps(action), teacher_actions=json.dumps(teacher[key]),
                   teacher_joint_agreement=action == teacher[key],
                   teacher_agent_agreement=sum(a == b for a, b in zip(action, teacher[key]))/2,
                   oracle_optimal=row["regret"] <= 1e-8,
                   original_inputs_identical=node["observations"][0] == node["observations"][1],
                   oracle_requires_unequal_actions=max(node["q95"])-max(diagonal) > 1e-8,
                   fit_state=regime == "ceiling" or row["split"] == "train")
    aggregate["regime"] = regime
    for split in ("all", "train", "heldout"):
        rows = individual if split == "all" else [r for r in individual if r["split"] == split]
        aggregate[split+"_regret"] = statistics.fmean(r["regret"] for r in rows)
        for key in ("oracle_optimal", "teacher_joint_agreement", "teacher_agent_agreement"):
            aggregate[split+"_"+key] = statistics.fmean(r[key] for r in rows)
    return indices, individual, aggregate


def summarize(rows, cfg, full):
    index = {(r["cell"],r["regime"],r["algorithm"],r["train_seed"]): r for r in rows}
    cells = ("nominal", *role.DIAGNOSTIC_CELLS)
    learn_cells, controls, paired = {}, {}, []
    for cell in cells:
        policy = [index[cell,"ceiling","monotonic_policy",s] for s in cfg["train_seeds"]]
        mean = statistics.fmean(r["all_regret"] for r in policy)
        successes = sum(r["all_oracle_optimal"] >= .99 for r in policy)
        learn_cells[cell] = dict(mean_all_state_regret=mean, seeds_at_99_percent_optimal=successes,
                                passed=(mean <= .10 and successes >= 8) if full else None)
        joint = [index[cell,"ceiling","joint_q",s] for s in cfg["train_seeds"]]
        regret = statistics.fmean(r["all_regret"] for r in joint)
        gap = statistics.fmean(r["undiscounted_policy_gap"] for r in joint)
        controls[cell] = dict(all_state_regret=regret, undiscounted_policy_gap=gap,
                              passed=(regret <= .10 and gap <= 2.) if full else None)
    for seed in cfg["train_seeds"]:
        reg = {a: statistics.fmean(index[c,"ceiling",a,seed]["all_regret"] for c in role.DIAGNOSTIC_CELLS) for a in ARMS}
        cost = {a: statistics.fmean(index[c,"ceiling",a,seed]["undiscounted_policy_cost"] for c in role.DIAGNOSTIC_CELLS) for a in ARMS}
        paired.append(dict(train_seed=seed, q_regret=reg["monotonic_q"], policy_regret=reg["monotonic_policy"],
                           hybrid_regret=reg["monotonic_hybrid"], hybrid_minus_q_regret=reg["monotonic_hybrid"]-reg["monotonic_q"],
                           hybrid_minus_q_cost=cost["monotonic_hybrid"]-cost["monotonic_q"]))
    nominal = lambda a: statistics.fmean(index["nominal","ceiling",a,s]["undiscounted_policy_cost"] for s in cfg["train_seeds"])
    gate = role.contrast_gate([r["hybrid_minus_q_regret"] for r in paired], statistics.fmean(r["q_regret"] for r in paired),
                             [r["hybrid_minus_q_cost"] for r in paired], nominal("monotonic_hybrid"), nominal("monotonic_q"), .10, .20, full)
    control = all(c["passed"] for c in controls.values()) if full else None
    return paired, dict(scientific_gate_applicable=full, primary=dict(
        learnability=dict(cells=learn_cells, passed=all(c["passed"] for c in learn_cells.values()) if full else None),
        objective=gate), positive_control=dict(cells=controls, passed=control),
        empirical_objective_attribution_supported=bool(control and gate["hypothesis_passed"]) if full else None,
        interpretation="Ceiling is allstates fit; split is transductive fixed-teacher transfer. No adaptation/generalization-to-new-environments claim.")


def run(args):
    if args.device != "cpu":
        raise ValueError("CPU-only protocol")
    torch.set_num_threads(1)
    cfg = settings(args.profile)
    audit = seed_audit(cfg)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    revision, dirty = base._git_state()
    expected = 3*len(REGIMES)*len(ARMS)*len(cfg["train_seeds"])
    manifest = dict(protocol_version=PROTOCOL, status="RUNNING", profile=args.profile, device="cpu",
                    started_at=datetime.now(UTC).isoformat(), git_revision=revision, git_dirty=dirty,
                    runtime=base._runtime_metadata(), train_seeds=cfg["train_seeds"], evaluation_seeds=cfg["evaluation_seeds"],
                    dataset_seeds=DATASET_SEEDS, sealed_test_evaluated=False, sealed_test_seeds=list(range(201,301)),
                    expected_models=expected, expected_evaluation_rows=expected*len(cfg["evaluation_seeds"]),
                    expected_optimizer_updates=expected*cfg["updates"])
    manifest["runtime"]["packages"]["ortools"] = version("ortools")
    write_json, write_csv = base._write_json, base._write_csv
    write_json(output/"manifest.json", manifest)
    write_json(output/"seed_audit.json", audit)
    write_json(output/"feature_contract.json", FEATURE_CONTRACT)
    write_json(output/"resolved_config.json", dict(settings=cfg, gamma=base.GAMMA, learning_rate=3e-4,
               cells={c:asdict(v) for c,v in base.cells(args.profile).items()}, arms=ARMS, regimes=REGIMES,
               hybrid_ce_weight=1., cf_weight=0., teacher_uses_full_population=True,
               split_transductive=True, ceiling_trains_all_states=True))
    progress, episodes, states_rows, metrics, counts, streams = [], [], [], [], [], []
    actual_models, feasibility = 0, {}
    try:
        for (cell, config), split_seed in zip(base.cells(args.profile).items(), DATASET_SEEDS):
            oracle = base.ExactOracle(config)
            original = base.dataset(oracle, split_seed)
            if oracle.bellman_residual() > 1e-9:
                raise RuntimeError("oracle Bellman residual")
            cell_dir = output/cell
            cell_dir.mkdir()
            witness = role.policy_feasibility(oracle, "monotonic_role_context", split_seed)
            write_json(cell_dir/"policy_feasibility.json", witness)
            teacher = teacher_labels(oracle, witness)
            teacher_indices = {k: oracle.nodes[k]["actions"].index(a) for k,a in teacher.items()}
            teacher_cost95 = oracle.policy_cost(teacher_indices, base.GAMMA)
            teacher_cost1 = oracle.policy_cost(teacher_indices, 1.)
            if abs(teacher_cost95 + max(oracle.nodes[oracle.initial]["q95"])) > 1e-8:
                raise RuntimeError("consistent optimal teacher does not attain discounted oracle cost")
            teacher_rows = [dict(state_id=oracle.key_to_id[k], actions=list(a)) for k,a in teacher.items()]
            write_json(cell_dir/"teacher_actions.json", teacher_rows)
            teacher_hash = hashlib.sha256((cell_dir/"teacher_actions.json").read_bytes()).hexdigest()
            feasibility[cell] = dict(status=witness["status"], assignment_verified=True,
                                     teacher_sha256=teacher_hash, state_count=len(teacher),
                                     discounted_policy_cost=teacher_cost95, undiscounted_policy_cost=teacher_cost1,
                                     undiscounted_oracle_gap=teacher_cost1 + max(oracle.nodes[oracle.initial]["q1"]))
            states, table = [], []
            for key in oracle.keys:
                node, sid = oracle.nodes[key], oracle.key_to_id[key]
                states.append(dict(state_id=sid, time=key[0], state=asdict(key[1]), observations=node["observations"], split=original["split"][key]))
                for action,q95,q1 in zip(node["actions"],node["q95"],node["q1"]):
                    table.append(dict(state_id=sid, actions=json.dumps(action), q95=q95, q1=q1))
            write_json(cell_dir/"oracle_states.json", states)
            write_csv(cell_dir/"oracle_q.csv", table)
            for regime in REGIMES:
                data = regime_data(original, oracle, teacher, regime)
                for seed in tqdm(cfg["train_seeds"], desc=f"policy supervision {cell}/{regime}", unit="seed"):
                    reference = model_for("monotonic_q",config,cfg,seed)
                    initial = reference.state_dict()
                    reference_count = sum(p.numel() for p in reference.parameters())
                    for arm in ARMS:
                        model = model_for(arm,config,cfg,seed)
                        count = sum(p.numel() for p in model.parameters())
                        if arm != "joint_q" and not all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items()):
                            raise RuntimeError("objective arms initialization differs")
                        if abs(count/reference_count-1) > (.10 if args.profile == "smoke" else .05):
                            raise RuntimeError("parameter matching failed")
                        counts.append(dict(cell=cell,regime=regime,algorithm=arm,train_seed=seed,parameters=count,
                                           agent_parameters=sum(p.numel() for p in model.agent.parameters()) if arm != "joint_q" else None,
                                           mixer_receives_gradient=arm not in ("monotonic_policy","joint_q"),
                                           reference_parameters=reference_count))
                        write_csv(output/"parameter_counts.csv",counts)
                        model, stream = fit(model,data,cfg,seed,arm,cell,regime,output,progress)
                        streams.append(dict(cell=cell,regime=regime,algorithm=arm,train_seed=seed,minibatch_sha256=stream))
                        write_csv(output/"minibatch_streams.csv",streams)
                        checkpoint = cell_dir/regime/arm/f"train_seed_{seed}"/"model.pt"
                        checkpoint.parent.mkdir(parents=True)
                        torch.save(dict(protocol_version=PROTOCOL,state_dict=model.state_dict(),algorithm=arm,objective=arm,
                                        regime=regime,train_seed=seed,config=asdict(config),settings=cfg,
                                        feature_contract=FEATURE_CONTRACT,target_mean=data["mean"],target_std=data["std"],
                                        teacher_sha256=teacher_hash,updates=cfg["updates"]),checkpoint)
                        restored = load_checkpoint(checkpoint)
                        if not torch.equal(base.predictions(model,data),base.predictions(restored,data)):
                            raise RuntimeError("checkpoint Q predictions differ")
                        if arm != "joint_q":
                            with torch.no_grad():
                                if not torch.equal(model.agent_q(data["x"]),restored.agent_q(data["x"])):
                                    raise RuntimeError("checkpoint agent predictions differ")
                        indices,individual,aggregate = evaluate(restored,arm,oracle,data,teacher,cell,regime,seed)
                        states_rows.extend(individual); metrics.append(aggregate)
                        write_csv(output/"state_metrics.csv",states_rows)
                        write_csv(output/"model_metrics.csv",metrics)
                        for evaluation in tqdm(cfg["evaluation_seeds"],desc=f"evaluate {cell}/{regime}/{arm}/{seed}",unit="episode",leave=False):
                            episodes.append(dict(cell=cell,regime=regime,algorithm=arm,train_seed=seed,**base.rollout(oracle,indices,evaluation)))
                        write_csv(output/"episodes.partial.csv",episodes)
                        actual_models += 1
        paired,summary = summarize(metrics,cfg,args.profile == "full")
        finals = {(r["cell"],r["regime"],r["algorithm"],r["train_seed"]):r["update"] for r in progress}
        batches = {}
        for r in streams:
            batches.setdefault((r["cell"],r["regime"],r["train_seed"]),set()).add(r["minibatch_sha256"])
        audits = dict(seed_audit=audit,complete_models=actual_models == expected,
                      complete_evaluations=len(episodes) == manifest["expected_evaluation_rows"],
                      exact_optimizer_updates=sum(finals.values()) == manifest["expected_optimizer_updates"],
                      updates_per_model_match=len(finals) == expected and all(v == cfg["updates"] for v in finals.values()),
                      unique_episode_keys=len({(r["cell"],r["regime"],r["algorithm"],r["train_seed"],r["eval_seed"]) for r in episodes}) == len(episodes),
                      zero_invalid_requests=all(r["invalid_requests"] == 0 for r in episodes),
                      exact_cost_reconciliation=all(abs(r["cost_reconciliation_error"]) < 1e-9 for r in episodes),
                      finite_training_logs=all(math.isfinite(r[k]) for r in progress for k in ("loss","normalized_mse","policy_ce","gradient_norm")),
                      matched_minibatch_streams=len(batches) == 3*len(REGIMES)*len(cfg["train_seeds"]) and all(len(v) == 1 for v in batches.values()),
                      checkpoints_reload_identical=True,identical_objective_initialization=True,
                      original_mixer_at_construction=True,policy_only_mixer_unchanged=True,
                      verified_consistent_optimal_teachers=len(feasibility) == 3,
                      oracle_observation_and_bellman_audits=True,sealed_panel_closed=True)
        if not all(v for v in audits.values() if isinstance(v,bool)):
            raise RuntimeError(f"engineering audit failed: {audits}")
        summary.update(audits=audits,teachers=feasibility)
        write_csv(output/"paired_seed_metrics.csv",paired)
        write_csv(output/"episodes.csv",episodes)
        write_csv(output/"coordination.csv",episodes)
        write_json(output/"summary.json",summary)
        manifest.update(status="COMPLETED",completed_at=datetime.now(UTC).isoformat(),actual_models=actual_models,
                        actual_evaluation_rows=len(episodes),actual_optimizer_updates=sum(finals.values()),audits=audits)
        print(json.dumps(summary,indent=2))
    except BaseException as error:
        manifest.update(status="FAILED",error=f"{type(error).__name__}: {error}",actual_models=actual_models,actual_evaluation_rows=len(episodes))
        raise
    finally:
        manifest["outputs"] = sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())
        base._write_json(output/"manifest.json",manifest)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",required=True,choices=("smoke","full"))
    parser.add_argument("--device",default="cpu",choices=("cpu",))
    parser.add_argument("--output-dir",required=True)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
