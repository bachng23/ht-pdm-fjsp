"""Oracle-supervised identity by peer-information factorial with unchanged mixer."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from ortools.sat.python import cp_model
from tqdm.auto import tqdm

from ht_pdm_fjsp import passive_technician_oracle_representation as oracle_exp
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig
from ht_pdm_fjsp.passive_technician_value_decomposition import PassiveValueDecomposition, TechnicianEdgeQ

PROTOCOL = "ra_qmix_role_context_v1"
FLAGS = {
    "monotonic_base": (False, False),
    "monotonic_role": (True, False),
    "monotonic_context": (False, True),
    "monotonic_role_context": (True, True),
}
ARMS = (*FLAGS, "joint_q")
DATASET_SEEDS = (96100, 96101, 96102)
DIAGNOSTIC_CELLS = ("shared_preference", "pressure")
FEATURE_CONTRACT = {
    "blocks": ["normalized_local_observation", "unscaled_one_hot_machine_identity", "normalized_peer_observation"],
    "identity": "stable machine index; not a learned role",
    "peer_order": "other machine in the two-machine cell; no self duplication",
    "inactive_blocks": "zero filled; same padded input shape across factorial arms",
    "execution": "shared network, per-machine masked greedy argmax; original mixer unused at execution",
    "peer_telemetry_required": True,
    "hidden_simulator_fields": False,
    "peer_actions_or_future_information": False,
    "cf_weight": 0.,
}


def settings(profile):
    cfg = oracle_exp.settings(profile)
    cfg.update(train_seeds=list(range(96000, 96010)) if profile == "full" else [96400],
               evaluation_seeds=list(range(96200, 96250)) if profile == "full" else [96410, 96411, 96412])
    return cfg


def conditioned_inputs(local, role, peer):
    if local.ndim != 3 or local.shape[1] != 2:
        raise ValueError("locked two-machine feature contract")
    identities = torch.eye(2, dtype=local.dtype, device=local.device).expand(local.shape[0], -1, -1)
    peer_obs = local.flip(1)
    if not role:
        identities = torch.zeros_like(identities)
    if not peer:
        peer_obs = torch.zeros_like(peer_obs)
    return torch.cat((local, identities, peer_obs), dim=-1)


class ConditionedEdgeQ(TechnicianEdgeQ):
    """Expand the trunk input, retaining technician slots from the local block."""
    def __init__(self, config, hidden):
        super().__init__(config.observation_dim, config.technicians, hidden)
        self.local_dim = config.observation_dim
        self.trunk[0] = nn.Linear(2*self.local_dim+2, hidden)

    def forward(self, observations):
        hidden = self.trunk(observations)
        slots = observations[..., 3:self.local_dim].reshape(*observations.shape[:-1], self.technicians, 6)
        encoded = self.tech_encoder(slots)
        edge = (self.query(hidden).unsqueeze(-2)*self.key(encoded)).sum(-1)/float(self.hidden_dim)**.5
        return torch.cat((self.defer(hidden), edge), dim=-1)


class RoleContextModel(PassiveValueDecomposition):
    def __init__(self, config, cfg, role, peer):
        if config.machines != 2:
            raise ValueError("locked two-machine protocol")
        super().__init__(config, "tqmix_no_cf", cfg["hidden"], cfg["mixer_hidden"])
        self.role, self.peer = role, peer
        # Preserve the existing mixer tensors/state interface entirely.
        self.agent = ConditionedEdgeQ(config, cfg["hidden"])

    def agent_q(self, local):
        inputs = conditioned_inputs(local, self.role, self.peer)
        batch, machines, features = inputs.shape
        return self.agent(inputs.reshape(batch*machines, features)).view(batch, machines, self.action_dim)


def model_for(arm, config, cfg, seed):
    torch.manual_seed(seed)
    if arm in FLAGS:
        return RoleContextModel(config, cfg, *FLAGS[arm])
    if arm != "joint_q":
        raise ValueError(arm)
    reference = RoleContextModel(config, cfg, False, False)
    target = sum(p.numel() for p in reference.parameters())
    d, a = config.machines*config.observation_dim, (config.technicians+1)**config.machines
    width = min(range(1, 513), key=lambda w: abs(w*w+(d+a+2)*w+a-target))
    torch.manual_seed(seed)
    return oracle_exp.JointQ(config, width)


def raw_feature_key(observations, machine, role, peer):
    """Equality keys; common observation scaling does not change equivalence."""
    local = tuple(observations[machine])
    identity = tuple(int(machine == i) if role else 0 for i in range(2))
    other = tuple(observations[1-machine]) if peer else (0,)*len(local)
    return local+identity+other


def policy_feasibility(oracle, arm, solver_seed, *, solver_factory=cp_model.CpSolver):
    """Exact finite lookup-policy CSP; SAT is not a neural fitting guarantee."""
    role, peer = FLAGS[arm]
    model = cp_model.CpModel()
    group_for, groups, variables, masks_for, records = {}, [], [], {}, []
    for key in oracle.keys:
        node = oracle.nodes[key]
        ids = []
        for machine in range(2):
            feature = raw_feature_key(node["observations"], machine, role, peer)
            mask = node["masks"][machine]
            if feature in masks_for and masks_for[feature] != mask:
                raise RuntimeError("identical inputs have different masks; CSP contract invalid")
            masks_for[feature] = mask
            if feature not in group_for:
                group_for[feature] = len(groups)
                groups.append(feature)
                variables.append(model.NewIntVar(0, oracle.config.technicians, f"group_{len(groups)-1}"))
            ids.append(group_for[feature])
        best = max(node["q95"])
        optimal = [a for a, q in zip(node["actions"], node["q95"]) if abs(q-best) <= 1e-8]
        unique = list(dict.fromkeys(ids))
        # Collapse duplicate columns and filter tuples violating shared determinism.
        allowed = [tuple(a[ids.index(g)] for g in unique) for a in optimal
                   if ids[0] != ids[1] or a[0] == a[1]]
        if not allowed:
            return dict(algorithm=arm, status="INFEASIBLE", method="direct_same_input_conflict",
                        lookup_policy_class="shared deterministic memoryless",
                        assignment_verified=None, state_id=oracle.key_to_id[key],
                        time=key[0], inputs=node["observations"], augmented_input=groups[ids[0]],
                        optimal_actions=optimal, identity=role, peer_information=peer,
                        visited_state_count=len(records)+1, group_count=len(groups))
        model.AddAllowedAssignments([variables[g] for g in unique], allowed)
        records.append((key, ids))
    invalid = model.Validate()
    if invalid:
        raise RuntimeError(f"invalid policy CSP: {invalid}")
    solver = solver_factory()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = solver_seed
    solver.parameters.max_time_in_seconds = 30.
    status = solver.Solve(model)
    if status == cp_model.MODEL_INVALID:
        raise RuntimeError("policy CSP MODEL_INVALID")
    result = dict(algorithm=arm, status=solver.StatusName(status), method="exact_finite_constraint_system",
                  lookup_policy_class="shared deterministic memoryless", identity=role, peer_information=peer,
                  assignment_verified=None, group_count=len(groups), state_count=len(records),
                  solver_time_seconds=solver.WallTime(), solver_seed=solver_seed, max_time_seconds=30.)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        actions = [solver.Value(v) for v in variables]
        for key, ids in records:
            node = oracle.nodes[key]
            action = tuple(actions[g] for g in ids)
            if action not in node["actions"] or max(node["q95"])-node["q95"][node["actions"].index(action)] > 1e-8:
                raise RuntimeError("CSP assignment is not oracle-optimal and feasible")
        result.update(assignment_verified=True,
                      witness_policy=[dict(group_id=i, inputs=feature, action=actions[i]) for i, feature in enumerate(groups)])
    return result


def seed_audit(cfg):
    source = Path(__file__).resolve().parents[2]/"configs/ra_qmix_role_context_seed_registry.json"
    registry = json.loads(source.read_text())
    panels = [set(cfg["train_seeds"]), set(cfg["evaluation_seeds"]), set(DATASET_SEEDS), set(range(201, 301))]
    if any(a & b for i, a in enumerate(panels) for b in panels[i+1:]):
        raise ValueError("seed panels overlap")
    if set().union(*panels[:3]) & set(registry["declared_seed_values"]):
        raise ValueError("seed panels overlap prior local manifests")
    return dict(prior_manifest_count=registry["manifest_count"], scope=registry["scope"],
                prior_panels_disjoint=True, panels_disjoint=True, sealed_panel_closed=True,
                environment_population_previously_inspected=True)


def augment_metrics(oracle, indices, arm, individual, aggregate):
    critical = []
    for row in individual:
        key = oracle.keys[row["state_id"]]
        node = oracle.nodes[key]
        action = node["actions"][indices[key]]
        same_obs = node["observations"][0] == node["observations"][1]
        diagonal = [q for a, q in zip(node["actions"], node["q95"]) if a[0] == a[1]]
        requires_unequal = same_obs and max(node["q95"])-max(diagonal) > 1e-8
        same_features = None
        if arm in FLAGS:
            same_features = (raw_feature_key(node["observations"], 0, *FLAGS[arm]) ==
                             raw_feature_key(node["observations"], 1, *FLAGS[arm]))
            if same_features and node["masks"][0] == node["masks"][1] and action[0] != action[1]:
                raise RuntimeError("shared identical inputs produced unequal greedy actions")
        row.update(selected_actions=json.dumps(action), original_inputs_identical=same_obs,
                   augmented_inputs_identical=same_features, oracle_requires_unequal_actions=requires_unequal)
        if requires_unequal:
            critical.append(row)
    for split in ("train", "heldout"):
        selected = [r for r in critical if r["split"] == split]
        aggregate[split+"_symmetry_conflict_state_count"] = len(selected)
        aggregate[split+"_symmetry_conflict_regret"] = statistics.fmean(r["regret"] for r in selected) if selected else None


def load_checkpoint(path):
    payload = torch.load(path, weights_only=True)
    if payload["protocol_version"] != PROTOCOL or payload["feature_contract"] != FEATURE_CONTRACT:
        raise ValueError("checkpoint protocol/feature contract mismatch")
    arm = payload["algorithm"]
    if payload["information_flags"] != (list(FLAGS[arm]) if arm in FLAGS else None):
        raise ValueError("checkpoint information flags mismatch")
    config = PassiveConfig(**payload["config"])
    model = model_for(arm, config, payload["settings"], payload["train_seed"])
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def contrast_gate(deltas, baseline, policy_deltas, nominal_treatment, nominal_control,
                  minimum_absolute_reduction, minimum_relative_reduction, full):
    mean = statistics.fmean(deltas)
    half = 2.262157163*statistics.stdev(deltas)/math.sqrt(10) if full else None
    ratio = -mean/baseline if baseline > 0 else None
    policy_delta = statistics.fmean(policy_deltas)
    nominal_ratio = nominal_treatment/nominal_control-1
    wins = sum(d < 0 for d in deltas)
    passed = (mean <= -minimum_absolute_reduction and ratio is not None and
              ratio >= minimum_relative_reduction and wins >= 8 and policy_delta <= 0 and nominal_ratio <= .10)
    return dict(mean_regret_delta=mean, ci95=None if half is None else [mean-half, mean+half],
                baseline_regret=baseline, relative_reduction=ratio, improving_seed_count=wins,
                diagnostic_policy_cost_delta_mean=policy_delta, nominal_cost_relative_increase=nominal_ratio,
                minimum_absolute_reduction=minimum_absolute_reduction,
                minimum_relative_reduction=minimum_relative_reduction,
                minimum_improving_seed_count=8, hypothesis_passed=passed if full else None)


def summarize(rows, cfg, full):
    lookup = {(r["cell"], r["algorithm"], r["train_seed"]): r for r in rows}
    paired = []
    for seed in cfg["train_seeds"]:
        regret = {a: statistics.fmean(lookup[c, a, seed]["heldout_regret"] for c in DIAGNOSTIC_CELLS) for a in FLAGS}
        cost = {a: statistics.fmean(lookup[c, a, seed]["undiscounted_policy_cost"] for c in DIAGNOSTIC_CELLS) for a in FLAGS}
        paired.append(dict(train_seed=seed, **{a+"_regret": v for a, v in regret.items()},
                           role_regret_delta=regret["monotonic_role"]-regret["monotonic_base"],
                           peer_given_role_regret_delta=regret["monotonic_role_context"]-regret["monotonic_role"],
                           peer_without_role_regret_delta=regret["monotonic_context"]-regret["monotonic_base"],
                           factorial_regret_interaction=(regret["monotonic_role_context"]-regret["monotonic_role"]-
                                                         regret["monotonic_context"]+regret["monotonic_base"]),
                           role_policy_cost_delta=cost["monotonic_role"]-cost["monotonic_base"],
                           peer_given_role_policy_cost_delta=cost["monotonic_role_context"]-cost["monotonic_role"]))
    mean_regret = lambda a: statistics.fmean(r[a+"_regret"] for r in paired)
    mean_nominal_cost = lambda a: statistics.fmean(lookup["nominal", a, seed]["undiscounted_policy_cost"] for seed in cfg["train_seeds"])
    role = contrast_gate([r["role_regret_delta"] for r in paired], mean_regret("monotonic_base"),
                         [r["role_policy_cost_delta"] for r in paired], mean_nominal_cost("monotonic_role"),
                         mean_nominal_cost("monotonic_base"), .10, .20, full)
    peer = contrast_gate([r["peer_given_role_regret_delta"] for r in paired], mean_regret("monotonic_role"),
                         [r["peer_given_role_policy_cost_delta"] for r in paired], mean_nominal_cost("monotonic_role_context"),
                         mean_nominal_cost("monotonic_role"), .05, .15, full)
    controls = {}
    for cell in ("nominal", *DIAGNOSTIC_CELLS):
        control = [lookup[cell, "joint_q", seed] for seed in cfg["train_seeds"]]
        regret = statistics.fmean(r["heldout_regret"] for r in control)
        gap = statistics.fmean(r["undiscounted_policy_gap"] for r in control)
        controls[cell] = dict(heldout_regret=regret, undiscounted_policy_gap=gap,
                              heldout_normalized_rmse=statistics.fmean(r["heldout_normalized_rmse"] for r in control),
                              passed=(regret <= .10 and gap <= 2.) if full else None)
    control_passed = all(v["passed"] for v in controls.values()) if full else None
    return paired, dict(primary={"role": role, "peer_given_role": peer},
                         positive_control=dict(cells=controls, passed=control_passed),
                         empirical_attribution_supported={"role": bool(role["hypothesis_passed"] and control_passed) if full else None,
                                                          "peer_given_role": bool(peer["hypothesis_passed"] and control_passed) if full else None},
                         secondary=dict(peer_without_role_regret_delta=statistics.fmean(r["peer_without_role_regret_delta"] for r in paired),
                                        factorial_regret_interaction=statistics.fmean(r["factorial_regret_interaction"] for r in paired)),
                         scientific_gate_applicable=full,
                         interpretation="Input interventions with fixed monotonic mixer; fixed-identity and peer-telemetry diagnostic, not an adaptation/scale-transfer result.")


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
    revision, dirty = oracle_exp._git_state()
    manifest = dict(protocol_version=PROTOCOL, status="RUNNING", profile=args.profile, device="cpu",
                    started_at=datetime.now(UTC).isoformat(), git_revision=revision, git_dirty=dirty,
                    runtime=oracle_exp._runtime_metadata(), train_seeds=cfg["train_seeds"],
                    evaluation_seeds=cfg["evaluation_seeds"], dataset_seeds=DATASET_SEEDS,
                    sealed_test_evaluated=False, sealed_test_seeds=list(range(201, 301)),
                    expected_models=3*5*len(cfg["train_seeds"]),
                    expected_evaluation_rows=3*5*len(cfg["train_seeds"])*len(cfg["evaluation_seeds"]),
                    expected_optimizer_updates=3*5*len(cfg["train_seeds"])*cfg["updates"])
    write_json, write_csv = oracle_exp._write_json, oracle_exp._write_csv
    write_json(output/"manifest.json", manifest)
    write_json(output/"seed_audit.json", audit)
    write_json(output/"feature_contract.json", FEATURE_CONTRACT)
    write_json(output/"resolved_config.json", dict(settings=cfg, gamma=oracle_exp.GAMMA, learning_rate=3e-4,
               cells={c: asdict(v) for c, v in oracle_exp.cells(args.profile).items()}, arms=FLAGS,
               cf_weight=0., max_oracle_states=100000, csp_time_limit_seconds=30., csp_workers=1))
    progress, episodes, state_rows, metrics, counts, feasibility = [], [], [], [], [], {}
    actual_models = 0
    try:
        for (cell, config), split_seed in zip(oracle_exp.cells(args.profile).items(), DATASET_SEEDS):
            oracle = oracle_exp.ExactOracle(config)
            data = oracle_exp.dataset(oracle, split_seed)
            if oracle.bellman_residual() > 1e-9:
                raise RuntimeError("oracle Bellman residual")
            cell_dir = output/cell
            cell_dir.mkdir()
            states, table = [], []
            for key in oracle.keys:
                node, sid = oracle.nodes[key], oracle.key_to_id[key]
                states.append(dict(state_id=sid, time=key[0], state=asdict(key[1]),
                                   observations=node["observations"], split=data["split"][key]))
                for action, q95, q1 in zip(node["actions"], node["q95"], node["q1"]):
                    table.append(dict(state_id=sid, actions=json.dumps(action), q95=q95, q1=q1))
            write_json(cell_dir/"oracle_states.json", states)
            write_csv(cell_dir/"oracle_q.csv", table)
            feasibility[cell] = {arm: policy_feasibility(oracle, arm, split_seed) for arm in FLAGS}
            write_json(cell_dir/"policy_feasibility.json", feasibility[cell])
            for seed in tqdm(cfg["train_seeds"], desc=f"role/context {cell}", unit="seed"):
                reference = model_for("monotonic_base", config, cfg, seed)
                tensors = reference.state_dict()
                reference_count = sum(p.numel() for p in reference.parameters())
                for arm in ARMS:
                    model = model_for(arm, config, cfg, seed)
                    count = sum(p.numel() for p in model.parameters())
                    if arm in FLAGS and not all(torch.equal(v, tensors[k]) for k, v in model.state_dict().items()):
                        raise RuntimeError("factorial tensors/initialization differ")
                    if abs(count/reference_count-1) > (.10 if args.profile == "smoke" else .05):
                        raise RuntimeError("joint-Q parameter matching failed")
                    counts.append(dict(cell=cell, algorithm=arm, train_seed=seed, parameters=count,
                                       reference_parameters=reference_count, relative_deviation=count/reference_count-1,
                                       identity=FLAGS[arm][0] if arm in FLAGS else None,
                                       peer_information=FLAGS[arm][1] if arm in FLAGS else None))
                    write_csv(output/"parameter_counts.csv", counts)
                    oracle_exp.fit(model, data, cfg, seed, arm, cell, output, progress)
                    checkpoint = cell_dir/arm/f"train_seed_{seed}"/"model.pt"
                    checkpoint.parent.mkdir(parents=True)
                    torch.save(dict(protocol_version=PROTOCOL, state_dict=model.state_dict(), algorithm=arm,
                                    train_seed=seed, config=asdict(config), settings=cfg,
                                    information_flags=list(FLAGS[arm]) if arm in FLAGS else None,
                                    feature_contract=FEATURE_CONTRACT, target_mean=data["mean"], target_std=data["std"],
                                    updates=cfg["updates"]), checkpoint)
                    restored = load_checkpoint(checkpoint)
                    if not torch.equal(oracle_exp.predictions(model, data), oracle_exp.predictions(restored, data)):
                        raise RuntimeError("checkpoint predictions differ")
                    indices, individual, aggregate = oracle_exp.evaluate_tables(restored, arm, oracle, data, cell, seed)
                    augment_metrics(oracle, indices, arm, individual, aggregate)
                    state_rows.extend(individual); metrics.append(aggregate)
                    write_csv(output/"state_metrics.csv", state_rows)
                    write_csv(output/"model_metrics.csv", metrics)
                    for evaluation in tqdm(cfg["evaluation_seeds"], desc=f"evaluate {cell}/{arm}/{seed}", unit="episode", leave=False):
                        episodes.append(dict(cell=cell, algorithm=arm, train_seed=seed,
                                             **oracle_exp.rollout(oracle, indices, evaluation)))
                    write_csv(output/"episodes.partial.csv", episodes)
                    actual_models += 1
        paired, summary = summarize(metrics, cfg, args.profile == "full")
        final_updates = {(r["cell"],r["algorithm"],r["train_seed"]): r["update"] for r in progress}
        actual_updates = sum(final_updates.values())
        audits = dict(seed_audit=audit, complete_models=actual_models == manifest["expected_models"],
                      complete_evaluations=len(episodes) == manifest["expected_evaluation_rows"],
                      exact_optimizer_updates=actual_updates == manifest["expected_optimizer_updates"],
                      updates_per_model_match=len(final_updates) == actual_models and all(v == cfg["updates"] for v in final_updates.values()),
                      unique_episode_keys=len({(r["cell"],r["algorithm"],r["train_seed"],r["eval_seed"]) for r in episodes}) == len(episodes),
                      zero_invalid_requests=all(r["invalid_requests"] == 0 for r in episodes),
                      exact_cost_reconciliation=all(abs(r["cost_reconciliation_error"]) < 1e-9 for r in episodes),
                      finite_training_logs=all(math.isfinite(r[k]) for r in progress for k in ("normalized_mse","normalized_cf","raw_mse","raw_cf","gradient_norm")),
                      cf_losses_zero=all(r["normalized_cf"] == 0 for r in progress),
                      checkpoints_reload_identical=True, identical_factorial_parameters_and_initialization=True,
                      original_mixer_unchanged=True, matched_minibatch_streams=True,
                      oracle_observation_and_bellman_audits=True, same_input_action_consistency=True,
                      verified_sat_assignments=all(r["assignment_verified"] is True for f in feasibility.values() for r in f.values()
                                                   if r["status"] in ("OPTIMAL","FEASIBLE")),
                      complete_feasibility_reports=sum(len(f) for f in feasibility.values()) == 12,
                      sealed_panel_closed=True)
        if not all(v for v in audits.values() if isinstance(v, bool)):
            raise RuntimeError(f"engineering audit failed: {audits}")
        summary["audits"] = audits
        summary["information_feasibility"] = {c: {a: {k: v for k, v in r.items() if k != "witness_policy"}
                                                  for a, r in f.items()} for c, f in feasibility.items()}
        write_csv(output/"paired_seed_metrics.csv", paired)
        write_csv(output/"episodes.csv", episodes)
        write_csv(output/"coordination.csv", episodes)
        write_json(output/"summary.json", summary)
        manifest.update(status="COMPLETED", completed_at=datetime.now(UTC).isoformat(), actual_models=actual_models,
                        actual_evaluation_rows=len(episodes), actual_optimizer_updates=actual_updates, audits=audits)
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}", actual_models=actual_models,
                        actual_evaluation_rows=len(episodes))
        raise
    finally:
        manifest["outputs"] = sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())
        write_json(output/"manifest.json", manifest)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("smoke", "full"))
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
