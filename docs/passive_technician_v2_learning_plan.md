# Passive-technician v2 learning experiment plan

## Material Passport

- Origin skill: experiment-agent
- Origin mode: plan
- Origin date: 2026-09-25
- Verification status: UNVERIFIED
- Version label: code_plan_v1

## Research question

Can decentralized machine agents learn to coordinate maintenance timing and
technician selection when technicians are passive shared resources with fixed
eligibility and service times?

The environment is the candidate selected by the locked v2.1 calibration:
`m4__synchronized__mixed_eligibility` at Git commit
`fd77f1b45f3fa3921edf75314d120269341e1074`.

## Hypotheses

- **H1 (valid learning):** every learned-policy evaluation has zero invalid
  actions and physical-invariant violations, and episode return equals negative
  objective within `1e-9`.
- **H2 (CTDE benefit):** selected MAPPO has lower mean reporting-panel
  objective than selected PS-IPPO across paired training seeds.
- **H3 (execution-information value):** centralized PPO is an information
  reference; its difference from MAPPO estimates the value of global execution
  information under the same training budget, without claiming a
  capacity-matched architectural effect.
- **H4 (heuristic threshold):** at least one decentralized learned policy has a
  lower mean objective than the locked reservation-aware heuristic on the
  reporting panel.

H1 is a hard validity gate. H2--H4 are falsifiable scientific outcomes and do
not determine whether the run is technically complete.

## Locked environment

- Four machine agents and two passive technicians.
- Horizon 36, failure age 6, maximum age 10, failure probability 0.30.
- Initial machine ages `(0, 0, 0, 0)`.
- Service times:

  | Machine | Technician 0 | Technician 1 |
  |---|---:|---:|
  | M0 | 2 | 5 |
  | M1 | 2 | 3 |
  | M2 | 3 | 2 |
  | M3 | 5 | 2 |

- Eligibility: M0 only technician 0; M3 only technician 1; M1 and M2 both.
- Technicians have no policy, learning state, reward, or action.
- Machine actions are `wait`, request technician 0, or request technician 1,
  with the simulator action mask enforced before sampling.
- Objective and transition versions remain
  `downtime_queue_terminal_v2` and `passive_technician_v2`.

## Information structure and conditions

All learned conditions receive the same team reward and use a shared actor
across machines.

1. **PS-IPPO:** actor and critic receive only each machine's local observation.
   The team reward is copied to each agent critic.
2. **MAPPO:** decentralized actor receives the same local observation; a
   centralized critic receives the joint global state during training only.
3. **Centralized PPO reference:** both actor and critic receive the global state
   and the actor emits all four machine actions. This is an information upper
   reference rather than a decentralized policy.

The local observation contains the machine's time, age, mode, assignment,
service remainder, and fixed properties plus availability, remaining workload,
queue length, queue position, and ownership for both technicians. It does not
contain other machines' ages, modes, or actions. The global critic state is the
concatenation of all four local observations.

Fixed baselines are:

- `reactive_fastest`;
- `threshold_fastest`;
- `workload_aware`;
- `reservation_aware`, which handles machines with fewer eligible technicians
  first and then minimizes virtual workload plus service time.

## Training, checkpoint selection, and stopping rule

Each learned condition uses the same PPO hyperparameters except for critic and
actor information structure. Training counts joint environment transitions.

| Profile | Training seeds | Joint steps per condition-seed | Vector envs |
|---|---:|---:|---:|
| smoke | `11000` | 1,024 | 4 |
| pilot | `11000:11003` | 50,000 | 8 |
| full | `11000:11005` | 300,000 | 8 |

Locked hyperparameters:

- five checkpoints at 20%, 40%, 60%, 80%, and 100% of budget;
- rollout length 128 joint steps per vector environment;
- PPO epochs 4, minibatch 256 environment-steps;
- learning rate `3e-4`, gamma `0.99`, GAE lambda `0.95`;
- clip range `0.2`, entropy coefficient `0.01`, value coefficient `0.5`;
- maximum gradient norm `0.5`;
- two hidden layers of width 64 for actors and 128 for centralized critics;
- reward scale `0.05` during training only;
- deterministic argmax evaluation.

For every algorithm and training seed, all five checkpoints are evaluated on
the checkpoint-selection panel. The checkpoint with lowest mean objective is
selected; exact ties choose the earliest checkpoint. Selected checkpoints are
then evaluated on the disjoint reporting panel. Training does not stop early.

## Seed panels

- Environment training episodes use deterministic seeds derived from training
  seed, vector-environment index, and completed episode count. They cannot
  overlap the panels below.
- Checkpoint-selection seeds: `9100:9130`.
- Reporting seeds: `9130:9200`.
- Sealed test seeds: `9200:9300`, recorded but never evaluated here.

Smoke uses selection seeds `9100:9103` and reporting seeds `9130:9133`.
Pilot uses `9100:9110` and `9130:9150`.

## Endpoints and statistical unit

Primary endpoint: mean reporting-panel objective per independent training seed.
The primary contrast is paired MAPPO minus PS-IPPO across the five full-profile
training seeds. A two-sided 95% t interval is reported over training-seed
differences.

Secondary endpoints:

- objective differences from reservation-aware and threshold-fastest;
- failures, downtime, waiting, PM/CM starts, terminal cost;
- technician utilization and utilization imbalance;
- slower-technician request fraction and proposal contention;
- selected checkpoint milestone and validation objective;
- training return, entropy, policy/value loss, approximate KL, and clip
  fraction;
- centralized PPO minus MAPPO as a descriptive information contrast.

Reporting episodes within a training seed are repeated measurements, not
independent algorithm replicates. Claims about training stability use training
seed as the unit.

## Hard gates

The run is technically valid only if:

1. every expected training replicate, checkpoint, selection evaluation, and
   reporting evaluation is present;
2. saved models reload and reproduce deterministic actions;
3. all learned actions satisfy masks and all state audits pass;
4. maximum reward/objective identity error is at most `1e-9`;
5. all selected checkpoints belong to the locked five milestones;
6. progress is shown with `tqdm` for training and evaluation;
7. the sealed panel remains closed.

## Artifact schema

Each run refuses a non-empty output directory and writes:

```text
artifacts/<timestamped-run-id>/
├── manifest.json
├── resolved_config.json
├── training_progress.csv
├── training_episodes.csv
├── checkpoint_evaluations.csv
├── checkpoint_selection.csv
├── episodes.partial.csv
├── episodes.csv
├── coordination.csv
├── summary.json
└── <algorithm>/train_seed_<seed>/
    ├── checkpoints/*.pt
    ├── selected_model.pt
    └── final_model.pt
```

## Interpretation boundary

This full profile is still development evidence because the environment was
selected using `9100:9200`. The sealed panel may be opened only after model
architecture, training budget, checkpoint-selection rule, and analysis code
have all been frozen in a later confirmatory protocol. The centralized PPO
condition differs in both execution information and parameterization, so its
contrast with MAPPO is descriptive.
