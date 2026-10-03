# Consistent-policy supervision diagnostic — locked v1, 2026-10-04

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: user-authorized implementation, tests and Mac smoke; full lab run handed to user.
- Protocol: ra_qmix_policy_supervision_v1, locked before smoke/full outcomes.
- Base commit: 0710ca83663df6173ae9e2a08cb74a528e03c875.

## Question and design
With ID+peer fixed, can the shared agent learn a consistent oracle-optimal assignment? Does adding decision supervision to full-Q fitting improve decisions without replacing the mixer?

Three factorial objective arms, identical shared ConditionedEdgeQ and original monotonic technician-aware mixer, all features active, CF=0:
- monotonic_q: normalized whole-Q MSE, unchanged baseline.
- monotonic_policy: masked local cross-entropy against one consistent SAT witness; mixer registered but receives no gradient and remains at initialization.
- monotonic_hybrid: normalized whole-Q MSE + 1.0 × masked local cross-entropy.
A fourth joint_q MSE diagnostic has central joint selection and matched total capacity within5% (smoke10%); it is not a comparable decentralized policy.

Each cell has a single full-population SAT teacher constructed BEFORE training, CP-SAT one worker, deterministic cell dataset seed,30s limit, tie tolerance1e-8. Abort UNKNOWN/UNSAT/invalid or any non-optimal/invalid assignment. Store its whole lookup map and selected joint actions. Do not choose tie labels independently across states: labels must agree for identical augmented local inputs. Teacher is the same across objectives, regimes and training seeds; this tests one fixed teacher, not sensitivity to alternative SAT solutions.

Two regimes:
- ceiling: all state-action rows trained; normalization uses all these rows. Tests finite-table learnability, NOT held-out generalization.
- split: 80/20 observation-equivalence-group split stratified by time; training-only normalization. Whole action tables of a state stay in one split. Teacher was constructed using ALL states, including held-out optimal-action constraints: this is explicitly a transductive fixed-teacher transfer diagnostic, NOT an independent held-out oracle generalization claim. No held-out labels enter gradient batches in split regime.

Architecture, tensors/initialization, Adam3e-4, clipping10, batch256, and 10000 updates are shared. For each regime/cell/seed, the same dedicated seed+17 RNG samples the same training state-action rows in all four arms. CE reads the sampled row's state and both masked local teacher labels, mean across agents/rows; therefore CE state sampling is action-row-count weighted, not uniform unique-input sampling. Mixed loss weight1.0 is locked; no tuning after smoke. ID is fixed machine index; peer telemetry is an execution requirement. Policy-only full-Q RMSE is uninterpretable as calibration (mixer untrained), reported descriptive only.

Cells unchanged from prior finite known population: N2 K2 horizon6 max_age5, nominal failure_age3,p=.45,service((2,3),(3,2)); shared_preference same except service((2,3),(2,3)); pressure failure_age2,p=.65,service((3,4),(3,4)). FIFO and all costs unchanged. Gamma=.95 oracle decisions, gamma=1 exact deployment costs evaluated separately. Smoke horizon3/max_age4, hidden16,batch64,160updates; full hidden64,mixer64.

## Locked hypotheses and endpoints
Training seeds are inference units (10), not states or episodes. Endpoints use FINAL checkpoint only.

H-learn (ceiling monotonic_policy): in EACH of three cells, mean all-state one-step oracle regret <=.10 AND >=8/10 seeds have >=99% all-state oracle-optimal joint decisions (regret<=1e-8). Optimality, not exact teacher agreement, is the primary metric because ties can differ. Teacher agreement is secondary. H-learn does not require deployment/gamma1 optimality; report exact costs as separate endpoints.

H-objective (ceiling monotonic_hybrid minus monotonic_q): equal mean weighting of shared_preference/pressure all-state regret; paired delta<=-.10, relative reduction>=20%, >=8/10 improving seeds. Additionally exact undiscounted deployed cost averaged across those cells must not increase, and nominal mean cost <=1.10×q baseline. Supportive paired Student-t CI95%, df9 t=2.262157163, not an extra gate. These are practical gates, not p-value significance claims.

Positive control: ceiling joint_q EACH cell all-state regret<=.10 AND mean exact undiscounted initial-policy gap<=2.0. Empirical attribution for H-objective requires its gate AND control. H-learn is an absolute finite-table capacity check; a FAIL is unresolved optimization/architecture/teacher difficulty, NOT an impossibility proof. Prior experiments' FAIL gates stay unchanged.

Split regime secondary only: training/held-out regret, oracle-optimal fraction, teacher joint/agent agreement, exact costs, critical symmetric-state actions. No confirmatory generalization claim; no best-cell/seed selection or threshold change. Policy vs Q contrast is descriptive; hybrid locked contrast isolates adding supervision more directly. Test ceiling first conceptually, but execute every regime/arm irrespective of scientific outcomes.

## Seeds, stopping, artifact schema
New full training97000–97009; rollouts97200–97249; dataset/solver97100/97101/97102; smoke97400/97410–97412. Prior local manifest registry frozen before outcomes, includes role/context run. Sealed201–300 closed. Fresh seeds do not imply fresh environment population.

Fixed full budget: 3cells×2regimes×4arms×10seeds=240models,2.4millionupdates,12000episodes. Smoke24models,3840updates,72episodes. No early scientific stopping, automatic retry/resume, or best-checkpoint selection. Exceptions/nonfinite/engineering audit failures retain FAILED artifacts; empty/nonexistent new UTC timestamp output only. Training and evaluation tqdm mandatory.

Root manifest, resolved_config, feature_contract, seed_audit, parameter_counts.csv, minibatch_streams.csv, training_progress.csv, state_metrics.csv, model_metrics.csv, paired_seed_metrics.csv, episodes.partial.csv/episodes.csv/coordination.csv, summary.json. Per-cell oracle_states.json/oracle_q.csv/policy_feasibility.json/teacher_actions.json; per-cell/regime/arm/train_seed_N/model.pt. Checkpoints include protocol, objective/regime, config/settings, feature contract, normalization and teacher SHA256; runtime records the OR-Tools version. Teacher discounted/undiscounted costs are stored separately to avoid assuming gamma=.95 decisions minimize gamma=1 costs. State metrics include selected actions, teacher actions/agreement, original input symmetry and conflict flags, fit-state status. Model metrics include all/train/heldout regret, optimal fraction and teacher agreement; split heldout is transfer to a teacher constructed transductively. Exact costs from DP; episodes are sampled support only.

Engineering smoke gate: counts/finalupdates/logs finite, bitwise checkpoint prediction/agent reload, same initialization/parameter counts/minibatch hashes across objectives, original mixer unchanged at construction, policy-only mixer unchanged after training, SAT teacher consistency/optimality, zero invalid/cost reconciliation errors, masked execution valid, sealed panel closed, outputs complete. Scientific gates disabled in smoke. Tests exercise teacher consistency/ties, losses/masks/gradient paths, matching budgets/streams, normalization isolation, summaries/control/nominal guards, save/load contracts and failure/no-overwrite behavior.

## Interpretation and handoff
No online RL/adaptation, CF, new environments, learned roles, permutation/scale transfer, or communication-cost claim. Allstates SAT proves lookup feasibility only; neither failed fit nor positive control alone identifies the mixer as causal culprit. Same nominal parameter counts can include untrained parameters in policy-only; explicitly report trainable/active roles.

Canonical workflow: tests + real Mac smoke → verified commit/push → exact compound CPU lab command for user → exact rsync -avhP --partial command for user. Assistant never SSH/full-launches/transfers. Raw prior results preserved. This plan authorizes code creation as requested by user, overriding the ARS executor's generic restriction against generating scripts.
