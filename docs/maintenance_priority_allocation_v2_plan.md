## Material Passport

- Type: locked protocol amendment, `maintenance_priority_allocation_v2`.
- Status: PLANNED before implementation or v2 outcome inspection.
- Authorization: user requested the three agreed corrections and a lab handoff.
- Parent: v1 plan and commit `31b2b9e`; v1 smoke artifacts are preserved. This is a new run, not a resume or a reanalysis of v1.

## Hypotheses and four deployment arms

Retain v1 H1: learned_both must reduce equally weighted N3/N4 nominal/pressure mean cost by at least 5% against EACH learned control, win at least 8 of 10 paired TRAINING seeds, have the 97.5% two-sided Bonferroni paired-t CI upper below zero (df9, critical 2.685010847), and keep nominal mean cost within 110% of each control. Both contrasts must pass. Report effect size and CI even when this practical-benefit gate fails. Failure is not evidence that learning a decision has zero value.

Four deployment arms: learned_both, fixed_priority, fixed_allocation, risk_skill_rule. Only the first three train. The existing deterministic rule is now an explicit fourth baseline in family tables, paired contrasts and exact-headroom reports. It still uses its observable urgency trigger and shortest-duration/restoration allocation; it supplies no labels. There is no extra training budget for the rule.

The rule contrast is secondary descriptive, with mean paired differences, relative reduction, wins, nominal change and 95% paired-t CI across training seeds (df9, critical 2.262157163). Do not convert it to a third confirmatory hypothesis or change the two-contrast multiplicity correction. A combined descriptive practical gate also records whether H1 passes AND learned_both beats the rule in primary and nominal cost, without claiming a third significance test. A single rule cost panel is reused against the ten learned models; the uncertainty is across training seeds CONDITIONAL on the fixed evaluation panel, not ten independent rule trainings or population-level configuration uncertainty.

## STOP and fixed priority contract

For fixed_priority, rank ALL currently feasible machines using the same failed/wait/age/risk score as v1. This arm fixes only machine order, not timing or urgency eligibility. At each decoder prefix expose only the highest-ranked feasible machine and STOP to the actor's categorical distribution. Its conditional serve-versus-STOP logits remain learned, as does technician allocation. STOP is legal even with failed demand/free technicians. Healthy low-risk machines remain eligible. On-policy log probabilities and entropy include this learned binary choice, and sequence replay verifies the rule order only on selected pairs. The full learner keeps all feasible machines plus STOP; fixed_allocation keeps learned machine/STOP choice and deterministic technician choice.

Thus the priority contrast tests learned identity/order versus rule order within a learned timing policy. It is not a claim to isolate all preventive-maintenance timing. STOP terminates the current matching; fixed_priority cannot skip a ranked machine to serve a lower-ranked one. Fixed-rule ties use physical index as a documented deterministic tie-break, not a learned ID priority. All pairs still execute simultaneously; no decoder-time clock advances. Encoders, parameter counts, physical interaction budgets, optimization schedules and public information are unchanged.

## Exact headroom and interpretation

Retain the N2/K2 exact Bellman oracle and selection/allocation regret decomposition. Add per-policy cost, optimum, absolute gap, gap/cost (maximum possible relative reduction against that deployed policy in THIS exact cell), and whether an unrestricted optimal policy could improve that cost by 5%. Add learned_both-vs-each-control/rule exact differences with reference headroom.

Report the rule's ceiling even before learned model diagnostics. This report cannot alter the 5% gate, budget, action rules, seeds or stopping. It cannot establish headroom on N3/N4, feasibility within a restricted control architecture, or a need for MARL. The oracle shares the simulator dynamics; it is not independent real-world validation. Exact diagnostics remain evaluation-only and never select checkpoints or train the actor.

## Seeds, budget and stopping

Fresh full training seeds 116000–116009; development 117000–117019; evaluation 118000–118049. Fresh smoke training 119000, development 119020, evaluation 119010–119012. Keep disjoint historical registry and explicitly reserve the parent v1 ranges 110000–110009, 111000–111019, 112000–112049, 114000, 114010–114012,114020. Include readable local manifest hashes and declared seeds in a v2 frozen registry; scope excludes unseen external runs. Existing sealed 201–300 and 601–700 stay closed. Configuration/environment/action/minibatch streams use the unchanged SHA256 role separation.

Each training seed is evaluated on all 50 common independent held-out episode seeds in each family; each episode seed defines configuration, initial state and shock grid. Statistical unit is the training seed after averaging episodes and the four primary families. Episode outcomes are not treated as independent learned-policy replications.

Keep all v1 physics, observations, training population, family definitions and settings: CPU-only, 120000 physical transitions/model, 250 complete-episode rollouts, 4000 optimizer steps/model, gamma=lambda=1, no tuning/early stop/checkpoint selection/resume. 30 trained models, 3.6m steps, 300000 training episodes, 7500 rollout logs, 120000 optimizer steps, 10500 learned test episodes, 600 learned development episodes, 350 rule test +20 rule development episodes, 31 exact rows. Smoke: 3 models, 144 training steps, 24 optimizer steps, 36 training episodes, 63 learned test +3 development episodes, 22 rule episodes, 4 exact rows. Smoke scientific gates remain null. No full run on Mac.

## Artifacts and gates

Retain the v1 artifacts and streaming partial outputs. Add reference_contrasts.csv, exact_headroom.csv, exact_headroom_contrasts.csv, exact_headroom_summary.json, and rule decision/per-machine traces under reference_decisions.csv and reference_machine_metrics.csv. Summaries include all four family cost tables. Freeze v2 plan and seed registry alongside source at startup, recording hashes. Checkpoint protocol version changes to v2; reject v1 checkpoints. Full runner refuses a dirty checkout before generating test configurations or creating artifacts.

Engineering gates: all v1 gates plus reference-key completeness/uniqueness, exact regret and headroom reconciliation, learned fixed-priority STOP probability and gradients, feasible nonurgent selection, fixed ordering during replay, rule contrast independence-unit disclosure, and full dirty-checkout rejection. Smoke must finish, reload all models identically, validate every trace, match expected artifact counts, and leave sealed panels closed. Tests and one timestamped Mac smoke precede commit/push; after successful push hand off exactly one lab compound command and one rsync command. Never launch full or transfer on the user's behalf.
