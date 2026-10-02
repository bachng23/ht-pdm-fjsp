# RA-QMIX artifact diagnostic — locked v1 (2026-10-02)

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Verification Status: protocol locked before diagnostic replay; existing endpoint results already observed, so all mechanism findings are exploratory.
- Version Label: ra_qmix_artifact_diagnostic_v1

## Hypotheses and contrasts
H1: post-boundary nominal regression in curriculum_replay is attributable in the recorded objective to maintenance and/or queue costs rather than only failures. Report all four cost deltas regardless of direction. H2: divergence emerges after the shared 240192-step boundary, with changes in request age, preventive starts, technician choices and Q margins. These are descriptive mechanism associations, not causal claims. No architecture changes, retraining, gamma/RNG changes, oracle experiments or permutation interventions in this run.

Source is the completed ra_qmix_curriculum_confirmation_full_cpu_20261001T081927Z, source commit 33ab6196f0dd4d4e12038b74b65fd3c5782096f7. Require exact protocol, seeds, settings, configs and all source checkpoints before replay. Source files are read-only and SHA256 checked before/after.

## Seeds and budget
Full: training seeds 84600–84609; evaluation seeds 84700–84749; regimes nominal_100 and curriculum_replay; steps 240192/360288/480384; all four source scenarios. 12000 replay episodes and 60 source checkpoints. Smoke selects train 84600, eval 84700–84702, all regimes/scenarios/checkpoints: 72 episodes and 6 checkpoints. Smoke is engineering evidence only. No new training seeds; sealed 201–300 stays closed. Stop after fixed replay panel; never select checkpoints. Failure/nonfinite/replay mismatch aborts and leaves FAILED manifest; no automatic retry or overwrite.

## Metrics and analysis
Primary: nominal paired curriculum-minus-control objective delta and decomposition per training seed/checkpoint; difference of that delta from shared boundary. Average common eval episodes within each training seed first, report seed deltas, across-seed mean/SD. No inference treating evaluation episodes as independent training runs; no p-value or causal verdict. Secondary: same per-scenario contrast, failures, preventive/corrective starts, waiting and requests; per-decision age/failure/queue/busy context, chosen technician, min service time, chosen service time, feasible action count and greedy top-two Q margin (null when only one feasible action). Longer chosen service time is descriptive, not proof of bad assignment.

Stream source training_progress.csv, aggregate by regime/seed/phase/checkpoint interval: TD/raw CF/weighted CF/total loss, epsilon endpoints and optimizer update endpoints; retain null for unlogged gradient norms and replay-state features. Export source adaptation_diagnostics/training_schedule unchanged to document replay ratios. No reconstruction of historical gradients from checkpoint gradients.

## Gates and artifacts
Replay every metric also present in source episodes.csv within absolute tolerance 1e-6; exact coverage/unique keys, masks feasible, zero invalid requests, finite Q values/costs, cost reconciliation, boundary state_dict equality, immutable source hashes. All checkpoints loaded strictly using source architecture, no checkpoint copies modified. Full gates include all source checkpoint and CSV expected coverage. Training is N/A; tqdm for multi-seed replay and training-log reading.

New timestamped output: manifest.json (status, protocol/source SHA/runtime/seeds/settings/artifact list), source_hashes.json, benchmark_config.json, resolved_config.json, episodes.csv, decisions.csv, seed_summary.csv, paired_deltas.csv, training_log_summary.csv, adaptation_diagnostics.csv, training_schedule.csv, summary.json, diagnostic_report.md. Missing metrics appear null and in report. No production-availability claim from current recovery-at-start semantics.

## Workflow
Dedicated codex/ra-qmix-artifact-diagnostic branch based on source commit; targeted and repository tests, CPU Mac smoke with real source checkpoints; commit/push verified files only; user runs full CPU replay on lab and rsync -avhP --partial afterward. No SSH/full-run/transfer by assistant. Progress is written after each completed checkpoint; no resume contract.
