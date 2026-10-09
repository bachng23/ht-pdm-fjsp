# Maintenance training coverage v1 — locked protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: user-authorized implementation and local smoke
- Origin Date: 2026-10-09 Asia/Taipei
- Verification Status: UNVERIFIED (scientific full result pending)
- Version Label: maintenance_training_coverage_v1
- Parent commit: 09e325f377a0e34bcf220e15e2a247e46ee1a126

## Question and intervention

Does including compatibility-mask episodes in training reduce the cooperative
policy's excessive waiting and its cost gap to centralized RL on skill-mask?
This is a six-cell seed-paired controlled coverage intervention, not a new MARL
architecture and not proof that MARL is required. Replacing specialized exposure
with mask exposure tests this specific mixture at fixed budget, not additive data.

Three inherited algorithms: central_matching_ppo, cooperative_local_ppo,
cooperative_full_ppo. Two schedules, repeated deterministically every five episodes:
legacy = [specialized, specialized, specialized, specialized, nominal];
covered = [specialized, skill_mask, specialized, skill_mask, nominal].
Thus nominal positions/configurations/shocks are identical between regimes;
all algorithms share the same physical configuration/shock seeds per episode.
Mixtures are 80/20/0 versus 40/20/40 specialized/nominal/skill-mask.
No homogeneous exposure in either arm. No pretraining or old checkpoint reuse.

Freeze WaitingEnv physics, prices (failure15/down6/overdue12), waiting limit4,
features, hidden64, PPO losses, actor/critic contracts and greedy serial decoder.
Equal tensor counts do not imply equal active policy capacity; preserve and report
inherited information/actor/critic contracts rather than claim capacity matching.
Each cell/seed receives 491520 physical ticks, 40960 complete H12 episodes,
320 rollouts of1536, 32 vector envs, 4 epochs, minibatch192 and10240 optimizer steps.
Use FINAL checkpoint320 for every arm: no development checkpoint selection,
guard-based fallback, early stopping, tuning or test-based replacement.
Intermediate weights at0/80/160/240/320 retained for diagnostics only; no test
rollouts on intermediate weights. Within-seed initial tensors must match all six cells.

## Seed splits and limits

New full initialization/train replica seeds180000..180009; development shocks
184000..184009; fresh sealed test shocks185000..185019. Familiar development
profile labels161000..161015 and test labels163000..163031 map to the original
80/16/32 labeled profile partition. Those test profiles were opened in earlier
studies: only new shock panel is sealed, not unseen profile combinations or
permutation classes. Training uses the original80-profile train pool.
Role-derived training configurations/environment/action/minibatch RNG seeds use
inherited names with NEW replica seeds; reject collisions with reserved/historical
values. Full/smoke new panels reserved globally; explicit reused profile labels
are exempt only as profile identifiers, never allowed training RNG values.
Engineering smoke uses seed189000, dev profile165020/shock189020, test profile165010
and shocks189040..189041; every smoke physical profile belongs to FULL TRAIN pool.
Smoke horizons4/6, hidden16,256 ticks/model, two128 rollouts, epochs2/minibatch32.
No full test shock trajectory may be opened by smoke/tests.

## Evaluation and locked inference

Evaluate all six cells on the same four conditions and H12/H24,32 profiles x20
shocks. H12 isolates condition transfer at trained horizon; H24 adds horizon
transfer and remains separately labeled. Fresh shocks are common across arms and
prefix-coupled across horizons. Rule chosen per condition by minimum development
H12 cost among the four inherited rules, deterministic name tie-break; joint
rollout uses that continuation with32 forecast scenarios. Rules/planner are
contextual baselines, not optimality bounds. They never see realized future shocks.
No rule selection using test. Final learned policies also evaluated on development
H12 for diagnostics, without weight selection. One common rule-generated development
trajectory per condition supplies warm latency measurements: one warm pass, three
measured passes, one CPU thread, all cells on same states. Communication reports
inherited logical payloads/rounds, not measured network performance.

Four PRIMARY tests on skill_mask H24, paired ten TRAIN REPLICA means conditional
on the fixed familiar-profile/fresh-shock panel. Do not treat episodes as replicas.
Bonferroni familywise0.05 across four tests gives two-sided98.75% paired-t intervals.
1-2: covered-minus-legacy objective for cooperative_local/full, respectively.
Success: mean priced reduction>=2%, >=8/10 negative paired differences, CI upper<0;
covered waiting-violation <=legacy+0.02 and terminal pending <=legacy+0.05/episode.
3-4: difference-in-differences gap closure for local/full relative to central:
[(coop_covered-central_covered)-(coop_legacy-central_legacy)]. Success: legacy
mean gap>0, mean gap closes>=25%, >=8/10 negative differences, CI upper<0 and
same within-policy waiting/pending guards. This can pass while central still wins;
report residual gap. If legacy gap<=0, closure criterion is NOT_APPLICABLE.
Gap closure alone can reflect central degradation: it is not evidence that MARL
improves. Interpret with the within-policy and central coverage contrasts.
For ALL four tests also require covered specialized H12 and nominal H12 cost
<=1.05 times respective legacy cost (mean preservation guards). These guards
are descriptive decision criteria, not extra hypothesis tests.
Central coverage effect, H12 mask effect, residual cooperative-minus-central gap,
other conditions/horizons, and all mechanism metrics are exploratory95% CIs.
Report all four tests including failures; no moving primary to H12 after seeing data.
CIs with n10 and degenerate/zero differences are limited; do not infer equivalence.

Mechanisms: component costs/failures/unavailability; episode waiting violation,
terminal pending; per-request served/censored counts, observed overdue ticks,
waiting-cost decomposition served versus unserved; served-only wait never used
alone as fairness evidence. Finite-horizon censoring does not prove permanent
starvation. Cost decomposition is descriptive, not attribution to critic or
credit assignment. Lower failure counts can accompany unserved broken machines.

## Stopping, audits and artifact schema

Full fixed budget:60 models,29491200 training ticks,2457600 training episodes,
614400 optimizer steps; no adaptive stop except execution error. Full runner
requires clean committed Linux checkout; human executes lab handoff. No resume;
new timestamped directory for each run, reject nonempty output.
Train and multi-seed eval show tqdm. Tests/local smoke precede commit/push.

Artifacts: manifest.json(status/commit/runtime/counts/hashes), benchmark_config.json,
resolved_config.json, seed_audit.json, profile_audit.json, evaluation_configs.json,
source_snapshot/, training_episodes.csv, training_progress.csv, training_coverage.json,
checkpoint_selection.json(final-only), reference_selection.json, development_episodes.csv,
episodes.partial.csv/episodes.csv, machine_metrics.csv, request_metrics.csv,
request_wait_decomposition.json, latency_states.json/latency.csv/latency_summary.json,
paired_seed_metrics.csv, summary.json, and regime/algorithm/train_seed/checkpoints plus
model.pt. Full tick traces intentionally omitted; smoke includes decisions.csv.
Derived request columns: overdue_ticks=max(wait-limit,0), observed_waiting_cost,
served_waiting_cost,censored_waiting_cost, all reconcile exactly with episode ticks/cost.
All row identities include training_regime; partial files retained on failure.
Exact schema is five snapshots per
model plus one final copy:360 checkpoint files full,24 smoke. manifest expected/actual
counts is authoritative; scientific test episodes317440 and5713920 physical intervals.

Smoke gates: process success, exact counts, mixture counts and seed pairing;
feasible actions/cost decomposition, complete-episode returns and PPO likelihood;
weights changed, bit-identical save/reload, all required CSV/JSON hashes; no full panel
opened, tqdm visible. Unit tests must also verify synthetic difference-in-differences,
missing/duplicate grid rejection, profile separation, legacy schedule equivalence,
and immutable output/failed manifest behavior. Smoke is engineering evidence only.

Interpretation permits negative coverage effect, central gains equally, persistent
MARL gap, or planner winning. Improvements support this mixture's value under
frozen algorithm and budget, not universal MARL value. Future tuning needs fresh
sealed shock seeds. Any protocol change is a named amendment before new full run.
