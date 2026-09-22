# MK01 safe-noop technician-conflict diagnostic

## Material Passport

- Origin skill: ARS experiment-agent
- Origin mode: plan/run
- Artifact: executable simulation experiment plan
- Instance: `ht_pdm_fjsp_brandimarte_mk01_v1`
- Architecture: six machine agents and two shared technicians
- Status: implementation and local smoke validation
- Scope: development diagnostic, not an algorithm-performance claim

## Question and locked hypotheses

This diagnostic tests whether a progress-safe no-op mask removes technician
conflicts caused by forcing idle machine agents into optional preventive
maintenance, while retaining a measurable conflict signal from voluntary
preventive and corrective requests.

The hypotheses are:

1. all safe-noop conditions have zero action-mask-forced preventive proposals
   and zero forced technician conflicts;
2. the threshold-0.18 safe-noop condition retains valid technician conflicts in
   at least 10% of seeds and at a rate of at least 1% per maintenance proposal;
3. its aggregate rejected-proposal rate is at most 20%.

The threshold-0.05 condition is descriptive. No monotonic-conflict hypothesis is
made because earlier maintenance can temporally disperse technician demand.

## Intervention and conditions

The legacy mask disables wait for every machine that has any feasible local
action. The safe-noop mask instead selects one rotating progress anchor whenever
there is no pending simulator event. It prefers an agent with feasible production
or corrective work, disables wait only for that anchor, and lets all peers yield.
This prevents an all-wait deadlock without forcing every idle peer to maintain.

Four paired conditions use the same deterministic joint SPT heuristic:

1. `legacy_threshold_018_joint_spt`: frozen legacy-mask control;
2. `noop_reactive_joint_spt`: safe no-op with no voluntary preventive action;
3. `noop_threshold_018_joint_spt`: intended safe-noop condition;
4. `noop_threshold_005_joint_spt`: descriptive maintenance-pressure condition.

Production routes, processing times, degradation, technician skills, service
times, resolver priority, and stochastic seeds are held fixed.

## Metrics and conflict taxonomy

Primary endpoint: valid technician conflicts in
`noop_threshold_018_joint_spt`, where valid excludes any conflict involving a
`forced_preventive` proposal.

Every technician conflict is assigned exactly once to:

- forced by action-mask maintenance;
- voluntary preventive–preventive;
- corrective–corrective;
- voluntary preventive–corrective;
- other, retained as an accounting alarm.

Secondary endpoints are episode conflict incidence, conflict rate per
maintenance proposal, rejected-proposal rate, maintenance proposals, production
conflicts, objective, makespan, failures, and preventive/corrective maintenance.
The independent unit is one environment seed; contrasts are paired by seed and
reported with two-sided 95% t intervals.

## Seed split and stopping rule

- Local macOS smoke: `61410:61415`, five seeds per condition.
- Full development panel: `61500:61700`, 200 already-opened seeds per condition.
- Future confirmation panel: `62000:62100`, kept unopened.

Stop after one full panel. Do not tune and rerun the same gate. A failed gate
triggers another development redesign; it does not authorize opening the future
panel or starting MARL training.

## Locked gate

All seven checks must pass:

1. all 800 episodes complete;
2. feasibility and conflict-accounting audits pass;
3. all safe-noop conditions have zero forced preventive proposals;
4. all safe-noop conditions have zero forced technician conflicts;
5. threshold 0.18 has valid conflicts in at least 10% of seeds;
6. its valid-conflict rate is at least 1% of maintenance proposals;
7. its aggregate rejected-proposal rate is at most 20%.

## Artifact schema

Each invocation rejects a non-empty output directory and writes:

- `benchmark_config.json`;
- `technician_noop_manifest.json`;
- final and incremental `.partial.csv` episode, decision, and coordination tables;
- `technician_noop_summary.json`.

Training/evaluation progress is displayed with `tqdm` over all condition-seed
episodes.

## Rented Ubuntu server command

Run this single command from the local Mac:

```bash
ssh -p 20002 root@194.93.49.15 'export PATH="/root/.local/bin:$PATH" && cd /root/ht-pdm-fjsp && git fetch origin && (git switch codex/mk01-safe-noop-diagnostic || git switch --track origin/codex/mk01-safe-noop-diagnostic) && git pull --ff-only && (command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh) && export PATH="/root/.local/bin:$PATH" && uv sync --frozen && set -o pipefail && RUN_ID="mk01_safe_noop_$(date -u +%Y%m%dT%H%M%SZ)" && mkdir -p "artifacts/$RUN_ID" && uv run ht-pdm-fjsp-technician-noop --profile full --output-dir "artifacts/$RUN_ID" 2>&1 | tee "artifacts/$RUN_ID.log"'
```

The experiment is simulator-bound and does not require CUDA.

## Pull all artifacts back to the Mac

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && rsync -avhP --partial -e "ssh -p 20002" root@194.93.49.15:/root/ht-pdm-fjsp/artifacts/ "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
