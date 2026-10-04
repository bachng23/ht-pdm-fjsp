# Maintenance coupling v1 — Mac verification and handoff

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent, run.
- Origin Date: 2026-10-04; Verification Status: VERIFIED engineering tests/smoke only.
- Branch: codex/maintenance-coupling-v1; locked protocol: maintenance_coupling_v1_plan.md.

## Final implementation verification
- `uv sync --frozen --offline --extra dev` completed on macOS Python 3.12.13, isolated worktree.
- Final new test file: 16 meaningful tests including completion-only recovery, service freeze, FIFO waiting/failures, failed-service cost accounting, congestion-aware assignment, independent un-memoized oracle tree (p=0/.37/1), hand-calculated optimal cost, physical-machine permutation, common shocks, cache cap, policy JSON round trip, complete runner artifacts/refused overwrite, fixed-timing attribution control and instance-level inference.
- Full repository test suite: **258 passed**, 29.40 seconds; one third-party tqdm warning.
- Final-code smoke: `artifacts/maintenance_coupling_smoke_20261004T132536Z/`. COMPLETED, 48 unique episodes, 80 audit decision rows, 16 exact-policy/oracle rows, zero invalid requests, reconciled costs, all audit gates passed. Full/sealed panel remains unopened. Source checksum is checked against the final implementation; smoke manifest records pre-commit dirty state because these new files were not committed yet.
- Initial three-policy engineering smoke retained separately; superseded by final four-policy smoke.
- C depth-3/tail-2, horizon-12 engineering probe on smoke seed 770100/771100: low/heterogeneous ~0.056 s and high/heterogeneous ~0.020 s for one episode. Not a full runtime estimate or scientific outcome; final added D control also passes runner tests.

## Interpretation limits
Smoke is engineering evidence; `primary_gate=false` is intentionally disabled in smoke and is not a negative research result. No full configuration outcomes or model training were run on the Mac. Full evaluation is 12 paired instances × four cells × 40 shock seeds × four policies = 7,680 episode rows. D is a fixed-timing/cost-aware lookahead control. Oracle uses a separate horizon-4 panel; its gaps cannot be combined with horizon-12 rollout costs. Stronger attribution requires C–B practical gate, positive pressure×skill interaction and positive D–C CI, as frozen in the protocol. No claim about MARL necessity or novelty follows automatically.

## CLI
```bash
uv run --frozen python -m ht_pdm_fjsp.maintenance_coupling --profile full --device cpu --output-dir artifacts/<new-timestamped-run>
```
Full run requires clean Git source; saves manifest with commit/source/protocol hashes. No new dependencies, no checkpoint prerequisite, no RL training. Policy parameters are frozen in policy.json after development tuning. Progress bars cover tuning, multi-seed evaluation and tiny oracle. Output directory must be fresh; interrupted artifacts remain incomplete and are not silently reused.

Follow docs/experiment_workflow.md: user launches full run from the Ubuntu terminal, then runs rsync on the Mac. Assistant does not SSH/run lab/transfer artifacts.
