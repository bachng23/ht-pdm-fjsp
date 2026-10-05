## Material Passport
- Type: cause-based problem formulation and algorithm design.
- User-confirmed research focus: learn maintenance priority and allocation of passive technicians; no fixed FIFO commitment.
- Status at proposal capture: PROPOSED. The subsequent implementation/protocol is recorded in `maintenance_priority_allocation_plan.md`; engineering verification is in `maintenance_priority_allocation_smoke_report.md`. No full scientific validation is claimed.
- Evidence: preceding oracle, role/context, policy-supervision and online TD/assignment experiments, especially `ra_qmix_heuristic_assignment_analysis_2026-10-05.md`.
- Revision: replaces the earlier FIFO-preserving proposal following the user's decision to focus on learned priority/allocation. Existing simulator, checkpoints and result reports remain unchanged.

## Research question
When technician capacity is scarce, which machines should receive preventive/corrective maintenance now, and which available technician should serve each selected machine, to minimize long-term team cost?

Technicians are passive resources with observable availability, compatibility, service duration and restoration characteristics. Machines are decision entities. A coordinated policy learns selection/priority and resource assignment; there is no learned technician agent and no permanently preassigned per-technician FIFO queue.

## Evidence and limits of causal diagnosis
Without ID, the original shared deterministic policy could not break symmetry in some small cells. Identity+peer context removed inspected lookup-policy infeasibility, and teacher-only ceilings demonstrated that the existing monotonic architecture can execute optimal policies in those cells. Monotonic mixing is therefore not established as the unique source of current failures.

The latest auxiliary addition reduced diagnostic cost but caused nominal starvation, and some supervised policies underperformed on training matrices. Correct oracle labels also did not ensure a good learned policy. TD/CE gradient interference, visitation coverage and approximation errors remain unseparated hypotheses.

A verified formulation mismatch exists in the old FIFO experiment: simultaneous same-technician requests are legal and collision_cost is not charged, while the heuristic restricted new requests to one per technician and optimized a short-horizon proxy. That restriction was not the physical service-capacity constraint. This mismatch does not alone explain nominal collapse, since the direct heuristic performs much better than its learned student there.

The redesign makes learned maintenance prioritization and actual service allocation the decisions, with physical capacity imposed explicitly. It changes the MDP; it is not a claim to have repaired the old MDP's learning failure or proved the source of every failure.

## Environment contract: pending demand and immediate dispatch
Maintain a shared candidate set rather than queues attached to technicians:
- A failed machine not under service has an outstanding corrective-maintenance demand. It stays pending until served.
- An operating machine not under service is eligible for preventive maintenance if technically compatible; policy may select it or leave it operating. Unselected preventive candidates are not committed maintenance orders.
- A machine under service cannot be selected again.
- New assignments may use only currently idle compatible technicians. Busy technicians retain their existing assignment until completion.
- At each physical decision time, choose a partial matching between candidate machines and idle technicians. Each machine and technician appears in at most one new pair. Unselected demands remain pending and can be reconsidered at the next decision time.
- Selected pairs start together; their decoder order does not determine a FIFO service order. Technician identity and machine identity are addresses for observation/action bookkeeping, not priority rules.

Capacity-one matching is now the actual physical constraint. There is no request reservation on a busy technician, no index-based resolution of simultaneous competing requests, and no collision penalty used as a substitute for coordinated feasible selection.

Waiting still exists: a failed unserved machine remains down and accumulates economic loss. Removing a FIFO container must not make waiting demands disappear or become cost-free. Waiting time is measured per demand; avoid adding a second generic queue penalty for the same downtime unless a distinct service-level cost is explicitly specified.

## Service timing and cost semantics
Align the maintenance abstraction with the repository's core assumption that maintenance occupies both the machine and technician:
1. Start: consume technician and stop the selected machine; charge maintenance-start cost once.
2. During service: technician/machine remain occupied. Do not increase operating age or sample operating failure for a stopped machine. Track unavailability through the full occupied interval.
3. Completion: release technician and apply the configured repair/restoration to the machine's age/failure state. Repair effects occur at completion, not at start.
4. Operating machines accumulate operating age and may experience failure according to the observable/known hazard model. A new failure incurs its event cost once and creates pending repair demand.

The old passive prototype resets age/failure at service start and may age/fail the machine during service; its outputs are not directly comparable to this proposed contract. This correction is a proposed model definition, not a retroactive reclassification of old results.

Core objective is expected finite-horizon total maintenance expenditure + unavailability/production-loss proxy + failure-event costs. Unavailability includes failed waiting and service occupancy, counting a machine once per time interval even if it is both failed and under repair. Pending age/wait metrics are diagnostics; additional SLA penalties require an explicit separate interpretation. A maintenance-only prototype's loss proxy is not a claim to model FJSP makespan/production scheduling fully.

Use one physical clock and explicit remaining horizon. Define service start/completion/hazard boundaries and terminal handling before experiments. Internal decoder choices do not advance physical time, release technicians, apply repair, or draw future failure events. Exogenous shocks should be keyed by episode/time/machine for policy-independent comparisons.

## Chosen policy: entity actor with learned priority and conditional assignment
Encode machine states, technician states and compatibility/service edges with shared entity/edge networks. Relevant inputs include age/risk, failed status, pending duration, remaining service time, compatibility, service duration, restoration effect and time-to-go. Static parameters varying across configurations must be exposed equally to all compared policies. Avoid a fixed concatenated peer layout or learned per-index priority embeddings. A variable-size representation is an architectural capability, not proof of unseen-size transfer.

At each decision time decode a partial matching:
1. Choose a candidate machine or STOP. Its score can depend on all candidate machines and feasible technicians; this is the learned maintenance priority.
2. Conditional on the chosen machine, choose a compatible idle technician. This is the learned allocation.
3. Reserve those two entities within the temporary matching, update the decoder's context and repeat.
4. STOP submits the matching; if no feasible pair exists, waiting is the only action. At most min(candidate_count, idle_technician_count) pairs can be selected.

Policy factorization for a sampled matching sequence is:
`pi(sequence | observation) = product_r pi(machine_r_or_STOP | observation, chosen_pairs_<r) * pi(technician_r | machine_r, observation, chosen_pairs_<r)`.
The technician factor is omitted at STOP. The sequence is a policy's latent construction of the joint action; the environment executes the resulting matching simultaneously. Previous chosen pairs distinguish otherwise symmetric candidates, while hard masks ensure capacity/compatibility. Several construction orders may yield the same matching; log probabilities and learning must be defined over sampled construction sequences consistently.

Allow STOP even when a feasible pair exists: preventive maintenance or a start near terminal time can cost more than its benefit. Do not hard-code repair priority, require every failed machine to be selected immediately, or force all technicians to stay busy. Feasibility does not guarantee absence of starvation, optimality or robustness; those need cost and service-coverage evaluation.

Train an online joint actor–critic, initially PPO, using observed team returns. Actor log probability sums conditional selection/allocation terms; training uses its sampled action prefixes, matching inference. Critic estimates team value without monotonic action-utility factorization. Gamma=1 aligns this finite-horizon learning objective with reported undiscounted cost. No assignment-teacher CE or counterfactual self-consistency penalty is the main actor objective. Small exact solutions diagnose learning and optimality gaps, not supply mandatory action labels.

This is coordinated execution, requiring a common dispatch view or communication of selected reservations. It is not strict independent simultaneous decentralized QMIX execution. If the application later requires fully decentralized decisions without communication, that is a different execution constraint and needs a compatible design.

## Learning distribution and evaluation
Specify the operating envelope before training: short/long service, low/high hazard, heterogeneous compatibility, varying load and numbers of resources. Include nominal and congested operating ranges from the outset. Use separate configurations/seeds for development and evaluation; distinguish new combinations within the envelope from extrapolation to unseen sizes/ranges.

Evaluation must answer both parts of the research question:
- Cost and failure/downtime outcomes across nominal and congested families.
- Which machines are selected when capacity is insufficient, per-machine waiting/service coverage, and actionable failed demand left pending.
- Technician-choice regret on small diagnosable configurations, skill utilization and resource feasibility.
- Decision-time compute/communication and true environment-transition budget.

Use fixed risk/skill dispatch as a transparent reference and small exact solutions as a ceiling. Mechanism controls should hold environment, encoder/information, actor–critic objective, data distribution and budget constant while fixing versus learning selection/allocation; do not compare only the new recipe against an old simulator/checkpoint. All methods in a comparison must use this same matching/service/cost contract. Old FIFO experiments remain historical evidence rather than pooled performance baselines.

One predeclared study should determine whether learned priority/allocation improve total cost without nominal regression. If it fails, inspect the two learned decisions and the model definition against those diagnostics; do not treat lower collision counts or auxiliary loss as success.

## Implementation acceptance before any new full run
- Timeline checks for failed waiting, preventive occupancy, long repair, completion/restoration and new failure.
- Joint matching masks and decoder capacity, including no compatible idle resource and optional STOP.
- Reward reconciles with cost components and no duplicate unavailability charge.
- Changing entity indexing does not introduce a hidden FIFO/priority rule; symmetry handling and variable-size behavior are checked explicitly.
- Save/reload, source snapshot/provenance, immutable run directories, tqdm, sealed seeds and counts.
- Lock hypotheses, metrics, seeds, budget/stopping and artifacts; tests and local Mac smoke; then verified commit/push and human lab/rsync handoff per `docs/experiment_workflow.md`.

No code implementation or full run is performed by this design revision.

## Primary literature grounding
[QMIX (Rashid et al., ICML 2018)](https://proceedings.mlr.press/v80/rashid18a.html) uses monotonic factorization to align decentralized greedy actions with centralized maximization; it does not guarantee learning convergence or transfer.

[Multi-Agent Reinforcement Learning is a Sequence Modeling Problem (Wen et al., NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html) provides a primary precedent for autoregressive cooperative policies conditioned on preceding choices. This matching decoder is an adaptation proposal, not a claim to reproduce MAT, inherit its guarantees or introduce an already-validated novel algorithm.
