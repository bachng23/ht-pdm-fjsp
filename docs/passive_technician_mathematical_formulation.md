# Mathematical formulation: shared passive technicians

## 1. Research problem

There are (N) machines and (K) passive technicians over a finite horizon
(t=0,ldots,T-1). Each machine is a learning agent. Technicians do not learn
and have fixed service characteristics. At every step, each machine decides
whether to defer maintenance or request one technician. Requests compete for a
shared technician pool and are handled by FIFO queues.

The main research question is whether decentralized machine policies can learn
maintenance timing and technician selection under this shared-resource
constraint.

## 2. Sets and fixed parameters


- Machines: (mathcal M={1,ldots,N}).
- Technicians: (mathcal K={1,ldots,K}).
- Horizon: (T).
- Machine failure-age threshold: (A_i).
- Maximum tracked age: (X_i^{max}).
- Fixed service time: (d_{ik}inmathbb N_+), the duration when technician (k) services machine (i).
- Maintenance cost: (c^m_i).
- Downtime cost: (c^d_i).
- Failure event cost: (c^f_i).
- Queue waiting cost: (c^q_i).

The skill matrix is represented by (d_{ik}). A lower value means that
technician (k) is more capable for machine (i). An infeasible assignment can
be represented by (d_{ik}=+infty), although the current pilot uses finite
service times for every pair.

The main stress cell is (N=3,K=2), with

```text
             technician 0   technician 1
machine 0           2              4
machine 1           4              2
machine 2           3              3
```

The exact oracle cell is (N=2,K=2,T=6), using the same queue and failure
semantics.

## 3. State

At the beginning of step (t), the state is

[
s_t=(x_t,y_t,b_t,m_t,q_t).
]

For each machine (i):

- (x_{i,t}in{0,ldots,X_i^{max}}) is operating age;
- (y_{i,t}in{0,1}) indicates whether the machine is failed.

For each technician (k):

- (b_{k,t}in{t,t+1,ldots,T+max d}) is the time at which the technician becomes idle;
- (m_{k,t}inmathcal Mcup{-1}) is the machine currently being serviced;
- (q_{k,t}) is the FIFO queue of machine requests waiting for technician (k).

A machine appears at most once in the union of all queues. A machine is removed
from its queue exactly when service starts.

## 4. Actions and observation

Each machine chooses

[
a_{i,t}in{0,1,ldots,K},
]

where (0) means defer and (k) means request technician (k). The joint
action is (a_t=(a_{1,t},ldots,a_{N,t})).

The decentralized observation contains local machine state, fixed capability
information, and public resource state. For each technician (k), the local
feature block contains availability, remaining busy time, whether the
technician is servicing this machine, current queue length, queue position,
and fixed service time (d_{ik}):

[
o_{i,t}=(t,x_{i,t},y_{i,t},{mathbf z}_{i,k,t})_{kinmathcal K},
]

where

[
{mathbf z}_{i,k,t}=(mathbf 1[b_{k,t}le t], b_{k,t}-t,
 mathbf 1[m_{k,t}=i], |q_{k,t}|,
 operatorname{pos}(i,q_{k,t}), d_{ik}).
]

Independent PPO receives only (o_{i,t}). A centralized critic may receive the
joint observation during training, while execution remains decentralized.

## 5. FIFO transition semantics

Given state (s_t) and joint request (a_t), the environment performs these
operations in order.

1. Every nonzero request is appended to the selected technician's queue if the
   machine is not already queued.
2. Every idle technician pops the oldest machine from its queue and starts that
   service immediately.
3. For each started job ((i,k)), set
   (x_{i,t+1}=0), (y_{i,t+1}=0), and (b_{k,t+1}=t+d_{ik}).
4. Every machine that did not start service ages by one, capped at
   (X_i^{max}). A queued machine continues aging while it waits.
5. A nonfailed machine that reaches its threshold fails with probability
   (p_i(x_{i,t+1})). The current pilot uses

[
p_i(x)=egin{cases}
0,&x<A_i,\\
p_i^{mathrm{fail}},&xge A_i,
end{cases}
]

with (p_i^{mathrm{fail}}=0.45). Failure draws are independent conditional
on the state and are seeded per episode.

6. A failed machine remains failed until service starts. A technician remains
   unavailable until its service completion time.

The Rodríguez reproduction mode replaces steps 1--2 for duplicate requests:
all simultaneous duplicate assignments are marked invalid and receive a fixed
penalty. The priority mode accepts the lowest-index requester. These modes are
ablations; FIFO is the main research semantics.

## 6. Objective

The default finite-horizon objective is to minimize expected cumulative cost:

[
J(pi)=mathbb E_pileft[sum_{t=0}^{T-1} c(s_t,a_t,s_{t+1})ight],
]

where

[
c_t =sum_i c_i^d y_{i,t}
      +sum_{(i,k)	ext{ starts at }t} c_i^m
      +sum_i c_i^f mathbf 1{i	ext{ fails at }t}
      +sum_i c_i^q mathbf 1{i	ext{ remains queued at }t}.
]

Lower cost is better. The locked current value is (c_i^q=0.5), so queue delay
affects the environment reward and is optimized by every baseline. Collision
count remains a diagnostic under FIFO semantics; collision cost is applied only
by the priority and invalid-collision ablations.

## 7. Exact finite-horizon oracle

For the small (N=2,K=2) cell, define (V_t(s)) as the optimal expected
remaining cost:

[
V_T(s)=0,
]

[
V_t(s)=min_{ainmathcal A(s)}
 left[c(s,a)+sum_{s'}P(s'|s,a)V_{t+1}(s')ight].
]

The transition probability (P) is obtained by enumerating the independent
failure-event outcomes after the deterministic FIFO service update. The oracle
uses the exact same state, queue, service-time, failure, and cost definitions as
the learned policies.

## 8. Learning problem

The simulator is a finite-horizon cooperative Dec-POMDP:

[
(mathcal I,mathcal S,{mathcal O_i},{mathcal A_i},P,r,T),
]

with shared reward (r_t=-c_t). Independent PPO learns
(pi_i(a_i|o_i)) for each machine. Centralized PPO learns a joint factorized
policy with a centralized value function. The skill-aware FIFO dispatcher is a
fixed nonlearning policy, and the exact DP is the small-instance oracle.

## 9. Locked comparison cells


- Main stress cell: (3) machines, (2) technicians, FIFO queues, 10 training seeds, 100 evaluation seeds.
- Exact oracle cell: (2) machines, (2) technicians, six-step horizon.
- Rodríguez ablation: same stress cell with invalid duplicate assignment.
- Priority resolver: diagnostic only, used to quantify machine-index bias.

Primary metric is mean final objective per training seed on common evaluation
seeds. Secondary metrics are failures, service jobs, queue waiting, proposal
conflicts, invalid proposals, technician utilization, and machine-level service
counts.
