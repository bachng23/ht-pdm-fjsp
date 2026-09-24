# Passive technician baseline suite

The main stress cell has three machines and two heterogeneous passive
technicians with FIFO queues. The exact expected dynamic program uses a
two-machine, two-technician, six-step cell with the same transition code. The
locked objective includes a queue-waiting cost of 0.5 per queued request-step,
and every learner receives the exact environment cost without an
algorithm-specific collision surcharge. Local observations include technician
availability, remaining busy time, queue position, and machine-technician
service time. The fixed baselines are random feasible and skill-aware FIFO
dispatch. Learned
baselines are tabular independent Q-learning, independent PPO (the
Rodríguez-style mechanism baseline), and centralized PPO as a coordination
upper bound.

The primary metric is total objective cost on the common 100-seed evaluation
panel, averaged within training seed. Secondary metrics are failures, jobs,
queue waiting, proposal collisions, and invalid actions. Training seeds are
11--20; evaluation seeds are 101--200; sealed seeds remain closed. Full
training uses 5,000 episodes per seed with fixed PPO hyperparameters and no
adaptive stopping. The smoke profile uses one training seed, three evaluation
seeds, and eight episodes to verify checkpoint save/load, artifact schema,
progress output, and seed recording.

The exact oracle is an expected-cost finite-horizon DP on the small cell. It is
not used to claim that the larger stress cell is solved exactly.
