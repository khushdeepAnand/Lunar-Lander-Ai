# Interview questions and answers

**What is reinforcement learning?** An agent improves a policy by acting in an
environment and receiving rewards. Here a landing attempt is one episode.

**How does it learn from rewards?** Replay stores `(state, action, reward,
next_state, terminated, truncated)`; gradient descent moves `Q(state, action)`
toward the terminal-aware Bellman target.

**Supervised vs. reinforcement learning?** There is no ground-truth label for
the correct engine choice. The agent discovers useful choices from delayed
rewards and exploration.

**Markov decision process?** A model of state, actions, transition dynamics,
reward and discount. The eight observed values approximate the current state.

**States, actions, rewards, policy?** Eight measurements; four discrete engine
commands; native environment reward; greedy action `argmax Q(s, a)`.

**Bellman equation?** The optimal action value equals immediate reward plus the
discounted maximum value available in the next state (unless truly terminal).

**What is a Q-value?** Estimated discounted future reward if the agent takes a
given action in the current state and subsequently follows a good policy.

**Why a neural network?** Eight continuous state values create far too many
possible states for a tabular Q-value lookup. An MLP generalizes across states.

**What is DQN?** Q-learning with a neural function approximator and separate
target network; this project uses vanilla DQN, **not Double DQN**.

**Why replay and target networks?** Replay breaks correlation between successive
steps. A temporarily frozen target network limits moving-target instability.

**What is epsilon-greedy?** Choose a random action with probability epsilon;
otherwise choose the highest-Q action. This code decays epsilon by environment
steps and evaluates greedily.

**Why is DQN unstable?** Off-policy bootstrapping with a neural approximator
can diverge; terminal rewards also produce large errors. Frozen targets, replay,
Huber loss and gradient clipping mitigate this but do not guarantee convergence.

**Reward shaping?** Extra guidance added to the training reward. The baseline
uses only native reward. The optional wrapper logs its potential-difference
bonus separately; evaluation reports native returns.

**How is success measured?** `terminated`, final native reward `+100`, both
leg-contact flags true, and normalized horizontal offset `≤0.2`. A body crash
can have both legs touching, so legs alone are insufficient. Off-pad rewarded
rest and time-limit endings are separate outcomes.

**How do you know it learned?** Compare an independently seeded greedy
evaluation of saved weights against the measured random baseline; inspect
landing and crash rates, not just training returns. Do not claim learning from
the two-episode smoke run.

**What if action space changes?** The network output size and env validation
must change; if actions become continuous, vanilla DQN is no longer suitable.

**How would you improve it?** First run multiple seeds. Then test Double DQN,
observation scaling or replay variants one at a time against the native baseline.

**Limitations?** A single short training run does not prove consistent landings;
the `|x|≤0.2` detector is a conservative observation-based pad proxy; a single
seed does not measure run-to-run variance. Replay contents are not persisted,
so checkpoints preserve weights and optimizer but do not support exact replay
buffer restoration.
