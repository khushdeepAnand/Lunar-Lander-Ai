# Reinforcement Learning Notes

Every statement below is grounded in code that actually exists in this
repository or in the installed Gymnasium source. Where a fact comes from reading
the environment's implementation, the file and line are cited.

**Environment version: Gymnasium 1.4.0, `LunarLander-v3`.**

---

## 1. The task as a Markov Decision Process

| MDP element | This project |
|---|---|
| **State** `s` | 8-dimensional `float32` vector: position, velocity, angle, angular velocity, two leg contacts (§2) |
| **Action** `a` | `Discrete(4)`: nothing, left thruster, main thruster, right thruster |
| **Reward** `r` | Dense potential-based shaping plus a ±100 terminal bonus (§3) |
| **Transition** | Box2D rigid-body physics, deterministic given the action, plus a small random initial impulse applied on `reset()` |
| **Discount** `γ` | `0.99`, configurable |
| **Horizon** | 1000 steps, enforced by Gymnasium's `TimeLimit` wrapper |
| **Policy** `π(a \| s)` | `argmax_a Q_θ(s, a)`, with ε-greedy exploration during training only |

The eight-value observation does **not** include the terrain profile, so it is
not strictly a complete Markov state for arbitrary trajectories; it is a
compact observation sufficient for many practical policies. Ground-contact
flags provide some local terrain information. The replay buffer stores
transitions from many different generated episodes to learn a policy that
generalizes across starting conditions.

---

## 2. Observation space (verified, not assumed)

Read from `gymnasium/envs/box2d/lunar_lander.py:624-634`, with `FPS = 50` and
`SCALE = 30`:

```python
state = [
    (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),           # 0
    (pos.y - (helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),   # 1
    vel.x * (VIEWPORT_W / SCALE / 2) / FPS,                                # 2
    vel.y * (VIEWPORT_H / SCALE / 2) / FPS,                                # 3
    self.lander.angle,                                                     # 4
    20.0 * self.lander.angularVelocity / FPS,                              # 5
    1.0 if self.legs[0].ground_contact else 0.0,                           # 6
    1.0 if self.legs[1].ground_contact else 0.0,                           # 7
]
assert len(state) == 8
```

| Index | Meaning | Actual scale (measured at runtime) |
|---|---|---|
| 0 | Horizontal offset from pad centre | normalised by half the viewport width; declared `±2.5` |
| 1 | Height above the landing pad | `0` at pad height, ≈ `1.40` at spawn; declared `±2.5` |
| 2 | Horizontal velocity | `vx * half_width / FPS`; declared `±10` |
| 3 | Vertical velocity | `vy * half_height / FPS`; declared `±10` |
| 4 | Lander angle | **raw radians, unnormalised**; declared `±2π` |
| 5 | Angular velocity | `20 * ω / FPS`; declared `±10` |
| 6 | Left leg ground contact | `0.0` or `1.0` |
| 7 | Right leg ground contact | `0.0` or `1.0` |

The declared bounds are what `env.observation_space.low` and `.high` actually
report (recorded in `docs/experiments.md`, Experiment 000). They are much
looser than the reachable range, so they must **not** be used for
normalisation.

### ⚠️ A correction worth knowing

You will find many tutorials and blog posts claiming that "LunarLander's
observation is unnormalised, which is why DQN is unstable on it". **That is
false**, and being able to say why is a good signal in an interview.

Indices 0–3 are explicitly divided by viewport dimensions and frame rate. The
genuinely mixed part is indices 4 and 5: the angle is raw radians (a scale up to
±3.14) while the angular velocity is multiplied by `20/FPS` (a scale around
±0.4). So the input is mixed-scale but bounded, and the baseline needs no
normalisation to work.

If a running mean/std normaliser is added later, it is a documented
**experiment** with its own config, compared against this baseline — not a
silent change.

---

## 3. Reward structure (verified from source)

From `lunar_lander.py:636-660`:

```python
reward = 0
shaping = (
    -100 * sqrt(state[0]**2 + state[1]**2)     # distance from the pad
    - 100 * sqrt(state[2]**2 + state[3]**2)    # speed
    - 100 * abs(state[4])                      # tilt
    + 10 * state[6]                            # left leg contact
    + 10 * state[7]                            # right leg contact
)
if self.prev_shaping is not None:
    reward = shaping - self.prev_shaping
self.prev_shaping = shaping

reward -= m_power * 0.30                       # main-engine fuel
reward -= s_power * 0.03                       # side-engine fuel

terminated = False
if self.game_over or abs(state[0]) >= 1.0:     # body contact, or left the viewport
    terminated = True
    reward = -100
if not self.lander.awake:                      # body has come to rest
    terminated = True
    reward = +100                              # this check runs last and wins
```

### 3.1 This is already potential-based shaping

The environment uses a difference of potentials, `Φ(s') - Φ(s)`. **This is not
the general discounted policy-invariant form**: for discount `γ` that form is
`γΦ(s') - Φ(s)`, with terminal potential handled consistently. Since the DQN
uses `γ=0.99`, the environment's native term is dense guidance, but claiming a
policy-invariance theorem for it would be incorrect.

Two practical consequences:

1. The native reward is already dense and already informative. That is what
   makes this environment learnable in a few hundred episodes rather than
   millions.
2. Any reward shaping added in `rewards.py` is a **second** shaping term layered
   on the first. If it is not also potential-based, it can change what the
   optimal policy is — which is exactly how shaping experiments go wrong.

### 3.2 The terminal reward is clean

On the final step, `reward` is **overwritten**, not accumulated: it becomes
exactly `+100` or exactly `-100`, and the per-step shaping delta and fuel cost
are discarded. This makes the terminal reward usable as a label — see §4.8 of
`BUILD_SPEC.md`.

### 3.3 The `+100`-wins edge case

The `not self.lander.awake` check runs *after* the crash check, so a lander that
crashes onto its fuselage and comes to rest in the same step receives `+100`.
This is a real quirk of the environment and the reason the landing detector is
defined with an explicit disagreement metric rather than `reward > 0`.

Also note that `game_over` is set by the **body** touching anything
(`ContactDetector.BeginContact`, `lunar_lander.py:63-68`); the legs are separate
bodies and only set `ground_contact`. Landing on the legs is fine; hitting the
ground with the fuselage is a crash.

---

## 4. Why a Q-network, and not a dynamics model

The naive framing is "train a neural network to predict where the rocket will
go". That is a **dynamics model**, and it solves the wrong problem: even a
perfect next-state predictor leaves you with the control question of *which
action to take*.

DQN instead learns `Q(s, a)`: the expected discounted sum of future rewards
obtained by taking action `a` in state `s`, and then acting optimally
thereafter.

```
Q(s, a) = E[ r_t + γ r_{t+1} + γ² r_{t+2} + ...  |  s_t = s, a_t = a, π = π* ]
```

Given `Q`, the control problem collapses to a single `argmax`:

```python
action = argmax_a Q(s, a)
```

No simulator rollout, no planning, no search. One forward pass of a small MLP
per decision. That is the entire reason the agent can act in real time.

The cost is that `Q` is only as good as the reward signal and the exploration
that generated the experience — which is what the rest of this document is
about.

---

## 5. The Bellman equation and the DQN target

`Q` satisfies the Bellman optimality equation, which is recursive:

```
Q*(s, a) = E[ r + γ · max_a' Q*(s', a') ]
```

This gives us a way to *learn* `Q` without knowing the transition probabilities:
treat the right-hand side as a regression target and move `Q(s,a)` toward it.

```
target = r + γ · (1 - terminated) · max_a' Q_target(s', a')
loss   = SmoothL1( Q_online(s, a) , target )
```

Three deliberate details:

- **`Q_target`, not `Q_online`, on the right-hand side.** Using the same network
  for both sides means the target moves every time the prediction moves. The
  network chases its own tail and the loss can diverge. A periodically-frozen
  target network is the standard fix (§7).
- **`(1 - terminated)`.** Known terminal states have no future, so
  `Q(s', ·) := 0` there. But this mask must key on **`terminated` only**, not on
  truncation — see §8.
- **SmoothL1 (Huber) loss.** Quadratic for small errors, linear for large ones.
  With `±100` terminal rewards, TD errors can be enormous; MSE would let a
  single crash dominate the gradient and destabilise the whole update. Huber
  bounds the influence of outliers.

---

## 6. The DQN training loop, step by step

This mirrors the actual sequence in `BUILD_SPEC.md` §4.10.

```
for each episode:
    s = env.reset()                          # seeded only on the FIRST reset
    for each step:
        ε = linear decay from 1.0 to 0.05 over 150,000 steps
        a = argmax Q_online(s) with probability (1 - ε)     # exploit
          = random action      with probability ε           # explore
        s', r, terminated, truncated, info = env.step(a)
        buffer.add(s, a, r, s', terminated, truncated)      # store BOTH flags
        if buffer is warm enough:
            batch = buffer.sample(64)                        # decorrelated samples
            q_sa   = Q_online(batch.s).gather(1, batch.a)    # predicted value of taken action
            target = batch.r + γ(1 - batch.terminated) max Q_target(batch.s')   # no grad
            loss   = SmoothL1(q_sa, target)
            loss.backward()
            clip_grad_norm_(params, 10.0)                    # bound the update
            optimizer.step()                                 # Adam, lr 5e-4
            if updates % 1000 == 0: Q_target ← copy of Q_online
        s = s'
        if terminated or truncated: break
```

---

## 7. Why experience replay and a target network

**Experience replay** breaks the temporal correlation that would otherwise
destroy training. Consecutive steps in a single episode are almost identical —
same region of state space, highly correlated reward. Mini-batches of such
consecutive samples have very low variance, and the gradients they produce are
biased toward whatever the agent happened to be doing in the last few seconds.

Sampling uniformly from a large buffer makes each batch approximately
`i.i.d.`, which is the assumption the gradient estimate relies on. It also lets
a single transition be reused many times, which matters a great deal because
environment interaction is the expensive part.

The buffer here is a preallocated ring buffer of NumPy arrays. `capacity * (2*8
+ 1 + 1 + 2)` floats with a 100,000 capacity is roughly 15 MB. A Python list of
tuples would be hundreds of megabytes and orders of magnitude slower to sample.

**The target network** is a frozen copy of the online network, used only to
compute the bootstrap value `max_a' Q_target(s', a')`. It is synchronised by hard
copy every `target_update_freq` gradient steps. Without it, the regression
target depends on the very parameters being updated, and the classic DQN
divergence appears. The target network's parameters are explicitly excluded from
the optimiser and its outputs are computed under `torch.no_grad()`.

The textbook name for this — function approximation in a bootstrapping,
off-policy update — is **the deadly triad**. Each element is fine alone; together
they remove the convergence guarantees that tabular Q-learning has. Target
networks and replay are the two standard stabilisers.

---

## 8. Truncation is not termination

This is the single most commonly botched detail in DQN implementations, and the
easiest one to be asked about in an interview.

Gymnasium's API returns two separate booleans:

| Flag | Meaning | Bootstrap? |
|---|---|---|
| `terminated=True` | A genuine terminal state. The lander crashed or landed safely; there are no future rewards. | **No** — `Q(s', ·) := 0` |
| `truncated=True` | The episode was cut at 1000 steps by the `TimeLimit` wrapper. The world did not end. | **Yes** — keep bootstrapping |

The correct target mask is therefore:

```python
target = reward + gamma * (1.0 - terminated) * max_next_q
```

Treating truncation as terminal would teach the agent that the universe ends at
step 1000. Any state from which a landing was still reachable would be
systematically undervalued — destroying exactly the long-horizon credit
assignment we are trying to learn. The agent would learn to give up near the
time limit.

Two distinctions that must not be blurred in the code:

- **Both flags end the episode loop.** A truncated episode stops stepping.
- **Only `terminated` ends bootstrapping.** A truncated transition still
  contributes `r + γ max Q(s')`.

Note also from the source (`lunar_lander.py:665`) that the environment itself
returns `truncated=False` unconditionally. `truncated=True` is produced by the
`TimeLimit` wrapper that `gym.make` installs. If you construct the environment
without `max_episode_steps`, truncation never fires at all and episodes can run
forever.

`truncated` is still stored on every transition, because "how often does the
agent run out of time instead of landing or crashing?" is part of an honest
evaluation.

---

## 9. Reward function vs. reward shaping

These are different things and conflating them makes results uninterpretable.

| | Reward **function** | Reward **shaping** |
|---|---|---|
| Definition | What success means; defines the task | A modification to the learning signal intended to make the task easier to learn |
| Changes the optimal policy? | It *defines* it | Should not (if potential-based) |
| Reported after training? | Always | Never — evaluation reports native return |
| In this project | The environment's own reward | Optional `RewardShapingWrapper`, off by default |

Why the discipline matters: if you train on `native + bonus` and then report the
return you got, you cannot tell whether the agent learned the task or learned the
wrapper. So:

- `info["native_reward"]` and `info["shaping_bonus"]` are logged as separate
  columns on every step.
- `EpisodeResult` carries `native_return` and `shaped_bonus` as separate fields.
- Every evaluation reports **native** return, landing rate, crash rate and
  truncation rate.

### Shaping implemented here

The optional wrapper uses one weight `w` and a bounded potential computed from
observed horizontal pad offset and angle:

```text
Phi(s) = -w * [min(|x|, 1) + min(|angle|, 1)]
bonus  = gamma * Phi(next_s) - Phi(s)
Phi(terminated state) = 0
```

It does not add a survival bonus, and the actual native reward remains in
`info["native_reward"]`. It is **disabled** for the default baseline. The
wrapper's existence alone is not evidence that it improves learning; that
requires a controlled experiment using equal budgets and held-out seeds.

### Shaping signals that are forbidden, and why

| Forbidden | Failure mode it creates |
|---|---|
| Reward per step survived | Incentivises hovering forever instead of landing |
| Reward for approaching the pad *without* a terminal bonus | Encourages loitering near the pad; the native reward already covers approach |
| Reward for fuel remaining | Encourages doing nothing |
| Any bonus keyed on absolute step count | Hovering exploit |
| Any bonus that fires on `truncated` | Rewards running out the clock |

The general rule: a shaping term must not be maximisable by a policy that never
lands.

---

## 10. Why DQN training becomes unstable

Specific mechanisms, and the mitigation in this codebase:

1. **The deadly triad** — bootstrapping + function approximation + off-policy
   learning. *Mitigation:* target network, replay buffer (§7).
2. **Maximisation bias.** `max_a' Q_target(s', a')` uses the maximum of noisy
   estimates, so it is biased upward, and the bias accumulates. *Mitigation:*
   Double DQN, a documented Phase-16 extension — select the action with the
   online network, evaluate it with the target network.
3. **Large TD errors from ±100 terminal rewards.** A crash can produce a TD error
   of order 100. *Mitigation:* SmoothL1 loss bounds the gradient per sample, and
   gradient clipping bounds the total step.
4. **Correlated samples.** *Mitigation:* uniform sampling from a large buffer.
5. **Premature exploitation.** If ε decays to ~0 before useful behaviour is
   found, the agent locks into a poor policy — classically "hover and do
   nothing". *Mitigation:* linear decay over 150,000 steps, and a floor of
   `0.05` rather than `0`.
6. **Insufficient warm-up.** Learning from a nearly-empty buffer produces
   high-variance, biased targets. *Mitigation:* `learning_starts = 1000`.
7. **Scale mismatch in the input.** *Mitigation:* the baseline tolerates it;
   normalisation is a documented experiment (§2).

---

## 11. How landing success is measured

Training return alone is **not** evidence that the agent can land. A policy that
hovers competently can accumulate a respectable return without ever touching
down. So success is measured as an explicit outcome classification per episode
(`metrics.classify_outcome`, `BUILD_SPEC.md` §4.8):

| Outcome | Condition |
|---|---|
| `landed` | `terminated`, terminal reward `+100`, both legs in contact, and horizontal pad offset at most `0.2` |
| `crashed` | `terminated` and the terminal reward is `-100` |
| `off_pad_or_unsafe` | `terminated` with `+100`, but fails the pad/leg-contact test |
| `truncated` | the episode ended without `terminated` (the 1000-step limit) |

The final observation's leg contacts are recorded as an **independent
cross-check**, and the fraction of terminated episodes where the reward and the
leg-contact proxy disagree is reported as `detector_disagreement_rate`. That
metric exists because of the `+100`-wins edge case in §3.3: "rewarded +100" and
"landed on both legs" are not definitionally the same thing, and any success
rate must state which of the two it reports.

Evaluation is run separately from training, over multiple independently seeded
episodes, with exploration disabled (`argmax`), no weight updates, and its own
seed range. Reported metrics: mean, median, std, min and max return, landing
success rate, crash rate, truncation rate, mean episode length, and the number
of episodes.

`std_return` is the standard deviation across evaluation episodes. It is **not**
a confidence interval, and it says nothing about training-run-to-run variance —
that requires multiple seeds, which is a Phase-16 extension.

---

## 12. Named concepts, for quick recall

| Term | One-line meaning in this project |
|---|---|
| **Agent** | The DQN policy `argmax_a Q_θ(s,a)` |
| **Environment** | `LunarLander-v3`: Box2D physics + reward + episode boundaries |
| **Episode** | One landing attempt, up to 1000 steps |
| **Return** | Reported episode return is the **undiscounted** sum of native rewards; the learning target discounts future values by gamma |
| **Policy** `π` | State → action mapping |
| **Q-value** | Expected return of taking `a` in `s`, then acting optimally |
| **Bellman target** | `r + γ(1-terminal)·max Q_target(s')` |
| **Bootstrapping** | Using the current value estimate to improve itself |
| **On-policy / off-policy** | DQN is off-policy: replay stores behaviour-policy data while the target is greedy |
| **Deadly triad** | Bootstrapping + function approximation + off-policy |
| **Replay buffer** | Fixed-capacity store of transitions, uniformly sampled |
| **Target network** | Frozen copy used to compute stable targets |
| **ε-greedy** | Explore at random with probability ε, exploit otherwise |
| **Reward shaping** | Extra learning signal added without redefining success |
| **Potential-based shaping** | Shaping of the form `Φ(s') - Φ(s)`; preserves optimal policies |
| **Truncation** | Episode cut by the time limit; the world did not end |

---

## References

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement
  learning.* Nature 518, 529–533. — the original DQN.
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*, 2nd
  ed. — the deadly triad, MDPs, TD learning.
- Ng, A., Harada, D. & Russell, S. (1999). *Policy invariance under reward
  transformations.* — potential-based shaping, which LunarLander's dense reward
  happens to satisfy.
- Van Hasselt, H., Guez, A. & Silver, D. (2016). *Deep reinforcement learning
  with double Q-learning.* — the Phase-16 Double DQN extension.
- Gymnasium documentation, `LunarLander`:
  https://gymnasium.farama.org/environments/box2d/lunar_lander/
