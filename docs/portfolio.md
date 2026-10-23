# Lunar AI | Python · PyTorch · Gymnasium · Streamlit · Plotly

## Overview and problem

A local, reproducible Deep Q-Network project controlling a physics-based lunar
lander from eight observations and four discrete thruster actions. The challenge
is learning a policy from returns, not labelled examples of good actions.

## Technical approach

The online MLP outputs one Q-value per action. A frozen target network supplies
Bellman bootstrap targets; a preallocated replay buffer samples uncorrelated
transitions. A linear epsilon schedule explores during training, while evaluation
uses greedy actions with no parameter updates. Smooth L1 loss and gradient
clipping reduce the impact of large terminal TD errors. The bootstrap mask uses
`terminated`, not `terminated or truncated`: a time limit ends the rollout, not
the underlying state value.

## Engineering challenges and decisions

- Python 3.12 is pinned because Python 3.14 requires compiling Box2D on Windows.
- The evaluation detector does not confuse two legs touching with a safe landing:
  the 30-episode random baseline had **7** body crashes with both legs in contact.
- A safe landing requires terminal `+100`, both legs contacting the ground, and
  `|x| ≤ 0.2` (the observed normalized pad half-width). Any rewarded rest outside
  this definition is reported separately as `off_pad_or_unsafe`.
- The environment's native reward is already shaped. An optional, separately
  logged potential-difference wrapper is disabled by default.
- Generated training logs and run artifacts stay local. One small checkpoint,
  successful GIF and two charts are committed with documented provenance.

## Actual results

| Experiment | Episodes | Mean native return | Safe landings | Evidence |
|---|---:|---:|---:|---|
| Uniform random actions | 30 | −182.61 | 0/30 | `docs/experiments.md`, Experiment 000 |
| Native DQN checkpoint (300 training episodes; epsilon 0.82) | 100 held-out | −108.77 | 0/100 | `docs/experiments.md`, Experiment 001; 100/100 time-limit truncations |
| Native DQN **best checkpoint (episode 750 of 900)** | 100 held-out | **252.91** | **80/100** | Experiment 002; 0 crashes, 4 timeouts, 16 rewarded off-pad/unsafe |
| Optional shaped agent (300 training episodes; exploratory) | 100 held-out | −160.33 | 0/100 | Experiment 003; **96 crashes**, not evidence of full-budget regression |

These are outcomes from saved checkpoints on distinct evaluation seeds, not
invented benchmark numbers. **One training seed** is insufficient to claim
reliable training across reruns. The 80% rate describes the saved episode-750
checkpoint on the 100 held-out episodes; the episode-900 policy regressed.

## Lessons learned

Checking the *actual* Gymnasium source and runtime spaces exposed incorrect
popular assumptions about observation scaling, reward shaping and touchdown
detection. A return curve alone cannot prove landing; the physical outcome and
time-limit frequency need independent measurement.

## Future improvements

Compare at least five full-budget training seeds per arm, evaluating both final
and best checkpoints on 100 held-out seeds per run, before making any claim
about reward shaping. The matched 300-episode runs are under-trained and exploratory.

## Interview explanation (30 seconds)

“I built a DQN that reads eight state values and predicts the expected return of
each of four thruster actions. It collects transitions in replay memory and
learns from a target network so that the regression target does not move on
every gradient step. During training it explores with a decaying epsilon;
evaluation is greedy and separately seeded. I distinguish genuine termination
from the 1000-step time limit in the Bellman target and count landings only when
the lander rests on both legs on the pad. That guards against a false success
rate you can get by just looking at the terminal reward or leg flags.”

## Resume entry (only verified claims)

**Lunar AI — Reinforcement Learning Simulation and Analytics Platform | Python, PyTorch, Gymnasium, Streamlit, Plotly**

- Implemented a configurable DQN with experience replay, target-network
  synchronization, epsilon-greedy exploration and terminal-aware Bellman targets.
- Built a local mission-control dashboard with real Gymnasium frame playback,
  telemetry, subprocess training/stop controls, checkpoint inspection and
  interactive artifact-backed analytics while retaining the CLI.
- Tested physics integration, learning updates, checkpoint round-trips and a
  safe-landing classifier that distinguishes crashes and time-limit endings.
- Evaluated the best native-reward checkpoint on 100 held-out episodes: **80%
  safe pad landings**, mean native return **252.91** (one training seed;
  `results/eval_20261015T163534138250Z_20261015T171500184929Z.json`,
  source checkpoint `checkpoints/20261015T163534138250Z/best.pt`).
- Logged optional shaping separately from native reward; documented why the
  under-trained 300-episode comparison does not establish a shaping effect.

The dashboard was verified by headless UI test, local browser capture and a
short real worker run. It does not provide exact training resume or a
multi-user training queue. Screenshots: `docs/assets/dashboard-overview.png`
and `docs/assets/dashboard-simulation.png`.
