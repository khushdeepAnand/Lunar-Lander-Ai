# Experiments

A log of every experiment run in this project. One entry per experiment, in
order. Each entry records the configuration, the exact command, the environment
it ran in, the **raw** results, and an interpretation.

**Rules for this document.**

- Every number here is copied from real program output. Nothing is estimated,
  extrapolated, or remembered.
- Raw output is preserved in `logs/` and `results/`; this file summarises it and
  links to it.
- Negative and null results are recorded, not hidden. An experiment that failed
  to improve anything is still a result.
- Any claim in the README or `docs/portfolio.md` must trace back to an entry here.

---

## Experiment 000 — Environment verification and Phase 1 setup

**Date:** 2026-10-15
**Phase:** 1
**Status:** complete

### Purpose

Prove that the Box2D physics simulation actually runs in this environment
*before* any learning code is written. If the agent later fails to learn, this
entry is what lets us attribute the failure to the agent rather than to a broken
simulation, a wrong package version, or a silently different observation space.

### Machine

| Item | Value |
|---|---|
| OS | Windows (`win32`), 64-bit |
| Python | 3.12.10 (`%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe`) |
| Also installed | Python 3.13; Python 3.14 (this is the `py` launcher default — see risk 1) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4096 MiB VRAM |
| Display driver | 581.95, reporting CUDA 13.0 |
| C++ toolchain | **absent** — no `vswhere`, no MSVC build tools |
| SWIG | **not on PATH** |
| Git | 2.55.0.windows.2 |
| GitHub CLI | `C:\Program Files\GitHub CLI\gh.exe` present |
| Project root | `$HOME\code\lunar-lander-ai` (deliberately outside OneDrive) |
| Virtualenv | `<project>\.venv`, Python 3.12.10 |

### Why the project lives outside OneDrive

The user's Desktop resolves to `$HOME\OneDrive\Desktop`. Placing the
project there would put a ~2.5 GB PyTorch install and a `.git` directory inside
OneDrive sync. Two concrete failure modes were avoided:

1. OneDrive "Files On-Demand" can *dehydrate* installed `.pyd`/`.dll` files,
   after which `import torch` fails with unrelated-looking DLL errors.
2. OneDrive's file locking during `git` operations is a known cause of repository
   corruption.

A Desktop shortcut (`Lunar Lander AI.lnk`) points at the real folder, so access
is unchanged.

### Dependency-resolution risks found *before* installing

These were found by querying the package indexes, not by encountering a failure.

#### Risk 1 — Python 3.14 cannot build Box2D on this machine

The `py` launcher defaults to Python **3.14** here. The real metadata from the
Gymnasium 1.4.0 sdist is:

```toml
box2d = [
    "pygame-ce >=2.1.3",
    "box2d ==2.3.10; python_version < '3.14'",     # prebuilt wheels exist
    "box2d-py ==2.3.8; python_version >= '3.14'",  # sdist only
    "swig ==4.*; python_version >= '3.14'",
]
```

Evidence gathered:

```
pip index versions box2d-py        -> 2.3.8, 2.3.7, 2.3.5, 2.3.4, 2.3.1
pip download box2d-py --only-binary=:all:
  -> ERROR: Could not find a version that satisfies the requirement box2d-py
     (from versions: none)
Box2D-2.3.10-cp312-cp312-win_amd64.whl   -> confirmed present on PyPI
```

`box2d-py` ships **no** Windows wheels at all. So on 3.14 the install would
require SWIG *and* MSVC — neither of which is present. On 3.12 it resolves to the
prebuilt `Box2D 2.3.10` wheel and needs no compiler.

**Decision:** target CPython 3.12 and encode it as
`requires-python = ">=3.12,<3.14"`.

#### Risk 2 — PyPI has no Windows CPython 3.12 wheel for recent PyTorch

`pip index versions torch` reports 2.14.1 as the newest release, but querying
the PyPI JSON API for `cp312` `win_amd64` files returns nothing for 2.9.1
through 2.14.1. Windows CUDA builds are published only on the PyTorch index:

```
https://download.pytorch.org/whl/cu130/torch/
  -> torch-2.14.1+cu130-cp312-cp312-win_amd64.whl   (confirmed present)
```

`cu130` matches the driver's reported CUDA 13.0.

**Decision:** install PyTorch only via
`--index-url https://download.pytorch.org/whl/cu130`.

### Environment contract (verified from Gymnasium 1.4.0 source)

Read directly from the installed source rather than from documentation.

**Registration** — `gymnasium/envs/__init__.py:94-99`:

```python
register(
    id="LunarLander-v3",
    entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
    max_episode_steps=1000,
    reward_threshold=200,
)
```

**Observation** — `gymnasium/envs/box2d/lunar_lander.py:624-634`, with
`FPS = 50`, `SCALE = 30`:

| Index | Meaning | Actual scale |
|---|---|---|
| 0 | Horizontal offset from pad centre | normalised by half viewport width, ≈ `[-1, 1]` |
| 1 | Height above the pad | normalised by half viewport height; `0` at pad height |
| 2 | Horizontal velocity | `vx * half_width / FPS` |
| 3 | Vertical velocity | `vy * half_height / FPS` |
| 4 | Lander angle | **raw radians, unnormalised** |
| 5 | Angular velocity | `20 * ω / FPS` |
| 6 | Left leg ground contact | `0.0` / `1.0` |
| 7 | Right leg ground contact | `0.0` / `1.0` |

**Correction to a common claim.** It is widely repeated that LunarLander's
observation is unnormalised and that this explains DQN instability here. Indices
0–3 are explicitly normalised by viewport size and frame rate. Only indices 4
and 5 are on a different scale. Any project that "fixes" this with a normaliser
is solving a problem it does not have.

**Action space** — `Discrete(4)`: `0` nothing, `1` left thruster,
`2` main thruster, `3` right thruster.

**Reward** — `lunar_lander.py:636-660`:

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
    reward = shaping - self.prev_shaping       # potential-based shaping
self.prev_shaping = shaping

reward -= m_power * 0.30                       # main-engine fuel
reward -= s_power * 0.03                       # side-engine fuel

terminated = False
if self.game_over or abs(state[0]) >= 1.0:     # body contact, or left the viewport
    terminated = True
    reward = -100                              # OVERWRITES the step reward
if not self.lander.awake:                      # body came to rest
    terminated = True
    reward = +100                              # OVERWRITES, and this check wins
```

Three consequences that constrain the rest of the project:

1. **The dense reward is already potential-based shaping.**
   `reward = shaping(s') - shaping(s)` is a difference of potentials. At the
   DQN discount gamma=0.99, the policy-invariant form would instead use
   `gamma * shaping(s') - shaping(s)`; do **not** claim an invariance proof for
   the native term. The reward is already dense and shaped, so any *additional*
   shaping in `rewards.py` must be treated as a serious experiment rather than a
   free improvement.
2. **The terminal reward replaces rather than accumulates**, so on the final
   step it is exactly `+100.0` or `-100.0`. This makes it usable as a clean
   landing label.
3. **The `+100` check runs last and wins.** A lander that crashes onto its
   fuselage and comes to rest in the same step is rewarded `+100`. So
   "reward > 0" and "landed on both legs" are *not* definitionally the same
   thing, and the landing detector must report a disagreement rate rather than
   assume they agree.

**Episode boundaries** — the environment itself returns `truncated=False`
unconditionally (`lunar_lander.py:665`). `truncated=True` is produced by
Gymnasium's `TimeLimit` wrapper (`gymnasium/wrappers/common.py:130-136`):

```python
observation, reward, terminated, truncated, info = self.env.step(action)
self._elapsed_steps += 1
if self._elapsed_steps >= self._max_episode_steps:
    truncated = True
```

Note that the wrapper does **not** clear `terminated`, so both flags can be true
on step 1000. The landing classifier must therefore check `terminated` first and
treat truncation as the fallback.

### Setup commands run

```powershell
New-Item -ItemType Directory -Force "$HOME\code\lunar-lander-ai"
py -3.12 -m venv "$HOME\code\lunar-lander-ai\.venv"
& "$HOME\code\lunar-lander-ai\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$HOME\code\lunar-lander-ai\.venv\Scripts\python.exe" -m pip install torch `
    --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
& "$HOME\code\lunar-lander-ai\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

### Installed versions (verified, from `pip freeze`)

| Package | Version |
|---|---|
| torch | **2.14.1+cu130** |
| gymnasium | 1.4.0 |
| Box2D | 2.3.10 |
| pygame-ce | 2.5.8 |
| numpy | 2.5.4 |
| pandas | 3.0.6 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| tqdm | 4.70.1 |
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |
| coverage | 7.16.2 |
| ruff | 0.17.0 |

`torch.cuda.is_available()` returns **True**, device
`NVIDIA GeForce RTX 3050 Laptop GPU`. The CUDA build is working; the project
still defaults to `device: auto` with a CPU fallback, so the GPU remains an
optimisation and not a requirement.

`python -m pip install -r requirements.txt --dry-run` resolves the full pinned
set and exits 0, so the pin file is known-good rather than merely written down.
`ruff check .` passes with zero findings.

### Space contract, as measured at runtime

```
observation_space.shape  (8,)
observation_space.dtype  float32
observation_space.low    [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
observation_space.high   [ 2.5,  2.5,  10,  10,  6.2831855,  10, 1, 1]
action_space             Discrete(4)
max_episode_steps        1000
reset() info keys        []            # the box2d env returns an EMPTY info dict
```

Two details that matter downstream:

- **The declared bounds are loose.** The angle is declared as `±2π` and the
  angular velocity as `±10`, but a flying lander sits near `0`. Anything that
  normalises with `observation_space.low/high` would be dividing by ranges that
  are never approached, crushing the signal that actually matters.
- **`info` is empty.** `info["native_reward"]` does not exist until
  `RewardShapingWrapper` adds it, so `train.py` must use
  `info.get("native_reward", reward)`.

Sample observation immediately after `reset(seed=0)`:

```
[ 0.00570612  1.3990337   0.5779653  -0.5282997  -0.0066053  -0.13091765  0.  0. ]
```

Index 1 reads `1.399` at spawn, i.e. the lander starts about 1.4 (normalised)
above the pad. This **corrects an earlier estimate in this document** of
`[-0.3, 1]`; index 1 is not confined to `[-1, 1]`.

### Run 000a — 3 random-action episodes (the Phase 1 gate)

```powershell
python scripts/verify_setup.py --episodes 3 --seed 0
```

| # | return | steps | terminated | truncated | terminal reward | outcome |
|---|---|---|---|---|---|---|
| 0 | -230.79 | 114 | True | False | -100.0 | crashed |
| 1 | -188.58 | 97 | True | False | -100.0 | crashed |
| 2 | -136.35 | 86 | True | False | -100.0 | crashed |

| Metric | Value |
|---|---|
| Mean return | -185.24 |
| Median return | -188.58 |
| Std return (population) | 38.63 |
| Min / max return | -230.79 / -136.35 |
| Mean episode length | 99.0 steps |
| Landed | 0/3 (0.0%) |
| Crashed | 3/3 (100.0%) |
| Truncated by time limit | 0/3 (0.0%) |
| Leg-proxy disagreement | 0/3 |

Exit code 0, `VERIFY SETUP OK`.

### Run 000b — 30 random-action episodes (the baseline to beat)

```powershell
python scripts/verify_setup.py --episodes 30 --seed 12345
```

| Metric | Value |
|---|---|
| Episodes | 30 |
| **Mean return** | **-182.61** |
| Median return | -125.18 |
| Std return (population) | 125.40 |
| Min / max return | -571.25 / -42.72 |
| Mean episode length | 94.0 steps |
| **Landed** | **0/30 (0.0%)** |
| Crashed | 30/30 (100.0%) |
| Truncated by time limit | 0/30 (0.0%) |
| **Leg-proxy disagreement** | **7/30 (23.3%)** |

Exit code 0, `VERIFY SETUP OK`.

### Run 000c — offscreen rendering

```powershell
python scripts/verify_setup.py --episodes 2 --seed 7 --render rgb_array
```

`VERIFY SETUP OK`, exit code 0. `render_mode="rgb_array"` draws offscreen without
opening a window, which is what Phase 8 needs in order to record a demo video
without a human present.

Episode 0 of this run (`-194.71`, 87 steps, seed 7) reproduced **exactly** in a
separate invocation later, confirming that seeded episodes are deterministic.

### Discovery: "both legs touching" is a broken landing detector

The first version of `verify_setup.py` classified an episode as landed whenever
the final observation had both leg contacts set. Under that rule:

- the 30-episode random-policy run reported **4 landings**; and
- a different run reported a "landing" with a return of **-194.71**, which is
  impossible, because a safe landing pays `+100` and can never end deeply
  negative.

The cause is exactly the edge case identified in the source review above. In
`lunar_lander.py:654-660` the `+100` check runs *after* the crash check, and
`game_over` is set by the lander's **body** making contact, while the **legs** are
separate bodies that only set `ground_contact`. A lander can therefore crash
onto its fuselage and still be resting on both legs.

**Measured frequency: 7 of 30 episodes (23.3%)** in run 000b had `both legs
touching` while the terminal reward was `-100`. The naive rule would have
claimed a **~23% landing rate for a policy that never landed once.**

Raw evidence, run 000b, episodes that crashed with both legs down:
`2, 6, 11, 14, 15, 22, 24`.

**This changed the code and the spec:**

- `scripts/verify_setup.py` now classifies by the terminal reward — the
  environment's own definition of success — and prints the disagreement rate.
- `docs/BUILD_SPEC.md` §4.8 specifies `classify_outcome` with `terminal_reward`
  authoritative, leg contacts as a cross-check, and
  `detector_disagreement_rate` as a **required** reported metric.
- Every future evaluation must report the disagreement rate. In this
  environment it is not a rounding error; it is roughly a quarter of episodes.

### Interpretation

#### What this establishes

1. **The simulation works.** `LunarLander-v3` initialises, resets, steps,
   terminates, truncates and closes cleanly. Observation is `Box(8,)` `float32`;
   actions are `Discrete(4)`; the time limit is 1000 steps. `env.close()` is
   safely repeatable.
2. **The GPU path works.** `torch 2.14.1+cu130` sees the RTX 3050.
3. **Rendering works offscreen**, so a demo can be recorded without a display.
4. **The random-policy baseline is established** (below).

#### The baseline that every later result is compared against

> **Random policy — 30 episodes, seed 12345:**
> **mean return -182.61, median -125.18, std 125.40,**
> **range [-571.25, -42.72], mean length 94.0 steps,**
> **landing rate 0.0%, crash rate 100.0%, truncation rate 0.0%.**

How to read that number:

- A uniformly random policy **never lands** — not once in 30 episodes. It is the
  floor. Any trained agent reporting a landing rate above zero, and a mean
  return meaningfully above -183, has learned something real.
- The environment's own definition of "solved" is a mean return of **200** over
  100 consecutive episodes. That is a swing of roughly **383 points** from this
  baseline — the size of the gap the agent must close.
- Random episodes end quickly (~94 of 1000 steps) and always by crashing, which
  is expected: random thrusting drives the lander into the ground long before
  the time limit. The truncation branch is therefore rare at the start of
  training and should become *more* common as the agent learns to descend
  cautiously. A truncation rate rising while the crash rate falls is an early
  warning sign of the "hover and never land" local optimum.

#### Caveats on these numbers, stated deliberately

- **n = 30 is a small sample.** The standard deviation across episodes (125.40)
  is larger than the mean's distance from zero, so the mean has a wide
  confidence interval. Use ≥100 episodes before publishing a baseline.
- **The distribution is left-skewed.** The mean (-182.61) sits well below the
  median (-125.18) because a minority of episodes (-571.25, -450.33, -382.87)
  are catastrophic tumbles that fire the main engine heavily and accrue large
  fuel penalties. Reporting only the mean overstates the typical episode, which
  is why median and min/max are reported alongside it.
- **These are random-policy numbers only.** They are not a claim about any
  trained model, and no trained model exists yet.

#### Phase 1 gate: passed

| Gate | Status |
|---|---|
| Environment resets, steps, terminates, truncates, closes | ✅ |
| Observation/action spaces verified against the build spec | ✅ |
| GPU detected; CPU fallback preserved | ✅ |
| Offscreen rendering verified | ✅ |
| Random-policy baseline measured and recorded | ✅ |
| `ruff check .` clean | ✅ |
| `pip install -r requirements.txt --dry-run` resolves | ✅ |

Ready for Phase 2 (the Q-network).

---

*Subsequent experiments are appended below as they are run.*

---

## Experiment 001 — Native-reward DQN, 300 episodes

**Date:** 2026-10-15. **Environment:** Gymnasium 1.4.0, `LunarLander-v3`.
**Training seed:** 42. **Device:** CUDA, RTX 3050 Laptop GPU.
**Configuration:** `logs/config_20261015T084428850313Z.json`; native reward only,
no shaping. **Training log:** `logs/train_20261015T084428850313Z.csv`.
**Periodic evaluation:** `logs/eval_20261015T084428850313Z.csv`, 20 greedy
episodes at each checkpoint, evaluation seed range starting at 20260.

```powershell
python scripts/train.py --config configs/default.yaml --episodes 300 --set training.eval_interval=50 --set training.checkpoint_interval=50 --set training.eval_episodes=20
python scripts/evaluate.py --checkpoint checkpoints/20261015T084428850313Z/best.pt --episodes 100 --seed 20260
```

| Training episodes | Mean native training return | Safe landings |
|---|---:|---:|
| 1–100 | −153.33 | 0/100 |
| 101–200 | −114.28 | 0/100 |
| 201–300 | −105.54 | 0/100 |

Best checkpoint by the **actual** periodic-evaluation mean native return was
episode 300 (−109.82 over 20 episodes). A fresh evaluation of the saved
checkpoint on **100** independent, seeded greedy episodes measured:

| Metric | Measured value |
|---|---:|
| Mean native return | **−106.79** |
| Median native return | −101.13 |
| Population std | 28.00 |
| Best / worst native return | −50.90 / −172.01 |
| Mean episode length | **1000** (time limit) |
| Safe landings | **0/100 (0%)** |
| Crashes | **0/100 (0%)** |
| Time-limit truncations | **100/100 (100%)** |
| End-of-training epsilon | **0.820** |

**Held-out confirmation:** the `seed 20260` run above reused the first 20 seeds
that were used for checkpoint selection. A fresh evaluation using **100 episodes
from seed 40400** (no overlap with checkpoint-selection seeds) measured mean
native return **−108.77**, median **−106.39**, population std **27.96**, all
**100/100 truncated**, with **0/100 landings**. This is the honest independent
number to use in a portfolio. Output:
`results/eval_20261015T084428850313Z_20261015T085544077217Z.json`.

**Interpretation:** Training returns improved versus the sampled random-policy
baseline (−182.61), but the saved greedy agent **did not learn to land**. It
reached the 1000-step time limit on every evaluation episode. Reward improved
because it learned to survive, not because it solved the landing task. The
epsilon schedule was still near the exploratory start (`0.820`) after only
28,381 environment steps, far short of its 150,000-step decay target. This
checkpoint must not be used in a portfolio as a successful landing agent.

**Next experiment:** longer native-reward training, preserving the algorithm
and evaluation method; only after checking its result consider reward shaping.

---

## Experiment 002 — Native reward, 900 episodes (successful checkpoint)

**Date:** 2026-10-15. **Training seed:** 42. **Environment:** `LunarLander-v3`,
Gymnasium 1.4.0, PyTorch 2.14.1+cu130, RTX 3050 Laptop GPU. **Shaping:** off.
**Run ID:** `20261015T163534138250Z`. **Config:**
`logs/config_20261015T163534138250Z.json`. **Metadata:**
`logs/run_20261015T163534138250Z.json`.

```powershell
python scripts/train.py --config configs/default.yaml --episodes 900 --set agent.epsilon.decay_steps=60000 --set training.eval_interval=150 --set training.checkpoint_interval=150 --set training.eval_episodes=10
python scripts/evaluate.py --checkpoint checkpoints/20261015T163534138250Z/best.pt --episodes 100 --seed 40400
```

This is a **distinct configuration**, not a continuation of Experiment 001.
Changed exploration decay from 150,000 to 60,000 environment steps after the
300-episode run showed epsilon still at 0.82. Logged evaluation used 10 greedy
episodes per checkpoint on seed range 20260–20269; the published measurement
below uses **100 different seeds, 40400–40499**.

| Checkpoint episode | Mean native return (10 episodes) | Safe landings | Crashes | Timeouts | Rewarded off-pad/unsafe |
|---:|---:|---:|---:|---:|---:|
| 150 | −290.46 | 0% | 60% | 40% | 0% |
| 300 | 188.64 | 70% | 10% | 10% | 10% |
| 450 | 148.12 | 50% | 0% | 30% | 20% |
| 600 | 211.44 | 70% | 10% | 20% | 0% |
| **750 (best)** | **262.46** | **70%** | **0%** | **0%** | **30%** |
| 900 | 175.72 | 40% | 40% | 0% | 20% |

**Checkpoint selection:** highest *measured* mean native return on the
10-episode checkpoint evaluation. That picked episode 750. The final episode
900 checkpoint is **not** substituted for it; the policy visibly regressed.

**Held-out result**, evaluating saved episode-750 `best.pt` over 100 seeded
greedy episodes (population standard deviation):

| Metric | Measured value |
|---|---:|
| Mean native return | **252.91** |
| Median native return | 262.34 |
| Std native return | 50.82 |
| Best / worst native return | 310.52 / −59.98 |
| Mean episode length | 323.84 steps |
| **Safe pad landings** | **80/100 (80%)** |
| Crashes | 0/100 |
| Time-limit truncations | 4/100 |
| Rewarded off-pad/unsafe rest | 16/100 |

The stricter safe-landing detector requires `+100` on the final step **and**
both legs down **and** `|x|≤0.2`. The 16 rewarded but off-pad/unsafe cases are
not counted as landings. `landing_40400.gif` is real playback from this same
checkpoint: evaluation seed 40400, native return **282.32**, 376 steps, safely
landed. Frames come straight from the environment renderer (captured every
second simulated step, then Lanczos-upscaled to 900x600 for a smoother, larger
picture; the model still acted at every step). Checkpoint copied unchanged to
`examples/best_native.pt`. The replay was regenerated with the higher-quality
encoder and copied unchanged to `docs/assets/landing_40400.gif` (1,092,481
bytes, SHA256 `250f0abc64fbda6bb90f34f7d6d075bc1f66105c4c2e65bd39c9d2c92e7496f1`,
byte-identical to `results/landing_40400.gif`, verified 2026-10-15).

**Caveat:** this is one training seed and one 100-episode held-out evaluation.
It proves the saved policy lands often under the tested distribution; it does
**not** establish run-to-run variance, robustness to different physics, or
bit-exact reproducibility on other hardware.

**Aborted attempt:** another 1,500-episode process started earlier, then ended
without a completed training CSV. Partial checkpoints/config may remain in
ignored local directories. No result is attributed to that attempt.

---

## Experiment 003 — Optional reward shaping, under-trained exploratory comparison

**Date:** 2026-10-15. **Run ID:** `20261015T172209870911Z`.
**Seed:** 42. **Episodes:** 300. **Epsilon schedule:** 1.0 to 0.05 over
150,000 steps. **Evaluation seeds:** checkpoint selection 20260–20279;
holdout 40400–40499. **Only changed setting relative to Experiment 001:**
`reward_shaping.enabled=true`, `angle_penalty_weight=2.0`.
**Config:** `logs/config_20261015T172209870911Z.json`.

```powershell
python scripts/train.py --config configs/default.yaml --episodes 300 --set training.eval_interval=50 --set training.checkpoint_interval=50 --set training.eval_episodes=20 --set reward_shaping.enabled=true --set reward_shaping.angle_penalty_weight=2.0
python scripts/evaluate.py --checkpoint checkpoints/20261015T172209870911Z/best.pt --episodes 100 --seed 40400
```

Shaping bonus uses `gamma*Phi(next)-Phi(current)`, with terminal potential zero
and `Phi(s)=-2*[min(|x|,1)+min(|angle|,1)]`. Training logs **native return**,
**shaping bonus** and **training return** separately. Evaluation uses native
return only; the wrapper is **not** applied during evaluation.

| Metric (100 held-out episodes) | Native baseline (Experiment 001) | Shaped agent |
|---|---:|---:|
| Mean **native** return | −108.77 | **−160.33** |
| Median **native** return | −106.39 | −143.54 |
| Population std | 27.96 | 51.87 |
| Safe pad landings | 0/100 | 0/100 |
| Crashes | 0/100 | **96/100** |
| Timeouts | 100/100 | 4/100 |

**Interpretation (exploratory only):** The shaped run had lower measured native
return and more crashes at this 300-episode budget, but **both** runs ended with
epsilon near 0.82 and neither landed on the held-out seeds. These are highly
exploratory, under-trained policies; this comparison does **not** establish that
shaping regresses a trained policy. The native episode-300 policy timed out in
**100/100** held-out episodes: it learned to hover/survive, not land. The best
native 900-episode checkpoint was selected on just ten periodic episodes, a
noisy winner's-curse risk; evaluate both selected and final models in future
full-budget, multi-seed comparisons. Training comparison chart is
`docs/assets/experiment_comparison.png`; it compares only these 300-episode runs.

---

## Stage 3 — Preregistered full-budget paired study (in progress)

**Protocol, fixed before reading new study results:** training seeds 42–46;
native and shaped arm at each seed; 900 episodes **without early stopping**,
epsilon 1.0→0.05 over 60,000 environment steps, otherwise matching config.
Additional shaping uses weight 2.0 in the shaped arm only. Save checkpoints
every 50 episodes, including journey episodes 100, 300, 500, 750, 900; periodic
checkpoint selection uses **50 greedy episodes** starting at seed 20260 every
100 training episodes, instead of the historical 10. Evaluate both the selected
`best.pt` and unselected `final.pt` on identical held-out seeds 40400–40499,
using greedy policy and native reward, 100 episodes per checkpoint. Training
records go to `logs/config_stage3_<arm>_<seed>.json`,
`logs/train_stage3_<arm>_<seed>.csv`, `logs/eval_stage3_<arm>_<seed>.csv` and
`logs/run_stage3_<arm>_<seed>.json`; independent outcomes go to
`results/eval_stage3_<arm>_<seed>_<best|final>_*.{csv,json}`.

The initial `stage3_native_42` process was interrupted mid-training during a
concurrent clean-environment install; its partial CSV/checkpoints remain as
evidence but **cannot** be resumed exactly. The replacement run uses prefix
`stage3b`: `python scripts/stage3_study.py --prefix stage3b --seeds 42 43 44 45 46
--device cuda`. This command is **already running**; do not launch it again
concurrently. A new study must use a different `--prefix`. The script runs
sequentially and refuses to overwrite incomplete runs. Once complete it writes
`results/stage3b_summary.csv` with per-seed landing numerator, denominator,
Wilson 95% interval, and mean native return, and prints per-arm/checkpoint
mean and sample standard deviation of five training-seed landing rates. A
descriptive pooled Wilson interval ignores between-seed dependence and is
**not** a confidence interval on the mean across training seeds. Compare best
and final explicitly to expose optimistic selection on periodic evaluations.

| Arm | Checkpoint | Training seeds | Held-out n/checkpoint | Mean ± SD safe-landing rate | Wilson 95% |
|---|---|---:|---:|---|---|
| Native | selected best / final | Pending | Pending | Not measured yet | Not measured yet |
| Shaped | selected best / final | Pending | Pending | Not measured yet | Not measured yet |

**Do not replace the pending cells with estimates or the 300-episode samples.**
The original 900-episode native run (one seed, different periodic selection
sample size) remains historical evidence, not one member of this paired study.

**Setup gate (2026-10-18):** Original `.venv` on Windows CPython 3.12.10:
`pytest -q` **25 passed**, `ruff check .` **All checks passed**,
`verify_setup.py --episodes 3` **VERIFY SETUP OK** (random-action returns
−230.79, −188.58, −136.35; 3/3 crashed; mean −185.24). A separate clean
Python 3.12 CPU `.venv` installed `requirements-cpu.txt` and the editable
dashboard/dev extras, `pip check` said **No broken requirements found**,
`verify_setup.py --episodes 1` said **VERIFY SETUP OK**, and the expanded
suite after Stage 3 code changes passed **27 tests** with Ruff clean. The
clean install does not prove a macOS installation; wheel listings were checked
independently and are documented in the README.

### Actual failure-gallery capture (separate from the paired study)

On 2026-10-18, `scripts/evaluate.py --checkpoint examples/best_native.pt
--episodes 100 --seed 40400 --device cpu --failure-gallery 3` recorded
`results/eval_examples_20261018T171503571915Z.{csv,json}`. The CPU run
measured **80/100** pad landings, mean native return **252.845** and mean
episode length **326.72** steps. These are not the historical CUDA-backed
252.913/323.84 measurements, though the outcome counts agree. The first three
non-landings were seeds **40403 (off-pad/unsafe, 392 steps)**, **40406
(off-pad/unsafe, 258 steps)** and **40414 (off-pad/unsafe, 371 steps)**.
Frame strips and provenance are in `results/gallery/eval_examples_20261018T171503571915Z/`;
the PNG tiles are sampled from the real Gymnasium renderer, not scripted
animation or predicted trajectories. This CPU replay is a failure-mode example,
**not** a second training seed or a new measurement of shaping.
