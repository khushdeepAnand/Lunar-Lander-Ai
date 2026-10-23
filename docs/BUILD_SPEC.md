# BUILD_SPEC.md — original design proposal for the Lunar Lander AI project

> **Historical planning document.** This was written before the implementation.
> Some proposed function signatures, file names, metrics and optional options
> differ from the shipped code. **README.md and the tested source files are
> authoritative.** In particular checkpoint restoration does not restore the
> replay buffer, the shipped landing detector uses a conservative pad proxy,
> and only one optional potential-shaping weight is implemented. Do not cite
> planned features here as completed work.

**Purpose.** This is the authoritative build specification. It exists so that the
implementation can be written by any capable coding model (or a human) without
re-deriving design decisions, and so the result is verifiable rather than
merely plausible. Where this document conflicts with a guess, this document
wins — *except* where it says "verify at runtime", in which case the runtime
observation wins and this document must be corrected.

**Status of this document.** Environment facts in section 2 were verified on
this machine by inspection. Facts in section 3 must be confirmed by running
`scripts/verify_setup.py` and the results recorded in `docs/experiments.md`.

---

## 1. Non-negotiables

These are the things that are easy to get subtly wrong. A reviewer who knows
DQN will look here first.

1. **`terminated` and `truncated` are stored separately** and are never
   collapsed into one boolean. See §6.1.
2. **Target values are computed under `torch.no_grad()`.** Gradients must never
   flow into the target computation. See §6.2.
3. **The target network is never passed to the optimizer** and is never in
   training mode when it matters. See §6.3.
4. **The native environment reward is always logged separately** from any
   shaping bonus. Evaluation always reports native return. See §6.4.
5. **No fabricated numbers anywhere.** Not in the README, not in docstrings,
   not in `docs/portfolio.md`. Every figure traces to a file in `logs/` or
   `results/`.
6. **The environment's declared shapes are verified at runtime** and a
   mismatch raises with the actual and expected values in the message. See §3.
7. **Nothing in the project requires a GPU.** `device: auto` must fall back to
   CPU cleanly.

---

## 2. Verified environment facts (this machine)

Confirmed by direct inspection, not assumed.

| Fact | Value | Consequence |
|---|---|---|
| `python` on PATH | `%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe`, **3.12.10** | Correct target interpreter |
| `py` launcher default | **3.14** | Never use bare `py -m venv`; use `py -3.12` |
| Project root | `$HOME\code\lunar-lander-ai` | Off OneDrive by design |
| Virtualenv | `<root>\.venv` | Outside OneDrive; torch will not be synced/dehydrated |
| GPU | RTX 3050 Laptop, **4 GB VRAM**, driver 581.95, CUDA 13.0 | `cu130` wheels are the match |
| MSVC / `vswhere` | **absent** | Any source build will fail; we must stay on wheels |
| SWIG on PATH | absent | same |
| Long paths enabled | `1` | Deep `node_modules`-style errors not a concern |
| `gh` CLI | `C:\Program Files\GitHub CLI\gh.exe` | Available for later repo publication |

### 2.1 The Python 3.14 trap (read this)

`pip install "gymnasium[box2d]"` resolves its extra differently per interpreter.
The real metadata from the Gymnasium 1.4.0 sdist:

```toml
box2d = [
    "pygame-ce >=2.1.3",
    "box2d ==2.3.10; python_version < '3.14'",     # prebuilt wheels exist
    "box2d-py ==2.3.8; python_version >= '3.14'",  # sdist only -> needs swig + MSVC
    "swig ==4.*; python_version >= '3.14'",
]
```

`Box2D-2.3.10-cp312-cp312-win_amd64.whl` was confirmed present on PyPI.
`box2d-py` has **no** Windows wheels at all (verified: `pip download box2d-py
--only-binary=:all:` → "from versions: none").

Therefore: **this project targets CPython 3.12 and forbids 3.14.** `pyproject.toml`
encodes this as `requires-python = ">=3.12,<3.14"`.

### 2.2 The PyTorch wheel trap

`torch 2.14.1` is the newest release on PyPI, but PyPI serves **no** Windows
`cp312` wheel for it. Windows CUDA builds are published only on the PyTorch
index:

```
--index-url https://download.pytorch.org/whl/cu130
torch-2.14.1+cu130-cp312-cp312-win_amd64.whl    # confirmed to exist
```

Install torch **only** via that index URL. A plain `pip install torch` on this
machine will either fail or silently give a CPU build.

---

## 3. Environment facts (verify at runtime, then record)

`scripts/verify_setup.py` must print and assert the following. Values below are
the canonical LunarLander-v3 contract; the script's actual output is the
authority and goes into `docs/experiments.md`.

| Property | Expected | Action if different |
|---|---|---|
| `env.observation_space` | `Box(8,)`, dtype `float32` | Raise; do not "adapt" silently |
| `env.action_space` | `Discrete(4)` | Raise; DQN needs discrete actions |
| Max episode steps | 1000 (via `TimeLimit`) | Read it from the wrapper, don't hardcode |
| `terminated` on crash | `True` | — |
| `terminated` on safe landing | `True` | — |
| `truncated` on time limit | `True` | Must be distinguished from landing |

### 3.1 Observation semantics (indices 0–7)

**Verified** by reading `gymnasium/envs/box2d/lunar_lander.py` lines 624–634 of
the 1.4.0 source, not from documentation or memory. The state vector is built as:

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

with `FPS = 50` and `SCALE = 30`.

| Index | Meaning | Actual scale (measured) |
|---|---|---|
| 0 | Horizontal offset from pad centre | normalised by half the viewport width; declared bounds `±2.5`, observed ≈ `0.006` at spawn |
| 1 | Height above the landing pad | normalised by half the viewport height; `0` at pad height, ≈ `1.40` at spawn height, declared bounds `±2.5` |
| 2 | Horizontal velocity | rescaled: `vx * half_width / FPS`; declared bounds `±10` |
| 3 | Vertical velocity | rescaled: `vy * half_height / FPS`; declared bounds `±10` |
| 4 | Lander angle | **raw radians, not normalised**; declared bounds `±2π` (±6.2831855) |
| 5 | Angular velocity | `20 * ω / FPS` — partially rescaled only; declared bounds `±10` |
| 6 | Left leg ground contact | `0.0` / `1.0` |
| 7 | Right leg ground contact | `0.0` / `1.0` |

The declared bounds above are the **runtime-measured** values of
`env.observation_space.low` / `.high` (see `docs/experiments.md`, Experiment 000).
They are loose relative to what is actually observed — the angle is declared as
`±2π` but a flying lander sits near `0`.

> **Do not normalise using `observation_space.low` / `.high` here.** Those bounds
> are far wider than the reachable range, so scaling by them would crush the
> signal that actually matters.

> **Correction to a widespread misconception.** The common claim that
> "LunarLander's observation is unnormalised, which is why DQN is unstable
> here" is **wrong**. Indices 0–3 are explicitly normalised by viewport size and
> frame rate. What is actually mixed is indices 4 and 5: the angle is raw
> radians (a scale of ±3.14) while the angular velocity is multiplied by
> `20/FPS` (a scale of roughly ±0.4 for fast rotation).

Consequence: the input is mixed-scale but bounded, and the baseline works
without normalisation. Adding a running mean/std normaliser is a **documented
experiment** (Phase 16), never a silent change, and any such experiment must be
A/B'd against this baseline.

The returned array is `np.float32`.

### 3.2 Verified environment constants

| Constant | Value | Source |
|---|---|---|
| Registration id | `LunarLander-v3` | `gymnasium/envs/__init__.py:95` |
| `max_episode_steps` | `1000` | same |
| `reward_threshold` (solve) | `200` | same |
| Main-engine fuel cost | `0.30` per frame while firing | `lunar_lander.py:649-651` |
| Side-engine fuel cost | `0.03` per frame while firing | `lunar_lander.py:652` |
| Main engine power (discrete) | `m_power = 1.0` | `lunar_lander.py:536` |
| Side engine power (discrete) | `s_power` in `[0.5, 1.0]` | `lunar_lander.py:573-579` |

### 3.3 Action semantics

| Action | Effect | Fuel cost |
|---|---|---|
| 0 | Do nothing | 0 |
| 1 | Fire the **left** orientation engine | 0.03 / frame |
| 2 | Fire the **main** downward engine | 0.30 / frame |
| 3 | Fire the **right** orientation engine | 0.03 / frame |

Discrete mode sets `m_power = 1.0` for action 2 and `s_power` in `[0.5, 1.0]`
for actions 1 and 3.

### 3.4 Native reward structure (verified against Gymnasium 1.4.0)

**Verified** by reading `LunarLander.step`, `lunar_lander.py:636-660`. Do not
reproduce this from memory — the details matter.

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
    reward = shaping - self.prev_shaping       # <- potential-based shaping
self.prev_shaping = shaping

reward -= m_power * 0.30                       # main-engine fuel
reward -= s_power * 0.03                       # side-engine fuel

terminated = False
if self.game_over or abs(state[0]) >= 1.0:     # body contact, or left the viewport
    terminated = True
    reward = -100                              # <- OVERWRITES the step reward
if not self.lander.awake:                      # body has come to rest ("slept")
    terminated = True
    reward = +100                              # <- OVERWRITES, and this check wins
```

Four consequences that shape the rest of this design:

1. **The per-step term is potential-based shaping.**
   `reward = shaping(s') - shaping(s)` is a difference of potentials, but the
   policy-invariant discounted form is `gamma * shaping(s') - shaping(s)`.
   Do **not** claim the native term provably preserves the optimal policy for
   gamma=0.99. The native reward is *already* dense and *already*
   shaped. A wrapper in `rewards.py` is a **second** shaping term layered on a
   shaped signal — which is precisely why the baseline must be trained on the
   native reward first (§9, Phase 7).
2. **The terminal reward replaces, it does not accumulate.** On the final step
   `reward` is overwritten with exactly `-100` or `+100`, excluding the shaping
   delta and the fuel cost. A terminal transition therefore has only two
   possible reward values. This is useful: it makes the terminal reward a clean
   label.
3. **`game_over` is triggered by the lander *body* touching anything**
   (`ContactDetector.BeginContact`, `lunar_lander.py:63-68`). The leg fixtures
   are separate bodies and only set `ground_contact`. Touching down on the legs
   is not a crash; slamming the fuselage into the ground is.
4. **The `+100` check runs last, so it wins.** A lander that crashes onto its
   body *and* comes to rest simultaneously ends with `reward = +100`. This is a
   genuine environment edge case and the reason §4.8 specifies a validated proxy
   plus a disagreement metric rather than a naive `reward > 0` test.

Finally, note `return ..., terminated, False, {}` at `lunar_lander.py:665`: the
environment itself **never** sets `truncated`. `truncated=True` comes from the
`TimeLimit` wrapper installed by `gym.make`. The environment also returns an
empty `info` dict, so `info["native_reward"]` exists only once
`RewardShapingWrapper` adds it — `train.py` must therefore use
`info.get("native_reward", reward)`.

---

## 4. Module-by-module specification

### 4.1 `src/lunar_lander_ai/utils.py`

```python
def set_global_seed(seed: int) -> None          # random, numpy, torch (cpu+cuda)
def resolve_device(requested: str) -> torch.device   # "auto" | "cpu" | "cuda"
def project_root() -> Path                      # resolves relative to this file
def ensure_dir(path: Path) -> Path              # mkdir(parents=True, exist_ok=True)
def write_json(path: Path, payload: dict) -> None
def read_json(path: Path) -> dict
def utc_timestamp() -> str                      # "2026-10-15T13:22:26Z"
def new_run_id(config_name: str) -> str         # "default_2026-10-15T13-22-26Z"
```

Rules: no module-level mutable state, no hardcoded absolute paths, all paths
built with `pathlib`. `project_root()` must work regardless of cwd.

### 4.2 `src/lunar_lander_ai/config.py`

Frozen dataclasses (immutable after load, so nothing can mutate the config
mid-run):

```python
@dataclass(frozen=True) class EpsilonConfig:  start: float; end: float; decay_steps: int
@dataclass(frozen=True) class AgentConfig:
    hidden_sizes: tuple[int, ...]; learning_rate: float; gamma: float
    batch_size: int; buffer_capacity: int; learning_starts: int
    train_frequency: int; target_update_freq: int; grad_clip_norm: float
    loss: str; epsilon: EpsilonConfig
@dataclass(frozen=True) class EnvironmentConfig: id: str; continuous: bool
    render_mode: str | None; max_episode_steps: int
@dataclass(frozen=True) class RewardShapingConfig: enabled: bool; ...weights: float
@dataclass(frozen=True) class TrainingConfig: episodes: int; render: bool
    checkpoint_interval: int; eval_interval: int; eval_episodes: int
    log_interval: int; early_stop: bool; solved_window: int; solved_threshold: float
@dataclass(frozen=True) class PathsConfig: log_dir: Path; checkpoint_dir: Path; results_dir: Path
@dataclass(frozen=True) class Config: seed: int; device: str
    environment: EnvironmentConfig; agent: AgentConfig
    reward_shaping: RewardShapingConfig; training: TrainingConfig; paths: PathsConfig

def load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> Config
def to_serialisable(config: Config) -> dict[str, Any]
```

**`validate()` must raise `ValueError` with a specific message** when:

- `gamma` not in `(0.0, 1.0]`
- `learning_rate <= 0`
- `batch_size < 1`
- `buffer_capacity < batch_size`
- `epsilon.start` or `epsilon.end` outside `[0, 1]`
- `epsilon.end > epsilon.start` (decay would run backwards)
- `epsilon.decay_steps < 1`
- `target_update_freq < 1` or `train_frequency < 1`
- `grad_clip_norm <= 0`
- `loss not in {"smooth_l1", "mse"}`
- `hidden_sizes` empty or containing a non-positive value
- `environment.continuous is True` → DQN cannot act on a continuous action space;
  fail with a message naming the reason, not a shape error three layers down

CLI overrides use dotted keys: `--set agent.gamma=0.995`. Explicit flags
(`--episodes`, `--seed`, `--lr`, …) are sugar that map onto the same keys.

### 4.3 `src/lunar_lander_ai/environment.py`

```python
def make_env(config: EnvironmentConfig) -> gym.Env
def describe_spaces(env: gym.Env) -> dict[str, Any]
def max_episode_steps(env: gym.Env) -> int | None   # unwrap TimeLimit
```

`make_env` must:

1. `gym.make(config.id, render_mode=config.render_mode, continuous=config.continuous, max_episode_steps=config.max_episode_steps)`.
2. Assert `isinstance(env.observation_space, spaces.Box)` and `shape == (8,)`;
   otherwise raise `ValueError(f"Expected Box(8,) observation, got {env.observation_space}")`.
3. Assert `isinstance(env.action_space, spaces.Discrete)`;
   otherwise raise `ValueError(f"DQN requires a discrete action space, got {env.action_space}")`.
4. Record the real sizes and expose them so the agent is built from the
   *environment*, never from constants.

`LunarLander-v3` defaults to discrete. Never pass `continuous=True` in this
project.

### 4.4 `src/lunar_lander_ai/network.py`

```python
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_sizes: Sequence[int] = (128, 128)) -> None: ...
    def forward(self, obs: torch.Tensor) -> torch.Tensor: ...
```

- `nn.Linear(obs_dim, h0) → ReLU → nn.Linear(h0, h1) → ReLU → … → nn.Linear(h_last, n_actions)`.
- **No activation on the output layer.** Q-values are unbounded reals; a
  `ReLU`/`tanh` there would cap or clip the value function, which is a classic
  silent bug.
- Accept `(obs_dim,)` or `(B, obs_dim)`; return `(n_actions,)` or `(B, n_actions)`.
- Input is cast to `float32` on the model's device.
- Default PyTorch `Linear` initialisation is acceptable; do not hand-roll init
  unless it is justified in a comment.

**Why the network predicts action values rather than the next position.** The
network is not a dynamics model. We do not need to know where the rocket will
go; we need to know how good each *choice* is. `Q(s,a)` is the expected
discounted sum of future rewards from taking `a` in `s` and then behaving
optimally. `argmax_a Q(s,a)` is then exactly the control policy, with no
planning or simulator rollout required. Predicting next state would model the
environment and then still leave the control problem unsolved.

### 4.5 `src/lunar_lander_ai/replay_buffer.py`

```python
@dataclass(frozen=True)
class Batch:
    obs: torch.Tensor          # (B, obs_dim)  float32
    actions: torch.Tensor      # (B,)          int64
    rewards: torch.Tensor      # (B,)          float32
    next_obs: torch.Tensor     # (B, obs_dim)  float32
    terminated: torch.Tensor   # (B,)          float32 in {0.0, 1.0}
    truncated: torch.Tensor    # (B,)          float32 in {0.0, 1.0}
    native_rewards: torch.Tensor | None

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int,
                 device: torch.device | str = "cpu",
                 seed: int | None = None) -> None: ...
    def add(self, obs, action, reward, next_obs,
            terminated: bool, truncated: bool, native_reward: float | None = None) -> None: ...
    def sample(self, batch_size: int) -> Batch: ...
    def __len__(self) -> int: ...
```

Requirements:

- **Preallocated NumPy ring buffer**, not a Python list of tuples. Fixed
  `np.empty((capacity, obs_dim), dtype=np.float32)` for obs and next_obs,
  `np.int64` for actions, `np.float32` for rewards, `np.uint8` for the two
  flags. A list of a million tuples is ~GBs; this is ~15 MB.
- Write pointer wraps; `size = min(size + 1, capacity)`. Oldest transitions are
  overwritten — verify this in a test (§8).
- `sample()` raises `ValueError(f"Cannot sample {batch_size} from buffer of size {len(self)}")`
  when undersized. Callers must gate on `len(buffer) >= max(batch_size, learning_starts)`.
- Sample with a dedicated `np.random.Generator` (checkpointable, reproducible).
- Fancy-indexing NumPy already returns a copy, so the returned tensors cannot
  alias the still-mutating ring buffer. This must be true; if a view were
  returned, sampled batches would silently corrupt as the buffer wrapped.

### 4.6 `src/lunar_lander_ai/agent.py`

```python
class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int,
                 config: AgentConfig, device: torch.device,
                 seed: int | None = None) -> None: ...

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0,
                      *, greedy: bool = False) -> int: ...
    def compute_target(self, rewards, next_obs, terminated) -> torch.Tensor: ...
    def learn(self, batch: Batch) -> float: ...
    def sync_target(self) -> None: ...
    @property
    def update_count(self) -> int: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, payload: dict) -> None: ...
```

`__init__` creates:

- `policy_net = QNetwork(...).to(device)` — trained.
- `target_net = QNetwork(...).to(device)` — `load_state_dict(policy_net.state_dict())`,
  then `requires_grad_(False)` and `.eval()`.
- `optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.learning_rate)`.
  **Only `policy_net.parameters()`.** Verify in a test that the optimizer's
  parameter groups contain no target-network tensors.
- `loss_fn = nn.SmoothL1Loss()` when `config.loss == "smooth_l1"`, else `nn.MSELoss()`.

`select_action`:

```python
if greedy or self._rng.random() >= epsilon:
    with torch.no_grad():
        return int(self.policy_net(obs_tensor).argmax(dim=-1).item())
return int(self._rng.integers(self.n_actions))
```

The exploration RNG is a dedicated `np.random.Generator`, not the global NumPy
RNG, so that seeding the buffer/agent is independent of any other library use.

`learn` — the exact algorithm:

```python
q_values = self.policy_net(batch.obs)                       # (B, A)
q_sa = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)   # (B,)

target = self.compute_target(batch.rewards, batch.next_obs, batch.terminated)

loss = self.loss_fn(q_sa, target)
self.optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.grad_clip_norm)
self.optimizer.step()

self._update_count += 1
if self._update_count % self.config.target_update_freq == 0:
    self.sync_target()
return float(loss.item())
```

`compute_target` — vanilla DQN:

```python
with torch.no_grad():                                  # non-negotiable
    next_q = self.target_net(next_obs).max(dim=1).values
    return rewards + self.gamma * (1.0 - terminated) * next_q
```

Documented Phase-16 extension (Double DQN) replaces the inner line with:

```python
next_actions = self.policy_net(next_obs).argmax(dim=1, keepdim=True)   # select with online
next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)  # evaluate with target
```

Guarded by `agent.double_dqn: false` in the config. Do not enable it, and do
not describe the project as Double DQN, until it is implemented **and**
benchmarked against the vanilla baseline.

`sync_target` uses `target_net.load_state_dict(policy_net.state_dict())` under
`torch.no_grad()`. A hard (periodic copy) update is the baseline; Polyak
averaging is a documented alternative, not a silent substitution.

### 4.7 `src/lunar_lander_ai/rewards.py`

```python
class RewardShapingWrapper(gym.Wrapper):
    """Adds a bonus to the native reward and reports both separately."""
    def __init__(self, env: gym.Env, config: RewardShapingConfig) -> None: ...
    # step() returns (obs, native + bonus, terminated, truncated, info)
    # info["native_reward"] and info["shaping_bonus"] are always populated.
```

Rules:

- **Never** replaces `info["reward"]` semantics silently. The wrapper sets
  `info["native_reward"]` and `info["shaping_bonus"]` on every step, so the
  logger can attribute reward to its source.
- When `config.enabled is False`, `make_env` returns the raw env — no wrapper,
  no behaviour change, no overhead.
- Must not change `observation_space`, `action_space`, or the termination flags.
  Assert this in a test (§8).
- Legitimate signals only, from the real observation: upright angle
  (`-w·(θ² + 0.5·ω²)`), reduced impact speed, proximity to the pad, safe leg
  contact. All observable, none exploitable.

**Explicitly forbidden shaping**, and the reason:

| Forbidden | Why |
|---|---|
| Reward per step survived | Incentivises hovering forever instead of landing |
| Reward for approaching the pad *without* a terminal bonus | The native reward already does this; adding more just encourages loitering near the pad |
| Reward for fuel remaining | Encourages doing nothing |
| Any bonus keyed on an absolute step count | Hovering exploit |
| Reward that fires on `truncated` | Rewards running out the clock |

### 4.8 `src/lunar_lander_ai/metrics.py`

The single source of truth for outcome classification and summary statistics.

```python
OUTCOMES = ("landed", "crashed", "truncated")

@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    index: int
    return_: float            # sum of the reward actually fed to the agent
    native_return: float      # sum of info["native_reward"]; == return_ when unshaped
    shaped_bonus: float       # sum of info["shaping_bonus"]
    length: int
    terminated: bool
    truncated: bool
    terminal_reward: float    # reward on the FINAL step; exactly +100.0 or -100.0 if terminated
    outcome: str              # "landed" | "crashed" | "truncated"
    legs_down: bool           # cross-check: both leg contacts set on the final observation
    final_obs: tuple[float, ...]

def classify_outcome(*, terminated: bool, truncated: bool,
                     terminal_reward: float, final_obs: np.ndarray) -> str
def summarise(results: Sequence[EpisodeResult]) -> dict[str, float]
```

```python
def classify_outcome(*, terminated, truncated, terminal_reward, final_obs) -> str:
    # The environment itself never sets `truncated`; the `TimeLimit` wrapper
    # does, at step 1000. A truncated episode did not end the world.
    if not terminated:
        return "truncated"
    # On the final step the environment OVERWRITES the reward with exactly
    # +100 (body came to rest) or -100 (body contact, or left the viewport).
    # See §3.4. So `terminated` implies `terminal_reward` is +100.0 or -100.0.
    return "landed" if terminal_reward > 0.0 else "crashed"
```

`terminal_reward` is the authoritative signal because it *is* the environment's
own definition of success. The leg contacts of the final observation are
recorded alongside as an **independent cross-check**:

```python
legs_down = final_obs[6] > 0.5 and final_obs[7] > 0.5
disagrees = legs_down != (terminal_reward > 0.0)
```

`summarise` must report `detector_disagreement_rate` — the fraction of
terminated episodes where `disagrees` is True. It exists because of §3.4 point 4:
the environment's `+100` check runs last and can override a crash, so "rewarded
+100" and "landed on both legs" are not definitionally the same thing. A
non-trivial disagreement rate means any published "landing success rate" must
state which of the two definitions it is reporting. This is also a strong answer
to the interview question "how do you know the agent actually landed?".

`summarise` returns: `n_episodes`, `mean_return`, `median_return`,
`std_return`, `min_return`, `max_return`, `mean_length`, `success_rate`,
`crash_rate`, `truncation_rate`, `detector_disagreement_rate`. Use
`statistics`/NumPy with an explicit population-vs-sample decision; document
which one `std_return` is. State clearly that `std_return` is the standard
deviation across evaluation episodes and is **not** a confidence interval — it
says nothing about run-to-run variance, which needs multiple seeds.

### 4.9 `src/lunar_lander_ai/checkpoint.py`

```python
CHECKPOINT_VERSION = 1

def save_checkpoint(path: Path, agent: DQNAgent, config: Config, *,
                    episode: int, env_step: int, epsilon: float,
                    best_mean_return: float | None = None,
                    run_id: str | None = None) -> None: ...

def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[DQNAgent, dict]: ...
```

Checkpoint payload:

| Key | Contents |
|---|---|
| `version` | `CHECKPOINT_VERSION` |
| `policy_net` | online network `state_dict` |
| `target_net` | target network `state_dict` |
| `optimizer` | Adam state_dict (enables resume) |
| `episode`, `env_step`, `update_count` | counters |
| `epsilon` | exploration rate at save time |
| `best_mean_return` | the criterion that selected this file, or `None` |
| `config` | `to_serialisable(config)` — full reproducibility |
| `rng_state` | Python / NumPy / torch RNG states |
| `package_versions` | `torch`, `gymnasium`, `numpy`, `Box2D` versions |
| `created_utc`, `run_id` | provenance |

Requirements:

- **Atomic write**: write to `<name>.tmp` then `os.replace()` onto the final
  path. A Ctrl-C mid-save must not leave a corrupt `best.pt` behind.
- `load_checkpoint` raises `FileNotFoundError` with the attempted path, and
  `ValueError` on a version mismatch, naming both versions.
- Loading rebuilds the agent from the **stored config**, not from the config
  file on disk, so evaluation is reproducible even if the YAML changed later.
- Three distinct files, three distinct meanings, never conflated:
  - `latest.pt` — most recent checkpoint, for resuming.
  - `best.pt` — best mean native return observed at a periodic evaluation.
    Written **only** when that evaluation actually ran and produced a number.
  - `final.pt` — written once when training stops (including on early stop or
    `KeyboardInterrupt`).

### 4.10 `src/lunar_lander_ai/train.py`

```python
@dataclass
class EpisodeStats:
    episode: int; env_step: int; return_: float; native_return: float
    shaped_bonus: float; length: int; terminated: bool; truncated: bool
    outcome: str; epsilon: float; mean_loss: float | None; updates: int
    wall_time_s: float

@dataclass
class TrainingRun:
    run_id: str; config: Config; episodes: list[EpisodeStats]
    log_path: Path; best_mean_return: float | None

def linear_epsilon(step: int, cfg: EpsilonConfig) -> float: ...
def train(config: Config, *, smoke_test: bool = False) -> TrainingRun: ...
```

Epsilon schedule (linear, then floored):

```python
frac = min(1.0, step / cfg.decay_steps)
return cfg.start + frac * (cfg.end - cfg.start)
```

Episode loop, in order:

1. `obs, info = env.reset()` — **seeded only on the very first reset** of the
   training run; later episodes must vary. See §7.
2. `epsilon = linear_epsilon(env_step, cfg.epsilon)`.
3. Select an action with `agent.select_action(obs, epsilon)`.
4. `next_obs, reward, terminated, truncated, info = env.step(action)`.
5. `buffer.add(obs, action, reward, next_obs, terminated, truncated, native_reward=info.get("native_reward"))`.
6. `env_step += 1`.
7. If `len(buffer) >= max(batch_size, learning_starts)` and
   `env_step % train_frequency == 0`: `loss = agent.learn(buffer.sample(batch_size))`.
8. `obs = next_obs`; `native_return += info.get("native_reward", reward)`;
   `shaped_bonus += info.get("shaping_bonus", 0.0)`.
9. Break on `terminated or truncated` — **both** end the episode. Only
   `terminated` suppresses bootstrapping (§6.1).
10. Append `EpisodeStats`, classifying the outcome via `metrics.classify_outcome`.
11. Every `checkpoint_interval`: save `latest.pt`.
12. Every `eval_interval` (and only if `episodes >= eval_interval`): run
    `evaluate(...)` headless and greedy; if `mean_return > best_mean_return`,
    write `best.pt` and record the new best.
13. Every `log_interval`: print one progress line.
14. Early stop when the rolling mean over the last `solved_window` episodes
    reaches `solved_threshold` (default 200.0, matching the environment's own
    solve criterion).
15. Wrap the whole loop in `try/except KeyboardInterrupt`; on interrupt, still
    save `final.pt` and flush the log. Never lose a run to Ctrl-C.

Persistence, kept separate:

- `logs/train_<run_id>.csv` — one row per episode, one column per `EpisodeStats` field.
- `logs/run_<run_id>.json` — full config, seed, episode count, package versions, UTC start/end.
- `logs/eval_<run_id>.csv` — one row per periodic evaluation.

**Smoke test mode** (`--smoke-test`) must be fast and must never launch a real
run. It forces: `episodes=2`, `learning_starts=32`, `batch_size=32`,
`buffer_capacity=2000`, `eval_interval=0` (disabled), `render=False`,
`checkpoint_interval=0` (disabled), writes to a `smoke_<run_id>` log so it
cannot clobber real results, and prints `SMOKE TEST OK` plus the two episode
returns. Target: under ~20 seconds on CPU.

### 4.11 `src/lunar_lander_ai/evaluate.py`

```python
def evaluate(env: gym.Env, agent: DQNAgent, *, episodes: int, seed: int,
             greedy: bool = True, render: bool = False,
             shaping: RewardShapingWrapper | None = None) -> list[EpisodeResult]

def run_evaluation(*, config: Config, checkpoint: Path, episodes: int,
                   seed: int, greedy: bool, render_mode: str | None) -> dict
```

Requirements:

- `agent.policy_net.eval()` and `torch.no_grad()` around every forward pass.
- **No weight updates.** Assert `agent.update_count` is unchanged across the
  whole evaluation (a test does this).
- Each episode gets its own deterministic seed: `env.reset(seed=seed + i)`.
  This is deliberate and differs from training (§7).
- Records the seed of every episode in the output so any single episode can be
  replayed exactly with `play.py --seed`.
- Writes `results/eval_<label>_<timestamp>.json` plus a matching `.csv` of
  per-episode rows. Never overwrites a previous experiment.
- Reports all of §4.8's metrics plus the checkpoint path, checkpoint episode,
  config hash, and whether shaping was enabled.

### 4.12 `src/lunar_lander_ai/plotting.py`

```python
def load_episode_log(path: Path) -> pd.DataFrame
def load_eval_log(path: Path) -> pd.DataFrame
def plot_training_curves(df: pd.DataFrame, out_dir: Path, *,
                         window: int = 100, dpi: int = 150) -> list[Path]
```

Six charts, each a separate PNG in `results/`:

1. `episode_return.png` — raw per-episode return as a thin alpha line.
2. `rolling_return.png` — rolling mean over `window` episodes, plus the raw
   line faintly behind it, plus a horizontal line at the 200 solve threshold.
3. `epsilon_decay.png` — epsilon against environment step.
4. `loss.png` — training loss, raw plus rolling mean. Log scale if the range spans >2 decades.
5. `eval_success_rate.png` — landing success rate per evaluation checkpoint, with
   the number of episodes annotated (a 20-episode rate is noisy and the chart
   must say so).
6. `experiment_comparison.png` — native vs shaped reward agents, **only**
   generated when both logs exist.

Rules:

- Every chart: title, axis labels with units, legend, `tight_layout()`,
  `dpi=150`, saved as PNG.
- Raw values and rolling averages must be visually distinguishable (alpha and
  line width) and named in the legend. Never plot a rolling mean as if it were
  the raw return.
- **Must not crash on partial or empty logs.** A run interrupted at episode 12
  still has to plot. Guard every column access; skip a chart whose required
  column is absent and warn on stderr.
- `matplotlib.use("Agg")` before importing pyplot — the plotting scripts must
  run headless in a terminal.

### 4.13 `scripts/*.py`

Thin CLI wrappers only. All logic lives in `src/lunar_lander_ai/`. Each script
resolves `project_root()` and puts `src/` on `sys.path`, so scripts run without
an editable install.

---

## 5. Hyperparameters and why

| Parameter | Baseline | Rationale |
|---|---|---|
| `gamma` | 0.99 | Effective horizon ≈ 100 steps at 1% discount. The lander needs a few hundred steps of credit assignment; 0.99 is the standard value and works. Much lower and the agent cannot value a distant landing. |
| `learning_rate` | 5e-4 | Adam on a small MLP. 1e-3 tends to oscillate on LunarLander; 1e-4 learns too slowly within a realistic episode budget. |
| `batch_size` | 64 | Small enough to take many gradient steps per episode (important with a 1000-step time limit), large enough to average out single-transition noise. |
| `buffer_capacity` | 100,000 | ~100 episodes of experience. Must be far larger than `batch_size` to decorrelate samples; large enough to still hold varied early failures. |
| `learning_starts` | 1,000 | One full episode's worth. Learning from a near-empty buffer produces wildly biased early targets. |
| `train_frequency` | 1 | One gradient step per environment step. Cheap here because the network is tiny. |
| `target_update_freq` | 1,000 **gradient steps** | Hard sync. Too frequent ≈ no target network; too rare and targets drift. Unit matters — the config and code must agree it is gradient steps, not env steps. |
| `hidden_sizes` | `[128, 128]` | ~18k parameters. Enough capacity for an 8-D input, small enough to train in one sitting. |
| `epsilon` | 1.0 → 0.05 over 150,000 steps | Roughly the first 150 episodes. Decaying to 0.05 rather than 0 keeps a floor of exploration, which helps escape the "hover and do nothing" local optimum. Do not let it hit 0 by episode 50, or the agent locks into whatever it had learned by then. |
| `grad_clip_norm` | 10.0 | LunarLander's ±100 terminal rewards produce large TD errors and occasional loss spikes. Clipping bounds the update, not the learning. |
| `loss` | SmoothL1 (Huber) | Quadratic for small TD errors, linear for large ones — so the ±100 crashes do not dominate the gradient. This is the standard DQN choice. |
| `episodes` | 1,000 | A *starting* budget, not a promise. See §9. |

---

## 6. The four correctness criticals in detail

### 6.1 Truncation vs termination — why it changes the target

LunarLander wraps the task in a `TimeLimit`. Gymnasium therefore returns two
independent flags:

- `terminated=True` — the MDP reached a **genuine terminal state**: the lander
  crashed, or it landed safely. There are no future rewards. Bootstrapping must stop.
- `truncated=True` — the episode was cut at 1000 steps by the time limit. The
  world did not end; if we had let it continue, there would be more reward. The
  value estimate should **still bootstrap** from `s'`.

So the mask is `(1 - terminated)`, and this is why `truncated` must be stored:

```python
target = reward + gamma * (1.0 - terminated) * max_a' Q_target(s', a')
```

Treating truncation as terminal would teach the agent that the world ends at
step 1000, systematically underestimating the value of any state from which a
landing was still reachable — precisely the long-horizon behaviour we are
trying to learn. `truncated` is still stored on every transition, because
reporting the truncation rate is part of honest evaluation.

Both flags end the *episode loop*. Only `terminated` ends the *bootstrapping*.
Those are two different decisions and the code must not conflate them.

### 6.2 No gradients through the target

The target is a fixed regression label computed from a frozen network. If it
were differentiable w.r.t. the online network, the optimizer could reduce the
loss by moving the target toward the prediction — "chasing its own tail" —
instead of learning the value function. Hence:

```python
with torch.no_grad():
    next_q = self.target_net(next_obs).max(dim=1).values
```

Test: build a batch, call `compute_target`, assert `target.requires_grad is False`
and `target.grad_fn is None`.

### 6.3 Target network isolation

The target network is a frozen copy. It must be:

- in `eval()` mode,
- with `requires_grad_(False)` on all parameters,
- **absent from the optimizer's parameter groups**,
- updated only by `load_state_dict` from the online network (or Polyak averaging).

Test: iterate `optimizer.param_groups` and assert no tensor is an element of
`target_net.parameters()`.

### 6.4 Native vs shaped reward

The native reward is the environment's own objective and the only thing
evaluation reports. Shaping is an additive learning aid. Conflating them makes
results uninterpretable — you cannot tell whether the agent learned the task or
learned the wrapper. Therefore `info["native_reward"]` and
`info["shaping_bonus"]` are logged as separate columns, `native_return` and
`shaped_bonus` are separate fields on `EpisodeStats` and `EpisodeResult`, and
every evaluation reports native return.

A **reward function** defines what the task is; a **reward-shaping strategy** is
a modification to the learning signal intended to make that task easier to
learn, without changing what counts as success. The distinction is only
meaningful if the two are measured separately.

---

## 7. Reproducibility

Seed order, applied once at process start:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ.setdefault("PYTHONHASHSEED", str(seed))
```

Plus dedicated generators: `ReplayBuffer(seed=seed)` and
`DQNAgent(seed=seed)`. Do not route exploration randomness through global NumPy
state; that couples it to unrelated library calls.

**First-reset seeding, and only the first.** In training:

```python
obs, info = env.reset(seed=config.seed)     # first episode only
...
obs, info = env.reset()                     # every episode after
```

Re-seeding on every reset would replay the identical episode forever and the
agent would never see new states — a bug that looks like "the agent stopped
learning".

In evaluation, the opposite: `env.reset(seed=seed + i)` per episode, so each
episode is deterministic *and* distinct, and any episode can be replayed later
with `play.py --seed <seed + i>`.

Seeds are recorded in `logs/run_<run_id>.json` and in every evaluation JSON.

**Documented caveat, to be stated in the README:** identical seeds do not
guarantee identical results across operating systems, CPU/GPU, PyTorch
versions, or BLAS backends. Seeding makes a run auditable and mostly
repeatable on the same machine — it is not a proof of bit-exact determinism.

---

## 8. Test specification

Every test must run headless and fast. Tests that train or render are marked
`@pytest.mark.slow`.

**`tests/test_environment.py`**
- `make_env` returns `Box(8,)` / `Discrete(4)` and `describe_spaces` matches.
- One random-action episode runs to `terminated or truncated` without error.
- `max_episode_steps(env) == 1000` (or whatever the runtime reports) and the
  episode length never exceeds it.
- A `make_env` call with a deliberately wrong expectation raises a `ValueError`
  mentioning the actual shape.
- `env.close()` is idempotent.

**`tests/test_network.py`**
- Output shape is `(batch, n_actions)` for `(batch, 8)` input and `(n_actions,)`
  for `(8,)` input.
- Output layer has no activation: two different inputs give different Q-values
  and the outputs are not confined to a bounded range (guards against a stray
  `ReLU`/`sigmoid` on the head).
- `loss.backward()` populates `.grad` on every parameter.
- Changing `hidden_sizes` changes the parameter count, i.e. the config is
  actually threaded through.
- `.to("cpu")` round-trip preserves output equality.

**`tests/test_replay_buffer.py`**
- `len()` grows to `capacity` and stops.
- Inserting `capacity + k` transitions overwrites the oldest — retrieve and
  assert the surviving contents.
- `sample()` raises `ValueError` when `len(buffer) < batch_size`.
- Sampled shapes and dtypes are exactly as specified; `terminated`/`truncated`
  survive the round-trip as 0.0/1.0 floats.
- Sampled tensors do not alias the ring buffer: overwrite the buffer after
  sampling and confirm the previously sampled batch is unchanged.
- `native_reward=None` paths produce `native_rewards is None`.

**`tests/test_agent.py`**
- `select_action(greedy=True)` is deterministic and equals `argmax` of the
  network output.
- `select_action` with `epsilon=1.0` returns a valid action in range over many
  draws, and covers more than one distinct action.
- `compute_target(...).requires_grad is False`.
- For a hand-built batch, `compute_target` equals a hand-computed Bellman
  target to `1e-6`, for the three cases: non-terminal, `terminated=1`,
  `truncated=1` (must equal the non-terminal result — the §6.1 rule).
- `learn()` returns a finite float and **changes** at least one policy-net
  parameter.
- `learn()` does **not** change any target-net parameter.
- The optimizer holds no target-net parameters.
- After `target_update_freq` calls, target and policy weights are identical.
- `grad_clip_norm` is applied: with a huge synthetic TD error, the total
  gradient norm after clipping is `<= grad_clip_norm + 1e-6`.
- Invalid configs (negative lr, `gamma=1.5`, `epsilon.end > start`) raise `ValueError`.

**`tests/test_checkpoint.py`**
- Save → load → identical policy-net outputs on a fixed observation.
- Loaded agent's `update_count`, `episode`, and `epsilon` match what was saved.
- The stored config round-trips and is what the rebuilt agent uses.
- Missing file → `FileNotFoundError` naming the path.
- Wrong `version` → `ValueError` naming both versions.
- A failed save (simulate by writing to an invalid path) leaves no partial file
  at the target path.

**`tests/test_metrics.py`**
- `classify_outcome` covers: landed (`terminated`, both legs 1.0),
  crashed (`terminated`, legs not both 1.0), truncated (`truncated`, not
  terminated). Truncated must win when both flags are somehow set and
  `terminated` is False.
- `summarise` reproduces hand-computed mean/median/std/min/max on a known list.
- `summarise` on an empty list either raises a clear error or returns zeros —
  pick one and document it.
- Success + crash + truncation rates sum to 1.0.

**`tests/test_plotting.py`** (add to the listed set — it is required by §10 of
the brief, which asks that plotting be tested with complete *and* partial logs)
- A full synthetic log produces all six charts, each a non-empty PNG.
- A log missing the `loss` column skips that chart with a warning and still
  produces the others.
- A one-row log does not crash.
- An empty DataFrame is handled without an exception.

---

## 9. Build phases and acceptance gates

Do not advance past a gate that is red.

| Phase | Deliverable | Gate |
|---|---|---|
| 1 | Environment setup | `.venv` imports `gymnasium`, `torch`, `Box2D`; `scripts/verify_setup.py` prints the real obs/action spaces and runs one random-action episode; `env.close()` clean |
| 2 | `network.py` + `test_network.py` | tests green; output shape `(B, 4)` with no output activation |
| 3 | `replay_buffer.py` + `test_replay_buffer.py` | tests green including rollover and the no-alias test |
| 4 | `agent.py` + `test_agent.py` | tests green; hand-computed Bellman target matches to 1e-6 |
| 5 | `train.py` + `scripts/train.py` | `--smoke-test` finishes in < 20 s, prints `SMOKE TEST OK`, writes a real CSV with 2 rows |
| 6 | First real training run + `evaluate.py` | honest metrics exist; loss is finite; epsilon decays as configured; buffer fills |
| 7 | `rewards.py` + shaping experiment | baseline preserved; both agents evaluated by the identical procedure; both configs recorded |
| 8 | `plotting.py`, `play.py`, `scripts/plot_results.py` | real charts from real logs; playback shows the agent's own actions |
| 9 | Full test suite + Ruff | `pytest` green, `ruff check` clean with no blanket `# noqa` |
| 10 | README, docs, `docs/portfolio.md`, git history | a stranger can clone and run; no secrets, no venv, no large artifacts committed |

### Phase 6 diagnostics — if the agent does not improve

Check in this order; each is a real bug class, not a guess:

1. Are `terminated`/`truncated` stored separately and is the mask `1 - terminated`?
2. Is the target computed under `no_grad` and from the *target* net?
3. Is `q_sa` gathered with `batch.actions`, or is the whole Q-vector being
   regressed against a scalar target? (A shape mismatch that silently
   broadcasts is the classic version of this.)
4. Is `learning_starts` so small that the agent learns from a handful of
   transitions?
5. Is epsilon reaching its floor too early?
6. Are episode returns actually being recorded from the environment reward, or
   from something that drifts?
7. Is the environment resolving to the discrete space at all?

Only after all seven are clear is "train longer" a legitimate answer.

---

## 10. CLI contract

```powershell
# Setup (once)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/verify_setup.py

# Smoke test — must be fast, must not launch a real run
python scripts/train.py --episodes 2 --smoke-test

# Full training
python scripts/train.py --config configs/default.yaml

# Train with overrides
python scripts/train.py --config configs/default.yaml --episodes 500 --seed 7 --set agent.learning_rate=0.0003

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best.pt --episodes 100

# Watch the trained agent (opens a window — human must be present)
python scripts/play.py --checkpoint checkpoints/best.pt --episodes 5

# Replay one exact evaluation episode
python scripts/play.py --checkpoint checkpoints/best.pt --seed 20260

# Charts
python scripts/plot_results.py --log-dir logs
```

Shared flags: `--config`, `--set KEY=VALUE` (repeatable), `--seed`,
`--device {auto,cpu,cuda}`, `--episodes`, `--render`, `--quiet`.

`train.py` additionally: `--smoke-test`, `--lr`, `--gamma`, `--batch-size`,
`--buffer-capacity`, `--epsilon-start/--epsilon-end/--epsilon-decay`,
`--target-update`, `--hidden-sizes` (e.g. `128,128,64`),
`--checkpoint-interval`, `--eval-interval`, `--reward-shaping`, `--run-id`.

`evaluate.py` additionally: `--checkpoint`, `--greedy/--no-greedy`, `--label`,
`--output-dir`.

`play.py` additionally: `--checkpoint`, `--seed`, `--fps`, `--save-video PATH`.

PowerShell note: `Activate.ps1` is blocked by default execution policy on some
machines. `LocalMachine` policy here is `RemoteSigned`, so activation works. If
it ever fails, either run
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or skip
activation entirely and call `.\.venv\Scripts\python.exe` directly.

---

## 11. Post-baseline extensions

Only after Phase 9 is green. Each is a separate, documented experiment with its
own recorded config — never a silent change to the baseline.

1. **Double DQN** — decouple action selection (online net) from evaluation
   (target net) to reduce maximisation bias. ~3 lines, flag-guarded, must be
   A/B'd against the vanilla baseline.
2. **Dueling DQN** — split the head into `V(s)` + `A(s,a)`.
3. **Prioritised experience replay** — sample by TD-error, correct with importance weights.
4. **Observation normalisation** — running mean/std wrapper; the raw scales here
   are a plausible instability source and this is a legitimate experiment.
5. **Hyperparameter search** — small grid over `lr`, `gamma`, `target_update_freq`,
   `hidden_sizes`; multiple seeds; report mean ± std, not a single lucky run.
6. **Multi-seed confidence intervals** — ≥5 seeds; the honest way to report.
7. **Profiling** — where the wall-clock actually goes (env stepping vs gradients).
8. **Streamlit dashboard** — optional; reads logs, launches evaluation, never
   auto-runs training on load.
9. **Baseline comparison** — an untrained random policy and a hand-written
   heuristic controller, so the learned policy has a reference point.
10. **Reward-shaping strategy comparison** — the full matrix, native returns only.

Do not label the baseline as any of these. A vanilla DQN with a hard target
update is not Double DQN, and calling it that in an interview will not survive
the first follow-up question.
