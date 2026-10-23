# Experiment Methodology: Multi-Seed DQN Evaluation

This document describes the methodology for the multi-seed DQN experiments in the Lunar AI project. It covers the experimental design, evaluation protocol, statistical methods, and reproducibility guarantees.

---

## 1. Experimental Design

### 1.1 Objective

Estimate the distribution of DQN performance on LunarLander-v3 across independent training runs with different random seeds. This addresses the fundamental limitation of single-seed results: one training run cannot establish repeatability of learning.

### 1.2 Design: Independent Training Seeds

**Key principle**: Five evaluation seeds are **not** equivalent to five independently trained agents.

- **Training seeds**: 5 distinct seeds (default: 42, 43, 44, 45, 46)
- **Each seed**: Trains a completely independent agent from scratch
- **Same configuration**: Identical hyperparameters, environment settings, training budget
- **Evaluation**: Every trained agent evaluated on the **same held-out seed range**

This design isolates *between-seed variability* (training stochasticity) from *within-seed variability* (episode-level stochasticity of a fixed policy).

### 1.3 Arms

Two experimental arms (configurable):

| Arm | Reward Shaping | Angle Penalty Weight |
|-----|----------------|---------------------|
| Native | Disabled | 0.0 |
| Shaped | Enabled | 2.0 |

Both arms use the **same** training budget and evaluation protocol. The shaped arm adds a potential-based shaping term (difference of discounted potentials) to the native reward. Evaluation **always** uses native reward only.

---

## 2. Training Protocol

### 2.1 Configuration (Default)

| Parameter | Value |
|-----------|-------|
| Episodes | 900 |
| Early stopping | Disabled |
| Epsilon decay | 1.0 → 0.05 over 60,000 env steps |
| Periodic evaluation | Every 100 episodes, 50 greedy episodes |
| Checkpointing | Every 50 episodes |
| Learning starts | 1,000 transitions |
| Batch size | 64 |
| Replay capacity | 100,000 |
| Target update freq | 1,000 gradient steps |
| Network | 2-layer MLP (128, 128) |
| Optimizer | Adam, lr=5e-4 |
| Loss | Smooth L1 (Huber) |
| Gradient clip | 10.0 |
| Discount (γ) | 0.99 |

### 2.2 Random Number Generation

- **Python RNG**: `random.seed(seed)`
- **NumPy RNG**: `np.random.seed(seed)` (legacy) + `np.random.default_rng(seed)` for replay buffer
- **PyTorch RNG**: `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`
- **Environment**: `env.reset(seed=seed)` for first episode only; subsequent episodes use unseeded reset

**Separation of RNGs**: The replay buffer uses its own `np.random.Generator` (seeded independently). The agent's action selection uses its own `np.random.Generator`. Training loops use the global NumPy/PyTorch RNGs. This prevents cross-contamination of random streams.

### 2.3 Checkpoints Saved

- `latest.pt` — most recent (overwritten)
- `episode_XXXX.pt` — journey checkpoints (every 50 episodes)
- `best.pt` — highest mean native return at periodic evaluation
- `final.pt` — end of training

Each checkpoint includes: model weights, optimizer state, training RNG states (Python, NumPy, PyTorch, CUDA), config, episode count, step count, timestamp, package versions.

---

## 3. Evaluation Protocol

### 3.1 Held-Out Seeds

- **Seed range**: 40400–40499 (100 seeds, distinct from training seeds and periodic eval seeds)
- **Episodes per agent**: 100 (configurable)
- **Policy**: Greedy (ε=0), no exploration
- **Reward**: Native only (shaping wrapper **not** applied during evaluation)
- **Rendering**: Headless (no window)

### 3.2 Metrics Recorded Per Episode

| Field | Description |
|-------|-------------|
| `episode` | 0–99 (index within evaluation batch) |
| `seed` | Held-out seed used (40400 + episode) |
| `native_return` | Sum of native rewards |
| `length` | Episode length in steps |
| `terminated` | True if terminated (landed/crashed) |
| `truncated` | True if time-limited (1000 steps) |
| `terminal_reward` | Final step reward (+100 or -100 or 0) |
| `outcome` | Classified outcome (see below) |
| `legs_down` | Both leg contacts > 0.5 |

### 3.3 Outcome Classification

The `classify()` function in `metrics.py` implements a **strict** landing detector:

```python
def classify(terminated: bool, terminal_reward: float, obs) -> str:
    if not terminated:
        return "truncated"
    if terminal_reward == 100.0 and obs[6] > 0.5 and obs[7] > 0.5 and abs(obs[0]) <= 0.2:
        return "landed"
    return "crashed" if terminal_reward == -100.0 else "off_pad_or_unsafe"
```

**Critical**: Gymnasium awards +100 for *any* sleeping lander, including off-pad. The detector requires:
1. `terminated == True` (lander came to rest)
2. `terminal_reward == 100.0` (environment's success signal)
3. **Both legs** in contact (`obs[6] > 0.5 and obs[7] > 0.5`)
4. **Pad proximity**: `|x| ≤ 0.2` (normalized horizontal offset)

This prevents the "leg-proxy disagreement" where a crashed lander rests on its legs (measured at ~23% in random policy).

### 3.4 Summary Statistics (Per Agent)

Computed by `summarise()` in `metrics.py`:

| Metric | Formula |
|--------|---------|
| `episodes` | Count (100) |
| `mean_return` | Mean of native_return |
| `median_return` | Median of native_return |
| `std_return` | Population std (pstdev) |
| `best_return` | Max native_return |
| `worst_return` | Min native_return |
| `mean_length` | Mean episode length |
| `landed_rate` | Count(landed) / episodes |
| `crashed_rate` | Count(crashed) / episodes |
| `truncated_rate` | Count(truncated) / episodes |
| `off_pad_or_unsafe_rate` | Count(off_pad_or_unsafe) / episodes |

---

## 4. Aggregation Across Training Seeds

### 4.1 Per-Seed Metrics

For each training seed, we record the summary statistics above plus Wilson 95% CI for landing rate.

### 4.2 Overall Statistics (Across Seeds)

| Metric | Formula |
|--------|---------|
| `n_seeds` | Number of completed training seeds |
| `mean_safe_rate` | Mean of per-seed `landed_rate` |
| `std_safe_rate` | Sample std of per-seed `landed_rate` |
| `mean_native_return` | Mean of per-seed `mean_return` |
| `std_native_return` | Sample std of per-seed `mean_return` |
| `mean_episode_length` | Mean of per-seed `mean_length` |
| `total_landed` | Sum of per-seed landed counts |
| `total_evaluated` | Sum of per-seed episode counts |

### 4.3 Confidence Intervals

Two types of intervals are reported:

#### A. t-Distribution CI (Between-Seed Variability)
For the mean across training seeds (e.g., mean landing rate):
- Standard error: `SE = std / sqrt(n_seeds)`
- Critical value: `t_{0.975, n_seeds-1}` (SciPy `stats.t.ppf`)
- CI: `mean ± t_crit * SE`

**Interpretation**: This captures uncertainty about the *mean performance across training seeds*. It does NOT capture episode-level variability within a single trained agent.

#### B. Wilson Score Interval (Per-Seed Landing Rate)
For each seed's landing proportion (binomial):
- Wilson 95% CI: accounts for binomial sampling variability
- Formula: standard Wilson with continuity correction
- Pooled Wilson (descriptive only): combines all seeds' numerators/denominators

**Interpretation**: This captures uncertainty about the *true landing probability* for a given trained agent, given 100 evaluation episodes.

**Important**: The pooled Wilson interval **ignores between-seed dependence** and is reported as descriptive only. It is NOT a confidence interval on the mean across seeds.

---

## 5. Reproducibility

### 5.1 What Is Guaranteed

- **Same seed → same trajectory** on the same machine, same Python/PyTorch/NumPy versions, same BLAS backend
- **Configuration is recorded** in `logs/config_<run_id>.json` for every run
- **Metadata is recorded** in `logs/run_<run_id>.json` (status, timestamps, PID, package versions, device)
- **Checkpoints include full RNG state** for exact resume (though replay buffer is not saved)

### 5.2 What Is NOT Guaranteed

- **Bit-identical results across machines/OSes/BLAS backends**: Floating-point non-determinism in CuDNN, MKL, etc.
- **Exact resume of training**: Replay buffer contents and environment RNG state are not checkpointed. Loading a checkpoint for *evaluation* works perfectly; *training resume* continues from the same weights but with a fresh replay buffer.
- **Numerical equivalence across PyTorch versions**: Different versions may have different kernels.

### 5.3 Resume Behavior

The multi-seed runner (`scripts/multi_seed_experiment.py`) implements resume by checking `logs/run_<run_id>.json`:

- If `status == "completed"` and `episodes_completed == target_episodes`: **skip** (already done)
- If config exists but status is incomplete: **refuse to overwrite** (manual inspection required)
- If no artifacts exist: **start fresh**

This prevents accidental data loss while allowing campaigns to be split across sessions.

---

## 6. Statistical Limitations & Caveats

1. **Small n (5 seeds)**: Low statistical power. CIs will be wide. Results are indicative, not definitive.
2. **No causal claims**: This is an observational comparison of two configurations. Without random assignment and blinding, we cannot claim reward shaping *causes* any difference.
3. **Multiple comparisons**: Comparing native vs shaped on multiple metrics inflates Type I error. No correction applied.
4. **Selection bias**: The "best" checkpoint is selected on noisy periodic evaluations (50 episodes). This introduces winner's curse. We report both "best" and "final" checkpoints to expose this.
5. **Episode-level vs seed-level variability**: A single trained agent evaluated on 100 episodes has within-seed variance. Five trained agents have between-seed variance. These are distinct and must not be conflated.
6. **Environment specifics**: Results apply to LunarLander-v3 with Gymnasium 1.4.0, Box2D 2.3.10. Different versions may yield different results.

---

## 7. Running an Experiment

### 7.1 Quick Smoke Test

```bash
python scripts/multi_seed_experiment.py --smoke-test --seeds 42 43 --prefix smoketest
```
Runs 2 episodes per seed (native + shaped arms), completes in ~30 seconds.

### 7.2 Full Campaign

```bash
python scripts/multi_seed_experiment.py --seeds 42 43 44 45 46 --device auto --prefix myexperiment
```
Runs 900 episodes × 5 seeds × 2 arms = 90,000 training episodes total. Expect several hours on GPU.

### 7.3 Analysis

```bash
python -m lunar_lander_ai.analysis --results-dir results --log-dir logs --prefix myexperiment --output-dir results/analysis
```
Generates interactive HTML charts and `findings_summary.md` in `results/analysis/`.

---

## 8. Data Artifacts

| Location | Content |
|----------|---------|
| `logs/config_<run>.json` | Full resolved configuration |
| `logs/run_<run>.json` | Metadata: status, timestamps, PID, versions, seed |
| `logs/train_<run>.csv` | Per-episode training logs (native_return, epsilon, loss, outcome, etc.) |
| `logs/eval_<run>.csv` | Periodic evaluation summaries (every 100 episodes) |
| `checkpoints/<run>/best.pt` | Best checkpoint by periodic eval |
| `checkpoints/<run>/final.pt` | Final checkpoint |
| `results/eval_<label>_<timestamp>.csv` | Per-episode held-out evaluation (100 eps) |
| `results/eval_<label>_<timestamp>.json` | Summary statistics for held-out eval |
| `results/<prefix>_summary.csv` | Aggregated per-seed + overall summary |
| `results/analysis/*.html` | Interactive Plotly visualizations |
| `results/analysis/findings_summary.md` | Auto-generated findings summary |

---

## 9. References

- Mnih et al., "Human-level control through deep reinforcement learning" (Nature, 2015)
- Wilson, "Probable inference, the law of succession, and statistical inference" (JASA, 1927)
- Gymnasium documentation: LunarLander-v3 environment specification
- PyTorch documentation: Random number generation and reproducibility