"""Evaluation outcomes from actual termination and observed pad contact."""

from math import sqrt
from statistics import mean, median, pstdev


def compute_wilson(successes: int, n: int) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (95% CI)."""
    if n == 0:
        return 0.0, 1.0
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    z = 1.959963984540054  # 97.5th percentile of standard normal
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def classify(terminated: bool, terminal_reward: float, obs) -> str:
    """Require rest, both legs and pad proximity for a genuine landing.

    Gymnasium awards +100 for any sleeping lander, including off-pad landings.
    The pad spans normalized x approximately [-0.2, 0.2] in LunarLander-v3.
    """
    if not terminated:
        return "truncated"
    if terminal_reward == 100.0 and obs[6] > 0.5 and obs[7] > 0.5 and abs(obs[0]) <= 0.2:
        return "landed"
    return "crashed" if terminal_reward == -100.0 else "off_pad_or_unsafe"


def summarise(rows: list[dict]) -> dict:
    """Summarise native returns and explicit mutually exclusive outcomes."""
    if not rows:
        raise ValueError("Cannot summarise zero episodes")
    rewards = [row["native_return"] for row in rows]
    count = len(rows)
    return {
        "episodes": count,
        "mean_return": mean(rewards),
        "median_return": median(rewards),
        "std_return": pstdev(rewards),
        "best_return": max(rewards),
        "worst_return": min(rewards),
        "mean_length": mean(row["length"] for row in rows),
        **{f"{name}_rate": sum(row["outcome"] == name for row in rows) / count
           for name in ("landed", "crashed", "truncated", "off_pad_or_unsafe")},
    }
