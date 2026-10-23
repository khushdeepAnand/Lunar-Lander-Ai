"""Phase 1 environment verification for the Lunar Lander AI project.

This script is deliberately standalone: it depends only on Gymnasium, NumPy and
torch, not on the ``lunar_lander_ai`` package. Its job is to prove that the
physics simulation actually works on this machine *before* any learning code
exists, so that a later failure can be attributed to the agent rather than to
the environment.

It verifies, and prints:

* the interpreter and the versions of every package the project depends on;
* the real ``observation_space`` and ``action_space`` of the environment;
* the real time limit, read out of Gymnasium's ``TimeLimit`` wrapper;
* one or more full episodes driven by uniformly random actions, classified by
  outcome.

It exits non-zero if any expectation in the project's build specification is
violated, so it is safe to use as a setup gate.

Usage (from the project root, with the virtualenv active)::

    python scripts/verify_setup.py
    python scripts/verify_setup.py --episodes 30 --seed 12345
    python scripts/verify_setup.py --render rgb_array
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

# The project's build specification fixes these. They are asserted, not assumed:
# if a future Gymnasium release changes the environment, this script fails loudly
# instead of letting the agent silently receive the wrong shapes.
EXPECTED_OBS_SHAPE = (8,)
EXPECTED_N_ACTIONS = 4

ACTION_DESCRIPTIONS = {
    0: "do nothing",
    1: "fire left orientation engine",
    2: "fire main (downward) engine",
    3: "fire right orientation engine",
}

OBSERVATION_DESCRIPTIONS = (
    "horizontal position x          ",
    "vertical position y            ",
    "horizontal velocity vx         ",
    "vertical velocity vy           ",
    "angle (radians, 0 = upright)   ",
    "angular velocity               ",
    "left leg ground contact (0/1)  ",
    "right leg ground contact (0/1) ",
)


@dataclass(frozen=True)
class EpisodeSummary:
    """The outcome of a single random-action episode."""

    return_: float
    length: int
    terminated: bool
    truncated: bool
    terminal_reward: float
    final_obs: tuple[float, ...]


def classify(summary: EpisodeSummary) -> tuple[str, bool]:
    """Classify an episode's outcome, and whether the leg proxy agreed.

    The terminal reward is authoritative. LunarLander overwrites the reward on
    the final step with exactly ``+100`` (the body came to rest) or ``-100``
    (body contact, or left the viewport). "Both legs touching" is only a proxy:
    a lander that *crashed* onto its fuselage can also be resting on its legs,
    so a naive leg-contact test reports false landings. This function returns
    both so the disagreement rate can be measured rather than assumed.

    Args:
        summary: the recorded outcome of one episode.

    Returns:
        A ``(outcome, leg_proxy_agreed)`` pair, where ``outcome`` is one of
        ``"landed"``, ``"crashed"`` or ``"truncated"``.
    """
    legs_down = summary.final_obs[6] > 0.5 and summary.final_obs[7] > 0.5

    if not summary.terminated:
        # Covers truncation by the TimeLimit wrapper, and any episode that did
        # not terminate. The world did not end.
        return "truncated", True

    landed = summary.terminal_reward > 0.0
    return ("landed" if landed else "crashed"), (legs_down == landed)


def _print_versions() -> None:
    """Print the interpreter and every relevant package version."""
    print("=" * 78)
    print("INTERPRETER AND PACKAGES")
    print("=" * 78)
    print(f"  python        {sys.version.split()[0]}")
    print(f"  executable    {sys.executable}")
    print(f"  numpy         {np.__version__}")
    print(f"  gymnasium     {gym.__version__}")

    try:
        import torch

        print(f"  torch         {torch.__version__}")
        print(f"  torch.cuda    available={torch.cuda.is_available()}", end="")
        if torch.cuda.is_available():
            print(f"  device={torch.cuda.get_device_name(0)}")
        else:
            print()
    except ImportError:
        print("  torch         NOT INSTALLED")

    try:
        import Box2D  # noqa: F401  (the import side effect is the point)

        print(f"  Box2D         {getattr(Box2D, '__version__', 'unknown')}")
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"  Box2D         NOT INSTALLED ({exc})")


def _get_max_episode_steps(env: gym.Env) -> int | None:
    """Read the time limit out of the wrapper stack rather than hardcoding it."""
    current: gym.Env | None = env
    while current is not None:
        limit = getattr(current, "_max_episode_steps", None)
        if limit is not None:
            return int(limit)
        current = getattr(current, "env", None)
    return None


def _run_episode(env: gym.Env, rng: np.random.Generator) -> EpisodeSummary:
    """Play one episode using uniformly random actions."""
    obs, _info = env.reset()
    total_reward = 0.0
    terminal_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = int(rng.integers(env.action_space.n))
        obs, reward, terminated, truncated, _info = env.step(action)
        terminal_reward = float(reward)
        total_reward += terminal_reward
        steps += 1

    return EpisodeSummary(
        return_=total_reward,
        length=steps,
        terminated=bool(terminated),
        truncated=bool(truncated),
        terminal_reward=terminal_reward,
        final_obs=tuple(float(v) for v in np.asarray(obs).ravel()),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the verification and return a process exit code."""
    parser = argparse.ArgumentParser(
        description="Verify that the LunarLander simulation works on this machine.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="random-action episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="seed for action sampling and the env")
    parser.add_argument(
        "--render",
        choices=("none", "human", "rgb_array"),
        default="none",
        help="'none' is headless; 'human' opens a window; 'rgb_array' draws offscreen",
    )
    parser.add_argument("--env-id", default="LunarLander-v3", help="Gymnasium environment id")
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    _print_versions()

    render_mode = None if args.render == "none" else args.render
    print()
    print("=" * 78)
    print(f"ENVIRONMENT  {args.env_id}  (render_mode={render_mode!r})")
    print("=" * 78)

    env = gym.make(args.env_id, render_mode=render_mode)
    exit_code = 0

    try:
        obs_space = env.observation_space
        act_space = env.action_space
        print(f"  observation_space.shape {getattr(obs_space, 'shape', None)}")
        print(f"  observation_space.dtype {getattr(obs_space, 'dtype', None)}")
        print("  observation_space.low   " + np.array2string(np.asarray(obs_space.low)))
        print("  observation_space.high  " + np.array2string(np.asarray(obs_space.high)))
        print(f"  action_space            {act_space}")

        limit = _get_max_episode_steps(env)
        print(f"  max_episode_steps       {limit}")

        # --- Assert the contract from docs/BUILD_SPEC.md section 3 ----------
        if not isinstance(obs_space, gym.spaces.Box):
            print(f"  FAIL expected a Box observation space, got {type(obs_space).__name__}")
            return 1
        if tuple(obs_space.shape) != EXPECTED_OBS_SHAPE:
            print(f"  FAIL expected observation shape {EXPECTED_OBS_SHAPE}, got {obs_space.shape}")
            exit_code = 1
        if not isinstance(act_space, gym.spaces.Discrete):
            print(f"  FAIL expected a Discrete action space, got {type(act_space).__name__}")
            return 1
        if int(act_space.n) != EXPECTED_N_ACTIONS:
            print(f"  FAIL expected {EXPECTED_N_ACTIONS} actions, got {act_space.n}")
            exit_code = 1

        print()
        print("  observation (index -> meaning)")
        for index, meaning in enumerate(OBSERVATION_DESCRIPTIONS):
            print(f"    [{index}] {meaning}")
        print("  actions")
        for action_index, description in ACTION_DESCRIPTIONS.items():
            print(f"    {action_index} -> {description}")

        sample_obs, sample_info = env.reset(seed=args.seed)
        print()
        print(f"  sample observation after reset: {np.asarray(sample_obs)}")
        print(f"  reset() info keys             : {sorted(sample_info.keys())}")

        rng = np.random.default_rng(args.seed)
        print()
        print("=" * 78)
        print(f"RANDOM-ACTION EPISODES  (n={args.episodes}, seed={args.seed})")
        print("=" * 78)
        print(
            f"  {'#':>4}  {'return':>9}  {'steps':>5}  {'term':>5}  "
            f"{'trunc':>5}  {'term_r':>7}  outcome"
        )

        summaries: list[EpisodeSummary] = []
        outcomes: list[str] = []
        disagreements = 0

        for index in range(args.episodes):
            summary = _run_episode(env, rng)
            summaries.append(summary)
            outcome, agreed = classify(summary)
            outcomes.append(outcome)
            if not agreed:
                disagreements += 1
            flag = "" if agreed else "  <- leg proxy DISAGREES"
            print(
                f"  {index:>4}  {summary.return_:>9.2f}  {summary.length:>5}  "
                f"{str(summary.terminated):>5}  {str(summary.truncated):>5}  "
                f"{summary.terminal_reward:>7.1f}  {outcome}{flag}"
            )

        returns = np.array([s.return_ for s in summaries], dtype=np.float64)
        lengths = np.array([s.length for s in summaries], dtype=np.float64)
        total = len(summaries)
        landed = outcomes.count("landed")
        crashed = outcomes.count("crashed")
        truncated_n = outcomes.count("truncated")

        print()
        print("  --- summary ---")
        print(f"  mean return          : {returns.mean():.2f}")
        print(f"  median return        : {float(np.median(returns)):.2f}")
        print(f"  std return (pop.)    : {returns.std(ddof=0):.2f}")
        print(f"  min / max return     : {returns.min():.2f} / {returns.max():.2f}")
        print(f"  mean episode length  : {lengths.mean():.1f} steps")
        print(f"  landed               : {landed}/{total}  ({100.0 * landed / total:.1f}%)")
        print(f"  crashed              : {crashed}/{total}  ({100.0 * crashed / total:.1f}%)")
        pct_truncated = 100.0 * truncated_n / total
        print(f"  truncated by time    : {truncated_n}/{total}  ({pct_truncated:.1f}%)")
        print(f"  leg-proxy disagreement: {disagreements}/{total}")

        if disagreements:
            print()
            print("  NOTE: the 'both legs touching' rule disagreed with the terminal")
            print("        reward on at least one episode. A lander that crashed onto")
            print("        its fuselage can also be resting on its legs, so leg contact")
            print("        alone is NOT a landing detector. The terminal reward is")
            print("        authoritative; see docs/BUILD_SPEC.md section 4.8.")

        print()
        print("  Flag semantics:")
        print("    terminated=True  -> genuine terminal state (crash or safe landing).")
        print("    truncated=True   -> hit the time limit; the world did not actually end.")
        print("    Both end the episode. Only `terminated` stops the Bellman bootstrap.")

        if not any(s.terminated or s.truncated for s in summaries):
            print("  FAIL no episode ever ended; the loop guard is suspect")
            exit_code = 1

    finally:
        env.close()
        # close() must be safely repeatable.
        env.close()

    print()
    print("VERIFY SETUP OK" if exit_code == 0 else "VERIFY SETUP FAILED")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
