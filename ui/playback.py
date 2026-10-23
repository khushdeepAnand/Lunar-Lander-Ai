"""Single-session, one-step-at-a-time Gymnasium playback."""

from functools import lru_cache
from pathlib import Path

import torch

from lunar_lander_ai.checkpoint import load
from lunar_lander_ai.environment import make_env
from lunar_lander_ai.metrics import classify

ACTIONS = ("No engine", "Left orientation engine", "Main engine", "Right orientation engine")


@lru_cache(maxsize=4)
def trusted_policy(checkpoint: Path, modified_ns: int):
    """Cache a trusted, immutable CPU policy until its local file changes."""
    return load(checkpoint, torch.device("cpu"))


class Playback:
    """A real environment and greedy policy; each call advances exactly one step."""

    def __init__(self, checkpoint: Path, seed: int):
        """Load a trusted local checkpoint and create an RGB-rendering environment."""
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        self.agent, payload = trusted_policy(checkpoint, checkpoint.stat().st_mtime_ns)
        if (payload["config"]["environment"]["id"] != "LunarLander-v3"
                or self.agent.n_actions != 4
                or self.agent.policy.net[0].in_features != 8):
            raise ValueError("Checkpoint needs LunarLander-v3 with 8 observations and 4 actions")
        self.env = make_env(payload["config"]["environment"], "rgb_array")
        try:
            self.obs, _ = self.env.reset(seed=seed)
            self.frame = self.env.render()
        except Exception:
            self.env.close()
            raise
        self.steps, self.total, self.reward = 0, 0.0, 0.0
        self.action = None
        self.q_values = None
        self.outcome = None
        self.trail = [self.obs.copy()]  # actual observed states of this episode

    def step(self) -> None:
        """Select the actual action, apply it, and record native telemetry."""
        if self.outcome is not None:
            return
        with torch.no_grad():
            decision_state = torch.as_tensor(self.obs, dtype=torch.float32,
                                             device=self.agent.device)
            self.q_values = self.agent.policy(decision_state).cpu().tolist()
        self.action = int(max(range(4), key=self.q_values.__getitem__))
        self.obs, self.reward, terminated, truncated, _ = self.env.step(self.action)
        self.frame = self.env.render()
        self.steps += 1
        self.total += self.reward
        self.trail.append(self.obs.copy())
        if terminated or truncated:
            self.outcome = classify(terminated, self.reward, self.obs)
            self.close()

    def close(self) -> None:
        """Release the Box2D/Pygame resources."""
        self.env.close()
