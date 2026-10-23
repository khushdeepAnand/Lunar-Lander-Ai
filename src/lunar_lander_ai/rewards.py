"""Optional potential-difference shaping; baseline uses unmodified native reward."""

import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    """Add the change in a bounded potential, not a per-step survival bonus."""

    def __init__(self, env: gym.Env, config: dict, gamma: float):
        """Keep native and additional rewards separately inspectable."""
        super().__init__(env)
        self.weight = float(config["angle_penalty_weight"])
        self.gamma = gamma
        self.previous = 0.0

    def potential(self, obs) -> float:
        """Penalize off-centre, tilted states via a bounded potential."""
        return -self.weight * (min(abs(float(obs[0])), 1) + min(abs(float(obs[4])), 1))

    def reset(self, **kwargs):
        """Reset potential for the new episode."""
        obs, info = self.env.reset(**kwargs)
        self.previous = self.potential(obs)
        return obs, info

    def step(self, action):
        """Use discounted potential difference; zero bonus at real terminal states."""
        obs, native, terminated, truncated, info = self.env.step(action)
        current = 0.0 if terminated else self.potential(obs)
        bonus = self.gamma * current - self.previous
        self.previous = current
        return obs, native + bonus, terminated, truncated, {
            **info, "native_reward": native, "shaping_bonus": bonus,
        }
