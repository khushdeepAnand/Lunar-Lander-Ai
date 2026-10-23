"""Create the discrete Gymnasium LunarLander environment."""

import gymnasium as gym


def make_env(config: dict, render_mode: str | None = None) -> gym.Env:
    """Create an environment and reject incompatible action or observation spaces."""
    if config["continuous"]:
        raise ValueError("DQN requires discrete actions; continuous=True is unsupported")
    env = gym.make(
        config["id"],
        continuous=False,
        render_mode=render_mode,
        max_episode_steps=config["max_episode_steps"],
    )
    if not isinstance(env.observation_space, gym.spaces.Box) or env.observation_space.shape != (8,):
        env.close()
        raise ValueError(f"Expected Box(8,) observation, got {env.observation_space}")
    if not isinstance(env.action_space, gym.spaces.Discrete) or env.action_space.n != 4:
        env.close()
        raise ValueError(f"Expected Discrete(4) actions, got {env.action_space}")
    return env
