"""Load and validate the experiment configuration."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    """Validated configuration loaded from YAML."""

    values: dict

    def __getitem__(self, key: str):
        """Access one configuration section."""
        return self.values[key]


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load YAML and apply dotted-key CLI overrides."""
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for key, value in (overrides or {}).items():
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown configuration key: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"Unknown configuration key: {key}")
        node[parts[-1]] = value
    agent = data["agent"]
    eps = agent["epsilon"]
    checks = {
        "learning_rate": agent["learning_rate"] > 0,
        "gamma": 0 < agent["gamma"] <= 1,
        "batch_size": agent["batch_size"] > 0,
        "buffer_capacity": agent["buffer_capacity"] >= agent["batch_size"],
        "learning_starts": agent["learning_starts"] >= agent["batch_size"],
        "target_update_freq": agent["target_update_freq"] > 0,
        "train_frequency": agent["train_frequency"] > 0,
        "grad_clip_norm": agent["grad_clip_norm"] > 0,
        "hidden_sizes": bool(agent["hidden_sizes"])
        and all(isinstance(n, int) and n > 0 for n in agent["hidden_sizes"]),
        "epsilon": 0 <= eps["end"] <= eps["start"] <= 1 and eps["decay_steps"] > 0,
        "episodes": data["training"]["episodes"] > 0,
        "eval_episodes": data["training"]["eval_episodes"] > 0,
        "max_episode_steps": data["environment"]["max_episode_steps"] > 0,
        "angle_penalty_weight": data["reward_shaping"]["angle_penalty_weight"] >= 0,
    }
    for name, valid in checks.items():
        if not valid:
            raise ValueError(f"Invalid configuration: {name}")
    if data["environment"]["continuous"]:
        raise ValueError("DQN requires discrete actions; continuous must be false")
    if agent["loss"] not in ("smooth_l1", "mse"):
        raise ValueError("loss must be smooth_l1 or mse")
    if data["reward_shaping"]["enabled"] and not data["reward_shaping"]["angle_penalty_weight"]:
        raise ValueError("Enabled reward shaping requires a positive angle_penalty_weight")
    if data["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    return Config(data)
