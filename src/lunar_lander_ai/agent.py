"""Vanilla Deep Q-Network learning algorithm."""

import numpy as np
import torch
from torch import nn

from lunar_lander_ai.network import QNetwork
from lunar_lander_ai.replay_buffer import Batch


class DQNAgent:
    """Online and frozen target networks with epsilon-greedy control."""

    def __init__(self, obs_dim: int, n_actions: int, config: dict, device: torch.device, seed: int):
        """Build networks and optimise only the online network."""
        self.device, self.n_actions, self.config = device, n_actions, config
        self.policy = QNetwork(obs_dim, n_actions, config["hidden_sizes"]).to(device)
        self.target = QNetwork(obs_dim, n_actions, config["hidden_sizes"]).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config["learning_rate"])
        self.loss_fn = nn.SmoothL1Loss() if config["loss"] == "smooth_l1" else nn.MSELoss()
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """Explore with probability epsilon, otherwise take argmax Q."""
        if np.asarray(obs).shape != (self.policy.net[0].in_features,):
            raise ValueError("Observation has the wrong shape")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return int(self.policy(tensor).argmax().item())

    def compute_target(self, batch: Batch) -> torch.Tensor:
        """Bootstrap after a truncation, never after genuine termination."""
        with torch.no_grad():
            next_q = self.target(batch.next_obs).max(dim=1).values
            return batch.rewards + self.config["gamma"] * (1 - batch.terminated) * next_q

    def learn(self, batch: Batch) -> float:
        """Apply one clipped Huber-loss gradient update and periodic hard sync."""
        q = self.policy(batch.obs).gather(1, batch.actions[:, None]).squeeze(1)
        loss = self.loss_fn(q, self.compute_target(batch))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["grad_clip_norm"])
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config["target_update_freq"] == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())
