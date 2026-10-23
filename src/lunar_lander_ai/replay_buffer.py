"""Preallocated uniform experience replay with separate episode-end flags."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    """Tensors returned by replay sampling."""

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    """Fixed-size ring buffer; time-limit truncation does not imply termination."""

    def __init__(self, capacity: int, obs_dim: int, seed: int, device: torch.device) -> None:
        """Allocate transition arrays and a private sampling RNG."""
        if capacity < 1 or obs_dim < 1:
            raise ValueError("capacity and obs_dim must be positive")
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.truncated = np.empty(capacity, dtype=np.float32)
        self.capacity, self.device = capacity, device
        self.position = self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return self.size

    def add(
        self, obs, action: int, reward: float, next_obs, terminated: bool, truncated: bool
    ) -> None:
        """Copy one transition into the ring."""
        i = self.position
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Sample independent indices, copying results onto the selected device."""
        if batch_size < 1 or batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} from buffer of size {self.size}")
        indices = self.rng.choice(self.size, batch_size, replace=False)

        def tensor(array):
            return torch.as_tensor(array[indices].copy(), device=self.device)

        return Batch(
            tensor(self.obs), tensor(self.actions), tensor(self.rewards),
            tensor(self.next_obs), tensor(self.terminated), tensor(self.truncated),
        )
