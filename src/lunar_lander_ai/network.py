"""Fully-connected action-value network."""

from collections.abc import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    """Map one or more observations to one Q-value per discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int]) -> None:
        """Construct the MLP without constraining its final Q-values."""
        super().__init__()
        if obs_dim < 1 or n_actions < 1 or not hidden_sizes or any(n < 1 for n in hidden_sizes):
            raise ValueError("obs_dim, n_actions and every hidden size must be positive")
        layers: list[nn.Module] = []
        previous = obs_dim
        for size in hidden_sizes:
            layers += [nn.Linear(previous, size), nn.ReLU()]
            previous = size
        layers.append(nn.Linear(previous, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict action values for a single observation or a batch."""
        return self.net(obs)
