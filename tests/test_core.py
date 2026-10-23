"""Focused mathematical and data-handling regression tests."""

import numpy as np
import pytest
import torch

from lunar_lander_ai.agent import DQNAgent
from lunar_lander_ai.config import load_config
from lunar_lander_ai.environment import make_env
from lunar_lander_ai.network import QNetwork
from lunar_lander_ai.replay_buffer import ReplayBuffer


def test_environment():
    """Verify runtime spaces and a complete random episode."""
    cfg = load_config("configs/default.yaml")
    env = make_env(cfg["environment"])
    try:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (8,) and env.action_space.n == 4
        for _ in range(1000):
            obs, reward, term, trunc, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward)
            if term or trunc:
                break
        else:
            pytest.fail("episode exceeded the 1000 step limit")
    finally:
        env.close()


def test_network_shape_and_gradient():
    """The output is one unconstrained value per action, with gradients."""
    net = QNetwork(8, 4, [32, 16])
    assert net(torch.zeros(8)).shape == (4,)
    output = net(torch.zeros(5, 8))
    assert output.shape == (5, 4)
    output.sum().backward()
    assert all(p.grad is not None for p in net.parameters())


def test_replay_rollover_and_flags():
    """Stored samples are copies and preserve both episode-end flags."""
    buf = ReplayBuffer(3, 8, 1, torch.device("cpu"))
    with pytest.raises(ValueError):
        buf.sample(1)
    for i in range(4):
        buf.add(np.full(8, i), i % 4, float(i), np.full(8, i + 1), i == 3, i == 2)
    assert len(buf) == 3
    batch = buf.sample(3)
    assert sorted(batch.rewards.tolist()) == [1.0, 2.0, 3.0]
    assert batch.terminated.sum() == 1 and batch.truncated.sum() == 1
    buf.add(np.full(8, 99), 0, 99, np.full(8, 99), False, False)
    assert 99 not in batch.rewards.tolist()


def test_bellman_target_and_learning():
    """Truncation bootstraps; termination does not; target stays frozen."""
    cfg = load_config("configs/default.yaml")["agent"] | {"target_update_freq": 100}
    agent = DQNAgent(8, 4, cfg, torch.device("cpu"), 1)
    assert all(not p.requires_grad for p in agent.target.parameters())
    buf = ReplayBuffer(3, 8, 1, torch.device("cpu"))
    for i in range(3):
        buf.add(np.ones(8), 1, 2.0, np.ones(8), i == 0, i == 1)
    batch = buf.sample(3)
    targets = agent.compute_target(batch)
    assert not targets.requires_grad
    next_q = agent.target(batch.next_obs).max(dim=1).values.detach()
    expected = batch.rewards + 0.99 * (1 - batch.terminated) * next_q
    torch.testing.assert_close(targets, expected)
    assert targets[batch.terminated.bool()].item() == 2.0
    before = [p.detach().clone() for p in agent.policy.parameters()]
    frozen = [p.detach().clone() for p in agent.target.parameters()]
    assert np.isfinite(agent.learn(batch))
    assert any(not torch.equal(p, old)
               for p, old in zip(agent.policy.parameters(), before, strict=True))
    assert all(torch.equal(p, old)
               for p, old in zip(agent.target.parameters(), frozen, strict=True))


def test_invalid_config():
    """Reject invalid DQN hyperparameters before building an environment."""
    with pytest.raises(ValueError, match="gamma"):
        load_config("configs/default.yaml", {"agent.gamma": 2.0})


def test_target_sync_and_truncation():
    """Hard sync occurs on gradient-update count; time limit still bootstraps."""
    cfg = load_config("configs/default.yaml")["agent"] | {"target_update_freq": 2}
    agent = DQNAgent(8, 4, cfg, torch.device("cpu"), 0)
    buf = ReplayBuffer(2, 8, 0, torch.device("cpu"))
    buf.add(np.ones(8), 2, 3, np.ones(8), False, True)
    buf.add(np.ones(8), 1, 3, np.ones(8), True, False)
    batch = buf.sample(2)
    target = agent.compute_target(batch)
    truncated_value = target[batch.truncated.bool()].item()
    assert truncated_value != 3  # the truncated transition retains a bootstrap
    assert target[batch.terminated.bool()].item() == 3
    agent.learn(batch)
    agent.learn(batch)
    assert agent.updates == 2
    assert all(torch.equal(a, b) for a, b in
               zip(agent.policy.parameters(), agent.target.parameters(), strict=True))
