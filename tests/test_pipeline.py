"""Checkpoint, metrics, shaping, plotting and CLI integration tests."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from lunar_lander_ai.agent import DQNAgent
from lunar_lander_ai.checkpoint import load, save
from lunar_lander_ai.config import load_config
from lunar_lander_ai.environment import make_env
from lunar_lander_ai.evaluate import evaluate
from lunar_lander_ai.metrics import classify, summarise
from lunar_lander_ai.plotting import plot
from lunar_lander_ai.rewards import RewardShapingWrapper
from lunar_lander_ai.train import epsilon_at, train


def test_outcomes_and_summary():
    """Contact alone cannot turn a body crash into a landing."""
    both_legs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    assert classify(True, -100, both_legs) == "crashed"
    assert classify(True, 100, both_legs) == "landed"
    assert classify(False, 0, both_legs) == "truncated"
    assert classify(True, 100, np.array([.5, 0, 0, 0, 0, 0, 1, 1])) == "off_pad_or_unsafe"
    rows = [{"native_return": x, "length": n, "outcome": o} for x, n, o in
            [(-10., 10, "crashed"), (100., 20, "landed")]]
    stats = summarise(rows)
    assert stats["mean_return"] == 45 and stats["landed_rate"] == .5
    assert stats["crashed_rate"] == .5 and stats["mean_length"] == 15
    with pytest.raises(ValueError):
        summarise([])


def test_epsilon():
    """Exploration declines to a floor, not to zero."""
    cfg = {"start": 1., "end": .05, "decay_steps": 100}
    assert epsilon_at(0, cfg) == 1
    assert epsilon_at(50, cfg) == pytest.approx(.525)
    assert epsilon_at(200, cfg) == pytest.approx(.05)


def test_checkpoint(tmp_path):
    """Save/load preserves action values and rejects unknown formats."""
    cfg = load_config("configs/default.yaml").values
    agent = DQNAgent(8, 4, cfg["agent"], torch.device("cpu"), 42)
    path = tmp_path / "model.pt"
    save(path, agent, cfg, 7, 100)
    rebuilt, payload = load(path, torch.device("cpu"))
    torch.testing.assert_close(agent.policy(torch.ones(8)), rebuilt.policy(torch.ones(8)))
    assert payload["episode"] == 7 and payload["steps"] == 100
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.pt", torch.device("cpu"))
    payload["version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="version"):
        load(path, torch.device("cpu"))


def test_shaping_separate_and_spaces():
    """Wrapper does not change spaces and logs native plus shaping separately."""
    cfg = load_config("configs/default.yaml").values
    env = RewardShapingWrapper(make_env(cfg["environment"]),
                               {"angle_penalty_weight": 1.}, .99)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (8,) and env.action_space.n == 4
        _, reward, _, _, info = env.step(0)
        assert reward == pytest.approx(info["native_reward"] + info["shaping_bonus"])
    finally:
        env.close()


def test_plot_partial(tmp_path):
    """One-row logs lacking loss still produce usable real plots."""
    pd.DataFrame([{"episode": 1, "native_return": -100., "epsilon": 1.}]).to_csv(
        tmp_path / "train_partial.csv", index=False)
    paths = plot(tmp_path, tmp_path / "out")
    assert len(paths) == 3 and all(path.stat().st_size > 0 for path in paths)


def test_smoke_and_eval(tmp_path):
    """The two-episode training interface writes real logs and usable weights."""
    cfg = load_config("configs/default.yaml").values
    folders = ("log_dir", "checkpoint_dir", "results_dir")
    cfg["paths"] = {key: str(tmp_path / key) for key in folders}
    output = train(cfg, smoke=True)
    assert output["episodes"] == 2
    agent, saved = load(tmp_path / "checkpoint_dir" / output["name"] / "final.pt",
                        torch.device("cpu"))
    before = [p.detach().clone() for p in agent.policy.parameters()]
    rows, summary = evaluate(agent, saved["config"], 2, seed=777)
    assert len(rows) == summary["episodes"] == 2
    assert all(torch.equal(p, old)
               for p, old in zip(agent.policy.parameters(), before, strict=True))
    metadata = json.loads((tmp_path / "log_dir" / f"config_{output['name']}.json").read_text())
    assert metadata["seed"] == 42
