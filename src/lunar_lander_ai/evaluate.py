"""Independent greedy evaluation of saved or in-memory DQN policies."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from lunar_lander_ai.environment import make_env
from lunar_lander_ai.metrics import classify, summarise


def evaluate(agent, config: dict, episodes: int, seed: int, render_mode=None,
             frame_callback=None, failure_callback=None,
             failure_limit=0) -> tuple[list[dict], dict]:
    """Evaluate native returns without exploration or network updates."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = make_env(config["environment"], render_mode)
    rows = []
    agent.policy.eval()
    failures = 0
    try:
        for index in range(episodes):
            obs, _ = env.reset(seed=seed + index)
            total, length, ended, terminal_reward = 0.0, 0, False, 0.0
            frames = []
            while not ended:
                with torch.no_grad():
                    action = agent.select_action(obs, epsilon=0)
                obs, reward, terminated, truncated, _ = env.step(action)
                if frame_callback is not None:
                    frame_callback(env.render())
                if failure_callback is not None and failures < failure_limit and (
                    length % 50 == 0 or terminated or truncated
                ):
                    frames.append(env.render())
                    frames = frames[-6:]
                total += reward
                terminal_reward = reward
                length += 1
                ended = terminated or truncated
            row = {
                "episode": index, "seed": seed + index, "native_return": total,
                "length": length, "terminated": terminated, "truncated": truncated,
                "terminal_reward": terminal_reward,
                "outcome": classify(terminated, terminal_reward, obs),
                "legs_down": bool(obs[6] > .5 and obs[7] > .5),
            }
            rows.append(row)
            if failure_callback is not None and row["outcome"] != "landed" and (
                failures < failure_limit
            ):
                failure_callback(row, frames)
                failures += 1
    finally:
        env.close()
    return rows, summarise(rows)


def save_results(rows: list[dict], summary: dict, output_dir: Path, label: str):
    """Write per-episode CSV and summary JSON under unique timestamped paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    stem = output_dir / f"eval_{label}_{tag}"
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return stem
