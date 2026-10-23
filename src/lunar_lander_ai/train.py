"""Training loop, progress logs and periodic evaluation."""

import csv
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import gymnasium
import numpy as np
import torch

from lunar_lander_ai.agent import DQNAgent
from lunar_lander_ai.checkpoint import save
from lunar_lander_ai.environment import make_env
from lunar_lander_ai.evaluate import evaluate
from lunar_lander_ai.metrics import classify
from lunar_lander_ai.replay_buffer import ReplayBuffer
from lunar_lander_ai.rewards import RewardShapingWrapper


def device_for(name: str) -> torch.device:
    """Select CPU or CUDA, failing explicitly on unsupported requests."""
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    use_cuda = name == "cuda" or (name == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def epsilon_at(step: int, config: dict) -> float:
    """Linear epsilon schedule based on environment interactions."""
    progress = min(step / config["decay_steps"], 1)
    return config["start"] + progress * (config["end"] - config["start"])


def train(config: dict, smoke: bool = False, run_id: str | None = None) -> dict:
    """Train a baseline DQN and log real episodes and evaluations."""
    config = json.loads(json.dumps(config))  # independent copy; never mutate the caller's config
    if smoke:
        config["training"].update(episodes=2, checkpoint_interval=0, eval_interval=0,
                                  render=False, early_stop=False)
        config["agent"].update(batch_size=32, learning_starts=32, buffer_capacity=2000)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)  # tiny MLP: threading overhead otherwise dominates
    tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    name = run_id or f"{'smoke_' if smoke else ''}{tag}"
    logs = Path(config["paths"]["log_dir"])
    checkpoints = Path(config["paths"]["checkpoint_dir"]) / name
    logs.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    (logs / f"config_{name}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metadata = {
        "started_utc": datetime.now(UTC).isoformat(), "run_id": name,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "gymnasium": gymnasium.__version__, "numpy": np.__version__,
        "seed": seed, "device": config["device"],
        "shaping_enabled": config["reward_shaping"]["enabled"], "status": "running",
        "pid": os.getpid(),
    }
    meta_path = logs / f"run_{name}.json"
    train_path = logs / f"train_{name}.csv"
    eval_path = logs / f"eval_{name}.csv"
    stop_path = logs / f"stop_{name}"
    metadata.update(config_path=str(logs / f"config_{name}.json"),
                    train_log=str(train_path), evaluation_log=str(eval_path),
                    checkpoint_dir=str(checkpoints))
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    steps, best, rows, eval_rows = 0, float("-inf"), [], []
    error = None
    stopped = False
    env = None
    try:
        device = device_for(config["device"])
        metadata["device"] = str(device)
        env = make_env(config["environment"], "human" if config["training"]["render"] else None)
        if config["reward_shaping"]["enabled"]:
            env = RewardShapingWrapper(env, config["reward_shaping"], config["agent"]["gamma"])
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                         config["agent"], device, seed)
        replay = ReplayBuffer(config["agent"]["buffer_capacity"], env.observation_space.shape[0],
                              seed, device)
        for episode in range(1, config["training"]["episodes"] + 1):
            if stop_path.exists():
                stopped = True
                break
            started = time.perf_counter()
            obs, _ = env.reset(seed=seed if episode == 1 else None)
            total = native = bonus = 0.0
            losses = []
            length = 0
            while True:
                epsilon = epsilon_at(steps, config["agent"]["epsilon"])
                action = agent.select_action(obs, epsilon)
                next_obs, reward, terminated, truncated, info = env.step(action)
                replay.add(obs, action, reward, next_obs, terminated, truncated)
                total += reward
                native_reward = info.get("native_reward", reward)
                native += native_reward
                bonus += info.get("shaping_bonus", 0.0)
                steps += 1
                length += 1
                if len(replay) >= config["agent"]["learning_starts"] and (
                    steps % config["agent"]["train_frequency"] == 0
                ):
                    losses.append(agent.learn(replay.sample(config["agent"]["batch_size"])))
                obs = next_obs
                if terminated or truncated:
                    break
            row = {
                "episode": episode, "env_steps": steps, "native_return": native,
                "training_return": total, "shaping_bonus": bonus, "length": length,
                "outcome": classify(terminated, native_reward, obs),
                "terminated": terminated, "truncated": truncated,
                "epsilon": epsilon_at(steps, config["agent"]["epsilon"]),
                "mean_loss": mean(losses) if losses else "", "updates": agent.updates,
                "rolling_return": mean(r["native_return"] for r in rows[-99:] + [
                    {"native_return": native}
                ]),
                "duration_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            # Flush each completed episode so a separate UI process can show real progress.
            with train_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())
                if episode == 1:
                    writer.writeheader()
                writer.writerow(row)
            if stop_path.exists():
                stopped = True
                break
            interval = config["training"]["checkpoint_interval"]
            if interval and episode % interval == 0:
                save(checkpoints / "latest.pt", agent, config, episode, steps)
                save(checkpoints / f"episode_{episode:04d}.pt", agent, config, episode, steps)
            interval = config["training"]["eval_interval"]
            if interval and episode % interval == 0:
                _, stats = evaluate(agent, config, config["training"]["eval_episodes"],
                                     seed=20_260)
                evaluation = {"episode": episode, **stats}
                eval_rows.append(evaluation)
                with eval_path.open("a", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=evaluation.keys())
                    if len(eval_rows) == 1:
                        writer.writeheader()
                    writer.writerow(evaluation)
                if stats["mean_return"] > best:
                    best = stats["mean_return"]
                    save(checkpoints / "best.pt", agent, config, episode, steps)
                agent.policy.train()
            if episode % config["training"]["log_interval"] == 0 or smoke:
                print(f"episode={episode} return={native:.1f} epsilon={row['epsilon']:.3f} "
                      f"loss={row['mean_loss']} outcome={row['outcome']}")
            if (config["training"]["early_stop"] and episode >= config["training"]["solved_window"]
                    and row["rolling_return"] >= config["training"]["solved_threshold"]
                    and eval_rows and eval_rows[-1]["episode"] == episode
                    and eval_rows[-1]["landed_rate"] >= 0.8):
                break
    except KeyboardInterrupt:
        stopped = True
        print("Interrupted: saving completed episodes and final checkpoint.")
    except Exception as exc:
        error = exc
    finally:
        if env is not None:
            env.close()
        try:
            if rows:
                save(checkpoints / "final.pt", agent, config, rows[-1]["episode"], steps)
                save(checkpoints / "latest.pt", agent, config, rows[-1]["episode"], steps)
        except Exception as exc:
            if error is None:
                error = exc
        metadata.update(finished_utc=datetime.now(UTC).isoformat(),
                        episodes_completed=len(rows), steps_completed=steps,
                        final_epsilon=epsilon_at(steps, config["agent"]["epsilon"]),
                        status="failed" if error else "stopped" if stopped else "completed",
                        error=str(error) if error else None)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        stop_path.unlink(missing_ok=True)
    if error:
        raise error
    print(f"run={name} episodes={len(rows)} steps={steps} checkpoints={checkpoints}")
    if smoke:
        print("SMOKE TEST OK")
    return {"name": name, "episodes": len(rows), "steps": steps, "checkpoints": str(checkpoints)}
