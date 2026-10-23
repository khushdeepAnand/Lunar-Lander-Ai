"""Focused tests for local UI integrations and real environment playback."""

import json
import os

import gymnasium as gym
import pytest
import torch
from streamlit.testing.v1 import AppTest
from ui.data import (
    ROOT,
    checkpoint_in_record,
    checkpoints,
    comparable_reward_runs,
    configuration_differences,
    source_run,
    table,
    wilson,
)
from ui.jobs import TrainingJobs
from ui.playback import Playback

from lunar_lander_ai.checkpoint import load
from lunar_lander_ai.config import load_config
from lunar_lander_ai.environment import make_env
from lunar_lander_ai.evaluate import evaluate
from lunar_lander_ai.train import train


def test_dashboard_starts_with_real_artifacts():
    """The landing page renders without inventing values or starting training."""
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Mission overview"
    assert checkpoints()


def test_overview_opens_real_simulation():
    """The primary call to action switches to the actual simulation page."""
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.title[0].value == "Live simulation"


def test_overview_empty_state(monkeypatch):
    """An empty archive never invents a checkpoint or training metrics."""
    monkeypatch.setattr("ui.data.artifacts", lambda *_args: [])
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert any(metric.label == "Selected model" and metric.value == "None"
               for metric in app.metric)
    assert any("Select a trusted local checkpoint" in message.value for message in app.info)


def test_wilson_small_sample():
    """A measured 8/10 is not presented as a certain 80% population rate."""
    low, high = wilson(8, 10)
    assert low < .8 < high
    with pytest.raises(ValueError):
        wilson(11, 10)


def test_archived_checkpoint_paths_and_comparison(tmp_path):
    """Archived Windows paths still match locally; controls differ only by shaping."""
    record = {"checkpoint": r"examples\best_native.pt"}
    assert checkpoint_in_record(record) == ROOT / "examples/best_native.pt"
    a = {"seed": 42, "agent": {"gamma": .99}, "environment": {"id": "LunarLander-v3"},
         "training": {"episodes": 300}, "device": "cpu", "reward_shaping": {"enabled": False}}
    b = {**a, "reward_shaping": {"enabled": True}}
    assert comparable_reward_runs(a, b)
    assert not comparable_reward_runs(a, {**b, "seed": 7})
    broken = tmp_path / "broken.csv"
    broken.write_text("a,b\n1,\"unterminated\n", encoding="utf-8")
    assert table(broken).empty
    assert source_run(ROOT / "examples/best_native.pt") == "20261015T163534138250Z"
    diff = configuration_differences({"native": a, "shaped": b})
    assert list(diff.Setting) == ["reward_shaping.enabled"]


def test_actual_playback_closes_environment():
    """A bundled checkpoint drives a real rendered step, with valid telemetry."""
    playback = Playback(ROOT / "examples/best_native.pt", 40400)
    try:
        assert playback.obs.shape == (8,)
        assert playback.frame.shape[2] == 3
        playback.step()
        assert playback.action in range(4)
        assert len(playback.q_values) == 4
        assert playback.action == max(range(4), key=playback.q_values.__getitem__)
        assert playback.steps == 1
        assert playback.total == playback.reward
    finally:
        playback.close()


def test_invalid_checkpoint_has_a_clear_error(tmp_path):
    """An absent model must not create a half-initialized environment."""
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        Playback(tmp_path / "missing.pt", 42)


def test_stop_preserves_metadata_and_rejects_invalid_config(tmp_path):
    """An already-requested stop leaves traceable status without running an episode."""
    config = load_config(ROOT / "configs/default.yaml").values
    config["paths"] = {"log_dir": str(tmp_path / "logs"),
                       "checkpoint_dir": str(tmp_path / "checkpoints"),
                       "results_dir": str(tmp_path / "results")}
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "stop_controlled").touch()
    outcome = train(config, run_id="controlled")
    assert outcome["episodes"] == 0
    meta = json.loads((tmp_path / "logs" / "run_controlled.json").read_text())
    assert meta["status"] == "stopped"
    assert not (tmp_path / "logs" / "stop_controlled").exists()
    manager = TrainingJobs()
    with pytest.raises(ValueError, match="epsilon"):
        manager.start({"agent.epsilon.end": 2.0})
    assert manager.process is None


def test_worker_start_failure_stays_visible(tmp_path, monkeypatch):
    """Even environment creation failures leave config, status and error on disk."""
    config = load_config(ROOT / "configs/default.yaml").values
    config["paths"] = {"log_dir": str(tmp_path / "logs"),
                       "checkpoint_dir": str(tmp_path / "checkpoints"),
                       "results_dir": str(tmp_path / "results")}

    def broken_environment(*_args, **_kwargs):
        raise RuntimeError("Box2D unavailable")

    monkeypatch.setattr("lunar_lander_ai.train.make_env", broken_environment)
    with pytest.raises(RuntimeError, match="Box2D unavailable"):
        train(config, run_id="failed")
    status = json.loads((tmp_path / "logs/run_failed.json").read_text())
    assert status["status"] == "failed" and "Box2D" in status["error"]
    assert (tmp_path / "logs/config_failed.json").exists()


@pytest.mark.slow
def test_stop_after_completed_episode_saves_valid_checkpoint(tmp_path, monkeypatch):
    """A stop during flight waits for the boundary, flushes CSV, then saves weights."""
    config = load_config(ROOT / "configs/default.yaml", {
        "training.episodes": 3, "training.checkpoint_interval": 0,
        "training.eval_interval": 0, "environment.max_episode_steps": 2, "device": "cpu",
    }).values
    config["paths"] = {"log_dir": str(tmp_path / "logs"),
                       "checkpoint_dir": str(tmp_path / "checkpoints"),
                       "results_dir": str(tmp_path / "results")}
    class StopOnBoundary(gym.Wrapper):
        def step(self, action):
            result = self.env.step(action)
            if result[2] or result[3]:
                (tmp_path / "logs/stop_stopped").touch()
            return result

    monkeypatch.setattr("lunar_lander_ai.train.make_env",
                        lambda cfg, render: StopOnBoundary(make_env(cfg, render)))
    result = train(config, run_id="stopped")
    assert result["episodes"] == len(table(tmp_path / "logs/train_stopped.csv")) == 1
    assert (tmp_path / "checkpoints/stopped/final.pt").is_file()
    assert not (tmp_path / "logs/stop_stopped").exists()
    assert json.loads((tmp_path / "logs/run_stopped.json").read_text())["status"] == "stopped"


@pytest.mark.slow
def test_periodic_evaluation_is_written(tmp_path):
    """A real one-episode training run publishes periodic evaluation and checkpoints."""
    config = load_config(ROOT / "configs/default.yaml", {
        "training.episodes": 1, "training.eval_interval": 1,
        "training.eval_episodes": 1, "training.checkpoint_interval": 1,
        "environment.max_episode_steps": 30, "device": "cpu",
    }).values
    config["paths"] = {"log_dir": str(tmp_path / "logs"),
                       "checkpoint_dir": str(tmp_path / "checkpoints"),
                       "results_dir": str(tmp_path / "results")}
    train(config, run_id="periodic")
    assert len(table(tmp_path / "logs/train_periodic.csv")) == 1
    evaluation = table(tmp_path / "logs/eval_periodic.csv")
    assert len(evaluation) == 1 and evaluation.iloc[0].episodes == 1
    assert (tmp_path / "checkpoints/periodic/latest.pt").exists()


def test_job_failure_is_reported(monkeypatch):
    """Nonzero exit is a failure, not a completed run."""
    manager = TrainingJobs()

    class FailedProcess:
        pid = 123

        def poll(self):
            return 1

    manager.process = FailedProcess()
    manager.run_id = "absent_test_run"
    assert manager.status()["status"] == "failed"


def test_orphaned_worker_after_server_restart(tmp_path):
    """Persisted running records with a dead PID are not claimed as live work."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run_ui_2026.json").write_text(
        json.dumps({"status": "running", "pid": 2_147_483_000,
                    "started_utc": "2026-10-18T00:00:00+00:00"}), encoding="utf-8")
    assert TrainingJobs(tmp_path).status()["status"] == "stale/orphaned"
    (logs / "run_ui_2027.json").write_text(
        json.dumps({"status": "running", "pid": os.getpid(),
                    "started_utc": "2026-10-18T00:00:00+00:00"}), encoding="utf-8")
    assert TrainingJobs(tmp_path).status()["status"] == "running"


@pytest.mark.slow
def test_real_failure_gallery_frames_from_environment():
    """Frame strips only use real rendered states from an actual failed rollout."""
    agent, payload = load(ROOT / "examples/best_native.pt", torch.device("cpu"))
    config = json.loads(json.dumps(payload["config"]))
    config["environment"]["max_episode_steps"] = 3
    failures = []
    rows, _ = evaluate(agent, config, 1, 40400, render_mode="rgb_array",
                       failure_callback=lambda row, frames: failures.append((row, frames)),
                       failure_limit=1)
    assert rows[0]["outcome"] == "truncated"
    assert failures[0][0]["seed"] == 40400
    assert failures[0][1][-1].shape[2] == 3


@pytest.mark.slow
def test_worker_completes_real_short_run(tmp_path):
    """A subprocess writes valid metadata and a streaming training CSV."""
    manager = TrainingJobs(tmp_path)
    run = manager.start({"training.episodes": 1, "training.eval_interval": 999,
                         "training.checkpoint_interval": 999, "device": "cpu"})
    try:
        manager.process.wait(timeout=60)
        status = manager.status()
        assert status["status"] == "completed", manager.log.read_text(errors="replace")
        assert status["episodes_completed"] == 1
        assert (tmp_path / "logs" / f"train_{run}.csv").exists()
        assert status["pid"] == manager.process.pid
        assert status["started_utc"]
    finally:
        if manager.process.poll() is None:
            manager.process.terminate()
