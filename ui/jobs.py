"""One local training subprocess with disk-backed progress and graceful stop."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from ui.data import ROOT, read_json

from lunar_lander_ai.config import load_config
from lunar_lander_ai.train import device_for


class TrainingJobs:
    """Own one process for this local dashboard server."""

    def __init__(self, root: Path = ROOT):
        """Initialize without launching work."""
        self.root = root
        self.process = None
        records = sorted((root / "logs").glob("run_ui_*.json"), reverse=True)
        self.run_id = records[0].stem.removeprefix("run_") if records else None
        self.log = None

    @staticmethod
    def alive(pid: int | None) -> bool:
        """Check PID existence; the recorded start time remains visible for PID-reuse diagnosis."""
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False

    def active_on_disk(self) -> bool:
        """Ignore damaged old metadata but block a second live dashboard worker."""
        for path in (self.root / "logs").glob("run_ui_*.json"):
            try:
                info = read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if info.get("status") == "running" and self.alive(info.get("pid")):
                return True
        return False

    def start(self, overrides: dict) -> str:
        """Validate before spawning; prevent competing dashboard runs."""
        if self.status()["status"] == "running" or self.active_on_disk():
            raise RuntimeError("A dashboard training job is already running")
        config = load_config(ROOT / "configs/default.yaml", overrides).values
        device_for(config["device"])
        run_id = "ui_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        logs = self.root / "logs"
        logs.mkdir(exist_ok=True)
        config["paths"] = {"log_dir": str(logs),
                            "checkpoint_dir": str(self.root / "checkpoints"),
                            "results_dir": str(self.root / "results")}
        config_file = logs / f"job_{run_id}.yaml"
        config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
        output = logs / f"output_{run_id}.txt"
        with output.open("w", encoding="utf-8") as file:
            self.process = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(config_file),
                 "--run-id", run_id], cwd=self.root, stdout=file, stderr=subprocess.STDOUT,
            )
        self.run_id, self.log = run_id, output
        return run_id

    def status(self) -> dict:
        """Report process and persisted metadata; errors remain readable after exit."""
        if self.run_id is None:
            return {"status": "idle"}
        path = self.root / "logs" / f"run_{self.run_id}.json"
        try:
            info = read_json(path) if path.exists() else {}
        except (OSError, ValueError, json.JSONDecodeError):
            info = {}
        code = self.process.poll() if self.process is not None else None
        status = info.get("status", "running" if code is None else "failed")
        pid = self.process.pid if self.process is not None else info.get("pid")
        if code is None and status == "running" and not (
            self.process is not None or self.alive(pid)
        ):
            status = "stale/orphaned"
        if code is None and self.process is not None:
            status = "running"
        if code == 0 and status == "running":
            status = "failed"  # worker exited without finalizing its metadata
        if code not in (None, 0):
            status = "failed"
        return {**info, "status": status, "run_id": self.run_id,
                "pid": pid, "returncode": code}

    def stop(self) -> None:
        """Ask the worker to finish its current episode then checkpoint and exit."""
        if self.status()["status"] == "running":
            (self.root / "logs" / f"stop_{self.run_id}").touch()
