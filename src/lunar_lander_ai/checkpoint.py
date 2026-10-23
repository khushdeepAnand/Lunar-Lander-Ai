"""Atomic, versioned checkpoint persistence."""

import codecs
import random
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import gymnasium
import numpy as np
import torch

from lunar_lander_ai.agent import DQNAgent

VERSION = 1


def save(path: Path, agent: DQNAgent, config: dict, episode: int, steps: int) -> None:
    """Atomically persist model, optimizer, config and training RNG states."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION, "config": config, "episode": episode, "steps": steps,
        "policy": agent.policy.state_dict(), "target": agent.target.state_dict(),
        "optimizer": agent.optimizer.state_dict(), "updates": agent.updates,
        "agent_rng": agent.rng.bit_generator.state,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "created_utc": datetime.now(UTC).isoformat(),
        "package_versions": {
            "torch": torch.__version__, "gymnasium": gymnasium.__version__,
            "numpy": np.__version__,
        },
    }
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_payload(path, device):
    """Prefer allow-listed ``weights_only`` loading; fall back for legacy NumPy RNG state.

    Legacy V1 files embed a NumPy array, which the strict loader rejects unless its few
    globals are allow-listed. If the strict attempt fails for any reason we fall back to
    full unpickling, which is acceptable ONLY for trusted, project-generated local files.
    """
    core = getattr(np, "_core", None) or np.core
    allowed = [np.ndarray, np.dtype, core.multiarray._reconstruct, codecs.encode]
    try:
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        if isinstance(path, BytesIO):
            path.seek(0)
        return torch.load(path, map_location=device, weights_only=False)


def load(path: Path | BytesIO, device: torch.device, *, optimizer: bool = False):
    """Load trusted local payload; legacy NumPy RNG state needs pickle deserialization."""
    if isinstance(path, Path) and not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = _read_payload(path, device)
        if payload.get("version") != VERSION:
            raise ValueError(f"Checkpoint version {payload.get('version')} != supported {VERSION}")
        cfg = payload["config"]
        output = f"net.{2 * len(cfg['agent']['hidden_sizes'])}.weight"
        agent = DQNAgent(payload["policy"]["net.0.weight"].shape[1],
                         payload["policy"][output].shape[0], cfg["agent"], device, cfg["seed"])
        agent.policy.load_state_dict(payload["policy"])
        agent.target.load_state_dict(payload["target"])
        agent.updates = payload["updates"]
        agent.rng.bit_generator.state = payload["agent_rng"]
        if optimizer:
            agent.optimizer.load_state_dict(payload["optimizer"])
        return agent, payload
    except (KeyError, IndexError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Incompatible checkpoint {path}: {exc}") from exc
