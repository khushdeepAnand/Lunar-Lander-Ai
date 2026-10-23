"""Read project-owned experiment artifacts; never deserialize arbitrary uploads."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_in_record(record: dict) -> Path | None:
    """Interpret archived Windows-style paths on every supported host OS."""
    value = record.get("checkpoint")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def comparable_reward_runs(first: dict, second: dict) -> bool:
    """Require identical recorded methods except the experimental shaping switch."""
    keys = ("seed", "device", "agent", "environment", "training")
    return (first.get("reward_shaping", {}).get("enabled") is False
            and second.get("reward_shaping", {}).get("enabled") is True
            and all(first.get(key) == second.get(key) for key in keys))


def configuration_differences(configs: dict[str, dict]) -> pd.DataFrame:
    """List only leaf settings whose recorded values differ between selected runs."""
    def flatten(value: dict, prefix: str = "") -> dict:
        result = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            result.update(flatten(item, name) if isinstance(item, dict) else {name: item})
        return result

    flat = {run: flatten(config) for run, config in configs.items()}
    rows = []
    for key in sorted(set().union(*(set(f) for f in flat.values()))):
        values = {run: str(settings.get(key, "—")) for run, settings in flat.items()}
        if len(set(values.values())) > 1 and not key.startswith("paths."):
            rows.append({"Setting": key, **values})
    return pd.DataFrame(rows)


def artifacts(folder: str, pattern: str) -> list[Path]:
    """Return matching project-local artifacts, newest first."""
    return sorted((ROOT / folder).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def read_json(path: Path) -> dict:
    """Read a local JSON object; reject malformed or non-object records."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Not a JSON object: {path.name}")
    return data


def table(path: Path) -> pd.DataFrame:
    """Read a CSV, tolerating an in-progress empty log."""
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError, OSError):
        return pd.DataFrame()


def wilson(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Expected 0 <= successes <= total and total > 0")
    z = 1.96
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * (p * (1 - p) / total + z * z / (4 * total * total)) ** .5
    radius *= 1 / (1 + z * z / total)
    return centre - radius, centre + radius


def checkpoints() -> list[Path]:
    """List only bundled and project-generated local models."""
    return artifacts("checkpoints", "*/*.pt") + artifacts("examples", "*.pt")


def source_run(checkpoint: Path) -> str | None:
    """Associate a showcase copy only when bytes match a real saved checkpoint."""
    if checkpoint.parent.name != "examples":
        return checkpoint.parent.name
    with checkpoint.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").digest()
    for candidate in artifacts("checkpoints", "*/best.pt"):
        with candidate.open("rb") as file:
            if hashlib.file_digest(file, "sha256").digest() == digest:
                return candidate.parent.name
    return None
