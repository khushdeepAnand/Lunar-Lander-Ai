"""Headless charts from training and evaluation CSV files."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot(log_dir: Path, out_dir: Path, window: int = 100) -> list[Path]:
    """Plot available metrics; tolerate short or incomplete runs."""
    files = sorted(log_dir.glob("train_*.csv"))
    if not files:
        raise FileNotFoundError(f"No training CSV files in {log_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for file in files:
        df = pd.read_csv(file)
        if df.empty:
            continue
        episode = df["episode"]

        def chart(name, ylabel, raw, rolling=None, *, episode=episode, file=file):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(episode, raw, alpha=.35 if rolling is not None else 1,
                    linewidth=1, label="Raw")
            if rolling is not None:
                ax.plot(episode, rolling, linewidth=2, label=f"Rolling mean ({window})")
            ax.set(title=name.replace("_", " ").title(), xlabel="Training episode", ylabel=ylabel)
            ax.grid(alpha=.2)
            ax.legend()
            fig.tight_layout()
            path = out_dir / f"{file.stem}_{name}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)

        if "native_return" in df:
            chart("episode_return", "Native return", df["native_return"])
            chart("rolling_return", "Native return", df["native_return"],
                  df["native_return"].rolling(window, min_periods=1).mean())
        if "epsilon" in df:
            chart("epsilon_decay", "Exploration probability", df["epsilon"])
        if "mean_loss" in df and df["mean_loss"].notna().any():
            chart("loss", "Mean training loss", df["mean_loss"],
                  df["mean_loss"].rolling(window, min_periods=1).mean())
        evaluation = log_dir / file.name.replace("train_", "eval_")
        if evaluation.is_file():
            ev = pd.read_csv(evaluation)
            if not ev.empty and "landed_rate" in ev:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(ev["episode"], ev["landed_rate"], marker="o", label="Safe landings")
                n_eval = int(ev["episodes"].iloc[0]) if "episodes" in ev else "unknown"
                ax.set(title=f"Evaluation Landing Rate ({n_eval} episodes/checkpoint)",
                       xlabel="Training episode",
                       ylabel="Landing fraction (independent evaluation episodes)", ylim=(0, 1))
                ax.legend()
                fig.tight_layout()
                path = out_dir / f"{file.stem}_eval_success_rate.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
                paths.append(path)
    # Draw this chart only if a native and a shaped experiment both exist.
    comparable = {}
    for file in files:
        metadata = log_dir / file.name.replace("train_", "config_").replace(".csv", ".json")
        if not metadata.is_file():
            continue
        config = json.loads(metadata.read_text(encoding="utf-8"))
        kind = "shaped" if config["reward_shaping"]["enabled"] else "native"
        key = (config["training"]["episodes"], config["seed"],
               config["agent"]["epsilon"]["decay_steps"])
        comparable.setdefault(key, {})[kind] = file
    pairs = [(key, runs) for key, runs in comparable.items()
             if "native" in runs and "shaped" in runs]
    if pairs:
        (budget, seed, _), by_type = sorted(pairs, key=lambda pair: pair[0][0])[-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        for kind, file in by_type.items():
            df = pd.read_csv(file)
            if not df.empty and "native_return" in df:
                ax.plot(df["episode"], df["native_return"].rolling(
                    window, min_periods=1).mean(), label=f"{kind}: rolling native return")
        ax.set(title=f"Native vs Shaped Training ({budget} episodes, seed {seed})",
               xlabel="Training episode",
               ylabel="Rolling mean native return")
        ax.legend()
        fig.tight_layout()
        comparison = out_dir / "experiment_comparison.png"
        fig.savefig(comparison, dpi=150)
        plt.close(fig)
        paths.append(comparison)
    return paths
