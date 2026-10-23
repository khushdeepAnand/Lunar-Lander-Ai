"""Evaluate a saved DQN checkpoint without updating weights."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_lander_ai.checkpoint import load  # noqa: E402
from lunar_lander_ai.evaluate import evaluate, save_results  # noqa: E402
from lunar_lander_ai.train import device_for  # noqa: E402


def main():
    """Load a checkpoint and report native-reward metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--failure-gallery", type=int, default=0, metavar="N",
                        help="Save frame strips for the first N non-landings")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not settings.get("greedy", True):
        parser.error("Evaluation only supports greedy action selection; set greedy: true")
    checkpoint = args.checkpoint or Path(settings["checkpoint"])
    episodes = args.episodes if args.episodes is not None else settings["episodes"]
    seed = args.seed if args.seed is not None else settings["seed"]
    if not 0 <= args.failure_gallery <= 10:
        parser.error("--failure-gallery must be between 0 and 10")
    agent, payload = load(checkpoint, device_for(args.device))
    failures = []

    def record_failure(row, frames):
        """Hold only a handful of actual frames until the unique result stem is known."""
        failures.append((row, frames))

    rows, summary = evaluate(agent, payload["config"], episodes, seed,
                             render_mode="rgb_array" if args.failure_gallery else None,
                             failure_callback=record_failure if args.failure_gallery else None,
                             failure_limit=args.failure_gallery)
    summary.update(checkpoint=str(checkpoint), checkpoint_episode=payload["episode"],
                   seed=seed, shaping=payload["config"]["reward_shaping"]["enabled"])
    path = save_results(rows, summary, Path(settings["output_dir"]), checkpoint.parent.name)
    if failures:
        gallery = Path(settings["output_dir"]) / "gallery" / path.name
        gallery.mkdir(parents=True, exist_ok=True)
        manifest = {"checkpoint": str(checkpoint), "episodes": []}
        for row, frames in failures:
            if not frames:
                continue
            name = f"seed_{row['seed']}.png"
            tiles = [Image.fromarray(frame).resize((300, 200)) for frame in frames]
            strip = Image.new("RGB", (300 * len(tiles), 200))
            for index, tile in enumerate(tiles):
                strip.paste(tile, (index * 300, 0))
            strip.save(gallery / name)
            manifest["episodes"].append({"seed": row["seed"], "outcome": row["outcome"],
                                         "steps": row["length"], "image": name})
        (gallery / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {path}.csv and .json\n{summary}")


if __name__ == "__main__":
    main()
