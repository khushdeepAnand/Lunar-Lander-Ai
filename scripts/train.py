"""Run a headless DQN training experiment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_lander_ai.config import load_config  # noqa: E402
from lunar_lander_ai.train import train  # noqa: E402


def main():
    """Parse train options and start the run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", help="Unique dashboard run identifier")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    args = parser.parse_args()
    import yaml

    overrides = {}
    for entry in args.set:
        if "=" not in entry:
            parser.error("--set requires KEY=VALUE")
        key, value = entry.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    for key in ("episodes", "seed", "device"):
        value = getattr(args, key)
        if value is not None:
            overrides["training.episodes" if key == "episodes" else key] = value
    train(load_config(args.config, overrides).values, smoke=args.smoke_test, run_id=args.run_id)


if __name__ == "__main__":
    main()
