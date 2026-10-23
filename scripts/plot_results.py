"""Generate saved PNG training curves from real CSV logs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_lander_ai.plotting import plot  # noqa: E402


def main():
    """Choose log/output paths and generate available charts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    for path in plot(args.log_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
