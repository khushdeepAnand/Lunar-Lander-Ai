"""Watch the saved agent's actual actions in LunarLander."""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_lander_ai.checkpoint import load  # noqa: E402
from lunar_lander_ai.evaluate import evaluate  # noqa: E402
from lunar_lander_ai.train import device_for  # noqa: E402


def main():
    """Display pygame playback, or render frames offscreen."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260)
    parser.add_argument("--render-mode", choices=("human", "rgb_array"), default="human")
    parser.add_argument("--save-gif", type=Path,
                        help="Save actual rendered frames as a GIF (use --render-mode rgb_array)")
    args = parser.parse_args()
    if args.save_gif and args.render_mode != "rgb_array":
        parser.error("--save-gif requires --render-mode rgb_array")
    agent, payload = load(args.checkpoint, device_for("auto"))
    frames = []
    callback = frames.append if args.save_gif else None
    rows, summary = evaluate(agent, payload["config"], args.episodes, args.seed,
                             args.render_mode, callback)
    if args.save_gif and frames:
        args.save_gif.parent.mkdir(parents=True, exist_ok=True)
        # Keep every second simulated frame so motion reads smoothly; the model
        # still acted at every step. Upscale with LANCZOS for a crisp picture.
        # Frames come straight from the real environment renderer, never redrawn.
        images = []
        for frame in frames[::2]:
            image = Image.fromarray(frame).resize((900, 600), Image.Resampling.LANCZOS)
            images.append(image.convert("P", palette=Image.Palette.ADAPTIVE,
                                        dither=Image.Dither.FLOYDSTEINBERG))
        images[0].save(args.save_gif, save_all=True, append_images=images[1:],
                       duration=50, loop=0, optimize=True)
        print(f"Saved actual-model playback: {args.save_gif} "
              f"({len(images)} frames, 20 fps)")
    for row in rows:
        print(f"seed={row['seed']} return={row['native_return']:.2f} outcome={row['outcome']}")
    print(summary)


if __name__ == "__main__":
    main()
