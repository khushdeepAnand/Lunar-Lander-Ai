"""Run the preregistered paired, full-budget shaping study without replacing old artifacts."""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.data import wilson  # noqa: E402

from lunar_lander_ai.checkpoint import load  # noqa: E402
from lunar_lander_ai.evaluate import evaluate, save_results  # noqa: E402
from lunar_lander_ai.train import device_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def study(seeds: list[int], device: str, prefix: str) -> None:
    """Sequential paired training and genuinely held-out evaluation per arm/seed."""
    if not re.fullmatch(r"[a-z0-9_]+", prefix):
        raise ValueError("Prefix must contain only lowercase letters, digits and underscores")
    rows = []
    for seed in seeds:
        for arm in ("native", "shaped"):
            run = f"{prefix}_{arm}_{seed}"
            config_path = ROOT / "logs" / f"config_{run}.json"
            status_path = ROOT / "logs" / f"run_{run}.json"
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") != "completed" or status.get("episodes_completed") != 900:
                    raise RuntimeError(f"Run {run} is incomplete; inspect it before retrying")
            else:
                if config_path.exists():
                    raise RuntimeError(f"Run {run} has existing artifacts; refusing overwrite")
                command = [sys.executable, str(ROOT / "scripts/train.py"), "--run-id", run,
                           "--seed", str(seed), "--device", device, "--episodes", "900"]
                settings = {"agent.epsilon.decay_steps": 60000,
                            "training.early_stop": False,
                            "training.eval_interval": 100,
                            "training.eval_episodes": 50,
                            "training.checkpoint_interval": 50,
                            "reward_shaping.enabled": arm == "shaped",
                            "reward_shaping.angle_penalty_weight": 2.0 if arm == "shaped" else 0.0}
                for key, value in settings.items():
                    command.extend(("--set", f"{key}={json.dumps(value)}"))
                subprocess.run(command, cwd=ROOT, check=True)
            for kind in ("best", "final"):
                checkpoint = ROOT / "checkpoints" / run / f"{kind}.pt"
                agent, payload = load(checkpoint, device_for("cpu"))
                result_tag = f"{prefix}_{arm}_{seed}_{kind}"
                existing = sorted((ROOT / "results").glob(f"eval_{result_tag}_*.json"))
                if existing:
                    if len(existing) != 1:
                        raise RuntimeError(f"Multiple evaluations for {result_tag}")
                    summary = json.loads(existing[0].read_text(encoding="utf-8"))
                else:
                    observations, summary = evaluate(agent, payload["config"], 100, 40400)
                    summary.update(checkpoint=str(checkpoint.relative_to(ROOT)),
                                   checkpoint_episode=payload["episode"], seed=40400,
                                   shaping=arm == "shaped", policy="greedy")
                    save_results(observations, summary, ROOT / "results", result_tag)
                n = summary["episodes"]
                successes = round(summary["landed_rate"] * n)
                low, high = wilson(successes, n)
                rows.append({"seed": seed, "arm": arm, "checkpoint": kind,
                             "episode": payload["episode"], "landed": successes,
                             "n": n, "safe_rate": successes / n,
                             "wilson_low": low, "wilson_high": high,
                             "mean_native_return": summary["mean_return"]})
                print(f"{run} {kind}: {successes}/{n}", flush=True)
    target = ROOT / "results" / f"{prefix}_summary.csv"
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for arm in ("native", "shaped"):
        for kind in ("best", "final"):
            group = [r for r in rows if r["arm"] == arm and r["checkpoint"] == kind]
            count = sum(r["landed"] for r in group)
            low, high = wilson(count, sum(r["n"] for r in group))
            spread = stdev(r["safe_rate"] for r in group) if len(group) > 1 else float("nan")
            print(f"{arm} {kind}: {mean(r['safe_rate'] for r in group):.3f} ± "
                  f"{spread:.3f} across {len(group)} runs; "
                  f"descriptive pooled Wilson 95% [{low:.3f}, {high:.3f}] "
                  "(does not account for between-seed dependence)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--prefix", default="stage3", help="Unique run family after interruption")
    args = parser.parse_args()
    study(args.seeds, args.device, args.prefix)
