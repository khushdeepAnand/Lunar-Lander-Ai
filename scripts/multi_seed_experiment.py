"""Reproducible multi-seed experiment runner for Lunar Lander DQN.

Runs N independent training seeds with identical configuration, evaluates each
trained agent on the same held-out seed range, and aggregates results with
confidence intervals. Supports resume (skips completed runs) and smoke testing.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_lander_ai.checkpoint import load  # noqa: E402
from lunar_lander_ai.evaluate import evaluate, save_results  # noqa: E402
from lunar_lander_ai.train import device_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
EVAL_SEED_START = 40400
EVAL_EPISODES = 100
TRAINING_EPISODES = 900


def sanitize_prefix(prefix: str) -> str:
    """Validate prefix contains only safe characters."""
    if not re.fullmatch(r"[a-z0-9_]+", prefix):
        raise ValueError("Prefix must contain only lowercase letters, digits and underscores")
    return prefix


def run_training(
    seed: int,
    run_id: str,
    device: str,
    config_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Run a single training job via subprocess; return metadata dict."""
    config_path = ROOT / "logs" / f"config_{run_id}.json"
    status_path = ROOT / "logs" / f"run_{run_id}.json"

    if status_path.exists():
        with status_path.open(encoding="utf-8") as f:
            status = json.load(f)
        completed = status.get("status") == "completed"
        episodes_done = status.get("episodes_completed") == TRAINING_EPISODES
        if completed and episodes_done:
            print(f"  [{run_id}] Already completed, skipping")
            return status
        if config_path.exists():
            raise RuntimeError(
                f"Run {run_id} has existing artifacts but incomplete; "
                "inspect before retrying"
            )

    command = [
        sys.executable, str(ROOT / "scripts/train.py"),
        "--run-id", run_id,
        "--seed", str(seed),
        "--device", device,
        "--episodes", str(TRAINING_EPISODES),
    ]
    for key, value in config_overrides.items():
        command.extend(("--set", f"{key}={json.dumps(value)}"))

    print(f"  [{run_id}] Starting training on seed {seed}...")
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - start
    print(f"  [{run_id}] Completed in {elapsed:.1f}s")

    with status_path.open(encoding="utf-8") as f:
        return json.load(f)


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: dict,
    label: str,
) -> tuple[list[dict], dict]:
    """Evaluate a single checkpoint on held-out seeds."""
    agent, payload = load(checkpoint_path, device_for("cpu"))
    rows, summary = evaluate(agent, config, EVAL_EPISODES, EVAL_SEED_START)
    summary.update(
        checkpoint=str(checkpoint_path.relative_to(ROOT)),
        checkpoint_episode=payload["episode"],
        seed=EVAL_SEED_START,
        shaping=config["reward_shaping"]["enabled"],
        policy="greedy",
    )
    save_results(rows, summary, ROOT / "results", label)
    return rows, summary


def compute_wilson(successes: int, n: int) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (95% CI)."""
    if n == 0:
        return 0.0, 1.0
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    from math import sqrt
    z = 1.959963984540054  # 97.5th percentile of standard normal
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def aggregate_per_seed_results(results: list[dict]) -> dict[str, Any]:
    """Compute per-seed and overall statistics from a list of evaluation summaries."""
    if not results:
        return {}

    # Per-seed metrics
    per_seed = []
    for r in results:
        n = r["episodes"]
        landed = round(r["landed_rate"] * n)
        wilson_low, wilson_high = compute_wilson(landed, n)
        per_seed.append({
            "seed": r.get("train_seed", "unknown"),
            "checkpoint": r.get("checkpoint", "unknown"),
            "landed": landed,
            "n": n,
            "safe_rate": landed / n,
            "wilson_low": wilson_low,
            "wilson_high": wilson_high,
            "mean_native_return": r["mean_return"],
            "median_native_return": r["median_return"],
            "std_native_return": r["std_return"],
            "mean_episode_length": r["mean_length"],
            "crash_rate": r["crashed_rate"],
            "timeout_rate": r["truncated_rate"],
            "off_pad_rate": r["off_pad_or_unsafe_rate"],
        })

    # Overall statistics across training seeds
    rates = [s["safe_rate"] for s in per_seed]
    returns = [s["mean_native_return"] for s in per_seed]
    lengths = [s["mean_episode_length"] for s in per_seed]

    overall = {
        "n_seeds": len(per_seed),
        "mean_safe_rate": mean(rates) if rates else 0.0,
        "std_safe_rate": stdev(rates) if len(rates) > 1 else 0.0,
        "mean_native_return": mean(returns) if returns else 0.0,
        "std_native_return": stdev(returns) if len(returns) > 1 else 0.0,
        "mean_episode_length": mean(lengths) if lengths else 0.0,
        "total_landed": sum(s["landed"] for s in per_seed),
        "total_evaluated": sum(s["n"] for s in per_seed),
    }

    # Pooled Wilson CI (descriptive only; ignores between-seed dependence)
    total_landed = overall["total_landed"]
    total_n = overall["total_evaluated"]
    pooled_low, pooled_high = compute_wilson(total_landed, total_n)
    overall["pooled_wilson_low"] = pooled_low
    overall["pooled_wilson_high"] = pooled_high

    return {"per_seed": per_seed, "overall": overall}


def write_summary_csv(aggregated: dict[str, Any], output_path: Path) -> None:
    """Write per-seed and overall summary to CSV."""
    per_seed = aggregated.get("per_seed", [])
    overall = aggregated.get("overall", {})

    # Per-seed rows
    fieldnames = list(per_seed[0].keys()) if per_seed else []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_seed)

        # Overall row
        if overall:
            overall_row = {
                "seed": "OVERALL",
                "checkpoint": "MEAN",
                "landed": overall.get("total_landed", 0),
                "n": overall.get("total_evaluated", 0),
                "safe_rate": overall.get("mean_safe_rate", 0.0),
                "wilson_low": overall.get("pooled_wilson_low", 0.0),
                "wilson_high": overall.get("pooled_wilson_high", 0.0),
                "mean_native_return": overall.get("mean_native_return", 0.0),
                "median_native_return": "",
                "std_native_return": overall.get("std_native_return", 0.0),
                "mean_episode_length": overall.get("mean_episode_length", 0.0),
                "crash_rate": "",
                "timeout_rate": "",
                "off_pad_rate": "",
            }
            writer.writerow(overall_row)


def run_multi_seed_experiment(
    seeds: list[int],
    device: str,
    prefix: str,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the full multi-seed experiment campaign."""
    prefix = sanitize_prefix(prefix)
    if smoke:
        print("SMOKE TEST MODE: Running 2 episodes per seed")
        eval_interval = 1
        checkpoint_interval = 0
        decay_steps = 100
    else:
        eval_interval = 100
        checkpoint_interval = 50
        decay_steps = 60000

    config_overrides = {
        "agent.epsilon.decay_steps": decay_steps,
        "training.early_stop": False,
        "training.eval_interval": eval_interval,
        "training.eval_episodes": 50 if not smoke else 1,
        "training.checkpoint_interval": checkpoint_interval,
    }

    all_results = []

    for seed in seeds:
        for arm in ("native", "shaped"):
            run_id = f"{prefix}_{arm}_{seed}"
            shaping_overrides = {
                **config_overrides,
                "reward_shaping.enabled": arm == "shaped",
                "reward_shaping.angle_penalty_weight": 2.0 if arm == "shaped" else 0.0,
            }

            # Training
            status = run_training(seed, run_id, device, shaping_overrides)
            if status.get("status") != "completed":
                print(f"  [{run_id}] Training failed or incomplete: {status.get('error')}")
                continue

            # Evaluate both best and final checkpoints
            for kind in ("best", "final"):
                checkpoint = ROOT / "checkpoints" / run_id / f"{kind}.pt"
                if not checkpoint.exists():
                    print(f"  [{run_id}] Checkpoint {kind}.pt not found, skipping")
                    continue

                label = f"{prefix}_{arm}_{seed}_{kind}"
                try:
                    _, summary = evaluate_checkpoint(checkpoint, status, label)
                    summary["train_seed"] = seed
                    summary["arm"] = arm
                    summary["checkpoint"] = kind
                    all_results.append(summary)
                    landed_pct = summary['landed_rate'] * 100
                    landed_n = round(summary['landed_rate'] * summary['episodes'])
                    ep_n = summary['episodes']
                    mean_ret = summary['mean_return']
                    print(
                        f"  [{run_id}] {kind}: {landed_pct:.1f}% landings "
                        f"({landed_n}/{ep_n}), mean return {mean_ret:.2f}"
                    )
                except Exception as e:
                    print(f"  [{run_id}] {kind} evaluation failed: {e}")

    # Aggregate and save
    aggregated = aggregate_per_seed_results(all_results)
    summary_path = ROOT / "results" / f"{prefix}_summary.csv"
    write_summary_csv(aggregated, summary_path)
    print(f"\nSummary written to {summary_path}")

    # Print overall results
    for arm in ("native", "shaped"):
        for kind in ("best", "final"):
            group = [r for r in all_results if r["arm"] == arm and r["checkpoint"] == kind]
            if not group:
                continue
            agg = aggregate_per_seed_results(group)
            o = agg["overall"]
            print(f"\n{arm.upper()} {kind.upper()} across {o['n_seeds']} seeds:")
            print(f"  Safe landing rate: {o['mean_safe_rate']:.3f} \u00b1 {o['std_safe_rate']:.3f}")
            print(
                f"  Pooled Wilson 95% CI: "
                f"[{o['pooled_wilson_low']:.3f}, {o['pooled_wilson_high']:.3f}]"
            )
            print(
                f"  Mean native return: {o['mean_native_return']:.2f} "
                f"\u00b1 {o['std_native_return']:.2f}"
            )
            print(f"  Mean episode length: {o['mean_episode_length']:.1f}")

    return {"results": all_results, "aggregated": aggregated, "prefix": prefix}


def main():
    """Entry point for the multi-seed experiment CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Training seeds (default: 42 43 44 45 46)")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--prefix", default="multiseed", help="Run family prefix")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2-episode test")
    args = parser.parse_args()

    if args.smoke_test:
        print("=" * 60)
        print("MULTI-SEED SMOKE TEST")
        print("=" * 60)
    else:
        print("=" * 60)
        print(f"MULTI-SEED EXPERIMENT: {args.prefix}")
        print(f"Seeds: {args.seeds}")
        print(f"Device: {args.device}")
        print("=" * 60)

    run_multi_seed_experiment(args.seeds, args.device, args.prefix, smoke=args.smoke_test)

    if args.smoke_test:
        print("\nSMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()
