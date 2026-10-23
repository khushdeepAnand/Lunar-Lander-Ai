"""Statistical analysis and visualization for multi-seed Lunar Lander experiments.

Uses Pandas for data aggregation and Plotly for interactive visualizations.
Generates publication-ready charts with uncertainty bands and statistical summaries.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_wilson_local(successes: int, n: int) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (95% CI) - local copy for analysis."""
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
        wilson_low, wilson_high = compute_wilson_local(landed, n)
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
    pooled_low, pooled_high = compute_wilson_local(total_landed, total_n)
    overall["pooled_wilson_low"] = pooled_low
    overall["pooled_wilson_high"] = pooled_high

    return {"per_seed": per_seed, "overall": overall}


def write_summary_csv(aggregated: dict[str, Any], output_path: Path) -> None:
    """Write per-seed and overall summary to CSV."""
    import csv

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


def load_experiment_results(results_dir: Path, prefix: str) -> pd.DataFrame:
    """Load all evaluation CSV files for a given experiment prefix."""
    files = sorted(results_dir.glob(f"eval_{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No evaluation CSVs found for prefix '{prefix}' in {results_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Extract metadata from filename
            stem = f.stem
            parts = stem.replace(f"eval_{prefix}_", "").split("_")
            if len(parts) >= 3:
                df["arm"] = parts[0]  # native or shaped
                df["train_seed"] = int(parts[1])
                df["checkpoint"] = parts[2]  # best or final
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        raise ValueError("No valid evaluation data found")

    return pd.concat(dfs, ignore_index=True)


def load_summary_csv(results_dir: Path, prefix: str) -> pd.DataFrame:
    """Load the aggregated summary CSV."""
    summary_path = results_dir / f"{prefix}_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
    return pd.read_csv(summary_path)


def compute_confidence_intervals(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Compute confidence intervals for key metrics grouped by arm and checkpoint."""
    from scipy import stats

    grouped = df.groupby(["arm", "checkpoint"]).agg(
        n_seeds=("safe_rate", "count"),
        mean_safe_rate=("safe_rate", "mean"),
        std_safe_rate=("safe_rate", "std"),
        mean_return=("mean_native_return", "mean"),
        std_return=("mean_native_return", "std"),
        mean_length=("mean_episode_length", "mean"),
    ).reset_index()

    # Standard error of the mean
    grouped["se_safe_rate"] = grouped["std_safe_rate"] / np.sqrt(grouped["n_seeds"])
    grouped["se_return"] = grouped["std_return"] / np.sqrt(grouped["n_seeds"])

    # t-distribution critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, grouped["n_seeds"] - 1)
    grouped["ci_safe_rate_low"] = grouped["mean_safe_rate"] - t_crit * grouped["se_safe_rate"]
    grouped["ci_safe_rate_high"] = grouped["mean_safe_rate"] + t_crit * grouped["se_safe_rate"]
    grouped["ci_return_low"] = grouped["mean_return"] - t_crit * grouped["se_return"]
    grouped["ci_return_high"] = grouped["mean_return"] + t_crit * grouped["se_return"]

    return grouped


def create_landing_rate_comparison(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create bar chart comparing landing rates across seeds, arms, and checkpoints."""
    fig = go.Figure()

    colors = {"native": "#1f77b4", "shaped": "#ff7f0e"}
    markers = {"best": "circle", "final": "square"}

    for arm in ["native", "shaped"]:
        for cp in ["best", "final"]:
            subset = summary_df[
                (summary_df["arm"] == arm) & (summary_df["checkpoint"] == cp)
            ].copy()
            if subset.empty:
                continue

            # Per-seed points
            fig.add_trace(go.Scatter(
                x=[f"Seed {s}" for s in subset["train_seed"]],
                y=subset["safe_rate"] * 100,
                mode="markers",
                name=f"{arm} {cp}",
                marker={
                    "color": colors[arm],
                    "symbol": markers[cp],
                    "size": 12,
                    "line": {"width": 1, "color": "white"}
                },
                showlegend=True,
                legendgroup=f"{arm}_{cp}",
            ))

            # Mean line
            mean_rate = subset["safe_rate"].mean() * 100
            fig.add_trace(go.Scatter(
                x=[f"Seed {s}" for s in subset["train_seed"]],
                y=[mean_rate] * len(subset),
                mode="lines",
                line={"color": colors[arm], "dash": "dash", "width": 2},
                name=f"{arm} {cp} mean",
                showlegend=False,
                legendgroup=f"{arm}_{cp}",
            ))

    fig.update_layout(
        title="Safe Landing Rate per Training Seed",
        xaxis_title="Training Seed",
        yaxis_title="Landing Rate (%)",
        yaxis={"range": [0, 105]},
        template="plotly_white",
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    out_path = output_dir / "landing_rate_comparison.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def create_landing_rate_with_ci(ci_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create grouped bar chart with confidence intervals for landing rates."""
    fig = go.Figure()

    colors = {"native": "#1f77b4", "shaped": "#ff7f0e"}
    arms = ci_df["arm"].unique()
    checkpoints = ci_df["checkpoint"].unique()

    x_positions = []
    x_labels = []
    for i, arm in enumerate(sorted(arms)):
        for j, cp in enumerate(sorted(checkpoints)):
            pos = i * (len(checkpoints) + 0.5) + j
            x_positions.append(pos)
            x_labels.append(f"{arm}<br>{cp}")

    for arm in sorted(arms):
        for cp in sorted(checkpoints):
            subset = ci_df[(ci_df["arm"] == arm) & (ci_df["checkpoint"] == cp)]
            if subset.empty:
                continue
            row = subset.iloc[0]
            pos = x_positions[len([p for p in x_labels if p == f"{arm}<br>{cp}"]) - 1]

            fig.add_trace(go.Bar(
                x=[pos],
                y=[row["mean_safe_rate"] * 100],
                error_y={
                    "type": "data",
                    "array": [(row["ci_safe_rate_high"] - row["mean_safe_rate"]) * 100],
                    "arrayminus": [(row["mean_safe_rate"] - row["ci_safe_rate_low"]) * 100],
                    "visible": True,
                    "thickness": 2,
                    "width": 8,
                },
                name=f"{arm} {cp}",
                marker_color=colors[arm],
                showlegend=True,
            ))

    fig.update_layout(
        title="Mean Safe Landing Rate with 95% CI Across Training Seeds",
        xaxis={
            "tickmode": "array",
            "tickvals": x_positions,
            "ticktext": x_labels,
        },
        yaxis_title="Landing Rate (%)",
        yaxis={"range": [0, 105]},
        template="plotly_white",
        height=500,
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    out_path = output_dir / "landing_rate_ci.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def create_return_distribution(episode_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create violin/box plot of episodic returns by arm and checkpoint."""
    fig = go.Figure()

    colors = {"native": "#1f77b4", "shaped": "#ff7f0e"}

    for arm in ["native", "shaped"]:
        for cp in ["best", "final"]:
            subset = episode_df[
                (episode_df["arm"] == arm) & (episode_df["checkpoint"] == cp)
            ]
            if subset.empty:
                continue

            fig.add_trace(go.Violin(
                y=subset["native_return"],
                name=f"{arm} {cp}",
                box={"visible": True},
                meanline={"visible": True},
                points="outliers",
                marker_color=colors[arm],
                scalegroup=arm,
                legendgroup=f"{arm}_{cp}",
                showlegend=True,
            ))

    fig.update_layout(
        title="Distribution of Episodic Native Returns (Held-out Evaluation)",
        yaxis_title="Native Return",
        template="plotly_white",
        height=600,
        violingroupgap=0.1,
        violingap=0.05,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    out_path = output_dir / "return_distribution.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def create_training_curves(log_dir: Path, prefix: str, output_dir: Path) -> list[Path]:
    """Create training curves with uncertainty bands from multiple seeds."""
    files = sorted(log_dir.glob(f"train_{prefix}_*.csv"))
    if not files:
        return []

    # Group by arm
    arms_data = {"native": [], "shaped": []}
    for f in files:
        stem = f.stem.replace(f"train_{prefix}_", "")
        parts = stem.split("_")
        if len(parts) >= 2:
            arm = parts[0]
            if arm in arms_data:
                df = pd.read_csv(f)
                df["train_seed"] = int(parts[1])
                arms_data[arm].append(df)

    paths = []
    for arm, dfs in arms_data.items():
        if not dfs:
            continue

        combined = pd.concat(dfs, ignore_index=True)

        # Compute rolling statistics across seeds at each episode
        # We need to align by episode number
        max_ep = combined["episode"].max()
        episodes = range(1, int(max_ep) + 1)

        stats_data = []
        for ep in episodes:
            ep_data = combined[combined["episode"] == ep]["native_return"]
            if len(ep_data) > 0:
                stats_data.append({
                    "episode": ep,
                    "mean": ep_data.mean(),
                    "std": ep_data.std() if len(ep_data) > 1 else 0,
                    "min": ep_data.min(),
                    "max": ep_data.max(),
                    "median": ep_data.median(),
                    "n_seeds": len(ep_data),
                })

        if not stats_data:
            continue

        stats_df = pd.DataFrame(stats_data)

        fig = go.Figure()

        # Uncertainty band (mean ± 1 std)
        fig.add_trace(go.Scatter(
            x=stats_df["episode"],
            y=stats_df["mean"] + stats_df["std"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=stats_df["episode"],
            y=stats_df["mean"] - stats_df["std"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)" if arm == "native" else "rgba(255, 127, 14, 0.2)",
            name="±1 std across seeds",
            showlegend=True,
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=stats_df["episode"],
            y=stats_df["mean"],
            mode="lines",
            line={"width": 2, "color": "#1f77b4" if arm == "native" else "#ff7f0e"},
            name=f"{arm} mean",
        ))

        # Individual seed traces (faded)
        for seed in combined["train_seed"].unique():
            seed_data = combined[combined["train_seed"] == seed]
            fig.add_trace(go.Scatter(
                x=seed_data["episode"],
                y=seed_data["native_return"],
                mode="lines",
                line={"width": 0.5, "color": "#1f77b4" if arm == "native" else "#ff7f0e"},
                opacity=0.3,
                name=f"{arm} seed {seed}",
                showlegend=False,
            ))

        fig.update_layout(
            title=f"Training Curves: {arm.capitalize()} Reward (Multi-Seed)",
            xaxis_title="Training Episode",
            yaxis_title="Native Return (per episode)",
            template="plotly_white",
            height=500,
            hovermode="x unified",
        )

        out_path = output_dir / f"training_curves_{arm}.html"
        fig.write_html(out_path, include_plotlyjs="cdn")
        paths.append(out_path)

    return paths


def create_outcome_comparison(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create stacked bar chart of outcome proportions."""
    # Melt the outcome rates
    outcome_cols = ["landed_rate", "crashed_rate", "truncated_rate", "off_pad_or_unsafe_rate"]
    outcome_labels = ["Landed", "Crashed", "Timeout", "Off-pad/Unsafe"]
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#9467bd"]

    fig = go.Figure()

    for arm in ["native", "shaped"]:
        for cp in ["best", "final"]:
            subset = summary_df[
                (summary_df["arm"] == arm) & (summary_df["checkpoint"] == cp)
            ]
            if subset.empty:
                continue

            means = subset[outcome_cols].mean() * 100
            for i, (col, label) in enumerate(zip(outcome_cols, outcome_labels, strict=False)):
                fig.add_trace(go.Bar(
                    x=[f"{arm}<br>{cp}"],
                    y=[means[col]],
                    name=label if arm == "native" and cp == "best" else None,
                    marker_color=colors[i],
                    legendgroup=label,
                    showlegend=(arm == "native" and cp == "best"),
                ))

    fig.update_layout(
        title="Outcome Composition by Arm and Checkpoint",
        yaxis_title="Proportion of Episodes (%)",
        template="plotly_white",
        height=500,
        barmode="stack",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    out_path = output_dir / "outcome_comparison.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def create_native_vs_shaped_return(log_dir: Path, prefix: str, output_dir: Path) -> Path | None:
    """Create comparison of native return during training for native vs shaped."""
    native_files = sorted(log_dir.glob(f"train_{prefix}_native_*.csv"))
    shaped_files = sorted(log_dir.glob(f"train_{prefix}_shaped_*.csv"))

    if not native_files or not shaped_files:
        return None

    # Use the longest-running matching pair
    native_dfs = [pd.read_csv(f) for f in native_files]
    shaped_dfs = [pd.read_csv(f) for f in shaped_files]

    native_combined = pd.concat(native_dfs, ignore_index=True)
    shaped_combined = pd.concat(shaped_dfs, ignore_index=True)

    fig = go.Figure()

    arms_data = [
        ("native", native_combined, "#1f77b4"),
        ("shaped", shaped_combined, "#ff7f0e"),
    ]
    for arm, combined, color in arms_data:
        # Compute rolling mean across seeds at each episode
        max_ep = combined["episode"].max()
        episodes = range(1, int(max_ep) + 1)

        rolling_means = []
        for ep in episodes:
            ep_data = combined[combined["episode"] == ep]["native_return"]
            if len(ep_data) > 0:
                rolling_means.append(ep_data.mean())

        if rolling_means:
            fig.add_trace(go.Scatter(
                x=list(episodes)[:len(rolling_means)],
                y=rolling_means,
                mode="lines",
                line={"width": 3, "color": color},
                name=f"{arm} rolling mean",
            ))

    fig.update_layout(
        title="Native Return During Training: Native vs Shaped Reward",
        xaxis_title="Training Episode",
        yaxis_title="Rolling Mean Native Return (across seeds)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )

    out_path = output_dir / "native_vs_shaped_training.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def create_efficiency_comparison(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create scatter plot of training efficiency: landing rate vs episode length."""
    fig = go.Figure()

    colors = {"native": "#1f77b4", "shaped": "#ff7f0e"}
    markers = {"best": "circle", "final": "square"}

    for arm in ["native", "shaped"]:
        for cp in ["best", "final"]:
            subset = summary_df[
                (summary_df["arm"] == arm) & (summary_df["checkpoint"] == cp)
            ].copy()
            if subset.empty:
                continue

            fig.add_trace(go.Scatter(
                x=subset["mean_episode_length"],
                y=subset["safe_rate"] * 100,
                mode="markers",
                name=f"{arm} {cp}",
                marker={
                    "color": colors[arm],
                    "symbol": markers[cp],
                    "size": 14,
                    "line": {"width": 1, "color": "white"}
                },
                text=[f"Seed {s}" for s in subset["train_seed"]],
                hovertemplate=(
                    "Seed: %{text}<br>Episode length: %{x:.0f}<br>Landing rate: %{y:.1f}%"
                ),
            ))

            # Mean point
            fig.add_trace(go.Scatter(
                x=[subset["mean_episode_length"].mean()],
                y=[subset["safe_rate"].mean() * 100],
                mode="markers",
                marker={
                    "color": colors[arm],
                    "symbol": markers[cp],
                    "size": 18,
                    "line": {"width": 2, "color": "black"}
                },
                name=f"{arm} {cp} mean",
                showlegend=False,
            ))

    fig.update_layout(
        title="Training Efficiency: Landing Rate vs Episode Length",
        xaxis_title="Mean Episode Length (steps)",
        yaxis_title="Safe Landing Rate (%)",
        template="plotly_white",
        height=500,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    out_path = output_dir / "efficiency_comparison.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def create_failure_analysis(episode_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create failure case analysis: return vs outcome type."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Native Return by Outcome", "Episode Length by Outcome"),
        horizontal_spacing=0.12,
    )

    colors = {
        "landed": "#2ca02c",
        "crashed": "#d62728",
        "truncated": "#1f77b4",
        "off_pad_or_unsafe": "#9467bd",
    }

    for outcome in ["landed", "crashed", "truncated", "off_pad_or_unsafe"]:
        subset = episode_df[episode_df["outcome"] == outcome]
        if subset.empty:
            continue

        fig.add_trace(go.Box(
            y=subset["native_return"],
            name=outcome.replace("_", " ").title(),
            marker_color=colors[outcome],
            boxpoints="outliers",
            showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Box(
            y=subset["length"],
            name=outcome.replace("_", " ").title(),
            marker_color=colors[outcome],
            boxpoints="outliers",
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(
        title="Failure Case Analysis: Returns and Lengths by Outcome Type",
        template="plotly_white",
        height=500,
        yaxis_title="Native Return",
        yaxis2_title="Episode Length",
    )

    out_path = output_dir / "failure_analysis.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def generate_findings_summary(
    ci_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> str:
    """Generate a concise, automatically generated findings summary."""
    lines = []
    lines.append("# Multi-Seed Experiment Findings Summary")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Seeds evaluated:** {summary_df['train_seed'].nunique()}")
    eps_per = summary_df['n'].iloc[0] if 'n' in summary_df.columns else 'N/A'
    lines.append(f"**Episodes per evaluation:** {eps_per}")
    lines.append("")

    # Overall comparison
    lines.append("## Overall Comparison (Best Checkpoint)")
    for arm in ["native", "shaped"]:
        subset = ci_df[(ci_df["arm"] == arm) & (ci_df["checkpoint"] == "best")]
        if not subset.empty:
            row = subset.iloc[0]
            lr = row['mean_safe_rate'] * 100
            ci_l = row['ci_safe_rate_low'] * 100
            ci_h = row['ci_safe_rate_high'] * 100
            mr = row['mean_return']
            ci_rl = row['ci_return_low']
            ci_rh = row['ci_return_high']
            lines.append(
                f"- **{arm.capitalize()}**: "
                f"Landing rate {lr:.1f}% (95% CI: [{ci_l:.1f}%, {ci_h:.1f}%]), "
                f"Mean return {mr:.1f} (95% CI: [{ci_rl:.1f}, {ci_rh:.1f}])"
            )
    lines.append("")

    # Statistical significance note
    lines.append("## Statistical Notes")
    lines.append("- Confidence intervals use t-distribution (small sample correction)")
    lines.append(
        "- Between-seed variability is captured; "
        "within-seed (episode-level) variability is separate"
    )
    lines.append("- Pooled Wilson intervals ignore between-seed dependence (descriptive only)")
    lines.append("- No causal claims about reward shaping without controlled experiment design")
    lines.append("")

    # Per-seed detail
    lines.append("## Per-Seed Results (Best Checkpoint)")
    for arm in ["native", "shaped"]:
        subset = summary_df[(summary_df["arm"] == arm) & (summary_df["checkpoint"] == "best")]
        if not subset.empty:
            lines.append(f"### {arm.capitalize()} Reward")
            for _, row in subset.iterrows():
                lines.append(f"- Seed {row['train_seed']}: "
                             f"{row['landed']}/{row['n']} landings ({row['safe_rate']*100:.1f}%), "
                             f"return {row['mean_native_return']:.1f}, "
                             f"length {row['mean_episode_length']:.0f}")
            lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("- Small number of training seeds (n=5) limits statistical power")
    lines.append(
        "- Results specific to LunarLander-v3, DQN architecture, "
        "and tested hyperparameters"
    )
    lines.append("- No guarantee of bit-identical reproduction across hardware/BLAS backends")
    lines.append(
        "- Episode-level variability (within a trained agent) "
        "is distinct from training-seed variability"
    )
    lines.append("")

    return "\n".join(lines)


def run_full_analysis(
    results_dir: Path,
    log_dir: Path,
    prefix: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run complete analysis pipeline and generate all visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    episode_df = load_experiment_results(results_dir, prefix)
    summary_df = load_summary_csv(results_dir, prefix)
    ci_df = compute_confidence_intervals(summary_df)

    # Generate visualizations
    paths = {}

    paths["landing_rate_comparison"] = create_landing_rate_comparison(summary_df, output_dir)
    paths["landing_rate_ci"] = create_landing_rate_with_ci(ci_df, output_dir)
    paths["return_distribution"] = create_return_distribution(episode_df, output_dir)
    paths["training_curves"] = create_training_curves(log_dir, prefix, output_dir)
    paths["outcome_comparison"] = create_outcome_comparison(summary_df, output_dir)
    paths["native_vs_shaped"] = create_native_vs_shaped_return(log_dir, prefix, output_dir)
    paths["efficiency"] = create_efficiency_comparison(summary_df, output_dir)
    paths["failure_analysis"] = create_failure_analysis(episode_df, output_dir)

    # Generate findings summary
    findings = generate_findings_summary(ci_df, episode_df, summary_df)
    findings_path = output_dir / "findings_summary.md"
    findings_path.write_text(findings, encoding="utf-8")
    paths["findings"] = findings_path

    # Save aggregated data as CSV for reproducibility
    ci_path = output_dir / "confidence_intervals.csv"
    ci_df.to_csv(ci_path, index=False)
    paths["confidence_intervals"] = ci_path

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze multi-seed experiment results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--prefix", default="multiseed")
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    paths = run_full_analysis(args.results_dir, args.log_dir, args.prefix, args.output_dir)
    for _name, _path in paths.items():
        pass
