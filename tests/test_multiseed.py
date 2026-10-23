"""Unit tests for multi-seed experiment functionality."""


import numpy as np
import pandas as pd
import pytest

from lunar_lander_ai.analysis import (
    aggregate_per_seed_results,
    compute_confidence_intervals,
    compute_wilson_local,
    generate_findings_summary,
    load_experiment_results,
    write_summary_csv,
)
from lunar_lander_ai.metrics import compute_wilson as metrics_wilson


def test_compute_wilson_basic():
    """Test Wilson score interval computation."""
    # Test basic cases
    low, high = compute_wilson_local(0, 10)
    assert low == 0.0
    assert high > 0.0

    low, high = compute_wilson_local(10, 10)
    assert low < 1.0
    assert high == pytest.approx(1.0)

    low, high = compute_wilson_local(5, 10)
    assert 0 < low < 0.5 < high < 1

    # Test edge cases
    with pytest.raises(ValueError):
        compute_wilson_local(-1, 10)
    with pytest.raises(ValueError):
        compute_wilson_local(11, 10)


def test_compute_wilson_matches_metrics():
    """Ensure analysis and metrics wilson functions match."""
    for s, n in [(0, 10), (5, 10), (8, 10), (10, 10)]:
        a_low, a_high = compute_wilson_local(s, n)
        m_low, m_high = metrics_wilson(s, n)
        assert abs(a_low - m_low) < 1e-10
        assert abs(a_high - m_high) < 1e-10


def test_aggregate_per_seed_results():
    """Test aggregation of per-seed results."""
    results = [
        {
            "episodes": 100,
            "landed_rate": 0.8,
            "mean_return": 250.0,
            "median_return": 260.0,
            "std_return": 50.0,
            "mean_length": 300.0,
            "crashed_rate": 0.0,
            "truncated_rate": 0.04,
            "off_pad_or_unsafe_rate": 0.16,
            "train_seed": 42,
            "checkpoint": "best",
        },
        {
            "episodes": 100,
            "landed_rate": 0.75,
            "mean_return": 240.0,
            "median_return": 250.0,
            "std_return": 55.0,
            "mean_length": 310.0,
            "crashed_rate": 0.05,
            "truncated_rate": 0.05,
            "off_pad_or_unsafe_rate": 0.15,
            "train_seed": 43,
            "checkpoint": "best",
        },
    ]

    aggregated = aggregate_per_seed_results(results)

    assert "per_seed" in aggregated
    assert "overall" in aggregated
    assert len(aggregated["per_seed"]) == 2

    overall = aggregated["overall"]
    assert overall["n_seeds"] == 2
    assert overall["mean_safe_rate"] == pytest.approx(0.775)
    assert overall["total_landed"] == 155  # 80 + 75
    assert overall["total_evaluated"] == 200


def test_aggregate_empty_results():
    """Test aggregation with empty results."""
    aggregated = aggregate_per_seed_results([])
    assert aggregated == {}


def test_write_summary_csv(tmp_path):
    """Test writing summary CSV with overall row."""
    aggregated = {
        "per_seed": [
            {
                "seed": 42,
                "checkpoint": "best",
                "landed": 80,
                "n": 100,
                "safe_rate": 0.8,
                "wilson_low": 0.71,
                "wilson_high": 0.87,
                "mean_native_return": 250.0,
                "median_native_return": 260.0,
                "std_native_return": 50.0,
                "mean_episode_length": 300.0,
                "crash_rate": 0.0,
                "timeout_rate": 0.04,
                "off_pad_rate": 0.16,
            }
        ],
        "overall": {
            "n_seeds": 1,
            "mean_safe_rate": 0.8,
            "std_safe_rate": 0.0,
            "mean_native_return": 250.0,
            "std_native_return": 0.0,
            "mean_episode_length": 300.0,
            "total_landed": 80,
            "total_evaluated": 100,
            "pooled_wilson_low": 0.71,
            "pooled_wilson_high": 0.87,
        },
    }

    output_path = tmp_path / "summary.csv"
    write_summary_csv(aggregated, output_path)

    df = pd.read_csv(output_path)
    assert len(df) == 2  # per-seed + overall
    assert str(df.iloc[0]["seed"]) == "42"
    assert str(df.iloc[1]["seed"]) == "OVERALL"


def test_load_experiment_results(tmp_path):
    """Test loading experiment results from CSV files."""
    # Create mock evaluation CSVs
    for seed in [42, 43]:
        for arm in ["native", "shaped"]:
            for cp in ["best", "final"]:
                df = pd.DataFrame({
                    "episode": range(5),
                    "seed": [40400 + i for i in range(5)],
                    "native_return": np.random.normal(200, 50, 5),
                    "length": np.random.randint(200, 500, 5),
                    "outcome": np.random.choice(["landed", "crashed", "truncated"], 5),
                })
                path = tmp_path / f"eval_multiseed_{arm}_{seed}_{cp}_20260101T000000000000Z.csv"
                df.to_csv(path, index=False)

    results_df = load_experiment_results(tmp_path, "multiseed")
    # 2 seeds * 2 arms * 2 checkpoints * 5 episodes = 40
    assert len(results_df) == 40
    assert "arm" in results_df.columns
    assert "train_seed" in results_df.columns
    assert "checkpoint" in results_df.columns
    # Check parsing worked correctly
    assert set(results_df["arm"].unique()) == {"native", "shaped"}
    assert set(results_df["train_seed"].unique()) == {42, 43}
    assert set(results_df["checkpoint"].unique()) == {"best", "final"}


def test_compute_confidence_intervals():
    """Test confidence interval computation."""
    # Need at least 2 seeds per group for CI computation
    summary_df = pd.DataFrame({
        "arm": ["native", "native", "native", "native", "shaped", "shaped", "shaped", "shaped"],
        "checkpoint": ["best", "best", "final", "final", "best", "best", "final", "final"],
        "train_seed": [42, 43, 42, 43, 42, 43, 42, 43],
        "safe_rate": [0.8, 0.78, 0.75, 0.73, 0.7, 0.68, 0.65, 0.63],
        "mean_native_return": [250, 248, 240, 238, 230, 228, 220, 218],
        "std_native_return": [50, 52, 55, 57, 60, 62, 65, 67],
        "mean_episode_length": [300, 302, 310, 312, 320, 322, 330, 332],
    })

    ci_df = compute_confidence_intervals(summary_df)
    assert len(ci_df) == 4
    assert "ci_safe_rate_low" in ci_df.columns
    assert "ci_safe_rate_high" in ci_df.columns
    assert "ci_return_low" in ci_df.columns
    assert "ci_return_high" in ci_df.columns
    # CI should contain the mean
    for _, row in ci_df.iterrows():
        assert row["ci_safe_rate_low"] <= row["mean_safe_rate"] <= row["ci_safe_rate_high"]
        assert row["ci_return_low"] <= row["mean_return"] <= row["ci_return_high"]


def test_generate_findings_summary():
    """Test findings summary generation."""
    ci_df = pd.DataFrame({
        "arm": ["native", "native", "shaped", "shaped"],
        "checkpoint": ["best", "final", "best", "final"],
        "mean_safe_rate": [0.8, 0.75, 0.7, 0.65],
        "ci_safe_rate_low": [0.71, 0.65, 0.60, 0.55],
        "ci_safe_rate_high": [0.87, 0.83, 0.78, 0.74],
        "mean_return": [250, 240, 230, 220],
        "ci_return_low": [230, 220, 210, 200],
        "ci_return_high": [270, 260, 250, 240],
    })

    episode_df = pd.DataFrame({
        "arm": ["native"] * 10 + ["shaped"] * 10,
        "checkpoint": ["best"] * 20,
        "train_seed": [42, 43] * 10,
        "native_return": np.random.normal(240, 50, 20),
        "length": np.random.randint(200, 500, 20),
        "outcome": np.random.choice(["landed", "crashed", "truncated", "off_pad_or_unsafe"], 20),
    })

    summary_df = pd.DataFrame({
        "train_seed": [42, 43],
        "arm": ["native", "native"],
        "checkpoint": ["best", "best"],
        "landed": [80, 75],
        "n": [100, 100],
        "safe_rate": [0.8, 0.75],
        "mean_native_return": [250, 240],
        "mean_episode_length": [300, 310],
    })

    findings = generate_findings_summary(ci_df, episode_df, summary_df)
    assert "# Multi-Seed Experiment Findings Summary" in findings
    assert "native" in findings.lower()
    assert "shaped" in findings.lower()
    assert "Limitations" in findings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
