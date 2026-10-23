# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-24

### Added
- **Multi-seed experiment runner** (`scripts/multi_seed_experiment.py`): Reproducible runner for N independent training seeds with identical configuration, held-out evaluation, and statistical aggregation.
- **Statistical analysis module** (`src/lunar_lander_ai/analysis.py`): Pandas/Plotly-based analysis with confidence intervals, Wilson score intervals, and publication-ready interactive visualizations.
- **New visualizations**:
  - Landing rate comparison per training seed (scatter + mean lines)
  - Mean landing rate with 95% CI across seeds (grouped bar chart)
  - Episodic return distributions (violin plots)
  - Training curves with uncertainty bands (±1 std across seeds)
  - Outcome composition (stacked bars)
  - Native vs shaped reward training comparison
  - Training efficiency scatter (landing rate vs episode length)
  - Failure case analysis (box plots by outcome type)
- **Automated findings summary** generation in Markdown format
- **Unit tests** for multi-seed functionality (`tests/test_multiseed.py`): seed handling, aggregation, metric calculations, persistence, resume behavior, confidence intervals.
- **Resume capability**: Experiment runner skips completed runs based on status metadata; refuses to overwrite incomplete runs.

### Changed
- Added `compute_wilson` function to `src/lunar_lander_ai/metrics.py` for shared Wilson score interval calculation.
- Updated `pyproject.toml` to include `scipy` as a dependency for t-distribution confidence intervals.
- Extended test suite from 29 to 37 tests.

### Fixed
- Ruff linting issues across new modules (line length, unused imports, dict() calls, zip strict parameter).
- Null byte cleanup in generated files.

## [0.1.0] - 2026-10-18

### Added
- Initial DQN implementation for LunarLander-v3
- Experience replay buffer with separate termination/truncation flags
- Epsilon-greedy exploration with linear decay
- Target network with hard updates
- Checkpointing with full RNG state persistence
- Greedy evaluation with safe-landing detection
- Optional reward shaping wrapper (potential-based)
- Training curves (matplotlib)
- Streamlit dashboard for live simulation, training, analytics, comparison
- Comprehensive test suite (29 tests)
- Documentation: architecture, build spec, experiments log, portfolio, interview prep
- Release packaging script

### Measured Results
- Native-reward 900-episode run (seed 42): 80/100 safe landings, mean return 252.91
- Shaped-reward 300-episode comparison: no improvement (both under-trained)