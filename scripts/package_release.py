#!/usr/bin/env python3
"""Create a portable release archive.

Two package types:
  --source-only     Source code, tests, docs, configs only (no experiment artifacts)
  --with-evidence   Source + available sample logs, checkpoints, evaluations, gallery

Both exclude: .venv, caches, nested ZIPs, secrets, Git internals, build artifacts.
"""

import argparse
import hashlib
import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Known experiment runs that may exist locally (from development machine)
KNOWN_RUNS = (
    "20261015T084428850313Z",
    "20261015T163534138250Z",
    "20261015T172209870911Z",
)

# Known evaluation stems that may exist locally
KNOWN_EVALUATIONS = (
    "20261015T084428850313Z_20261015T085544077217Z",
    "20261015T163534138250Z_20261015T171500184929Z",
    "20261015T172209870911Z_20261015T172654289150Z",
    "examples_20261015T175639362188Z",
    "examples_20261018T171503571915Z",
)

# Known gallery that may exist locally
KNOWN_GALLERY = ROOT / "results/gallery/eval_examples_20261018T171503571915Z"

# Files that must always be present for any valid package
REQUIRED_CORE = (
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-cpu.txt",
    "requirements-dev.txt",
    "src/lunar_lander_ai/__init__.py",
    "src/lunar_lander_ai/agent.py",
    "src/lunar_lander_ai/analysis.py",
    "src/lunar_lander_ai/checkpoint.py",
    "src/lunar_lander_ai/config.py",
    "src/lunar_lander_ai/environment.py",
    "src/lunar_lander_ai/evaluate.py",
    "src/lunar_lander_ai/metrics.py",
    "src/lunar_lander_ai/network.py",
    "src/lunar_lander_ai/plotting.py",
    "src/lunar_lander_ai/replay_buffer.py",
    "src/lunar_lander_ai/rewards.py",
    "src/lunar_lander_ai/train.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/play.py",
    "scripts/plot_results.py",
    "scripts/stage3_study.py",
    "scripts/multi_seed_experiment.py",
    "scripts/package_release.py",
    "scripts/verify_setup.py",
    "configs/default.yaml",
    "configs/evaluation.yaml",
    "tests/test_core.py",
    "tests/test_dashboard.py",
    "tests/test_hud.py",
    "tests/test_multiseed.py",
    "tests/test_pipeline.py",
    "ui/data.py",
    "ui/hud.py",
    "ui/jobs.py",
    "ui/playback.py",
    "ui/theme.py",
    "app.py",
    ".streamlit/config.toml",
    "docs/BUILD_SPEC.md",
    "docs/architecture.md",
    "docs/experiments.md",
    "docs/portfolio.md",
    "docs/interview.md",
    "docs/reinforcement-learning.md",
    "docs/dashboard-guide.md",
    "docs/ui-architecture.md",
    "docs/visual-design.md",
    "docs/experiment-methodology.md",
)

# Patterns of files to always exclude
EXCLUDE_PATTERNS = (
    ".venv/",
    "venv/",
    "env/",
    "ENV/",
    ".conda/",
    "__pycache__/",
    "*.py[cod]",
    "*$py.class",
    "*.so",
    "*.pyd",
    "*.egg-info/",
    ".eggs/",
    "build/",
    "dist/",
    "pip-wheel-metadata/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".coverage",
    ".coverage.*",
    "htmlcov/",
    "coverage.xml",
    ".tox/",
    ".nox/",
    ".vscode/",
    ".idea/",
    "*.swp",
    "*.swo",
    "Thumbs.db",
    "ehthumbs.db",
    "desktop.ini",
    ".DS_Store",
    ".env",
    ".env.*",
    "!/.env.example",
    ".git/",
    ".gitignore",
    "*.zip",
    "*.tmp",
    "logs/*.csv",
    "logs/*.json",
    "logs/*.jsonl",
    "logs/*.log",
    "logs/job_*.yaml",
    "logs/output_*.txt",
    "logs/stop_*",
    "logs/dashboard_evaluation.txt",
    "logs/**/*.csv",
    "logs/**/*.json",
    "checkpoints/*.pt",
    "checkpoints/*.pth",
    "checkpoints/*.tar",
    "checkpoints/**/*.pt",
    "checkpoints/**/*.pth",
    "checkpoints/**/*.tar",
    "checkpoints/**/*.tmp",
    "results/*.png",
    "results/*.jpg",
    "results/*.svg",
    "results/*.pdf",
    "results/*.mp4",
    "results/*.gif",
    "results/*.json",
    "results/*.md",
    "results/*.csv",
    "results/gallery/",
    "data/*.npz",
    "data/*.npy",
    "data/*.csv",
)


def match_any(path: Path, patterns: tuple[str, ...]) -> bool:
    """Check if path matches any glob pattern relative to ROOT."""
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        return False
    for pat in patterns:
        if pat.endswith("/"):
            if rel.startswith(pat) or f"/{pat}" in f"/{rel}/":
                return True
        else:
            if Path(rel).match(pat):
                return True
            # Also match against full path parts
            for part in Path(rel).parts:
                if Path(part).match(pat):
                    return True
    return False


def collect_core_files() -> set[Path]:
    """Collect all core source files that should be in every package."""
    files = set()
    for rel in REQUIRED_CORE:
        p = ROOT / rel
        if p.is_file():
            files.add(p)
    # Also include all .gitkeep files
    for gitkeep in ROOT.rglob(".gitkeep"):
        if not match_any(gitkeep, EXCLUDE_PATTERNS):
            files.add(gitkeep)
    return files


def collect_evidence_files() -> tuple[set[Path], list[str]]:
    """Collect available evidence files (logs, checkpoints, evaluations, gallery).

    Returns:
        (files, warnings) where warnings list any missing optional artifacts.
    """
    files = set()
    warnings = []

    # Add available run logs and checkpoints
    for run in KNOWN_RUNS:
        for prefix, suffix in (("config", "json"), ("train", "csv"),
                               ("eval", "csv"), ("run", "json")):
            p = ROOT / "logs" / f"{prefix}_{run}.{suffix}"
            if p.is_file():
                files.add(p)
            else:
                warnings.append(f"Missing evidence log: {p.relative_to(ROOT)}")
        ckpt = ROOT / "checkpoints" / run / "best.pt"
        if ckpt.is_file():
            files.add(ckpt)
        else:
            warnings.append(f"Missing evidence checkpoint: {ckpt.relative_to(ROOT)}")

    # Add available evaluations
    for stem in KNOWN_EVALUATIONS:
        for ext in ("csv", "json"):
            p = ROOT / "results" / f"eval_{stem}.{ext}"
            if p.is_file():
                files.add(p)
            else:
                warnings.append(f"Missing evidence evaluation: {p.relative_to(ROOT)}")

    # Add available gallery
    if KNOWN_GALLERY.is_dir():
        for png in KNOWN_GALLERY.glob("*.png"):
            files.add(png)
        manifest = KNOWN_GALLERY / "manifest.json"
        if manifest.is_file():
            files.add(manifest)
        else:
            warnings.append(f"Missing gallery manifest: {manifest.relative_to(ROOT)}")
    else:
        warnings.append(f"Missing evidence gallery: {KNOWN_GALLERY.relative_to(ROOT)}")

    return files, warnings


def sha256(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def package(
    output: Path,
    with_evidence: bool = False,
    strict: bool = False,
) -> tuple[int, list[str]]:
    """Build the release archive.

    Args:
        output: Destination ZIP file path.
        with_evidence: Include available experiment evidence.
        strict: If True and evidence requested, fail on missing evidence files.

    Returns:
        (file_count, warnings)
    """
    # Collect core files (always included)
    core_files = collect_core_files()

    # Collect evidence files if requested
    evidence_files = set()
    warnings = []
    if with_evidence:
        evidence_files, evidence_warnings = collect_evidence_files()
        warnings.extend(evidence_warnings)

    all_files = core_files | evidence_files

    # Filter out excluded patterns
    filtered_files = {f for f in all_files if not match_any(f, EXCLUDE_PATTERNS)}

    # Check for required core files
    missing_core = [rel for rel in REQUIRED_CORE if not (ROOT / rel).is_file()]
    if missing_core:
        raise FileNotFoundError(f"Required core files missing: {', '.join(missing_core)}")

    # Strict mode: fail if any requested evidence is missing
    if strict and with_evidence:
        missing_evidence = [w for w in warnings if w.startswith("Missing evidence")]
        if missing_evidence:
            raise FileNotFoundError(
                f"Strict mode: evidence files missing: {', '.join(missing_evidence)}"
            )

    # Check for duplicate archive paths
    archive_paths = {}
    for f in filtered_files:
        try:
            rel = f.relative_to(ROOT).as_posix()
        except ValueError:
            continue
        arc_path = f"LUNAR AI/{rel}"
        if arc_path in archive_paths:
            warnings.append(f"Duplicate archive path (will use first): {arc_path}")
        else:
            archive_paths[arc_path] = f

    output.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write via temp file
    fd, temporary = tempfile.mkstemp(
        prefix="lunar-release-", suffix=".zip", dir=output.parent
    )
    os.close(fd)

    try:
        with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            archive.writestr("LUNAR AI/", "")
            for arc_path, file in sorted(archive_paths.items()):
                archive.write(file, arc_path)

        # Validate ZIP integrity
        with zipfile.ZipFile(temporary) as archive:
            if damaged := archive.testzip():
                raise OSError(f"Corrupted release entry: {damaged}")
            # +1 for the directory entry "LUNAR AI/"
            if len(archive.namelist()) != len(archive_paths) + 1:
                raise OSError("Archive contains duplicate or missing entries")

        Path(temporary).replace(output)
    finally:
        Path(temporary).unlink(missing_ok=True)

    return len(archive_paths), warnings


def main() -> int:
    """Package a chosen path; never implicitly overwrite the user's desktop."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output", type=Path, help="Destination ZIP file")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Build source-only package (default if neither flag given)",
    )
    parser.add_argument(
        "--with-evidence",
        action="store_true",
        help="Include available experiment evidence (logs, checkpoints, evaluations, gallery)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested evidence files are missing",
    )
    args = parser.parse_args()

    if args.source_only and args.with_evidence:
        parser.error("--source-only and --with-evidence are mutually exclusive")

    with_evidence = args.with_evidence
    # Default to source-only if neither specified
    if not args.source_only and not args.with_evidence:
        with_evidence = False

    try:
        count, warnings = package(
            args.output, with_evidence=with_evidence, strict=args.strict
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    pkg_type = "with-evidence" if with_evidence else "source-only"
    print(f"Packaged {count} verified files in {args.output} [{pkg_type}]")

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
