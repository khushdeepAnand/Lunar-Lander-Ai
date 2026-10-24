"""Tests for the release packaging script."""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_release.py"


def run_package_release(args, cwd=None):
    """Run the package_release.py script and return (exit_code, stdout, stderr)."""
    cmd = [sys.executable, str(SCRIPT)] + args
    result = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True, timeout=60)
    return result.returncode, result.stdout, result.stderr


def test_source_only_package(tmp_path):
    """Test packaging from a clean source checkout (no evidence artifacts)."""
    output = tmp_path / "lunar-ai-source.zip"
    code, stdout, stderr = run_package_release([str(output), "--source-only"])
    assert code == 0, f"Exit {code}: {stderr}"
    assert "source-only" in stdout

    # Verify ZIP integrity and contents
    with zipfile.ZipFile(output, "r") as z:
        names = z.namelist()
        # Should have directory entry + core files
        assert len(names) > 50
        # Check required core files
        required = [
            "LUNAR AI/README.md",
            "LUNAR AI/LICENSE",
            "LUNAR AI/pyproject.toml",
            "LUNAR AI/src/lunar_lander_ai/__init__.py",
            "LUNAR AI/tests/test_core.py",
            "LUNAR AI/scripts/train.py",
        ]
        for req in required:
            assert req in names, f"Missing required file: {req}"
        # Ensure no excluded files
        for name in names:
            assert ".venv/" not in name
            assert "__pycache__/" not in name
            assert ".git/" not in name
            assert not name.endswith(".zip")
            assert not name.endswith(".tmp")


def test_package_excludes_secrets_and_caches(tmp_path):
    """Verify that secrets, caches, and virtual environments are excluded."""
    output = tmp_path / "test.zip"
    code, _, _ = run_package_release([str(output), "--source-only"])
    assert code == 0

    with zipfile.ZipFile(output, "r") as z:
        names = z.namelist()
        for name in names:
            # No virtual environments
            assert ".venv" not in name
            assert "venv" not in name or "vendor" in name  # allow "vendor" but not venv
            # No cache directories
            assert "__pycache__" not in name
            assert ".pytest_cache" not in name
            assert ".ruff_cache" not in name
            # No Git internals
            assert ".git/" not in name
            # No nested ZIPs
            assert not name.endswith(".zip")
            # No temp files
            assert not name.endswith(".tmp")


def test_package_no_duplicate_entries(tmp_path):
    """Verify archive has no duplicate paths."""
    output = tmp_path / "test.zip"
    code, _, _ = run_package_release([str(output), "--source-only"])
    assert code == 0

    with zipfile.ZipFile(output, "r") as z:
        names = z.namelist()
        # Check for duplicates
        assert len(names) == len(set(names)), "Duplicate entries found in archive"
        # Validate ZIP integrity
        assert z.testzip() is None, "Corrupted ZIP entry"


def test_package_handles_missing_evidence_gracefully(tmp_path):
    """Test that missing evidence produces warnings, not failures (non-strict mode)."""
    output = tmp_path / "test.zip"
    # Fresh clone won't have evidence artifacts
    code, stdout, stderr = run_package_release(
        [str(output), "--with-evidence"],
        cwd=tmp_path  # Run from empty temp dir to simulate fresh clone
    )
    # In non-strict mode, should succeed with warnings
    assert code == 0, f"Expected success with warnings, got exit {code}: {stderr}"
    assert "Warnings:" in stdout or "Missing evidence" in stdout


def test_package_strict_mode_fails_on_missing_evidence(tmp_path):
    """Test that strict mode fails when evidence is missing."""
    output = tmp_path / "test.zip"
    code, stdout, stderr = run_package_release(
        [str(output), "--with-evidence", "--strict"],
        cwd=tmp_path  # Run from empty temp dir
    )
    assert code != 0, "Strict mode should fail when evidence is missing"
    assert "Missing evidence" in stderr or "Missing evidence" in stdout


def test_package_detects_nested_zips(tmp_path):
    """Verify that nested ZIPs are not included in the archive."""
    # Create a fake nested zip in the source tree
    fake_zip = ROOT / "fake_nested.zip"
    fake_zip.write_text("dummy")

    try:
        output = tmp_path / "test.zip"
        code, _, _ = run_package_release([str(output), "--source-only"])
        assert code == 0

        with zipfile.ZipFile(output, "r") as z:
            names = z.namelist()
            assert "LUNAR AI/fake_nested.zip" not in names
    finally:
        fake_zip.unlink(missing_ok=True)


def test_package_handles_existing_output(tmp_path):
    """Test that existing output file is replaced atomically."""
    output = tmp_path / "test.zip"
    # First run
    code1, _, _ = run_package_release([str(output), "--source-only"])
    assert code1 == 0
    size1 = output.stat().st_size

    # Second run should overwrite
    code2, _, _ = run_package_release([str(output), "--source-only"])
    assert code2 == 0
    size2 = output.stat().st_size

    # Sizes should be equal (deterministic)
    assert size1 == size2, "Package not deterministic"


def test_package_deterministic(tmp_path):
    """Test that two runs produce identical archives."""
    output1 = tmp_path / "test1.zip"
    output2 = tmp_path / "test2.zip"

    code1, _, _ = run_package_release([str(output1), "--source-only"])
    code2, _, _ = run_package_release([str(output2), "--source-only"])
    assert code1 == 0 and code2 == 0

    # Compare ZIP contents (sorted)
    with zipfile.ZipFile(output1, "r") as z1, zipfile.ZipFile(output2, "r") as z2:
        names1 = sorted(z1.namelist())
        names2 = sorted(z2.namelist())
        assert names1 == names2, "Archive contents not deterministic"

        # Compare file sizes for each entry
        for name in names1:
            info1 = z1.getinfo(name)
            info2 = z2.getinfo(name)
            assert info1.file_size == info2.file_size, f"Size mismatch for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
