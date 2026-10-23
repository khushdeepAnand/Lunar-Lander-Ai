"""Create a portable source archive with a small, measured experiment showcase."""

import argparse
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("20261015T084428850313Z", "20261015T163534138250Z",
        "20261015T172209870911Z")
EVALUATIONS = (
    "20261015T084428850313Z_20261015T085544077217Z",
    "20261015T163534138250Z_20261015T171500184929Z",
    "20261015T172209870911Z_20261015T172654289150Z",
    "examples_20261015T175639362188Z",
    "examples_20261018T171503571915Z",
)


def package(output: Path) -> int:
    """Atomically replace output only after every archive entry passes CRC checks."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    )
    files = {ROOT / name.decode("utf-8") for name in tracked.split(b"\0") if name}
    for run in RUNS:
        files.update(ROOT / "logs" / f"{prefix}_{run}.{suffix}"
                     for prefix, suffix in (("config", "json"), ("train", "csv"),
                                            ("eval", "csv"), ("run", "json")))
        files.add(ROOT / "checkpoints" / run / "best.pt")
    for stem in EVALUATIONS:
        files.update(ROOT / "results" / f"eval_{stem}.{extension}"
                     for extension in ("csv", "json"))
    gallery = ROOT / "results/gallery/eval_examples_20261018T171503571915Z"
    files.update(gallery.glob("*.png"))
    files.add(gallery / "manifest.json")
    # Early runs did not persist a run summary; retain their real logs instead.
    files.discard(ROOT / "logs" / f"run_{RUNS[0]}.json")
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Release evidence missing: {', '.join(sorted(missing))}")
    files = {p for p in files if p.is_file()}
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="lunar-release-", suffix=".zip", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("LUNAR AI/", "")
            for file in sorted(files):
                archive.write(file, f"LUNAR AI/{file.relative_to(ROOT).as_posix()}")
        with zipfile.ZipFile(temporary) as archive:
            if damaged := archive.testzip():
                raise OSError(f"Corrupted release entry: {damaged}")
            if len(archive.namelist()) != len(files) + 1:
                raise OSError("Archive contains duplicate or missing entries")
        Path(temporary).replace(output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return len(files)


def main() -> None:
    """Package a chosen path; never implicitly overwrite the user's desktop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="Destination ZIP file")
    args = parser.parse_args()
    print(f"Packaged {package(args.output)} verified files in {args.output}")


if __name__ == "__main__":
    main()
