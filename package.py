"""Create the final code-only zip for HackerRank Orchestrate submission."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


REPO_ROOT = Path(__file__).resolve().parent
CODE_DIR = REPO_ROOT / "code"
OUTPUT_ZIP = REPO_ROOT / "submission_code.zip"
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "data",
}
EXCLUDED_FILE_NAMES = {".env"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pyd"}


def main() -> None:
    if not CODE_DIR.exists():
        raise SystemExit(f"missing code directory: {CODE_DIR}")

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    files = [path for path in CODE_DIR.rglob("*") if path.is_file() and not should_exclude(path)]
    if not files:
        raise SystemExit("no files found to package")

    with ZipFile(OUTPUT_ZIP, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(REPO_ROOT).as_posix())

    print(f"created {OUTPUT_ZIP.name}")
    print(f"files={len(files)}")


def should_exclude(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    parts = set(relative.parts)
    if parts & EXCLUDED_DIR_NAMES:
        return True
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    return False


if __name__ == "__main__":
    main()
