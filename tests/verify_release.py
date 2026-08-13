from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRECTORIES = ("data/raw", "data/processed", "data/derived", "src", "tests")
FORBIDDEN_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}


def main() -> None:
    for relative_path in REQUIRED_DIRECTORIES:
        if not (ROOT / relative_path).is_dir():
            raise RuntimeError(f"Missing directory: {relative_path}")

    source_files = list((ROOT / "src").rglob("*.py"))
    if not source_files:
        raise RuntimeError("No Python source files found.")
    for path in source_files:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        if "utils.plotting" in source or "matplotlib" in source.lower():
            raise RuntimeError(f"Plotting code found: {path}")

    csv_paths = list((ROOT / "data").rglob("*.csv"))
    if not csv_paths:
        raise RuntimeError("No CSV data files found.")
    for path in csv_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            if not next(csv.reader(handle), None):
                raise RuntimeError(f"CSV has no header: {path}")

    for path in ROOT.rglob("*"):
        if path.is_file() and (path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name == "plotting.py"):
            raise RuntimeError(f"Excluded release asset found: {path}")

    print(f"Validated {len(source_files)} Python files and {len(csv_paths)} CSV files.")


if __name__ == "__main__":
    main()
