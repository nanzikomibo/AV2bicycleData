from __future__ import annotations

import csv
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_contains_expected_directories() -> None:
    for relative_path in ["data/raw", "data/processed", "data/derived", "src", "tests"]:
        assert (ROOT / relative_path).is_dir()


def test_release_excludes_plotting_assets_and_code() -> None:
    forbidden_suffixes = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
    for path in (ROOT / "src").rglob("*"):
        if path.is_file():
            assert path.suffix.lower() not in forbidden_suffixes, path
            assert path.name != "plotting.py", path
            if path.suffix == ".py":
                text = path.read_text(encoding="utf-8")
                assert "utils.plotting" not in text
                assert "matplotlib" not in text.lower()


def test_python_sources_compile() -> None:
    sources = list((ROOT / "src").rglob("*.py"))
    assert sources
    for path in sources:
        py_compile.compile(path, doraise=True)


def test_csv_files_have_headers() -> None:
    csv_paths = list((ROOT / "data").rglob("*.csv"))
    assert csv_paths
    for path in csv_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            assert next(csv.reader(handle)), path
