from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(path), low_memory=False, **kwargs)


def save_csv(df: pd.DataFrame, path: Path | str, index: bool = False, **kwargs) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index, encoding="utf-8-sig", **kwargs)
    return path


def save_json(payload: dict, path: Path | str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def require_files(base: Path, names: Iterable[str]) -> list[Path]:
    paths = [base / name for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    return paths
