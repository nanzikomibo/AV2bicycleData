from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import DEFAULT_OUTPUT_DIR, JERK_ABNORMAL_THRESHOLD
from utils.io_utils import ensure_dir, save_csv
from utils.metrics import pair_level_metrics
from utils.stats_utils import trajectory_quality_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 1: AV2 trajectory and metric quality assessment.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--input", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = args.input or args.output_dir / "processed" / "argo_features.csv"
    tables = ensure_dir(args.output_dir / "tables" / "exp_4_1_quality")
    df = pd.read_csv(path, low_memory=False)

    quality = trajectory_quality_summary(df, "analysis_group", JERK_ABNORMAL_THRESHOLD)
    save_csv(quality, tables / "trajectory_quality_summary_by_group.csv")
    for state_col in ["interaction_state", "motion_state"]:
        state = df.groupby(["analysis_group", state_col], dropna=False).size().reset_index(name="frames")
        state["share"] = state["frames"] / state.groupby("analysis_group")["frames"].transform("sum")
        save_csv(state, tables / f"{state_col}_distribution.csv")

    print(f"[DONE] Experiment 1: {tables}")


if __name__ == "__main__":
    main()
