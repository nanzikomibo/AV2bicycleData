from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import DEFAULT_MAX_HEADWAY_S, DEFAULT_MAX_LATERAL_OFFSET_M, DEFAULT_MIN_DURATION_S, DEFAULT_OUTPUT_DIR
from utils.io_utils import ensure_dir, save_csv
from utils.metrics import PRIMARY_METRICS, apply_filters, pair_level_metrics
from utils.stats_utils import summarize_numeric, two_group_tests

PAIR_METRICS = [
    "mean_spacing", "median_time_headway", "mean_relative_speed", "mean_acceleration",
    "std_acceleration", "mean_jerk", "std_jerk", "mean_lateral_offset",
    "mean_abs_lateral_offset", "min_mttc", "mean_drac", "max_drac", "tit_3", "tit_1p5", "p_mttc_lt_1p5", "p_mttc_lt_3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 2: AV versus HV following a bicycle leader.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S)
    p.add_argument("--max-headway", type=float, default=DEFAULT_MAX_HEADWAY_S)
    p.add_argument("--max-lateral-offset", type=float, default=DEFAULT_MAX_LATERAL_OFFSET_M)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = args.input or args.output_dir / "processed" / "argo_features.csv"
    tables = ensure_dir(args.output_dir / "tables" / "exp_4_2_av_hv_bicycle")
    raw = pd.read_csv(path, low_memory=False)
    df = apply_filters(raw, args.min_duration, args.max_headway, args.max_lateral_offset)
    df = df[df["follower_group"].isin(["AV", "HV"]) & df["leader_group"].eq("bicycle")].copy()
    if set(df["analysis_group"].unique()) != {"AV-bicycle", "HV-bicycle"}:
        raise RuntimeError(f"Expected AV-bicycle and HV-bicycle groups, got {sorted(df['analysis_group'].unique())}")

    frame_summary = summarize_numeric(df, ["analysis_group"], PRIMARY_METRICS)
    pair = pair_level_metrics(df)
    pair_summary = summarize_numeric(pair, ["analysis_group"], PAIR_METRICS)
    tests = two_group_tests(pair, "analysis_group", "AV-bicycle", "HV-bicycle", PAIR_METRICS)
    sample = df.groupby("analysis_group").agg(frames=("pair_id", "size"), pairs=("pair_id", "nunique")).reset_index()
    save_csv(sample, tables / "sample_size_after_filtering.csv")
    save_csv(frame_summary, tables / "behavior_summary_frame_level.csv")
    save_csv(pair, tables / "behavior_pair_level_data.csv", float_format="%.10g")
    save_csv(pair_summary, tables / "behavior_summary_pair_level.csv")
    save_csv(tests, tables / "av_vs_hv_pair_level_tests.csv")

    print(f"[DONE] Experiment 2: {tables}")


if __name__ == "__main__":
    main()
