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
from utils.stats_utils import ols_hc3, summarize_numeric, two_group_tests

PAIR_METRICS = [
    "mean_spacing", "median_time_headway", "mean_relative_speed", "mean_acceleration",
    "std_acceleration", "mean_jerk", "std_jerk", "mean_abs_lateral_offset", "min_mttc", "mean_drac", "tit_3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 3: bicycle versus vehicle leader heterogeneity.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S)
    p.add_argument("--max-headway", type=float, default=DEFAULT_MAX_HEADWAY_S)
    p.add_argument("--max-lateral-offset", type=float, default=DEFAULT_MAX_LATERAL_OFFSET_M)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = args.input or args.output_dir / "processed" / "argo_features.csv"
    tables = ensure_dir(args.output_dir / "tables" / "exp_4_3_leader_heterogeneity")
    raw = pd.read_csv(path, low_memory=False)
    df = apply_filters(raw, args.min_duration, args.max_headway, args.max_lateral_offset)
    df = df[df["follower_group"].isin(["AV", "HV"]) & df["leader_group"].isin(["bicycle", "vehicle"])].copy()

    frame_summary = summarize_numeric(df, ["follower_group", "leader_group"], PRIMARY_METRICS)
    pair = pair_level_metrics(df)
    pair_summary = summarize_numeric(pair, ["follower_group", "leader_group"], PAIR_METRICS)
    tests = []
    for follower in ["AV", "HV"]:
        tests.append(two_group_tests(pair, "analysis_group", f"{follower}-bicycle", f"{follower}-vehicle", PAIR_METRICS))
    tests = pd.concat(tests, ignore_index=True)

    pair["leader_bicycle"] = pair["leader_group"].eq("bicycle").astype(int)
    pair["follower_av"] = pair["follower_group"].eq("AV").astype(int)
    pair["leader_bicycle_x_follower_av"] = pair["leader_bicycle"] * pair["follower_av"]
    regressions = ols_hc3(pair, PAIR_METRICS, level="pair")

    save_csv(df.groupby("analysis_group").agg(frames=("pair_id", "size"), pairs=("pair_id", "nunique")).reset_index(),
             tables / "sample_size_after_filtering.csv")
    save_csv(frame_summary, tables / "leader_heterogeneity_frame_summary.csv")
    save_csv(pair, tables / "leader_heterogeneity_pair_level_data.csv", float_format="%.10g")
    save_csv(pair_summary, tables / "leader_heterogeneity_pair_summary.csv")
    save_csv(tests, tables / "leader_type_pair_level_tests.csv")
    save_csv(regressions, tables / "leader_heterogeneity_ols_hc3.csv")

    print(f"[DONE] Experiment 3: {tables}")


if __name__ == "__main__":
    main()
