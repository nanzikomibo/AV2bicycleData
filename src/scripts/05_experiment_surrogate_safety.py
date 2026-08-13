from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import DEFAULT_MAX_HEADWAY_S, DEFAULT_MAX_LATERAL_OFFSET_M, DEFAULT_MIN_DURATION_S, DEFAULT_OUTPUT_DIR
from utils.io_utils import ensure_dir, save_csv
from utils.metrics import apply_filters, pair_level_metrics
from utils.stats_utils import low_mttc_pair_test, summarize_numeric, tail_probabilities


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 4: MTTC, DRAC, and TiT surrogate safety analysis (2D-TTC excluded).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S)
    p.add_argument("--max-headway", type=float, default=DEFAULT_MAX_HEADWAY_S)
    p.add_argument("--max-lateral-offset", type=float, default=DEFAULT_MAX_LATERAL_OFFSET_M)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = args.input or args.output_dir / "processed" / "argo_features.csv"
    tables = ensure_dir(args.output_dir / "tables" / "exp_4_4_safety")
    raw = pd.read_csv(path, low_memory=False)
    df = apply_filters(raw, args.min_duration, args.max_headway, args.max_lateral_offset)
    df = df[df["follower_group"].isin(["AV", "HV"]) & df["leader_group"].isin(["bicycle", "vehicle"])].copy()
    pair = pair_level_metrics(df)

    save_csv(tail_probabilities(df, "analysis_group", "mttc"), tables / "mttc_tail_probabilities_frame_level.csv")
    safety_metrics = ["min_mttc", "mean_drac", "max_drac", "tit_3", "tit_1p5", "p_mttc_lt_1p5", "p_mttc_lt_3"]
    save_csv(summarize_numeric(pair, ["analysis_group"], safety_metrics), tables / "mttc_drac_tit_pair_level_summary.csv")
    comparisons = [("AV-bicycle", "HV-bicycle"), ("AV-bicycle", "AV-vehicle"), ("HV-bicycle", "HV-vehicle")]
    tests = pd.concat([low_mttc_pair_test(pair, "analysis_group", a, b) for a, b in comparisons], ignore_index=True)
    save_csv(tests, tables / "low_mttc_pair_prevalence_tests.csv")
    save_csv(pair[["pair_id", "analysis_group"] + safety_metrics], tables / "mttc_drac_tit_pair_level_data.csv", float_format="%.10g")

    print(f"[DONE] Experiment 4: {tables}")


if __name__ == "__main__":
    main()
