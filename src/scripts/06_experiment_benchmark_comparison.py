from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import DEFAULT_MAX_HEADWAY_S, DEFAULT_MAX_LATERAL_OFFSET_M, DEFAULT_MIN_DURATION_S, DEFAULT_OUTPUT_DIR
from utils.io_utils import ensure_dir, save_csv
from utils.metrics import PRIMARY_METRICS, apply_filters, pair_level_metrics
from utils.stats_utils import summarize_numeric, tail_probabilities, two_group_tests

PAIR_METRICS = [
    "mean_spacing", "median_time_headway", "mean_relative_speed", "mean_acceleration",
    "std_acceleration", "mean_jerk", "std_jerk", "mean_abs_lateral_offset",
    "min_mttc", "mean_drac", "max_drac", "tit_3", "tit_1p5", "p_mttc_lt_1p5", "p_mttc_lt_3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 5: AV2 versus naturalistic benchmark comparison.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--argo-input", type=Path, default=None)
    p.add_argument("--benchmark-dir", type=Path, default=None)
    p.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S)
    p.add_argument("--max-headway", type=float, default=DEFAULT_MAX_HEADWAY_S)
    p.add_argument("--max-lateral-offset", type=float, default=DEFAULT_MAX_LATERAL_OFFSET_M)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    argo_path = args.argo_input or args.output_dir / "processed" / "argo_features.csv"
    benchmark_dir = args.benchmark_dir or args.output_dir / "processed" / "benchmark"
    benchmark_files = sorted(benchmark_dir.glob("*_features.csv"))
    if not benchmark_files:
        raise FileNotFoundError(f"No processed benchmark files found in {benchmark_dir}")
    tables = ensure_dir(args.output_dir / "tables" / "exp_4_5_benchmark_comparison")

    argo = pd.read_csv(argo_path, low_memory=False)
    argo = apply_filters(argo, args.min_duration, args.max_headway, args.max_lateral_offset)
    argo = argo[argo["follower_group"].isin(["AV", "HV"]) & argo["leader_group"].isin(["bicycle", "vehicle"])].copy()
    argo_pair = pair_level_metrics(argo)

    benchmark_pairs = []
    frame_summaries = []
    tail_tables = []
    presence_rows = []
    acc_arrays: list[np.ndarray] = []
    jerk_arrays: list[np.ndarray] = []
    for path in benchmark_files:
        print(f"[INFO] Experiment 5 reading {path.name}", flush=True)
        data = pd.read_csv(path, low_memory=False)
        filtered = apply_filters(data, args.min_duration, args.max_headway, args.max_lateral_offset)
        dataset_name = str(data["dataset_name"].iloc[0]) if len(data) else path.stem.replace("_features", "")
        pair = pair_level_metrics(filtered)
        pair["analysis_group"] = "Benchmark overall"
        benchmark_pairs.append(pair)
        frame_summaries.append(summarize_numeric(filtered, ["dataset_name"], PRIMARY_METRICS))
        tail_tables.append(tail_probabilities(filtered, "dataset_name", "mttc"))
        presence_rows.append({
            "dataset_name": dataset_name, "frames_after_filtering": len(filtered),
            "pairs_after_filtering": filtered["pair_id"].nunique(),
            "mean_acceleration": filtered["f_a_used"].mean(), "median_acceleration": filtered["f_a_used"].median(),
            "jerk_mean": filtered["f_jerk"].mean(), "jerk_std": filtered["f_jerk"].std(),
            "lateral_missing_ratio": filtered["lateral_offset"].isna().mean(),
        })
        acc_arrays.append(pd.to_numeric(filtered["f_a_used"], errors="coerce").dropna().to_numpy())
        jerk_arrays.append(pd.to_numeric(filtered["f_jerk"], errors="coerce").dropna().to_numpy())
        del data, filtered, pair
        gc.collect()

    benchmark_pair = pd.concat(benchmark_pairs, ignore_index=True)
    combined_pair = pd.concat([argo_pair, benchmark_pair], ignore_index=True, sort=False)
    combined_summary = summarize_numeric(combined_pair, ["analysis_group"], PAIR_METRICS)
    tests = []
    for group in ["AV-bicycle", "HV-bicycle", "AV-vehicle", "HV-vehicle"]:
        tests.append(two_group_tests(combined_pair, "analysis_group", group, "Benchmark overall", PAIR_METRICS))

    all_acc = np.concatenate(acc_arrays) if acc_arrays else np.array([])
    all_jerk = np.concatenate(jerk_arrays) if jerk_arrays else np.array([])
    overall_validation = pd.DataFrame([{
        "dataset_name": "Benchmark overall", "frames_with_acceleration": len(all_acc),
        "f_a_used_mean": np.mean(all_acc), "f_a_used_median": np.median(all_acc), "f_a_used_std": np.std(all_acc, ddof=1),
        "frames_with_jerk": len(all_jerk), "f_jerk_mean": np.mean(all_jerk),
        "f_jerk_median": np.median(all_jerk), "f_jerk_std": np.std(all_jerk, ddof=1),
    }])
    if not abs(overall_validation.loc[0, "f_a_used_mean"]) < 10 or overall_validation.loc[0, "f_jerk_std"] == 0:
        raise RuntimeError("Benchmark acceleration/jerk validation failed after filtering.")

    scale = combined_pair.groupby(["dataset_source", "dataset_name", "analysis_group"], dropna=False).agg(
        pairs=("pair_id", "nunique"), frames=("frames", "sum"), median_duration_s=("duration_s", "median"),
        mean_duration_s=("duration_s", "mean")
    ).reset_index()
    save_csv(scale, tables / "dataset_scale_and_duration.csv")
    save_csv(pd.DataFrame(presence_rows), tables / "benchmark_dataset_presence_and_validation.csv")
    save_csv(overall_validation, tables / "benchmark_overall_acceleration_jerk_validation.csv")
    save_csv(pd.concat(frame_summaries, ignore_index=True), tables / "benchmark_frame_summary_by_dataset.csv")
    save_csv(pd.concat(tail_tables, ignore_index=True), tables / "benchmark_mttc_tail_by_dataset.csv")
    save_csv(combined_pair, tables / "argo_vs_benchmark_pair_level_data.csv", float_format="%.10g")
    save_csv(combined_summary, tables / "argo_vs_benchmark_pair_level_summary.csv")
    save_csv(pd.concat(tests, ignore_index=True), tables / "argo_vs_benchmark_pair_level_tests.csv")

    print(f"[DONE] Experiment 5: {tables}")


if __name__ == "__main__":
    main()
