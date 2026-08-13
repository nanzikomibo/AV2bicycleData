from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import DEFAULT_OUTPUT_DIR
from utils.io_utils import ensure_dir, save_csv
from utils.metrics import apply_filters, pair_level_metrics
from utils.stats_utils import summarize_numeric

SUMMARY_METRICS = [
    "mean_spacing", "median_time_headway", "mean_relative_speed", "mean_acceleration",
    "std_acceleration", "mean_jerk", "std_jerk", "mean_abs_lateral_offset",
    "min_mttc", "mean_drac", "max_drac", "tit_3", "tit_1p5", "p_mttc_lt_1p5", "p_mttc_lt_3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 6: threshold robustness analysis; no IDM component.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--argo-input", type=Path, default=None)
    p.add_argument("--benchmark-dir", type=Path, default=None)
    return p.parse_args()


def settings() -> list[tuple[str, str, float, float, float, bool]]:
    return [
        ("baseline", "duration>=5s; headway<=60s; lateral<=2.5m", 5.0, 60.0, 2.5, False),
        ("duration_threshold", ">=3s", 3.0, 60.0, 2.5, False),
        ("duration_threshold", ">=5s", 5.0, 60.0, 2.5, False),
        ("duration_threshold", ">=8s", 8.0, 60.0, 2.5, False),
        ("headway_threshold", "<=30s", 5.0, 30.0, 2.5, False),
        ("headway_threshold", "<=60s", 5.0, 60.0, 2.5, False),
        ("lateral_threshold", "<=1.5m", 5.0, 60.0, 1.5, False),
        ("lateral_threshold", "<=2.0m", 5.0, 60.0, 2.0, False),
        ("lateral_threshold", "<=2.5m", 5.0, 60.0, 2.5, False),
        ("lateral_threshold", "no filter", 5.0, 60.0, 1e12, False),
        ("near_stationary_filter", "removed", 5.0, 60.0, 2.5, True),
    ]


def pairs_for_settings(data: pd.DataFrame, include_av2_groups: bool) -> list[pd.DataFrame]:
    outputs = []
    for setting_name, setting_value, duration, headway, lateral, remove_stationary in settings():
        filtered = apply_filters(data, duration, headway, lateral, remove_stationary)
        if include_av2_groups:
            filtered = filtered[
                filtered["follower_group"].isin(["AV", "HV"])
                & filtered["leader_group"].isin(["bicycle", "vehicle"])
            ]
        pair = pair_level_metrics(filtered)
        if not include_av2_groups:
            pair["analysis_group"] = "Benchmark overall"
        pair["setting_name"] = setting_name
        pair["setting_value"] = setting_value
        outputs.append(pair)
    return outputs


def main() -> None:
    args = parse_args()
    argo_path = args.argo_input or args.output_dir / "processed" / "argo_features.csv"
    benchmark_dir = args.benchmark_dir or args.output_dir / "processed" / "benchmark"
    benchmark_files = sorted(benchmark_dir.glob("*_features.csv"))
    tables = ensure_dir(args.output_dir / "tables" / "exp_4_6_robustness")

    argo = pd.read_csv(argo_path, low_memory=False)
    all_pairs = pairs_for_settings(argo, include_av2_groups=True)
    del argo
    gc.collect()
    for path in benchmark_files:
        print(f"[INFO] Experiment 6 reading {path.name}", flush=True)
        data = pd.read_csv(path, low_memory=False)
        all_pairs.extend(pairs_for_settings(data, include_av2_groups=False))
        del data
        gc.collect()

    pair_data = pd.concat(all_pairs, ignore_index=True, sort=False)
    summary = summarize_numeric(pair_data, ["setting_name", "setting_value", "analysis_group"], SUMMARY_METRICS)
    counts = pair_data.groupby(["setting_name", "setting_value", "analysis_group"], dropna=False).agg(
        n_pairs=("pair_id", "nunique"), n_frames=("frames", "sum")
    ).reset_index()
    summary = summary.merge(counts, on=["setting_name", "setting_value", "analysis_group"], how="left")
    summary = summary.rename(columns={"analysis_group": "group"})

    wide = summary.pivot_table(index=["setting_name", "setting_value", "metric"], columns="group", values="mean").reset_index()
    if {"AV-bicycle", "HV-bicycle"}.issubset(wide.columns):
        wide["AV_minus_HV_bicycle"] = wide["AV-bicycle"] - wide["HV-bicycle"]
    save_csv(pair_data, tables / "robustness_pair_level_data.csv", float_format="%.10g")
    save_csv(summary, tables / "robustness_threshold_summary.csv")
    save_csv(wide, tables / "robustness_av_minus_hv_bicycle.csv")

    print(f"[DONE] Experiment 6: {tables}")


if __name__ == "__main__":
    main()
