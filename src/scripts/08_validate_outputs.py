from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import DEFAULT_OUTPUT_DIR
from utils.io_utils import save_csv, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the completed analysis and write an audit report.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    raw_counts = pd.read_csv(out / "tables" / "preprocessing" / "argo_pair_counts_by_raw_type.csv")
    canonical = pd.read_csv(out / "tables" / "preprocessing" / "argo_pair_counts_by_analysis_group.csv")
    for raw_name, canonical_name in [("bicycle-AV", "AV-bicycle"), ("bicycle-HV", "HV-bicycle")]:
        raw_row = raw_counts.loc[raw_counts["pair_type"] == raw_name]
        canonical_row = canonical.loc[canonical["analysis_group"] == canonical_name]
        passed = len(raw_row) == 1 and len(canonical_row) == 1 and int(raw_row.iloc[0]["frames"]) == int(canonical_row.iloc[0]["frames"]) and int(raw_row.iloc[0]["pairs"]) == int(canonical_row.iloc[0]["pairs"])
        check(f"classification_{canonical_name}", passed,
              f"raw={raw_row[['frames','pairs']].to_dict('records')}; canonical={canonical_row[['frames','pairs']].to_dict('records')}")

    overview = pd.read_csv(out / "tables" / "preprocessing" / "argo_input_overview.csv").iloc[0]
    check("av2_lateral_offset_complete", float(overview["lateral_offset_missing_ratio"]) == 0.0,
          f"missing_ratio={overview['lateral_offset_missing_ratio']}")

    benchmark = pd.read_csv(out / "tables" / "exp_4_5_benchmark_comparison" / "benchmark_overall_acceleration_jerk_validation.csv").iloc[0]
    check("benchmark_acceleration_plausible", abs(float(benchmark["f_a_used_mean"])) < 10 and abs(float(benchmark["f_a_used_median"])) < 10,
          f"mean={benchmark['f_a_used_mean']}; median={benchmark['f_a_used_median']}")
    check("benchmark_jerk_nonconstant", float(benchmark["f_jerk_std"]) > 0,
          f"mean={benchmark['f_jerk_mean']}; std={benchmark['f_jerk_std']}")

    processed_headers = []
    for path in [out / "processed" / "argo_features.csv", *sorted((out / "processed" / "benchmark").glob("*_features.csv"))]:
        processed_headers.extend(pd.read_csv(path, nrows=0).columns.tolist())
    check("two_dimensional_ttc_absent", not any("2d" in str(col).lower() for col in processed_headers),
          "No 2D-TTC columns are present in processed data.")
    required_safety = {"mttc", "drac", "tit_deficit_3p0", "tit_deficit_1p5"}
    check("mttc_drac_tit_columns_present", required_safety.issubset(processed_headers),
          f"missing={sorted(required_safety - set(processed_headers))}")
    argo_safety = pd.read_csv(out / "processed" / "argo_features.csv", usecols=["mttc", "drac", "tit_deficit_3p0", "tit_deficit_1p5"])
    finite_mttc = argo_safety["mttc"].replace([float("inf"), float("-inf")], float("nan")).dropna()
    check("mttc_nonnegative", bool((finite_mttc >= 0).all()), f"minimum={finite_mttc.min() if len(finite_mttc) else 'none'}")
    check("drac_and_tit_nonnegative", bool((argo_safety[["drac", "tit_deficit_3p0", "tit_deficit_1p5"]] >= 0).all().all()),
          "DRAC and TiT deficits are non-negative.")

    expected_dirs = [
        "exp_4_1_quality", "exp_4_2_av_hv_bicycle", "exp_4_3_leader_heterogeneity",
        "exp_4_4_safety", "exp_4_5_benchmark_comparison", "exp_4_6_robustness",
    ]
    for name in expected_dirs:
        table_files = list((out / "tables" / name).glob("*.csv"))
        check(f"outputs_{name}", bool(table_files), f"tables={len(table_files)}")

    result = pd.DataFrame(checks)
    save_csv(result, out / "validation_report.csv")
    save_json({"all_checks_passed": bool(result["passed"].all()), "checks": checks}, out / "validation_report.json")
    if not result["passed"].all():
        failed = result.loc[~result["passed"], ["check", "detail"]]
        raise RuntimeError("Output validation failed:\n" + failed.to_string(index=False))
    print(f"[VALIDATED] All {len(result)} checks passed.")


if __name__ == "__main__":
    main()
