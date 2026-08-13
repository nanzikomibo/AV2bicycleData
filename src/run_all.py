from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import DEFAULT_ARGO_FILE, DEFAULT_OLD_DIR, DEFAULT_OUTPUT_DIR
from utils.io_utils import ensure_dir, save_json

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run preprocessing and statistical analyses.")
    p.add_argument("--argo-file", type=Path, default=DEFAULT_ARGO_FILE)
    p.add_argument("--old-dir", type=Path, default=DEFAULT_OLD_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--skip-preprocessing", action="store_true", help="Reuse processed files already under output-dir.")
    return p.parse_args()


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]
    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    started = datetime.now(timezone.utc)
    if not args.skip_preprocessing:
        run("01_prepare_argo_features.py", "--argo-file", args.argo_file, "--output-dir", args.output_dir)
        run("00_preprocess_benchmark.py", "--old-dir", args.old_dir, "--output-dir", args.output_dir)

    for script in [
        "02_experiment_trajectory_quality.py",
        "03_experiment_av_hv_bicycle_behavior.py",
        "04_experiment_leader_heterogeneity.py",
        "05_experiment_surrogate_safety.py",
        "06_experiment_benchmark_comparison.py",
        "07_experiment_robustness_analysis.py",
    ]:
        run(script, "--output-dir", args.output_dir)
    run("08_validate_outputs.py", "--output-dir", args.output_dir)

    packages = {}
    for name in ["pandas", "numpy", "scipy", "statsmodels"]:
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            packages[name] = f"unavailable: {exc}"
    save_json({
        "project": "leader-cyclist data processing and statistical analyses",
        "started_utc": started.isoformat(), "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "argo_file": str(args.argo_file.resolve()), "benchmark_dir": str(args.old_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()), "two_dimensional_ttc": "excluded",
        "idm": "removed",
    }, args.output_dir / "analysis_manifest.json")
    print(f"\n[DONE] Corrected analysis is available at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
