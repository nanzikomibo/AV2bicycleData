"""Repository paths and analysis settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
DATA_ROOT = REPOSITORY_ROOT / "data" / "raw"

DEFAULT_ARGO_FILE = (
    DATA_ROOT
    / "AV2_processed_new_leader_follower_dataset"
    / "av2_new_leader_follower_frame_level_dataset.csv"
)
DEFAULT_OLD_DIR = DATA_ROOT / "combine dataset"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs"

OLD_FILES = [
    "combined_bdp_mon_leader_follower.csv",
    "combined_bdp_tue_leader_follower.csv",
    "combined_frank_tue_leader_follower.csv",
    "combined_frank_wed_leader_follower.csv",
    "combined_hckstr_leader_follower.csv",
    "combined_nuk_leader_follower.csv",
    "combined_TUM_leader_follower.csv",
]

DEFAULT_MIN_DURATION_S = 5.0
DEFAULT_MAX_HEADWAY_S = 60.0
DEFAULT_MAX_LATERAL_OFFSET_M = 2.5
DEFAULT_NEAR_STATIONARY_SPEED_MPS = 0.5
BENCHMARK_DT_S = 0.04
JERK_ABNORMAL_THRESHOLD = 15.0
RANDOM_SEED = 42
