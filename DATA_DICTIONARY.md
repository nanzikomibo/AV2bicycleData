# Data dictionary

## Data tiers

`data/raw/` preserves the CSV inputs used by the workflow. `data/processed/` contains standardized frame-level features and one record per interaction pair. `data/derived/` contains summary statistics, hypothesis tests, validation audits, and robustness tables.

## Common fields

| Field | Meaning |
| --- | --- |
| `scenario_id` / `dataset_name` | Source scenario or benchmark dataset label. |
| `pair_id` | Interaction-pair identifier used to aggregate frames. |
| `frame` / `time_s` | Frame index and time in seconds. |
| `f_*` / `l_*` | Follower and leader variables, respectively. |
| `spacing` | Longitudinal bumper-to-bumper spacing in metres. |
| `lateral_offset` | Signed lateral separation in metres. |
| `time_headway` | Spacing divided by follower speed, in seconds. |
| `ttc` / `mttc` | Time-to-collision and modified time-to-collision, in seconds. |
| `drac` | Deceleration rate to avoid collision. |
| `tit_deficit_3p0` / `tit_deficit_1p5` | Frame-level time-integrated TTC deficits. |
| `duration_s` | Interaction-pair duration in seconds. |
| `analysis_group` | Follower-leader category used in analyses. |

The scripts in `src/utils/metrics.py` define the exact transformations and units.
