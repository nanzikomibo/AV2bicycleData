# Leader-cyclist interaction data and analysis code

This repository contains CSV datasets and Python scripts for processing leader-cyclist interaction trajectories and reproducing the tabular statistical analyses in the associated study. It deliberately excludes all figure-generation code and image outputs.

The complete CSV archive is available from Zenodo: [10.5281/zenodo.21914938](https://doi.org/10.5281/zenodo.21914938).

## Repository layout

- `data/raw/`: repository-provided CSV inputs, including AV2-derived leader-follower trajectories and benchmark inputs. The complete upstream AV2 scenarios are separate Parquet files.
- `data/raw/scenario_parquet/`: empty release directory with instructions for separately downloading official AV2 scenario Parquet files.
- `data/raw/rawdata_csv/`: one example/checking scenario CSV; it is not the complete upstream AV2 input collection.
- `data/processed/`: frame-level features and pair-level records produced by preprocessing.
- `data/derived/`: analysis tables used to report trajectory quality, behavior, safety, benchmark comparison, and robustness results.
- `src/`: preprocessing, metric construction, statistical analysis, and validation scripts.
- `tests/`: release-boundary checks.

## Requirements and execution

Use Python 3.10 or later. Install dependencies with `python -m pip install -r requirements.txt`, then run:

```bash
python src/run_all.py
```

The default paths read `data/raw/` and write tabular results to `outputs/`. Existing processed and derived CSV files are supplied so the published results can be inspected without rerunning the full pipeline. To reuse processed data, use `python src/run_all.py --skip-preprocessing`.

Run the release checks with:

```bash
python -m pytest tests -q
```

## Complete CSV archive

The complete raw, processed, and derived CSV archive is published on Zenodo: [10.5281/zenodo.21914938](https://doi.org/10.5281/zenodo.21914938). Download and extract the archive into the repository root so that the `data/raw/`, `data/processed/`, and `data/derived/` directory structure is retained.

## Official AV2 source data

The complete upstream input is the Argoverse 2 motion-forecasting training collection. Original `scenario_*.parquet` files are not included in this GitHub repository because the training split contains roughly 200,000 scenes and requires substantial download time and local storage. The complete AV2 motion-forecasting dataset contains 250,000 scenarios across its dataset splits.

The official public source is:

```text
s3://argoverse/datasets/av2/motion-forecasting/train/
```

After installing the AWS CLI, download one selected scenario without AWS credentials:

```bash
aws s3 cp --no-sign-request \
  s3://argoverse/datasets/av2/motion-forecasting/train/<SCENE_ID>/scenario_<SCENE_ID>.parquet \
  data/raw/scenario_parquet/scenario_<SCENE_ID>.parquet
```

Use `src/scripts/00_extract_av2_pairs_from_parquet.py` to extract AV/HV-following-bicycle interactions from downloaded Parquet files. It supports bounded downloads through `--download --start <N> --stop <N>` and processing existing local files through `--extract-only`. By default, extracted files are written to `outputs/from_official_parquet/`, so repository-provided CSVs are not overwritten. Pass the resulting frame-level file to the next step with `python src/run_all.py --argo-file outputs/from_official_parquet/av2_new_leader_follower_frame_level_dataset.csv`. See `data/raw/scenario_parquet/README.md` and the [official Argoverse download guide](https://argoverse.github.io/user-guide/getting_started.html) for details. Plan storage and network capacity before requesting the full collection; the GitHub release intentionally contains no original Parquet files.

## Time fields and privacy

Precise timestamps and frame times are retained because duration, speed, acceleration, time headway, TTC, MTTC, DRAC, and TiT calculations require temporal resolution. The files contain no direct personal identifiers, images, video, faces, license plates, or contact information. Timestamps are not intended for identity inference.

## Licenses

The Python code is available under the MIT License in `LICENSE`. Repository-added CSV data are available under CC BY 4.0; see `DATA_LICENSE.md`. External benchmark material remains subject to the original source terms.
