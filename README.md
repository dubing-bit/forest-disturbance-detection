# Forest Disturbance Detection

**English** | [简体中文](README.zh-CN.md)

## Code source

This repository contains code associated with the following paper:

Du, B., Yuan, Z., Bo, Y., and Zhang, Y. (2023). **A Combined Deep Learning and Prior Knowledge Constraint Approach for Large-Scale Forest Disturbance Detection Using Time Series Remote Sensing Data.** *Remote Sensing*, 15(12), 2963. [Paper](https://www.mdpi.com/2072-4292/15/12/2963) · [DOI](https://doi.org/10.3390/rs15122963).

The code detects forest disturbance years from annual spectral-index time series using a position-encoded Transformer and prior-knowledge constraints. NBR checkpoints for 7-, 9-, and 11-year windows are provided in [`models/`](models/). Study-area imagery and the original training and reference datasets are not included.

## Installation

Requirements: Python 3.10 or later, Git, and a CPU or a CUDA-capable GPU.

```bash
git clone https://github.com/dubing-bit/forest-disturbance-detection.git
cd forest-disturbance-detection
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For CUDA, install the appropriate PyTorch build using the [official installation instructions](https://pytorch.org/get-started/locally/), then install this project. `--device auto` selects CUDA when available and otherwise uses the CPU.

## Running examples

Run the following commands from the repository directory with the environment activated. `python -m forest_disturbance` can replace `forest-disturbance` in every example.

### Example data

Generate a synthetic annual NBR stack and training CSV, then run prediction:

```bash
python scripts/make_demo.py
forest-disturbance predict --input outputs/demo/annual_nbr.tif --output outputs/demo/disturbance_year.tif --checkpoint models/PET_NBR_W9.pth --start-year 2000
```

These synthetic files demonstrate the input and output formats.

### Raster prediction

Prepare a georeferenced multiband GeoTIFF containing one NBR band per consecutive year, ordered from oldest to newest. For example, a 2000–2020 stack contains 21 annual bands. The input must include its CRS, affine transform, and correct NoData or validity-mask information. Cloud masking, annual compositing, index calculation, gap filling, and smoothing must be completed upstream as needed.

This example uses an 11-year checkpoint, a temporal step of 4, and a slope threshold of −0.05:

```bash
forest-disturbance predict --input data/annual_nbr_2000_2020.tif --output outputs/disturbance_year.tif --checkpoint models/PET_NBR_W11.pth --start-year 2000 --step 4 --min-slope -0.05 --scale-factor 10000 --block-size 128 --batch-size 4096 --device auto
```

Input values are divided by `--scale-factor`; use `10000` for indices stored at that scale and `1` for normalized values. An optional `--mask data/forest_mask.tif` must be a single-band raster with the same dimensions, CRS, and affine transform. Only finite, valid, positive mask pixels are processed. A pixel with any missing or invalid year is excluded. An unmasked value of `0` is treated as an observation.

The output is a single-band `int16` GeoTIFF preserving the input grid and CRS:

| Value | Meaning |
| --- | --- |
| Positive integer, e.g. `2009` | Earliest detected disturbance calendar year |
| `0` | No accepted disturbance at an eligible pixel |
| `-32768` | NoData, an incomplete time series, or an excluded mask pixel |

### Training

Prepare a CSV with one fixed-length temporal window per row and a binary `label`: `0` for stable forest and `1` for a disturbance-containing window. A 9-year CSV uses this header:

```csv
t0,t1,t2,t3,t4,t5,t6,t7,t8,label
```

```bash
forest-disturbance train --csv outputs/demo/training_windows.csv --checkpoint outputs/model.pth --window-size 9 --scale-factor 1 --epochs 100 --patience 20 --seed 42
```

All columns except the label are temporal features unless `--feature-columns` is specified. Select feature columns explicitly when the CSV contains IDs, coordinates, or other metadata. Convert disturbance-year labels into binary window labels before training.

New checkpoints store the training scale factor, which prediction reads by default. This assumes the training CSV and prediction raster use the same stored units. Override prediction's `--scale-factor` when those units differ.

### Year evaluation

Prepare a CSV containing `reference` and `prediction` columns, with `0` for stable forest and positive integers for calendar years. Remove NoData samples before evaluation.

```bash
forest-disturbance evaluate --csv data/validation.csv --tolerance 0
```

The JSON output includes year accuracy, binary accuracy, disturbance precision, recall, F1, omission rate, and commission rate. `--tolerance 0` requires the same year; `--tolerance 1` allows a difference of one year. An event outside the tolerance counts as both an omission and a commission. Metrics with a zero denominator are reported as `null`.

## Deployment example

For a Linux server or workstation, install the repository in a virtual environment as described above. An activated shell is optional when using the environment's Python executable directly. For a checkout at `/srv/forest-disturbance-detection`, run:

```bash
/srv/forest-disturbance-detection/.venv/bin/python -m forest_disturbance predict \
  --input /data/annual_nbr_2000_2020.tif \
  --output /data/results/disturbance_year.tif \
  --checkpoint /srv/forest-disturbance-detection/models/PET_NBR_W11.pth \
  --start-year 2000 \
  --step 4 \
  --scale-factor 10000 \
  --block-size 128 \
  --batch-size 4096 \
  --device cpu \
  --threads 4
```

Replace the paths with the deployment's actual locations and ensure the output directory is writable. Use `--device cuda:0` for a configured CUDA environment. Reduce `--batch-size` if device memory is insufficient, or `--block-size` if host memory is insufficient. Add `--overwrite` to replace an existing output file.

## Parameters

### Shared prediction and training parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or an explicit device such as `cuda:0` |
| `--threads` | `min(4, logical CPU count)` | Number of PyTorch CPU threads |

### Prediction: `forest-disturbance predict`

| Parameter | Default | Description |
| --- | --- | --- |
| `--input` | Required | Annual, chronologically ordered multiband index GeoTIFF |
| `--output` | Required | Output disturbance-year GeoTIFF |
| `--checkpoint` | Required | Matching NBR checkpoint or a checkpoint from training |
| `--start-year` | Required | Calendar year represented by the first input band |
| `--window-size` | Inferred from checkpoint | Temporal window length; an explicit value must match the checkpoint |
| `--step` | `1` | Temporal window step; the paper uses 2, 2, and 4 for windows 7, 9, and 11 |
| `--scale-factor` | Checkpoint metadata, otherwise `10000` | Divide stored input values by this number |
| `--block-size` | `128` | Spatial tile side length in pixels |
| `--batch-size` | `4096` | Maximum number of temporal windows per batch |
| `--min-slope` | `-0.05` | Candidate slope must be strictly below this threshold |
| `--min-intercept` | `0.2` | Candidate intercept must be strictly above this threshold |
| `--min-drop` | `-0.1` | Candidate adjacent-year difference must be strictly below this threshold |
| `--padding` | `legacy` | `legacy`: source-program padding layout; `edge`: endpoint repetition; `linear`: endpoint extension by 0.001 per position |
| `--mask` | None | Aligned, single-band forest mask; valid positive values are included |
| `--overwrite` | Disabled | Allow replacement of an existing output |

### Training: `forest-disturbance train`

| Parameter | Default | Description |
| --- | --- | --- |
| `--csv` | Required | CSV of fixed-length temporal windows and binary labels |
| `--checkpoint` | Required | Destination for the best validation-loss checkpoint |
| `--window-size` | `9` | Temporal feature count; choose `7`, `9`, or `11` |
| `--feature-columns` | All columns except the label | Space-separated temporal columns, in time order |
| `--label-column` | `label` | Binary label column |
| `--scale-factor` | `1` | Divide stored training features by this number |
| `--epochs` | `100` | Maximum training epochs |
| `--batch-size` | `64` | Training and validation batch size |
| `--learning-rate` | `0.001` | Adam learning rate |
| `--validation-fraction` | `0.2` | Fraction of each class reserved for validation |
| `--patience` | `20` | Epochs without validation-loss improvement before stopping |
| `--seed` | `42` | Random seed |
| `--no-balance` | Disabled | Disable class-balanced sampling in the training set |
| `--overwrite` | Disabled | Allow replacement of an existing checkpoint |

### Evaluation: `forest-disturbance evaluate`

| Parameter | Default | Description |
| --- | --- | --- |
| `--csv` | Required | CSV containing reference and predicted calendar years |
| `--reference-column` | `reference` | Reference-year column |
| `--prediction-column` | `prediction` | Predicted-year column |
| `--tolerance` | `0` | Nonnegative allowed difference in years |

Each command also supports `--help`.

## References

Du, Bing, Zhanliang Yuan, Yanchen Bo, and Yusha Zhang. 2023. “A Combined Deep Learning and Prior Knowledge Constraint Approach for Large-Scale Forest Disturbance Detection Using Time Series Remote Sensing Data.” *Remote Sensing* 15 (12): 2963. [https://doi.org/10.3390/rs15122963](https://doi.org/10.3390/rs15122963).

## Citation and license

If you use this code in research, please cite the associated paper:

```bibtex
@article{du2023forest,
  title={A Combined Deep Learning and Prior Knowledge Constraint Approach for Large-Scale Forest Disturbance Detection Using Time Series Remote Sensing Data},
  author={Du, Bing and Yuan, Zhanliang and Bo, Yanchen and Zhang, Yusha},
  journal={Remote Sensing},
  volume={15},
  number={12},
  pages={2963},
  year={2023},
  doi={10.3390/rs15122963}
}
```

Citation metadata is available in [`CITATION.cff`](CITATION.cff). The code is distributed under the [Apache License 2.0](LICENSE); see [`NOTICE`](NOTICE) for attribution. Checkpoint information is available in [`models/README.md`](models/README.md).
