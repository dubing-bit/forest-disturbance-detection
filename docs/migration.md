# Migration from the supplied experiment directory

The public package is `forest_disturbance`. The original directory mixed multiple generations of classifiers, preprocessing experiments, local data paths, model objects, cached bytecode, and TensorBoard runs. The published source tree concentrates on the PETransformer image workflow and provides a portable CLI and Python API.

## Source mapping

The historical paths below are relative to the original `实验程序/forest_disturbance_detect/forest_disturbance/` directory. They describe provenance; these files are not runtime dependencies.

| Historical source | Current component | Change |
| --- | --- | --- |
| `model/PETransformerModel.py` | [`src/forest_disturbance/model.py`](../src/forest_disturbance/model.py) | Preserves PET parameter names and computation; adds architecture validation and portable, weights-only checkpoint loading. |
| `classification/Change_trendv2.py` | [`src/forest_disturbance/rules.py`](../src/forest_disturbance/rules.py) | Replaces pixel-level sorting and `np.polyfit` with array operations and closed-form coefficients. |
| `implement/Forest_disturbance_detection.py` | [`src/forest_disturbance/inference.py`](../src/forest_disturbance/inference.py) | Replaces row classification and pixel date loops with tiled input, bounded minibatches, and vectorized dating. |
| `classification/Read_image.py` | Rasterio I/O in `inference.py` | Reads and writes by spatial window, validates grid/CRS, and preserves georeferencing. |
| `Dataset/Get_data.py`, `train.py` | [`data.py`](../src/forest_disturbance/data.py), [`training.py`](../src/forest_disturbance/training.py) | Uses explicit pre-extracted CSV windows, stratified validation, and metadata checkpoints. |
| `savemodel/PET_NBR_W{7,9,11}.pth` | [`models/`](../models/) | Keeps the three supplied NBR PET state dictionaries; the accompanying manifest records integrity hashes. |

The earlier `TransformerModel.py`, root `model/transformer_len9*.pth`, random-forest experiments, and duplicate `程序/` scripts belong to other experiment paths. They are not interchangeable with PET checkpoints. NDMI, NDVI, TOA, and `PET_NBR_W1_11.pth` variants were also excluded from the public model selection because the source did not establish their intended relationship to the published NBR experiment.

## Behavior that needs attention

| Topic | Historical behavior | Current behavior |
| --- | --- | --- |
| Result encoding | Stored a zero-based temporal location; `0` also meant no event, losing first-year events. | Stores `start_year + location`; `0` means no event and `-32768` means skipped/NoData. Do not add a year offset again. |
| Temporal limits | Negative locations could index from the end of the output array; some out-of-range locations could fail. | Rejects dates outside the original input interval. |
| Default window/step | Main image example used W11/step4; other scripts used different settings. | Defaults to W9/step1. Set both explicitly for a particular historical experiment. The checkpoint window length must match. |
| Padding | Image code inserted right padding before the last observation. CSV code used a different construction. | `legacy` retains the image insertion layout by default; `edge` and `linear` are explicit alternatives. |
| Missing years and masks | No consistent eligibility contract across scripts. | A missing or invalid year skips the entire pixel; optional positive forest masks restrict processing. There is no automatic gap filling. |
| Model mode | Image inference used `eval()`; some CSV prediction scripts omitted it. | All prediction uses evaluation mode with gradients disabled. |
| Rule thresholds | The image path used intercept `0.2`, whereas a CSV prediction path used `0.0`. | Defaults follow the image path: slope `-0.05`, intercept `0.2`, immediate drop `-0.1`. |
| Numerical boundaries | Repeated `np.polyfit` and sorted candidate lists. | Float64 closed form and strict comparisons; rounding at an exact threshold can differ. |
| Checkpoints | Some scripts serialized whole model objects as `.pkl`. | Accepts PET tensor state dictionaries and new metadata checkpoints through `weights_only=True`. |

`legacy` padding is a compatibility setting for the source image workflow, not a guarantee of byte-for-byte reproduction of every historical script. Corrected year encoding, bounds checks, modern numerical libraries, and the original scripts' differing data preparation can change outputs. See [the method contract](method.md) before comparing maps.

## Training changes

Training now consumes one fixed-length temporal window per CSV row, with a binary label (`0` no change, `1` disturbance). The loader does not interpret a calendar-year label as a class label or automatically select a window around that year. Prepare windows and their labels upstream. Feature columns are read in their declared order; specify feature columns if the CSV also contains IDs or metadata. Scaling must agree with inference.

The validation split is stratified and seeded. Class balancing applies only to training; each validation example is evaluated once. The old program also weighted-sampled its validation set and rounded model scores before computing validation loss. The new trainer uses the model's log probabilities with negative log-likelihood, does not round scores, and saves the best validation-loss checkpoint from the first epoch onward. These changes mean a retrained model is not expected to reproduce the original weights or learning curve.

The built-in random split is for training monitoring and early stopping. It does not replace an independent spatial/temporal test set, a held-out geographic evaluation, or a disturbance-year accuracy assessment. Training hyperparameters and sample construction must be recorded for any scientific reproduction attempt.

## Public-tree cleanup

Before cleanup, the original source and weights are archived locally under the ignored `.publish/` directory. The public repository excludes:

- TensorBoard event files and run folders, cached bytecode, editor configuration, and duplicate script trees.
- Serialized full-model and random-forest objects unused by the package.
- Unused checkpoint variants and duplicate early Transformer weights.
- Hard-coded drive paths and personal-machine import modifications.

The local archive is not part of the public repository or installable distribution. It retains historical material for reference without requiring it to run the new package. The public checkpoint manifest preserves the identity of the three selected NBR files; exclusion of other experiments is not evidence that those experiments are scientifically invalid.

The article's open-access license and the repository's software license apply to different artifacts. See the repository license for code terms and [CITATION.cff](../CITATION.cff) for the research citation.
