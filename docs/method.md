# Method and implementation contract

This repository reorganizes the supplied research code into a Python package. It preserves the PETransformer classifier and the source program's prior-knowledge rules while replacing image-wide allocation and Python pixel loops with spatial tiles, model minibatches, and NumPy array operations.

## Relationship to the paper

The reference is Du, Yuan, Bo, and Zhang (2023), [A Combined Deep Learning and Prior Knowledge Constraint Approach for Large-Scale Forest Disturbance Detection Using Time Series Remote Sensing Data](https://doi.org/10.3390/rs15122963), *Remote Sensing*, 15(12), 2963. Relevant method sections were checked against the [publisher's PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-15-02963/article_deploy/remotesensing-15-02963.pdf).

The paper describes a two-stage method: classify fixed-length temporal windows with a position-encoded Transformer, then use temporal context to locate a disturbance year. Its S-DRI slope uses observations two years before and after the candidate, excluding the candidate itself. The tested window/stride pairs are 7/2, 9/2, and 11/4; the best reported combination is window 11 with S-DRI threshold −0.05. The reported 87.8% accuracy concerns the paper's reference data and evaluation, not this software release. See §§2.3–2.4 and Table 3.

## Input and output

- Input: a georeferenced raster with one band per consecutive year, ordered from oldest to newest. Each band contains the same single spectral index. Published weights are named for NBR; the software does not compute NBR, composite Landsat scenes, or reproduce the paper's upstream image preparation.
- Scaling: raw values are divided by `scale_factor`. `PredictConfig` defaults to `10000`; the CLI uses the checkpoint's scale metadata when available, otherwise `10000`. Use `1` for already scaled index values. Training CSV scaling is explicit and defaults to `1`.
- Eligibility: all years at a pixel must be finite and valid according to the raster mask/NoData metadata. A supplied forest mask must match the raster grid and CRS exactly; only positive, valid mask pixels are processed. No temporal interpolation or smoothing is performed.
- Output: one `int16` GeoTIFF band with the earliest accepted **calendar year**, `0` for an eligible pixel with no accepted event, and `-32768` for an incomplete or excluded pixel. CRS and affine transform are retained. `0` is a detection result, not a guarantee that the forest was undisturbed.

The first input band represents `start_year`. Unlike the historical zero-based output, an event in that first year can be represented. A result contains one year per pixel even if several windows suggest different events.

## Classifier

[`model.py`](../src/forest_disturbance/model.py) implements the state-dictionary-compatible PETransformer path from the original `model/PETransformerModel.py`. The supplied PET weights use one index feature, two classes, embedding dimension 64, one attention head, three encoder layers, and feed-forward dimension 128. Temporal maximum pooling, ReLU, and a final linear layer produce log probabilities. Class `1` indicates a disturbance-containing window; class `0` indicates no change.

Inference runs in evaluation mode with gradients disabled. Minibatches contain independent `(time, feature)` sequences; attention operates along each sequence's time axis. Changing minibatch or spatial tile size therefore does not intentionally change the method. Floating-point differences can still affect samples extremely close to class or rule boundaries.

Only `models/PET_NBR_W7.pth`, `models/PET_NBR_W9.pth`, and `models/PET_NBR_W11.pth` are distributed. Their names and tensor structure match the supplied PETransformer code, but a checkpoint has no embedded provenance establishing its exact training samples or association with a particular published accuracy. The loader infers dimensions from original PET state dictionaries and uses the source program's one-head architecture. New checkpoints include architecture and training metadata. Full pickled model objects are unsupported.

## Padding and temporal windows

For window length `W`, each end receives `W // 2` padding values after scaling. `PredictConfig` defaults to `W=9`, temporal step `1`, and `padding="legacy"`. The CLI infers `W` from its required checkpoint. These are software defaults; use `W=11, step=4` to select the paper's reported best window/stride pair, together with the matching W11 checkpoint.

Three padding modes are available:

| Mode | Behavior |
| --- | --- |
| `legacy` | Reproduces the historical image program's insertion layout. Left values decrease by `0.001` per padding position. The right padding copies `last + 0.001` **before** the final observation, because the source used `np.insert(..., -1, ...)`. |
| `edge` | Repeats the first and last observed values outside the series. |
| `linear` | Extends the first value downwards and the last value upwards by `0.001` per padding position. |

For example, padding `[a, b, c]` by two values on each side yields `[a-.002, a-.001, a, b, c+.001, c+.001, c]` in `legacy` mode. This unusual end layout is retained for compatibility; it shifts the final observation inside the padded sequence and can influence late detections. `edge` and `linear` have different boundary behavior and require separate validation. None of these modes creates real evidence about unobserved future years.

Window starts are `0, step, 2*step, ... < number_of_years` in padded coordinates. A local candidate `p` maps to input position `start + p - W//2`. Candidates outside the original input interval are discarded.

## Prior-knowledge rules

[`rules.py`](../src/forest_disturbance/rules.py) follows the original `classification/Change_trendv2.py`, rather than silently replacing its conditions with a new interpretation of the paper. For a window `x`, candidate positions are zero-based `p = 2, ..., W-3`. Define:

```text
drop = x[p] - x[p-1]
y = [x[p-2], x[p-1], x[p+1], x[p+2]]
slope = (-2*y[0] - y[1] + y[2] + 2*y[3]) / 10
intercept = mean(y) - 2*slope
```

These are the closed-form least-squares coefficients for the original regression times `[0, 1, 3, 4]`. The default tests are all strict:

```text
drop < -0.1
slope < -0.05
intercept > 0.2
```

Among candidates passing all conditions, the most negative immediate drop wins; equal drops select the earlier position. If no candidate passes, the window has no dated event. This is equivalent to the original ascending-drop search away from numerical threshold boundaries. The closed form removes per-pixel sorting and repeated `np.polyfit` calls. All coefficient calculations use float64.

The drop and intercept filters, and the strict slope comparison, are implementation choices inherited from the supplied image program. They are not additional claims validated by the paper. In particular, the paper's threshold description uses `<=`, while this code retains `<`.

## Computational changes and validation scope

[`inference.py`](../src/forest_disturbance/inference.py) reads one spatial tile, selects complete pixels, and gathers temporal windows into bounded minibatches. It evaluates the exact vectorized dating rules and temporal bounds first, then classifies only windows that could produce a valid event. The final condition is still classifier class 1 AND the same rule AND valid bounds; changing this order does not change the accepted dates for the sample-independent PET model in evaluation mode. Stable windows can skip neural computation entirely. Arbitrary custom models with cross-sample dependencies are outside this contract. Nonfinite scores are checked only for windows actually classified. It stores only the earliest accepted event per pixel. There is no Python loop over individual pixels, no image-wide temporal-window tensor, and no image-wide disturbance-year cube.

Batching changes execution granularity; it does not spatially downsample or omit eligible pixels. Memory use depends on tile size, temporal length, model size, and minibatch size. Runtime depends on hardware, disk access, valid-pixel fraction, and temporal settings. Synthetic correctness checks and timing comparisons cannot establish field accuracy or a universal speedup.

The original training/reference rasters and manually interpreted evaluation labels are absent from the supplied directory. This release therefore makes no claim to reproduce the paper's quantitative results. Independent, spatially and temporally appropriate test data are needed before interpreting scientific accuracy, including late-period and low-magnitude disturbances.
