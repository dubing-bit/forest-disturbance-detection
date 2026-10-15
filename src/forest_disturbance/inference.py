"""Bounded-memory batch inference and georeferenced raster streaming.

Only the current spatial tile and current minibatch are materialized. Windows
that cannot satisfy the final prior-knowledge conditions skip model execution.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable
import json
import math
import os
import tempfile

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

from .rules import detect_change

NODATA = -32768


@dataclass(frozen=True)
class PredictConfig:
    window_size: int = 9
    step: int = 1
    batch_size: int = 4096
    block_size: int = 128
    start_year: int = 2000
    scale_factor: float = 10000.0
    min_slope: float = -0.05
    min_intercept: float = 0.2
    min_drop: float = -0.1
    padding: str = "legacy"
    device: str = "cpu"

    def __post_init__(self):
        for name in ("window_size", "step", "batch_size", "block_size", "start_year"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.window_size < 5 or self.window_size % 2 != 1:
            raise ValueError("window_size must be odd and at least 5")
        if min(self.step, self.batch_size, self.block_size) < 1:
            raise ValueError("step, batch_size and block_size must be positive")
        if not math.isfinite(self.scale_factor) or self.scale_factor <= 0:
            raise ValueError("scale_factor must be finite and positive")
        if not all(math.isfinite(x) for x in (self.min_slope, self.min_intercept, self.min_drop)):
            raise ValueError("rule thresholds must be finite")
        if self.padding not in {"legacy", "edge", "linear"}:
            raise ValueError("padding must be legacy, edge or linear")
        if not 1 <= self.start_year <= 32767:
            raise ValueError("start_year must fit a positive int16 calendar year")


def pad_series(series: np.ndarray, window_size: int, mode: str = "legacy") -> np.ndarray:
    """Pad scaled (pixels, years) data, including exact historical insertion behavior."""
    half = window_size // 2
    if mode == "edge":
        return np.pad(series, ((0, 0), (half, half)), mode="edge")
    left = series[:, :1] - np.arange(half, 0, -1, dtype=np.float32) * 0.001
    if mode == "linear":
        right = series[:, -1:] + np.arange(1, half + 1, dtype=np.float32) * 0.001
        return np.concatenate((left, series, right), axis=1)
    if mode == "legacy":
        # np.insert(array, -1, last + .001) inserts BEFORE the last year each time.
        inserted = np.repeat(series[:, -1:] + 0.001, half, axis=1)
        return np.concatenate((left, series[:, :-1], inserted, series[:, -1:]), axis=1)
    raise ValueError("unknown padding mode")


def predict_series(model: torch.nn.Module, series: np.ndarray, config: PredictConfig) -> np.ndarray:
    """Return earliest calendar year, 0 for no event, or NODATA for incomplete pixels.

    Input shape is (pixels, years), in raw input units. All temporal values for a
    pixel must be finite. Class 1 means disturbance; class 0 means stable forest.
    """
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 2 or series.shape[1] == 0:
        raise ValueError("series must have shape (pixels, years), with at least one year")
    if config.start_year + series.shape[1] - 1 > 32767:
        raise ValueError("end year does not fit int16")
    model_config = getattr(model, "config", None)
    if model_config is not None and model_config.window_size != config.window_size:
        raise ValueError("prediction window size differs from the checkpoint")
    output = np.full(series.shape[0], NODATA, dtype=np.int16)
    valid = np.flatnonzero(np.isfinite(series).all(axis=1))
    if not valid.size:
        return output
    scaled = series[valid] / config.scale_factor
    if not np.isfinite(scaled).all():
        raise ValueError("scaled values overflow; check scale_factor")
    padded = pad_series(scaled, config.window_size, config.padding)
    starts = np.arange(0, series.shape[1], config.step)
    offsets = np.arange(config.window_size)
    earliest = np.full(valid.size, series.shape[1], dtype=np.int32)
    device = torch.device(config.device)
    model.to(device).eval()
    with torch.inference_mode():
        for begin in range(0, valid.size * starts.size, config.batch_size):
            flat = np.arange(begin, min(begin + config.batch_size, valid.size * starts.size))
            pixel = flat // starts.size
            temporal = flat % starts.size
            windows = np.ascontiguousarray(padded[pixel[:, None], starts[temporal, None] + offsets])
            # The final decision is classifier AND rule AND temporal bounds.
            # Applying the cheap necessary conditions first is equivalent for
            # the sample-independent PET model in eval mode and saves inference
            # on stable windows. Do not replace this with an approximate filter.
            local = detect_change(
                windows, min_slope=config.min_slope,
                min_intercept=config.min_intercept, min_drop=config.min_drop,
            )
            location = starts[temporal] + local - config.window_size // 2
            candidate = (local >= 0) & (location >= 0) & (location < series.shape[1])
            if not candidate.any():
                continue
            inputs = torch.from_numpy(windows[candidate]).unsqueeze(-1).to(device)
            logits = model(inputs)
            if logits.ndim != 2 or logits.shape != (int(candidate.sum()), 2):
                raise ValueError("model must return (batch, 2) class scores")
            if not torch.isfinite(logits).all():
                raise ValueError("model produced non-finite class scores")
            positive = logits.argmax(dim=1).eq(1).cpu().numpy()
            if not positive.any():
                continue
            np.minimum.at(earliest, pixel[candidate][positive], location[candidate][positive])
    output[valid] = np.where(
        earliest < series.shape[1], config.start_year + earliest, 0,
    ).astype(np.int16)
    return output


def predict_raster(
    input_path: str | Path, output_path: str | Path, model: torch.nn.Module,
    config: PredictConfig, *, mask_path: str | Path | None = None,
    overwrite: bool = False, progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Stream a chronological multiband index GeoTIFF to a one-band year map.

    Mask pixels must be positive, valid, and aligned exactly to the index stack.
    An atomic rename publishes the completed output; interrupted runs do not
    leave a partial result at the requested output path.
    """
    from contextlib import ExitStack

    source = Path(input_path).resolve()
    target = Path(output_path).resolve()
    if source == target or (mask_path and Path(mask_path).resolve() == target):
        raise ValueError("output must differ from the input and mask")
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    counts = {"valid_pixels": 0, "disturbed_pixels": 0, "skipped_pixels": 0}
    handle, temporary = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=".tif", dir=target.parent)
    os.close(handle)
    try:
        with ExitStack() as stack:
            src = stack.enter_context(rasterio.open(source))
            if src.crs is None:
                raise ValueError("input raster must declare a coordinate reference system")
            if config.start_year + src.count - 1 > 32767:
                raise ValueError("end year does not fit int16")
            mask = stack.enter_context(rasterio.open(mask_path)) if mask_path else None
            if mask and (mask.count != 1 or mask.shape != src.shape or mask.crs != src.crs
                         or mask.transform != src.transform):
                raise ValueError("forest mask must have one band and the same grid/CRS as the stack")
            profile = {
                "driver": "GTiff", "height": src.height, "width": src.width,
                "count": 1, "dtype": "int16", "crs": src.crs, "transform": src.transform,
                "nodata": NODATA, "compress": "deflate", "predictor": 2,
                "tiled": True, "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER",
            }
            dst = stack.enter_context(rasterio.open(temporary, "w", **profile))
            dst.set_band_description(1, "earliest_disturbance_year")
            dst.update_tags(
                method="PETransformer + prior knowledge", no_disturbance="0",
                configuration=json.dumps(asdict(config), sort_keys=True),
            )
            total = math.ceil(src.height / config.block_size) * math.ceil(src.width / config.block_size)
            completed = 0
            for row in range(0, src.height, config.block_size):
                for col in range(0, src.width, config.block_size):
                    window = Window(col, row, min(config.block_size, src.width - col),
                                    min(config.block_size, src.height - row))
                    cube = src.read(window=window, out_dtype="float32", masked=True)
                    values = np.asarray(cube.filled(np.nan)).reshape(src.count, -1).T
                    if mask:
                        forest = mask.read(1, window=window, masked=True)
                        include = (np.asarray((forest > 0).filled(False))
                                   & np.isfinite(forest.data)).ravel()
                        values[~include] = np.nan
                    result = predict_series(model, values, config).reshape(int(window.height), int(window.width))
                    dst.write(result, 1, window=window)
                    counts["valid_pixels"] += int(np.count_nonzero(result != NODATA))
                    counts["disturbed_pixels"] += int(np.count_nonzero(result > 0))
                    counts["skipped_pixels"] += int(np.count_nonzero(result == NODATA))
                    completed += 1
                    if progress:
                        progress(completed, total)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"output": str(target), "seconds": round(perf_counter() - started, 3), **counts}
