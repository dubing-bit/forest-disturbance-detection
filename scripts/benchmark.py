"""Compare historical row-model/scalar-rule prediction with minibatched mapping.

Uses the real supplied checkpoint and synthetic in-memory time series. Disk I/O
and checkpoint loading are excluded from both times; no pixelwise model-call
baseline is invented. The row baseline mirrors the original raster implementation.
"""

import argparse
from dataclasses import asdict
import json
import os
import platform
from time import perf_counter

import numpy as np
import torch

from forest_disturbance.inference import PredictConfig, pad_series, predict_series
from forest_disturbance.model import load_model
from forest_disturbance.rules import detect_change


def scalar_change(window, config):
    differences = np.insert(np.diff(window.astype(np.float64)), 0, 0)[2:-2]
    for offset in np.argsort(differences, kind="stable"):
        position = offset + 2
        if differences[offset] < config.min_drop:
            slope, intercept = np.polyfit(
                [0, 1, 3, 4], window[[position - 2, position - 1, position + 1, position + 2]], 1,
            )
            if slope < config.min_slope and intercept > config.min_intercept:
                return position
    return -1


def row_baseline(model, cube, config):
    years, height, width = cube.shape
    data = pad_series(cube.reshape(years, -1).T / config.scale_factor,
                      config.window_size, config.padding)
    result = np.zeros(height * width, dtype=np.int16)
    with torch.inference_mode():
        for start in range(0, years, config.step):
            for row in range(height):
                first = row * width
                windows = np.ascontiguousarray(data[first:first + width, start:start + config.window_size])
                predicted = model(torch.from_numpy(windows).unsqueeze(-1)).argmax(-1).numpy()
                for col in np.flatnonzero(predicted == 1):
                    local = scalar_change(windows[col], config)
                    position = start + local - config.window_size // 2
                    if local >= 0 and 0 <= position < years:
                        year = config.start_year + position
                        pixel = first + col
                        if result[pixel] == 0 or year < result[pixel]:
                            result[pixel] = year
    return result


def timed(function, repeats):
    times = []
    for _ in range(repeats):
        begin = perf_counter()
        result = function()
        times.append(perf_counter() - begin)
    return result, float(np.median(times))


def processor_name():
    name = platform.processor()
    if not name and os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        except OSError:
            pass
    return name or "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/PET_NBR_W9.pth")
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--years", type=int, default=21)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.height, args.width, args.years, args.threads, args.repeats) < 1:
        parser.error("dimensions, threads and repeats must be positive")
    torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint)
    model_batches = []
    model.register_forward_hook(lambda module, inputs, output: model_batches.append(len(inputs[0])))
    config = PredictConfig(window_size=model.config.window_size, batch_size=args.batch_size)
    rng = np.random.default_rng(42)
    cube = rng.normal(6500, 200, (args.years, args.height, args.width)).astype(np.float32)
    cube[args.years // 2:, :, args.width // 2:] -= 4500
    series = cube.reshape(args.years, -1).T
    predict_series(model, series[:32], config)  # warm both model and vectorized code
    model_batches.clear()
    baseline, baseline_seconds = timed(lambda: row_baseline(model, cube, config), args.repeats)
    baseline_calls = len(model_batches) // args.repeats
    baseline_neural_windows = sum(model_batches) // args.repeats
    model_batches.clear()
    batched, batched_seconds = timed(lambda: predict_series(model, series, config), args.repeats)
    batched_calls = len(model_batches) // args.repeats
    batched_neural_windows = sum(model_batches) // args.repeats
    np.testing.assert_array_equal(baseline, batched)
    windows = rng.uniform(-0.2, 0.9, (10000, config.window_size)).astype(np.float32)
    scalar, scalar_seconds = timed(
        lambda: np.array([scalar_change(row, config) for row in windows]), args.repeats,
    )
    vector, vector_seconds = timed(lambda: detect_change(windows), args.repeats)
    np.testing.assert_array_equal(scalar, vector)
    report = {
        "platform": platform.platform(), "processor": processor_name(),
        "logical_cpus": os.cpu_count(), "torch": torch.__version__, "numpy": np.__version__,
        "device": "cpu", "threads": args.threads, "repeats": args.repeats,
        "checkpoint": args.checkpoint, "shape_T_H_W": list(cube.shape),
        "configuration": asdict(config), "outputs_equal": True,
        "row_baseline_seconds": baseline_seconds, "batched_seconds": batched_seconds,
        "mapping_speedup": baseline_seconds / batched_seconds,
        "row_baseline_model_calls": baseline_calls,
        "batched_model_calls": batched_calls,
        "row_baseline_neural_windows": baseline_neural_windows,
        "batched_neural_windows": batched_neural_windows,
        "rules_samples": len(windows), "rules_scalar_seconds": scalar_seconds,
        "rules_vector_seconds": vector_seconds, "rules_speedup": scalar_seconds / vector_seconds,
        "scope": "Synthetic in-memory CPU mapping; excludes disk I/O and checkpoint loading. No accuracy claim.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
