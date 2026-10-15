"""Small GeoTIFF integration tests, including atomic publication on failure."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from forest_disturbance.inference import NODATA, PredictConfig, predict_raster, predict_series


class PositiveClassifier(torch.nn.Module):
    def forward(self, inputs):
        scores = torch.zeros((len(inputs), 2), device=inputs.device)
        scores[:, 1] = 1
        return scores


class FailingClassifier(PositiveClassifier):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("injected model failure")
        return super().forward(inputs)


class RasterInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.input = self.directory / "input.tif"
        self.mask = self.directory / "mask.tif"
        self.output = self.directory / "output.tif"
        self.transform = from_origin(500000, 3000000, 30, 30)
        self.config = PredictConfig(block_size=3, batch_size=11, start_year=1984)
        self.cube = np.full((9, 5, 7), 7000, dtype=np.float32)
        self.cube[3:, ::2, ::2] = 1000
        self.cube[2, 0, 1] = -9999
        self.cube[4, 4, 6] = np.nan
        self.write_raster(self.input, self.cube, nodata=-9999)

    def write_raster(self, path, values, *, nodata=None, transform=None, crs="EPSG:32649"):
        if values.ndim == 2:
            values = values[None]
        with rasterio.open(
            path, "w", driver="GTiff", count=values.shape[0], height=values.shape[1],
            width=values.shape[2], dtype=str(values.dtype), nodata=nodata,
            transform=self.transform if transform is None else transform, crs=crs,
        ) as dataset:
            dataset.write(values)

    def assert_no_partial_files(self):
        self.assertEqual(list(self.directory.glob(".output-*.tif")), [])

    def test_georeference_nodata_mask_and_edge_tiles(self):
        forest = np.ones((5, 7), dtype=np.float32)
        forest[0, 0] = 0
        forest[1, 1] = -9999
        forest[2, 2] = np.nan
        forest[2, 3] = np.inf
        self.write_raster(self.mask, forest, nodata=-9999)
        progress = []
        stats = predict_raster(
            self.input, self.output, PositiveClassifier(), self.config,
            mask_path=self.mask, progress=lambda done, total: progress.append((done, total)),
        )
        values = self.cube.copy()
        values[values == -9999] = np.nan
        series = values.reshape(9, -1).T
        series[~(np.isfinite(forest) & (forest > 0)).ravel()] = np.nan
        expected = predict_series(PositiveClassifier(), series, self.config).reshape(5, 7)
        with rasterio.open(self.output) as result, rasterio.open(self.input) as source:
            np.testing.assert_array_equal(result.read(1), expected)
            self.assertEqual(result.crs, source.crs)
            self.assertEqual(result.transform, source.transform)
            self.assertEqual(result.bounds, source.bounds)
            self.assertEqual(result.shape, (5, 7))
            self.assertEqual(result.count, 1)
            self.assertEqual(result.dtypes, ("int16",))
            self.assertEqual(result.nodata, NODATA)
            self.assertEqual(result.descriptions, ("earliest_disturbance_year",))
            self.assertEqual(json.loads(result.tags()["configuration"])["start_year"], 1984)
        self.assertEqual(progress, [(i, 6) for i in range(1, 7)])
        self.assertEqual(stats["valid_pixels"], int(np.count_nonzero(expected != NODATA)))
        self.assertEqual(stats["disturbed_pixels"], int(np.count_nonzero(expected > 0)))
        self.assertEqual(stats["skipped_pixels"], int(np.count_nonzero(expected == NODATA)))
        self.assertEqual(stats["valid_pixels"] + stats["skipped_pixels"], 35)
        self.assertGreater(stats["disturbed_pixels"], 0)
        self.assert_no_partial_files()

    def test_spatial_block_and_minibatch_sizes_do_not_change_output(self):
        expected = None
        for block_size, batch_size in ((1, 1), (3, 5), (20, 4096)):
            predict_raster(
                self.input, self.output, PositiveClassifier(),
                replace(self.config, block_size=block_size, batch_size=batch_size),
                overwrite=expected is not None,
            )
            with rasterio.open(self.output) as dataset:
                actual = dataset.read(1)
            if expected is None:
                expected = actual
            else:
                np.testing.assert_array_equal(actual, expected)

    def test_all_masked_pixels_skip_model_and_write_nodata(self):
        self.write_raster(self.mask, np.zeros((5, 7), dtype=np.uint8))
        model = FailingClassifier()
        stats = predict_raster(
            self.input, self.output, model, self.config, mask_path=self.mask,
        )
        self.assertEqual(model.calls, 0)
        self.assertEqual(stats["skipped_pixels"], 35)
        with rasterio.open(self.output) as dataset:
            np.testing.assert_array_equal(dataset.read(1), np.full((5, 7), NODATA))

    def test_existing_output_is_protected_without_overwrite(self):
        original = b"existing output must survive"
        self.output.write_bytes(original)
        with self.assertRaises(FileExistsError):
            predict_raster(self.input, self.output, PositiveClassifier(), self.config)
        self.assertEqual(self.output.read_bytes(), original)
        self.assert_no_partial_files()

    def test_failure_removes_partial_file_and_preserves_existing_output(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    self.output.write_bytes(b"old result")
                with self.assertRaisesRegex(RuntimeError, "injected model failure"):
                    predict_raster(
                        self.input, self.output, FailingClassifier(), self.config,
                        overwrite=existing,
                    )
                if existing:
                    self.assertEqual(self.output.read_bytes(), b"old result")
                    self.output.unlink()
                else:
                    self.assertFalse(self.output.exists())
                self.assert_no_partial_files()

    def test_progress_callback_failure_does_not_publish_partial_result(self):
        def fail_after_write(done, total):
            raise RuntimeError("injected progress failure")

        with self.assertRaisesRegex(RuntimeError, "injected progress failure"):
            predict_raster(
                self.input, self.output, PositiveClassifier(), self.config,
                progress=fail_after_write,
            )
        self.assertFalse(self.output.exists())
        self.assert_no_partial_files()

    def test_input_and_mask_cannot_be_used_as_output(self):
        before = self.input.read_bytes()
        with self.assertRaises(ValueError):
            predict_raster(self.input, self.input, PositiveClassifier(), self.config, overwrite=True)
        self.assertEqual(self.input.read_bytes(), before)
        self.write_raster(self.mask, np.ones((5, 7), dtype=np.uint8))
        mask_before = self.mask.read_bytes()
        with self.assertRaises(ValueError):
            predict_raster(
                self.input, self.mask, PositiveClassifier(), self.config,
                mask_path=self.mask, overwrite=True,
            )
        self.assertEqual(self.mask.read_bytes(), mask_before)

    def test_invalid_mask_grid_and_missing_crs_leave_no_output(self):
        cases = (
            {"values": np.ones((4, 7), dtype=np.uint8)},
            {"values": np.ones((2, 5, 7), dtype=np.uint8)},
            {"values": np.ones((5, 7), dtype=np.uint8), "crs": "EPSG:4326"},
            {"values": np.ones((5, 7), dtype=np.uint8),
             "transform": from_origin(500001, 3000000, 30, 30)},
        )
        for case in cases:
            with self.subTest(case=case):
                self.write_raster(self.mask, **case)
                with self.assertRaises(ValueError):
                    predict_raster(
                        self.input, self.output, PositiveClassifier(), self.config,
                        mask_path=self.mask,
                    )
                self.assertFalse(self.output.exists())
                self.assert_no_partial_files()
        self.write_raster(self.input, self.cube, crs=None)
        with self.assertRaises(ValueError):
            predict_raster(self.input, self.output, PositiveClassifier(), self.config)
        self.assertFalse(self.output.exists())
        self.assert_no_partial_files()


if __name__ == "__main__":
    unittest.main()
