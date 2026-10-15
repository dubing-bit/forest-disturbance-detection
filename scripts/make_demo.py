"""Generate synthetic input files for a quick installation/inference smoke test."""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    cube = rng.normal(6500, 100, (21, 24, 32)).astype(np.int16)
    cube[9:, 5:18, 8:24] -= 4500
    cube[:, 0, 0] = -9999
    with rasterio.open(
        args.directory / "annual_nbr.tif", "w", driver="GTiff", count=21,
        height=24, width=32, dtype="int16", nodata=-9999, crs="EPSG:32650",
        transform=from_origin(500000, 4500000, 30, 30),
    ) as dst:
        dst.write(cube)
        for index in range(21):
            dst.set_band_description(index + 1, str(2000 + index))
    stable = rng.normal(0.65, 0.02, (40, 9))
    changed = rng.normal(0.65, 0.02, (40, 9))
    changed[:, 4:] -= 0.45
    table = np.column_stack((np.concatenate((stable, changed)), np.repeat([0, 1], 40)))
    np.savetxt(args.directory / "training_windows.csv", table, delimiter=",", fmt="%.6f",
               header=",".join([f"t{i}" for i in range(9)] + ["label"]), comments="")
    print(f"Synthetic files written to {args.directory.resolve()}")
    print("This example tests the software pipeline; it does not validate mapping accuracy.")


if __name__ == "__main__":
    main()
