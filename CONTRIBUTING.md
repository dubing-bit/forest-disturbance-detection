# Contributing

Create a Python 3.10+ environment and run `python -m pip install -e ".[dev]"`.
Before submitting a change, run `ruff check src tests scripts`, `pytest -q`,
and `python -m build`. Include a minimal reproducible example for bugs.

Keep raster I/O bounded by spatial tiles and model inputs bounded by minibatches.
Changes to temporal padding, thresholds, window selection, model layout or output
year semantics need an independent scalar-reference regression test and a note in
the documentation. Benchmarks must report workload, hardware, batch/thread settings
and output equality. Synthetic performance does not establish scientific accuracy.

Never commit study imagery, private coordinates, credentials, machine-specific
paths, experiment logs or serialized full Python model objects. New contributions
are distributed under this repository's Apache-2.0 license.
