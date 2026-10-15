# Pretrained NBR checkpoints

These three files are copied byte-for-byte from the supplied source tree's
`forest_disturbance/savemodel/` directory. They contain PET model state dictionaries
for a single NBR index, 2 classes, embedding dimension 64, 1 attention head,
3 encoder layers, and feed-forward dimension 128. Class 0 means no disturbance;
class 1 means disturbance in the temporal window.

| File | Window | SHA-256 |
| --- | ---: | --- |
| PET_NBR_W7.pth | 7 | e223a42ea52ec1f4d05c93076ffdfd740ef6e8c26fbc925b90a255f799688a2e |
| PET_NBR_W9.pth | 9 | 3fff7433bbb64e36fab2b6f301c603511beba660f6101b3fe40881f417b7d0ae |
| PET_NBR_W11.pth | 11 | 6399e2dd77d8aba28595368ba9de80309fbba0a9ba6266e689dad9399cc9337d |

These raw checkpoints do not record training samples, split IDs, index scale,
or test metrics. Do not infer that a filename alone certifies reproduction of the
paper's reported accuracy. The original raster path divides stored index values
by 10000; use `--scale-factor 1` for already normalized index rasters and verify
the distribution for your own data. Use a separate validation set before mapping
new areas. New training checkpoints store architecture and scale metadata.

Other early Transformer, random forest, NDVI/NDMI, and TOA experiments are excluded
from this maintained NBR pipeline. See [migration](../docs/migration.md).
