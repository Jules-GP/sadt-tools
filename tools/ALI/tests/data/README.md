# Test data

Nothing is committed here. The suite builds its own 16³ synthetic CBCT volumes
with SimpleITK and its own four-triangle meshes with VTK, and stubs the agent
that walks the volume — so mode detection, DICOM recognition, weight discovery,
the landmark vocabulary, output naming, tree preservation, the run report and
every cross-argument rule are exercised without a checkpoint or a GPU.

That covers 57 of the 59 tests. The two that need real files are marked and
skipped by default.

## Staging the model bundles

Weights are never committed and never packaged. The server fetches them into
`/DATA/ALI/models`; the server's `scripts/data-manifest.yml` holds the URLs and
sizes.

| Bundle | Layout | Size | Where |
|---|---|---|---|
| `ALI_CBCT_Models` | `<landmark>/<scale>/*.pth`, scales `1` and `0-3` | 4.7 GB | ALI release, eight per-region archives unpacked side by side |
| `ALI_IOS_Models` | flat `Upper_O_model.pth`, `Lower_C_model.pth`, … | 0.9 GB | ALIDDM release v1.0.4 |

```bash
scripts/setup-models.sh --tool ALI      # from a slicer-remote-tool-server checkout
```

The two layouts are mutually exclusive, which is what lets the engine reject a
bundle of the wrong kind with a message rather than a crash midway through a
run.

## Running the real-model tests

Both are pointed at their data by environment variable, so nothing here has to
know where a deployment stages it:

```bash
# CBCT: the agent actually walks a volume. Needs a card.
SADT_ALI_CBCT_MODELS=/path/to/DATA/ALI/models/ALI_CBCT_Models \
SADT_ALI_SCAN=/path/to/DATA/ALI/testfiles/MG_test_scan.nii.gz \
uv run pytest -m "gpu and models"
```

| Variable | What |
|---|---|
| `SADT_ALI_CBCT_MODELS` | the CBCT bundle |
| `SADT_ALI_SCAN` | one real CBCT volume |
| `SADT_ALI_IOS_MODELS` | the IOS bundle |
| `SADT_ALI_MESH` | one **segmented** intra-oral mesh |

The IOS half additionally needs the `ios` extra, which compiles pytorch3d from
source and therefore a CUDA toolkit — `nvcc`, not just a driver. The Crown_Seg
chain test needs that tool built with *its* `segmentation` extra:

```bash
uv sync --extra ios                       # compiles pytorch3d, needs nvcc
cd ../Crown_Seg && uv sync --extra segmentation
```

`test_crownseg_output_feeds_straight_into_ali` then wants three fixtures under
this directory: `raw_arch.vtk` (an *unsegmented* intraoral scan),
`crownseg_checkpoint.pth`, and the `ALI_IOS_Models` bundle. The published
ALIDDM test mesh `T1_01_U_segmented.vtk` is already segmented and therefore
does **not** exercise the chain — Crown_Seg passes it straight through. Use a
raw arch, or none at all and let the test skip.

Write manual runs into the repository's gitignored `output/` directory, never
next to the inputs.
