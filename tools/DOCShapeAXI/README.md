# sadt-docshapeaxi

Grade a surface mesh with a shapeaxi classifier, and show what the model
looked at.

Served as `DOCShapeAXI`. Package `sadt_docshapeaxi`; the public surface is
`run()`.

## Provenance

| | |
|---|---|
| Upstream | `DOCShapeAXI/DOCShapeAXI.py`, `DOCShapeAXI_CLI/DOCShapeAXI_CLI.py` |
| Upstream commit | `ea37083` (2026-08-14), the last upstream commit touching the module |
| Algorithm modified | no -- the prediction and the GradCAM are upstream's, step for step |

## What it does

Two passes over a folder of surfaces:

* **Grade.** A `SaxiMHAFBClassification` (or `SaxiMHAFBRegression`) checkpoint
  reads each surface and returns one number per case, written to
  `files_<anatomy>_prediction.csv`.
* **Explain.** captum's GradCAM is taken on the last block of the model's
  convnet, resized to the renderer's 224x224 view, clipped to its 1st and 99th
  percentile, median-filtered, and written back onto the surface as a point
  array -- one per grade. A grade nobody can check is a grade nobody should act
  on, which is why this is on by default.

`data_type` and `task` are the two questions a clinician can answer; the
checkpoint, the network class and the number of classes all follow from them
through one table (`catalog.py`). Upstream spreads the same table across
`find_model_name`, `find_nn_type` and a `num_classes` attribute set as a side
effect of the first, and publishes all three as CLI arguments that must agree
with each other.

## What changed

**Nothing is downloaded.** Upstream fetches `model_path.json` from GitHub on
every run, reads a release URL out of it and pulls the `.ckpt` into the output
folder. Here the bundle is named by the caller and hosted by the deployment,
like every other model in this repository: a run that silently pulls whatever
revision the release currently points at is a result nobody can reproduce. A
test asserts the fetch did not come across.

**A task with no model is refused, not answered with another one.** Upstream's
`find_model_name` branches on the anatomy first, so a mandibular condyle
reaches `condyles_4_class` whether `task` says binary, severity or regression.
Only the airway has a model per task. Here the pair is resolved against the
table and an unsupported one is refused by name -- and the panel narrows the
options (`options_when`) so a client never provokes it.

**The manifest is rebuilt.** Upstream appends to `files_<type>.csv` with
`open(..., 'a')`, so re-running into the same output folder grades every
surface twice.

**Surfaces are found recursively.** Upstream's single `os.listdir` sees a flat
folder only, and takes `.vtk` alone; a cohort arrives as a folder of folders as
often as not.

**No WSL path translation.** `linux2windows_path` turned `C:\...` into
`/mnt/c/...` because the module ran the CLI inside WSL on Windows. The server
runs on the server.

**Dropped, with the reason.** The environment setup (`install_pytorch.py`, the
conda env, the WSL install), the progress log file the panel polled for a
percentage, and the model-download button: all of them are deployment or UI,
and none is a grading parameter.

**Nothing was missed by oversight.** Every CLI parameter is here or derived:
`input_dir` is `meshes`, `output_dir` is itself, `data_type` and `task` are
published, `model`/`nn`/`num_classes` are derived from those two, and
`log_path` went with the progress file.

## Versions

`torch==2.11.0+cu128`, `pytorch3d==0.7.9+pt2110cu128`, `shapeaxi>=2.0.2`,
`captum==0.9.0`, `opencv-python-headless==4.12.0.88`, `numpy==2.2.6`,
Python 3.11.

Two pins are worth the sentence they cost:

* **torch 2.11, not the 2.8.0 the imaging tools use.** This tool shares
  Crown_Seg's engine and its pytorch3d wheel tag names one torch exactly.
  Aligning them means revalidating both tools' outputs, so they stay together
  and the deployment carries two runtimes rather than three.
* **numpy 2.2.6, not 2.3.2.** `opencv-python-headless` 4.12 caps numpy below
  2.3, and OpenCV is here for one `cv2.resize` in the GradCAM path. captum
  0.8.0 caps it below 2.0 outright, which is why 0.9.0 is pinned instead.
  Upstream meets neither constraint: its shared conda env leaves numpy to
  whatever pip last resolved.

The shapeaxi engine is an **extra**, exactly as in Crown_Seg: a plain
`uv sync` imports the package and publishes the schema without a CUDA stack;
`uv sync --extra grading` builds the real thing.

## Validated against

| | |
|---|---|
| The catalog | yes -- every published (anatomy, task) pair resolves to a checkpoint, a network and a class count, and the layout's option narrowing is derived from the same table |
| Surface discovery and the manifest | yes -- on real `.vtk` files written by the tests |
| Checkpoint resolution | yes -- flat bundle, nested bundle, a file named directly, and a bundle missing the model |
| `pytest -m "not models"` | 18 passed |
| pyflakes | 0 |
| A real grading run | **not done.** Needs `--extra grading` and the hosted bundle. This is the step that would show whether the ported prediction and GradCAM still produce upstream's numbers, and it has not been run. |

## Working on it

```bash
uv sync                      # schema and the tests below
uv sync --extra grading      # the engine, ~5 GB
uv run pytest -q -m "not models"
uv run --no-project --with pyflakes -- python -m pyflakes src
.venv/bin/python ../../scripts/describe.py .
```
