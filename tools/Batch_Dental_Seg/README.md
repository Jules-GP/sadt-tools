# sadt-batchdentalseg

Segments teeth and jaw structures on a dental CT or CBCT, with the
DentalSegmentator family of nnUNet v2 models. One bundle per model; the bundle
you pick chooses the label table with it.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, path
`BATCHDENTALSEG/BATCHDENTALSEGLib/SegmentationWidget.py`, commit `6df3fab`
(2026-08-05), by way of `slicer-remote-tool-server`'s `tools/BatchDentalSeg/` —
whose history this repository carries, so `git log --follow` on
`src/sadt_batchdentalseg/pipeline.py` reaches back through it.

Upstream pins **not** kept, for the same reason as AMASSS: this package pins
torch 2.8.0+cu128 and nnunetv2 2.8.1, the stack the deployed server runs. See
"Versions".

Upstream is a 2940-line Qt widget and most of it is not this pipeline. Already
absent before this port, each for a stated reason: the queue table, the RAM
watchdog, killing nnUNet processes a crashed scan left behind, the "free
memory" button, the per-scan cool-down, restoring the queue from disk — all of
which exist because the widget runs inside Slicer on a clinician's laptop and
has to survive being out of memory. Also not ported: the runtime model download
from GitHub releases (a tool holding patient data does not make outbound calls
mid-run), the auto-crop (upstream applies it only when its RAM preflight fails,
and it changes what the network sees), the mirroring resolution (a button the
user presses after looking at the result), and the mesh exports.

Changes made by this port — the algorithm is untouched:

- **Scratch space lives under `output_dir`** (`.batchdentalseg_work/`, removed
  before returning). A tool must not write outside the directory it is given.
- **`segment()` returns the run report** instead of a `SegmentationRun`; tools
  no longer call each other.
- **Zip extraction removed.** The server unpacks archives before `run()`.
- **The GPU semaphore is gone** — each call is its own process now.
- **`device` is a `Literal["cuda", "cpu"]`**, so the schema publishes both
  options. `model` deliberately stays a plain `Path`: which bundles exist is a
  property of the deployment, not of this package, so the picker comes from the
  server's data listing rather than from a hard-coded set here that would go
  stale the moment a bundle is not staged.
- **`device` and `tile_step_size` are arguments**, not server settings.
  `run()` must not read the environment, and `tile_step_size` moves the
  segmentation. `BATCHDENTALSEG_MAX_GPU_JOBS` went with the semaphore.
- **`check_dependencies()` is gone.** It existed so a deployment missing torch
  reported that once per run rather than once per scan; the lockfile makes it
  unreachable.

## What it does

| | |
|---|---|
| Inputs | `scans`: one scan (`.nii`/`.nii.gz`/`.nrrd`/`.nrrd.gz`/`.gipl`/`.gipl.gz`) or a folder of them, searched recursively. `model`: the bundle. `output_dir`: where results go. |
| Outputs | One `<scan>_<ID>` label volume per scan, mirroring the input tree, plus `BatchDentalSeg_report.json`. With `separate_segments`, one binary file per label present. |
| Model files | A bundle directory holding `dataset.json`, `plans.json` and `fold_0/checkpoint_final.pth`. Fetched by the server into `/DATA/BatchDentalSeg/models`; see `tests/data/README.md` for staging one by hand. |
| GPU | Used when available; `device="cpu"` works and is much slower. |

Models, and what each labels:

| Bundle | Segments |
|---|---|
| `DentalSegmentator` | Adult. Upper Skull (maxilla included), Mandible, Upper Teeth, Lower Teeth, Mandibular canal |
| `PediatricDentalSeg` | Paediatric, the same five |
| `NasoMaxillaDentSeg` | Six — the maxilla is split out of the Upper Skull, which shifts every later value |
| `UniversalLab` | Every tooth individually in Universal numbering, deciduous included, plus Mandible, Maxilla and Mandibular canal |

Three things worth knowing before reading a result:

- **The bundle's folder name selects the model**, and the label table follows
  from it. That is deliberate: a separate "which labels" argument would let a
  caller pair bundle X with the labels of Y, and the result would be a
  plausible volume with every structure named wrong.
- **The label values are part of the trained weights**, not a presentation
  choice — they are the integers the network emits. Renaming a catalog entry is
  safe; renumbering one silently mislabels anatomy. The report ships the table
  next to the results, because the segmentation is a volume of integers and
  without it they mean nothing.
- **`separate_segments` writes only labels PRESENT in the scan.** A full
  UniversalLab run would otherwise produce 55 files per patient, most empty, and
  an empty mask is indistinguishable from a structure the model failed on.

### A cross-repo contract this repository cannot enforce

A key in `catalogs.MODELS` **must equal** the folder name the server's
`scripts/data-manifest.yml` downloads that bundle into. A key that drifts makes
an installed model unselectable. The server-side suite asserted this against the
manifest directly; that file lives in the other repository and a tool package
cannot reach it, so the check is gone and only this note remains.

## Versions

torch 2.8.0+cu128, nnunetv2 2.8.1, SimpleITK 2.5.6, numpy 2.3.2, Python 3.11 —
the stack the deployed server runs, which is where every BatchDentalSeg result
to date was produced. No vtk: the mesh exports are not ported.

The CUDA wheels come from an explicit index (`explicit = true` is load-bearing —
without it uv looks for every package on the PyTorch index); the venv is 7.2 GB.
The reasoning is the same as AMASSS's, at length in `tools/AMASSS/README.md`.

## Validated against

- **Input**: `DATA/AMASSS/testfiles/MG_test_scan.nii.gz`, one real CBCT.
- **Weights**: `PediatricDentalSeg`, staged from the upstream release named in
  the server's manifest and checked against its recorded sha256s and sizes.
- **Reference**: the pre-port implementation, run inside the deployed server
  container on the same scan, bundle and settings, on the same GPU, with
  `separate_segments=True`.
- **Result**: same output tree, same file names, identical `labels` table and
  identical `model` / `device` / `tile_step_size` / `separate_segments` /
  `summary` report fields.

As with AMASSS, nnUNet on CUDA is not bit-deterministic, so the reference was
run three times and this package three times before any conclusion was drawn.
Both sets scatter across the same states — 2 of the 9 port×reference pairs are
bit-identical on the label volume — and the worst port-vs-reference difference
is the *same number* as the worst reference-vs-reference difference on five of
the six files:

| File | Voxels | Worst ref vs ref | Worst port vs ref | Worst Dice |
|---|---|---|---|---|
| `_Seg` (labels) | 4 829 309 | 180 | 180 | 0.999982 |
| `_Upper-Skull` | 3 096 518 | 144 | 144 | 0.999977 |
| `_Mandible` | 1 223 918 | 34 | 35 | 0.999986 |
| `_Upper-Teeth` | 262 881 | 6 | 6 | 0.999989 |
| `_Lower-Teeth` | 227 033 | 6 | 6 | 0.999987 |
| `_Mandibular-canal` | 18 959 | 1 | 1 | 0.999974 |

- **Tolerance**: **Dice < 0.9999 against a reference is a regression**;
  above it is nnUNet's own CUDA noise floor, which the pre-port tool shares.

**GPU tests were run**: `uv run pytest -m models` with the PediatricDentalSeg
bundle on an RTX 6000 Ada — passed. CI skips them (`-m "not gpu"`); the other 24
tests stub `nnunet_runner.predict_folder` and need no checkpoint.

## Working on it

```bash
cd tools/Batch_Dental_Seg
uv sync                    # ~7.2 GB, CUDA wheels
uv run pytest -m "not gpu" # 24 tests, no GPU and no checkpoints needed
uv run pytest -m models    # a real bundle, see tests/data/README.md
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .
```
