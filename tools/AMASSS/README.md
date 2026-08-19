# sadt-amasss

Segments craniofacial structures on an oriented CBCT scan: mandible, maxilla,
cranial base, cervical vertebra, upper airway, skin, and the three masks AREG
consumes. One nnUNet v2 model per structure.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, path `AMASSS_CLI/`,
commit `21a62a8` (2026-05-22), by way of `slicer-remote-tool-server`'s
`tools/AMASSS/` -- whose history this repository carries, so `git log --follow`
on `src/sadt_amasss/pipeline.py` reaches back through it.

Upstream pins **not** kept: upstream declares torch 2.2.0 / torchvision 0.17.0 /
nnunetv2 2.8.0; this package pins torch 2.8.0+cu128 and nnunetv2 2.8.1. See
"Versions" for why.

Changes from upstream -- the algorithm is untouched, the envelope is not:

- **Scratch space lives under `output_dir`** (`.amasss_work/`, removed before
  returning). The server-side version took a scratch directory the server owned;
  a tool must not write outside the directory it is given.
- **`segment()` returns the run report** instead of a `SegmentationRun`. That
  class existed so another in-process tool could pick the files up directly;
  tools no longer call each other, so the reason is gone.
- **Zip extraction removed** from both the scan and model arguments. The server
  unpacks archives before `run()` is called.
- **The GPU semaphore is gone.** It serialised inference when every tool shared
  one process; a tool is now its own process and concurrency is the server's.
- **`device`, `tile_step_size` and `gpu_resampling` are arguments**, not server
  settings. `run()` must not read the environment, and the last two change the
  segmentation, so they belong next to the result.
- **VTK is a hard dependency**, so `vtk_export.is_available()` and the
  "install requirements.txt" guard around it are gone. The lockfile guarantees
  what the shared image could not.

The "FIX:" comments in `catalog.py` and `pipeline.py` record defects of the
original Slicer CLI corrected during the first port -- recursive folder scanning,
the missing label colours, the `CAN` code that matched nothing, the
`sys.exit(1)`, the batch that aborted on its last scan. They are unchanged here.

## What it does

| | |
|---|---|
| Inputs | `scans`: one CBCT (`.nii`/`.nii.gz`/`.nrrd`/`.nrrd.gz`/`.gipl`/`.gipl.gz`) or a folder of them, searched recursively. `model`: the bundle. `output_dir`: where results go. |
| Outputs | One `<scan>_<ID>_SegOut/` folder per scan holding the masks (and `.vtk` surfaces on request), plus `AMASSS_report.json`. |
| Model files | `<bundle>/<CODE>/**/*__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`, one subfolder per structure code. The shipped bundle is `AMASSS_Models/`, 2.1 GB, fetched by the server into `/DATA/AMASSS/models`. |
| GPU | Used when available; `device="cpu"` works and is much slower. CUDA falls back to CPU with a warning when no card is visible. |

Structure codes: `MAND`, `MAX`, `CB`, `CV`, `UAW`, `SKIN`, `CBMASK`,
`MANDMASK`, `MAXMASK` -- published as the argument's `choices`, so a client can
render them without a second declaration. The display names the old schema
published ("Cranial base", …) are still accepted but not offered.

`Literal` cannot be built from `catalog.STRUCTURE_CODES` (it takes literals
only), so the set is written twice and a test asserts the two agree. A structure
added to the catalog and not to `run()` would be unselectable from the client. `TEETH`, `RC` and `MCAN` are deliberately absent --
no model ships for them, and offering them produced either a KeyError during
surface export or a silent collision onto the mandible's label.

Three behaviours worth knowing before reading a result:

- **A structure with no model in the bundle does not fail the run.** It is listed
  in `structures_without_model` in the report. Check that field before
  concluding a structure was not present in the scan.
- **A single-structure run always writes the separate form**, whatever `merge`
  says: a "merged" volume of one structure is just that structure.
- **`gpu_resampling` and `tile_step_size` both move the mask.** The report
  records what was asked for, because a segmentation is only reproducible
  alongside them.

## Versions

Pinned to what the deployed server actually runs -- torch 2.8.0+cu128,
nnunetv2 2.8.1, SimpleITK 2.5.6, numpy 2.3.2, vtk 9.6.2, Python 3.11 -- rather
than to upstream's declared torch 2.2.0 / nnunetv2 2.8.0.

The reason is that no AMASSS result has ever been produced on upstream's pins.
The Slicer module installs into Slicer's shared interpreter, where the pins are
a truce with fifteen other modules; the server has run torch 2.8.0+cu128 since
the `lab-ai:2026.08` image, and every mask this tool has produced came from that
stack. Pinning 2.2.0 here would mean shipping an untested combination and having
nothing to validate it against. Upstream's numbers are recorded above, and
moving to them is available to whoever wants to revalidate.

The CUDA wheels come from an explicit index, and `explicit = true` is
load-bearing -- without it uv looks for every package on the PyTorch index:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

The resulting venv is 7.7 GB. See the repository README on deduplicating that
across tools at build time.

## Validated against

- **Input**: `DATA/AMASSS/testfiles/MG_test_scan.nii.gz`, one real CBCT.
- **Weights**: `DATA/AMASSS/models/AMASSS_Models`, structures MAND, MAX and CB,
  both merge modes.
- **Reference**: the pre-port implementation, run inside the deployed server
  container (`slicer-remote-tool-server-inference-1`, torch 2.8.0+cu128,
  nnunetv2 2.8.1) on the same scan, weights and settings, on the same GPU.
- **Result**: same geometry, same labels, same report fields (`device`,
  `gpu_resampling`, `tile_step_size`, `predicted_structures`, `merge_modes`,
  `summary` all equal), and masks within the reference implementation's own
  run-to-run spread.

**nnUNet on CUDA is not bit-deterministic**, so a single comparison would have
been meaningless. The reference was run four times and this package four times --
eight runs of identical code, identical package versions and the same card -- and
both sets scatter across the same handful of states. 5 of the 16 port×reference
pairs are bit-identical; the rest differ by the same margin two *reference* runs
differ by:

| Mask | Voxels | Worst reference vs reference | Worst port vs reference | Worst Dice |
|---|---|---|---|---|
| `_MAND` | 1 768 003 | 5 | 6 | 0.999998 |
| `_MAX` | 1 475 988 | 22 | 23 | 0.999992 |
| `_CB` | 1 900 614 | 41 | 41 | 0.999989 |
| `_MERGED` | 5 131 306 | 66 | 68 | 0.999994 |

- **Tolerance**: none of the difference is attributable to the repackaging --
  port-vs-reference is indistinguishable from reference-vs-reference, to within
  one or two voxels in five million. The cause is nnUNet's sliding-window
  accumulation, a float reduction CUDA does not order deterministically;
  asserting bit-identity would assert something the pre-port tool does not
  satisfy either. **Treat a Dice below 0.9999 against a reference as a
  regression**, and anything above it as this same noise floor.

**GPU tests were run**: `uv run pytest -m models` with the real bundle on an
RTX 6000 Ada -- passed, all three structures predicted, merged volume carrying
exactly labels {0, 1, 2, 4}. CI skips them (`-m "not gpu"`); the 48 remaining
tests stub `nnunet_runner.predict_folder` and need no checkpoint.

## Working on it

```bash
cd tools/AMASSS
uv sync                    # ~7.7 GB, CUDA wheels
uv run pytest -m "not gpu" # 48 tests, no GPU and no checkpoints needed
uv run pytest -m models    # the real bundle, see tests/data/README.md
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .
```
