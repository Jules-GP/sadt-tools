# sadt-ali

Places anatomical landmarks and writes Slicer markups files (`.mrk.json`). One
tool, two engines that share nothing but their output format:

- **CBCT** — one deep-RL agent per landmark walks the volume at 1 mm and then at
  0.3 mm until it converges on the point. 119 landmarks across four regions.
- **IOS** — per tooth, the mesh is rendered from a dozen viewpoints and a 2D
  UNet predicts masks that are projected back onto the surface.

Which engine runs is decided from the data, never from an argument.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, paths `ALI_CBCT/`,
`ALI_CBCT_utils/`, `ALI_IOS/` and `ALI_IOS_utils/`, by way of
`slicer-remote-tool-server`'s `tools/ALI/` — whose history this repository
carries, so `git log --follow` on `src/sadt_ali/cbct/engine.py` reaches back
through it to `a0ed474` (2026-07-31).

> **The upstream commit is not recorded.** The server-side port landed as
> `ADD ALI & CrownSeg` with no upstream SHA in the message and none anywhere in
> the tree, so there is no way to recover from this repository which upstream
> revision the algorithm was taken from. Every other row in
> [PROVENANCE.md](../../PROVENANCE.md) carries one. **This needs filling in by
> whoever made that port**, and until it is, "same as upstream" is a claim about
> code nobody can point at. The per-module mapping below is exact and was
> verified by reading; only the revision is missing.

| This package | Upstream |
|---|---|
| `cbct/engine.py` | `ALI_CBCT/ALI_CBCT.py` |
| `cbct/agent.py` | `ALI_CBCT_utils/agent.py` |
| `cbct/brain.py` | `ALI_CBCT_utils/brain.py` |
| `cbct/environment.py` | `ALI_CBCT_utils/environment.py` |
| `cbct/preprocess.py` | `ALI_CBCT_utils/preprocess.py` |
| `ios/engine.py` | `ALI_IOS/ALI_IOS.py` |
| `ios/surface.py` | `ALI_IOS_utils/surface.py` |
| `ios/render.py` | `ALI_IOS_utils/{render,mask_renderer,agent}.py` |

Upstream pins **not** kept: this package pins torch 2.8.0+cu128, monai 1.6.0,
itk 5.4.7 and Python 3.11. See "Versions" for why.

Changes from upstream — the algorithm is untouched, the envelope is not. The
first four were made during the server-side port and are unchanged here; the
rest are this migration's.

- **One markups file per scan**, holding every landmark found. Upstream wrote
  one file per anatomical region, so every downstream tool (ASO, AREG,
  AutoMatrix) had to recombine them by hand.
- **`display.visibility` is `true`.** Both upstream CLIs wrote `false`, which
  switches the markups *display node* off: Slicer loads the file, lists the node
  and draws nothing. Invisible inside the old Slicer module, which loaded nodes
  itself; fatal for anyone opening a returned file.
- **Scans are keyed by path relative to the input root**, not by base name. Two
  patients called `scan.nii.gz` in different folders used to overwrite each
  other, twice — in the working dictionary and again in the flat output folder.
- **Both impacted-canine spellings resolve** (`UR3OI` ≡ `UR3OIP`) and
  `group_of()` never raises. The unguarded `LABEL_GROUPS[...]` lookup this
  replaces threw a `KeyError` caught far above, and *nothing at all* was written
  for that scan — including every landmark already found.
- **Zip extraction removed.** The server unpacks archives before `run()` is
  called, with the bomb cap and `strip_single_root` that used to live in
  `ALILogic._extracted`.
- **Model-bundle auto-selection removed.** `model` was optional and the tool
  picked a hosted bundle matching the detected mode by walking `data_store`. A
  tool no longer resolves paths, so `model` is required and the server picks.
  The layout check survives: a bundle of the wrong kind is still refused with a
  message naming both kinds, which is what that code was really for.
- **CrownSeg is no longer called.** `ALILogic.ensure_segmented()` imported
  `tools.CrownSeg` in-process and segmented an unlabelled mesh on the fly.
  Tools do not call each other; `ios.engine.require_labels()` refuses the batch
  up front instead, naming `Crown_Seg` and the array it looked for. **This is
  the one behaviour change a user can see** — see "The Crown_Seg chain".
- **The GPU semaphore is gone.** Both engines held a
  `threading.BoundedSemaphore(ALI_MAX_GPU_JOBS)`, which serialised inference
  when every tool shared one process. A tool is its own process now, so an
  in-process limit caps nothing; the server holds it, across tools.
- **`device` and `search_seconds` are arguments**, not settings. `run()` must
  not read the environment.
- **`ui="tabs"` and the `groups` table are gone**, with the `ArgSpec` schema
  that carried them. See "What the client loses".

## What it does

| | |
|---|---|
| Inputs | `input`: one CBCT (`.nii`/`.nii.gz`/`.nrrd`/`.nrrd.gz`/`.gipl`/`.gipl.gz`), one intraoral surface (`.vtk`/`.stl`), or a folder of either, searched recursively. A DICOM series inside a folder is converted automatically. `model`: the bundle. `output_dir`: where results go. |
| Outputs | One `<scan>_lm_<ID>.mrk.json` per scan, mirroring the input's folder tree, plus `run_report.json`. |
| Model files | CBCT: `<bundle>/**/<landmark>/<scale>/*.pth`, scale folders named `1` and `0-3`, both required per landmark. IOS: flat checkpoints carrying an `O`/`C` token and an `Upper`/`Lower` one, e.g. `Upper_O_model.pth`. Fetched by the server into `/DATA/ALI/models`. |
| GPU | Used when available; `device="cpu"` works and is much slower — the per-landmark search budget defaults to 60 s on CPU against 15 s on CUDA for that reason. |

Three behaviours worth knowing before reading a result:

- **`landmarks` replaces `cbct_regions`, it does not narrow it.** Naming any
  landmark makes the region selection inert. That is what lets a caller ask for
  the seven points it needs instead of running 58 agents to use seven — one
  agent being a full two-scale walk of the volume.
- **A landmark missing from the bundle and a landmark that never converged are
  different things**, and look identical in the Slicer scene. The first is in
  `landmarks_without_model`, the second in that scan's `landmarks_failed`. They
  need opposite fixes: another bundle, or another scan.
- **An input holding both CBCT and IOS data is refused**, not half processed.

## The Crown_Seg chain

ALI's IOS engine needs meshes that already carry tooth labels. Upstream, and in
the server-side port, ALI segmented them itself by importing CrownSeg. It does
not any more, so the sequence is the server's to run:

```
Crown_Seg  →  ALI (IOS)
```

Crown_Seg's `run_report.json` lists every labelled mesh under `segmented_meshes`,
whether this run produced the labels or found them already there — so re-running
the chain on a mixed batch is cheap and safe. Feed its output directory straight
in as ALI's `input`.

Skipping it is not a silent failure: `require_labels()` checks the whole batch
before any weights load and refuses it with a message naming `Crown_Seg` and the
three array names it looked for. Checking up front matters — discovering it on
mesh 40 of 40 costs an hour of inference first.

`tests/test_integration.py` runs the real chain, each tool in its own venv, the
way the server does.

## What the client loses

The old `ArgSpec` schema published presentation metadata that `describe.py` has
no field for. Two are worth stating plainly, because a client written against
the old schema will look worse rather than break:

- **`landmarks` had `ui="tabs"` and a `groups` table**, so 119 check boxes
  rendered as four tabs matching the anatomical regions. The new schema
  publishes the 119 options as a flat `choices` list. The grouping still exists
  in `cbct.catalog.GROUP_LABELS`; nothing publishes it.
- **`section` is gone**, so the four collapsible boxes ("Inputs", "CBCT
  landmarks", "IOS landmarks", "Outputs") are no longer declared. Both
  selections are still always shown, and one is always inert.

Neither affects a result. Both affect how usable the panel is for the one tool
in this repository with a three-figure option count, and both are the same
question: whether the schema should carry layout hints at all. Raised here
rather than worked around.

## Versions

Pinned to what the deployed server actually runs — torch 2.8.0+cu128,
monai 1.6.0, itk 5.4.7, SimpleITK 2.5.6, vtk 9.6.2, numpy 2.3.2, Python 3.11 —
which is the same reasoning as [AMASSS](../AMASSS/README.md#versions): the
Slicer module installs into Slicer's shared interpreter, where the pins are a
truce with fifteen other modules, and no ALI result has ever been produced on
them. monai 1.6.0 and itk 5.4.7 are what the sibling tools already lock, so this
adds no runtime to the image.

The one place a version is load-bearing in the code:
`cbct/environment.py` uses monai's `EnsureChannelFirst`. Upstream branched on
`sys.version_info >= (3, 10)` to choose between it and `AddChannel`, which monai
removed years ago; at 1.6.0 only the former exists and the branch is gone.

### pytorch3d

Only the IOS engine needs it, so it sits behind an extra and a plain `uv sync`
stays fast — CI can import the package and publish its schema, and the CBCT
engine works with no pytorch3d at all.

```toml
[project.optional-dependencies]
ios = ["pytorch3d"]

[tool.uv.sources]
pytorch3d = { git = "https://github.com/facebookresearch/pytorch3d.git", tag = "v0.7.9" }

[tool.uv.extra-build-dependencies]
pytorch3d = ["torch"]
```

Same incantation as [Crown_Seg](../Crown_Seg/README.md), same tag, so the two
tools share one build:

- PyPI's newest pytorch3d is 0.7.4, with wheels for cp38–cp310 only, built
  against a torch generations older than ours. It is compiled from source.
- `extra-build-dependencies` puts torch into pytorch3d's isolated build
  environment. Its `setup.py` imports torch without declaring it, so without
  this the lock fails with `ModuleNotFoundError: No module named 'torch'`.
  `no-build-isolation-package` does **not** work here — it stops uv from
  providing torch rather than making it available.

`uv lock` resolves 56 packages in about a second. `uv sync --extra ios` then
compiles pytorch3d; deployment should do that once and keep the wheel in a local
`/wheels` directory rather than rebuilding it per image.

The venv is 8.8 GB synced without the extra. See the repository README on
deduplicating that across tools at build time.

## Validated against

- **Schema**: `describe.py` accepts the signature and publishes 9 arguments,
  `returns: path`, with `choices` on `cbct_regions` (4), `landmarks` (119),
  `ios_networks` (2) and `device` (2). Asserted out of process in
  `tests/test_integration.py` against the real venv.
- **Tests**: 57 passing, 1 skipped, 1 GPU test deselected. The agent is stubbed,
  so mode detection, DICOM recognition, weight discovery, the vocabulary, output
  naming, tree preservation, the run report, work-directory cleanup, the
  output-containment rule and every cross-argument rule run for real, with no
  checkpoint and no card.
- **Not yet validated against reference output.** No GPU run has been made:
  this workstation has no CUDA device, and the CBCT bundle (4.7 GB) and IOS
  bundle are not staged here. **The `gpu`/`models` tests have not been run**,
  and neither has a comparison against the pre-port implementation — which is
  what AMASSS, Batch_Dental_Seg and Surg_Mov_Pred each have and this does not.
  Everything needed to do it is in `tests/data/README.md`.

## Working on it

```bash
cd tools/ALI
uv sync                    # ~8.8 GB, CUDA wheels, no pytorch3d
uv run pytest -m "not gpu" # 57 tests, no GPU and no checkpoints needed
uv sync --extra ios        # compiles pytorch3d, needs nvcc
uv run pytest -m models    # the real bundles, see tests/data/README.md
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .
```
