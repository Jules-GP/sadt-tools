# sadt-automatrix

Apply a registration matrix somebody else computed to the scans and landmarks
it belongs to.

Served as `AutoMatrix`. Package `sadt_automatrix`; the public surface is
`run()`.

## Provenance

| | |
|---|---|
| Upstream | `Automatrix_CLI/Automatrix_CLI.py`, `AutoMatrix/AutoMatrix.py`, `AutoMatrix/AutoMatrix_Method/` |
| Upstream commit | `9318adb` (2026-07-20), the last upstream commit touching the module |
| Algorithm modified | no -- the resampling and the landmark transform are upstream's, argument for argument. What changed is around them; see "What changed" |

## What it does

AutoMatrix computes no registration. It takes transforms that AREG, ASO or a
mirroring matrix produced, works out which scan each one belongs to, and moves
a cohort through them.

**Pairing is by patient key**, and the key is a file name cut at a list of
markers: `MG01_T1_scan.nii.gz` and `MG01_MAND_Or.tfm` both key as `MG01`. The
scan list and the matrix list are cut at *different* markers -- a matrix is
also cut at `_Left`, `_Mirror` and `_SegOr`, a scan is also cut at `_Seg` and
`_Scan` -- which is what lets one patient's four region matrices all find their
way to the same scans. Both lists are reproduced marker for marker in
`inputs.py`, because any tidier key would silently re-pair every cohort in the
lab. A matrix argument naming one FILE instead of a folder is applied to every
patient, which is how a single mirroring matrix serves a whole batch.

**A volume takes the transform; a landmark takes its inverse.** An ITK
transform maps a point of the output space back into the input space, so the
resampler is handed it as it is and `.mrk.json` control points are handed
`GetInverse()`. That is upstream's code and it is also what makes a scan and
its landmarks land in the same place -- which is the assertion the test suite
is built around.

**Which grid the result lands on** is the one thing here with more than one
answer, and the report names it per case:

| grid | when |
|---|---|
| `chosen` | a `reference` volume was given -- the whole cohort lands on it and can be compared voxel to voxel |
| `scan` | no reference: each scan is resampled on its own grid |
| `mirror_source` | the matrix's name contains `mirror`, so it is applied in the scan's own space; mirroring onto somebody else's grid would move the patient as well as flip them |
| `composite_neighbour` | the transform is a chain (AREG), and the volume beside it -- `P1_transform.tfm` next to `P1.nii.gz` -- is the only grid it is valid on, overriding the reference |
| `composite_fallback` | that chain's neighbouring volume is missing; upstream warns and uses the scan's own grid, and so does this |

Segmentations resample nearest-neighbour, scans linear, and that is what
`is_segmentation` is for: a mask interpolated linearly comes back holding label
values that were never in it.

**Two matrices for one patient collide** unless `add_matrix_name` is set: both
results are written to the same name and the second replaces the first. That is
upstream's behaviour, it is left alone, and there is a test pinning it -- fixing
it would rename every file the pipelines downstream already expect.

| | |
|---|---|
| Inputs | `scans`: `.nii`, `.nii.gz`, `.nrrd` or `.mrk.json`, one file or a folder. `matrices`: `.tfm`, `.h5`, `.mat` or `.txt`, one file or a folder. |
| Options | `suffix`, `reference`, `add_matrix_name`, `from_areg`, `is_segmentation`. |
| Outputs | The input tree rebuilt under `output_dir`, one moved file per scan and matrix, plus `AutoMatrix_report.json`. |
| Model files | None. There is no network in this tool. |

## What changed

**Every CLI parameter is here, except the log path.** Upstream's
`Automatrix_CLI.xml` declares nine: `input_patient`, `input_matrix`,
`reference_file`, `suffix`, `matrix_name`, `fromAreg`, `output_folder`,
`log_path`, `is_seg`. Eight became arguments (`scans`, `matrices`, `reference`,
`suffix`, `add_matrix_name`, `from_areg`, `output_dir`, `is_segmentation`).
`log_path` is dropped: it exists so the Slicer progress bar can watch a file's
mtime tick, and the server reads the process's own output. Nothing was missed.

**"From AReg" now runs at all.** Upstream's branch reads
`args.matrix_lineEdit`, which the CLI never declares, so the first landmark
file that reaches it raises `AttributeError` and the run dies. It is wired to
the matrix folder here -- which is what the module's own widget puts in that
field -- and the region layout it walks (`Cranial Base`/`CBReg_matrix.tfm`,
`Maxilla`/`MAXReg_matrix.tfm`, `Mandible`/`MANDReg_matrix.tfm`, keyed on the
`_CB`, `_L` and `_U` markers) is upstream's, including the fact that `_L` maps
to the maxilla and `_U` to the mandible.

**Surface meshes are refused by name.** `.vtk`, `.vtp`, `.stl`, `.off` and
`.obj` are in the extension list the module validates its input against, and
upstream's CLI hands each of them to `sitk.ReadImage`, catches the failure and
writes a line to a log: no mesh has come out of AutoMatrix since the CLI
replaced the in-Slicer implementation at `7bc362b`. This is the same outcome
said where somebody reads it -- the file is named in the report with the reason,
and a mesh sitting beside a volume costs only itself.

Restoring meshes was considered and deliberately not done. The in-Slicer
version applied the transform through `slicer.util.loadTransform` and
`HardenTransform`, which silently converts LPS to RAS; a `.mrk.json` and a
`.vtk` off the same patient do not agree about which of the two they are in,
and there is no reference run here to settle it. A mirrored mesh looks right,
which is exactly why guessing is worse than refusing. **This is the one
decision in the port worth revisiting**, and it needs a real Slicer session and
a real mesh, not a rewrite.

**`.npy` matrices are refused by name**, for the same reason and more sharply.
Upstream's matrix search collects them and `sitk.ReadTransform` cannot open
one, so today they fail. The in-Slicer version did read them -- as a RAS
transform-*to*-parent applied without inversion, where every other format here
is an LPS resampling transform applied inverted. The two conventions disagree
by a coordinate flip *and* by a direction, so a `.npy` dropped into a folder of
`.tfm` would move a scan somewhere else entirely while looking like it worked.

**A missing reference volume is refused, not warned about.** Upstream logs
"No valid reference image provided" and resamples every scan on its own grid,
so a typo in the path produces a full cohort, a clean exit and results that
cannot be compared with each other. A `reference` that is not there is now a
`ToolInputError` before anything runs.

**A batch that produced nothing says so.** Upstream returns quietly with an
empty output folder when no matrix matched a key, when every matrix failed to
parse, or when the whole input was meshes. The guard here counts files
*written* -- not scans walked, not matrices read -- and raises `NothingWritten`
carrying the per-file reasons. `AutoMatrix_report.json` is written first, so
the reasons are on disk either way.

**The order is stable.** Upstream walks `glob.iglob`, whose order is the
filesystem's, so two runs on one folder could report the same cohort in
different orders. The grouping by extension is upstream's; the sort inside each
group is new.

**A file's extension is the longest declared one it ends with**, where upstream
used `''.join(Path(p).suffixes)`. The two agree on every ordinary name. They
disagree on `P.1_scan.nii.gz`, where `suffixes` returns `.1_scan.nii.gz` and
the output would be named off the wrong stem.

**Dropped, with the reason.** The Mirror check box, which downloads
`Mirror.zip` from a GitHub release and types the path into the matrix field: it
is a download, not a parameter, and the matrix it fetches is named as a file
like any other. The `<filter-progress>` prints, the file-vs-folder combo boxes
(the client's `path` picker takes both), the progress bar, the timer and the
dark-mode stylesheet: all UI over a local Slicer session.

## Versions

`SimpleITK==2.5.6`, Python 3.11. Upstream pins nothing -- the CLI imports
whatever SimpleITK the running Slicer has -- and 2.5.6 is what every sibling
tool here locks, so a matrix written by AREG and applied by AutoMatrix goes
through one ITK rather than two. `numpy` is a test dependency only: the tool
itself never touches an array.

## Validated against

| | |
|---|---|
| A known translation, on a volume | yes -- a 4 mm shift written as a `.tfm` comes back as a 4-voxel move of the cube, matching `np.roll` to 1e-4, and the bounding box lands exactly on `[16, 40)` |
| A volume and its landmarks agreeing | yes -- a landmark placed at the centre of the cube is still at the centre of the cube afterwards, to 1e-6 mm. This is what catches a transform applied in the wrong direction, which nothing else here would |
| Segmentation interpolation | yes -- a half-voxel shift of a 0/7 mask stays `{0, 7}` with `is_segmentation`, and provably does not without it |
| Which grid the result lands on | yes -- all five cases (`chosen`, `scan`, `mirror_source`, `composite_neighbour`, `composite_fallback`) asserted on the output volume's size |
| The pairing rule | yes -- the patient keys are pinned against upstream's marker chains |
| `pytest` | 27 passed |
| Against the pre-port CLI, on real data | **not done.** No side-by-side run of this package against `Automatrix_CLI.py` on a real cohort with real AREG matrices. Everything above is a synthetic fixture with a closed-form answer, which is strong for geometry and says nothing about an oblique direction matrix or a transform format that has not been tried |
| Through `POST /run/AutoMatrix` | **not done.** Not deployed; the schema generates and the client test drives it as a fixture, which is not the same as a round trip |
| The "From AReg" path, against a real AREG output tree | **not done.** It is tested against a tree the test builds, so what is verified is that the layout is walked as written -- not that a real AREG run puts its files there |

## Working on it

```bash
uv sync
uv run pytest -q
uv run --no-project --with pyflakes -- python -m pyflakes src tests
.venv/bin/python ../../scripts/describe.py .
```

No model bundle, no GPU, no network: the whole suite runs anywhere Python and
SimpleITK do.
