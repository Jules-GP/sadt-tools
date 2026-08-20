# sadt-greedyreg

Affine registration of a follow-up CBCT onto its baseline, with Greedy.

Served as `GreedyReg`. Package `sadt_greedyreg`; the public surface is `run()`.

## Provenance

| | |
|---|---|
| Upstream | `GreedyReg/GreedyReg.py`, `GreedyReg/GreedyReg_Method/Logic.py`, `GreedyReg_CLI/GreedyReg_CLI.py` |
| Upstream commit | `7ca116e` (2026-06-26), the last upstream commit touching the module |
| Algorithm modified | no -- see "What changed" |

## What it does

Two things upstream keeps on two tabs, and one composition of them.

**Greedy** solves an affine between the two timepoints and resamples the
follow-up onto the baseline's grid. The command handed to Greedy is upstream's,
argument for argument:

```
-d 3 -a -m <metric> -i <fixed> <moving> -o <warp>
-n 100x100x50x25 -e 0.5 -search 100 10 20 -dof <6|12> -ia <init> [-gm <mask>]
```

followed by `-d 3 -rf <fixed> -rm <moving> <output> -r <warp>`. The schedule,
the search, the 0.5 step and the NCC 4x4x4 window are all upstream's, and none
of them is exposed as an argument -- they are what every result to date was
produced with.

**Landmark** is upstream's Distant Registration: ALI_CBCT places a region's
landmarks on both scans, a rigid transform is fitted to the matched pairs by
SVD, and the follow-up is written out with that transform baked into its NIfTI
affine. No voxel is resampled and none is interpolated; the image simply says
where it is. Used when the two timepoints are too far apart for Greedy's search
window to find anything.

**Landmark + Greedy** runs the second and hands its transform to the first as
`-ia`. This is not a new algorithm: it is the sequence the module's own status
message instructs the user to perform by hand -- *"Distant registration
complete! Now run Automatic Registration to refine."* -- expressed as one call
instead of two, using the `initFolder` mechanism upstream already has for it.

Pairing is by patient key: the leading letters-and-digits of the file name,
upper-cased, exactly as `GreedyReg_CLI.ID_PATTERN` does it. `MG01_T1.nii.gz`
pairs with `MG01_T2.nii.gz`. Note what that excludes: `C_0001_T1.nii.gz` has no
key, because the pattern wants digits immediately after letters. That is
upstream's rule and it is kept -- changing it would silently re-pair every
folder that works today.

## What changed

**Greedy comes from a wheel, not a download.** Upstream fetches an ITK-SNAP
4.2.2 tarball from SourceForge on first use and calls `bin/greedy` as a
subprocess; on Windows it downloads a 150 MB NSIS installer and either pulls the
executable out with 7-Zip or silently installs and uninstalls ITK-SNAP. This
declares `picsl-greedy==1.4.0`, the same code published as a wheel by PICSL, and
calls `Greedy3D.execute()` with the identical command line. What that removes: a
network fetch at run time, an unpinned binary version, and three platform
branches. What it does not remove: the possibility that 1.4.0 and ITK-SNAP
4.2.2's build differ. Nothing here has been diffed against the ITK-SNAP binary,
and that is the open item in "Validated against".

**No torch, no monai, no pydicom pin.** Upstream's `Logic._aliRequiredLibs`
installs `monai`, `pydicom==2.2.2` and `dicom2nifti==2.3.0` into Slicer's shared
`site-packages` so that ALI_CBCT can run in-process. Those two pins are what
break Slicer's own DICOM modules, and they are gone: ALI_CBCT is reached through
the supervisor, in its own virtualenv, and this tool depends on nothing it uses.

**A batch of files, not of scene nodes.** Upstream's single-pair path works on
volumes loaded in Slicer; `t1`/`t2` take either two folders or two files, so the
single pair is expressible without a scene.

**Dropped, with the reason.** The manual alignment tab (interactive transform
handles, centering, hardening), the sensitivity demo, the "download Greedy" and
"install ALI libraries" buttons, and the model-download helper: all of them are
UI over a local Slicer session or over deployment state, and none of them is a
registration parameter. The Greedy binary is now a dependency and the ALI
bundle is named by the caller, so the two download paths have nothing left to
do.

**Nothing was missed by oversight.** Every registration parameter upstream
exposes -- metric, degrees of freedom, mask folder, init folder, region -- is an
argument here, and the report names what each case used.

## Versions

`picsl-greedy==1.4.0`, `nibabel==5.4.2`, `numpy==2.3.2`, Python 3.11.

Upstream pins none of these. nibabel is pip-installed on demand into Slicer, and
its CLI exits 1 with an install hint when it is missing; the Greedy binary is
whatever the ITK-SNAP 4.2.2 archive holds.

## Validated against

| | |
|---|---|
| Registration recovers a known translation | yes -- a 4-voxel (2.0 mm) shift on a synthetic volume comes back as 1.998 mm on the warp's translation, and the correlation with the baseline goes from 0.822 to 0.9999 |
| Landmark fit | yes -- a known rigid offset between two landmark sets is recovered exactly, and the aligned volume's voxels are bit-identical to the input's |
| `pytest -m "not gpu"` | 16 passed |
| Against the ITK-SNAP 4.2.2 binary | **not done.** No side-by-side run of `picsl-greedy` 1.4.0 against the binary upstream downloads, on a real pair. Until that exists, "algorithm modified: no" is a claim about the command line, not about the numbers. |
| Against a real patient pair, through `POST /run/GreedyReg` | **not done.** Needs the ALI bundle for the landmark modes and a real T1/T2 pair. |

## Working on it

```bash
uv sync
uv run pytest -q
uv run --no-project --with pyflakes -- python -m pyflakes src tests
.venv/bin/python ../../scripts/describe.py .
```

The landmark modes need no model bundle to test: ALI's output is planted
through a fake supervisor, which is what keeps the suite runnable in CI.
