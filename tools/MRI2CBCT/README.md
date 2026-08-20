# sadt-mri2cbct

Bring a TMJ MRI into its CBCT's space, one pipeline step at a time.

Served as `MRI2CBCT`. Package `sadt_mri2cbct`; the public surface is `run()`.

## Provenance

| | |
|---|---|
| Upstream | `MRI2CBCT/`, `MRI2CBCT_CLI/` |
| Upstream commit | `6d8e49b` (2026-07-17) |
| Algorithm modified | no -- the ported modules are upstream's, with the changes listed below, none of which touch what is computed |

## What it does

Upstream ships six CLI modules and a panel that launches them one at a time.
That is kept: a call runs ONE step, named by `step`.

| step | what it does | upstream |
|---|---|---|
| `Orient MRI` | writes a direction, a slice spacing and a centred origin into each MRI | `MRI2CBCT_ORIENT_CENTER_MRI` |
| `Resample` | resamples MRI, CBCT and segmentations to a common size or spacing, two timepoints at a time | `MRI2CBCT_RESAMPLE_CBCT_MRI` |
| `Approximate` | segments the condyle and brings the MRI roughly onto the CBCT | `MRI2CBCT_APPROX` |
| `LR crop` | splits each volume into left and right | `MRI2CBCT_LR_CROP` |
| `TMJ crop` | crops both modalities to the joint, from a condyle segmentation | `MRI2CBCT_TMJ_CROP` |
| `Register` | inverts, normalises and masks both modalities, then registers with elastix | `MRI2CBCT_REG` |

The sequence is deliberate and it is why this is not one call: a badly oriented
MRI is worth catching before an hour of registration, which is exactly what the
module's separate tabs are for.

## What changed

**Nothing in what is computed.** The ported modules are upstream's files. What
was done to them: the `sys.path.append("..")` stanza that let a Slicer CLI reach
its sibling package became ordinary relative imports; each module's private
stdout logging handler became `logging.getLogger(__name__)`; the
`<filter-progress>` prints went, and with them the **`time.sleep(0.5)` after
each file** -- pure pacing for Slicer's progress bar, and twenty seconds on a
cohort of forty scans.

**Two upstream defects, fixed.**

* `resample.py` logged with `f"Writing: {fobj["out"]}"`. Nested double quotes
  inside an f-string are a Python 3.12 feature; the deployment interpreter is
  3.11, where that file cannot even be imported. Slicer 5.13 ships 3.12, which
  is why upstream never saw it.
* `torchreg` is imported at module level by `crop_approximation.py` and is
  declared nowhere in the module's own dependency list. It is a declared
  dependency here.

**The pins are the deployed stack's.** Upstream's `MRI2CBCT.py:82-115` installs
`pydicom==2.2.2` and `dicom2nifti==2.3.0` into Slicer's shared `site-packages`,
which is what breaks Slicer's own DICOM modules; declares
`('nnunet_version', "==2.8.0")`, which names no package that exists; and pip
installs `torch==2.2.0` from the cu118 index, fighting the cu128 wheels
everything else here uses. A tool with its own virtualenv has none of those
problems, and none of that code came across.

**The eight normalisation numbers are eight arguments.** Upstream packs them
into one string read back with a `\d+` regex (`extract_values`). The string is
still what the ported code receives -- `extract_values` is unchanged and a test
pins the order -- but a client now renders eight labelled spin boxes instead of
asking for `[[0, 100, 10, 95], [0, 100, 10, 95]]`.

**Sizes and spacings are lists, not strings.** `resample_size` and `spacing`
are `list[int]` / `list[float]`; empty means "keep the scan's own", which is
what upstream's literal string `"None"` meant.

**Dropped, with the reason.** The manual approximation tab
(`ManualApprox_MRI2CBCT.py`, 488 lines of interactive transform handles), the
DICOM import and scene-node selection, and the model/test-file download buttons.
All are UI over a local Slicer session or over deployment state, and none is a
registration parameter.

**Nothing was missed by oversight.** Every parameter of the six CLI XMLs is an
argument here, except the two scratch folders (`tmp_fold`, `tempo_fold`), which
the tool now owns: a caller does not choose where a subprocess writes its
intermediates. `keep_temporary` is upstream's `tempo_fold` flag, kept, because
the intermediate volumes are what tell you WHICH stage went wrong.

## Versions

`torch==2.8.0+cu128`, `nnunetv2==2.8.1`, `torchreg==0.1.3`, `itk==5.4.7`,
`itk-elastix==0.25.4`, `SimpleITK==2.5.6`, `nibabel==5.4.2`, `numpy==2.3.2`,
`pandas==2.3.3`, `scipy==1.16.2`, `scikit-learn==1.7.2`, Python 3.11.

## Validated against

| | |
|---|---|
| Orient MRI | yes -- direction, slice spacing and centred origin all assert against the values upstream's `calculate_new_origin` computes |
| Resample | yes -- a requested size and a requested spacing are both reached, and neither given keeps the scan's own |
| LR crop | yes -- an MRI is halved on Z and a CBCT on X, which is upstream's asymmetry |
| `pytest -m "not models"` | 18 passed |
| pyflakes | 0 |
| Approximate, TMJ crop | **not run.** Both need the nnUNet condyle bundle, which is deployment state. |
| Register | **not run.** elastix on synthetic cubes converges to nothing meaningful; the reference is a clinical pair. This is the step that matters most and it is the one still unvalidated. |

## Working on it

```bash
uv sync
uv run pytest -q -m "not models"
uv run --no-project --with pyflakes -- python -m pyflakes src
.venv/bin/python ../../scripts/describe.py .
```
