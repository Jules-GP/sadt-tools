"""The six operations upstream puts on six tabs, behind one `step` argument.

Upstream ships MRI2CBCT as six CLI modules that the panel launches one at a
time, each with its own inputs and its own output folder, and the clinician
inspects the result before pressing the next one. That sequence is deliberate --
a badly oriented MRI is worth catching before an hour of registration -- so it
is kept: a call runs ONE step, and `step` says which.

What each step needs is checked here, up front. An argument the chosen step
never reads is left alone rather than refused, because the client hides it
(see layout.py) and a stale value in a hidden field is not the caller's fault.
"""

import json
import logging
import tempfile
from pathlib import Path

from .errors import StepFailed, ToolInputError

logger = logging.getLogger(__name__)

STEP_ORIENT = "Orient MRI"
STEP_RESAMPLE = "Resample"
STEP_APPROXIMATE = "Approximate"
STEP_LR_CROP = "LR crop"
STEP_TMJ_CROP = "TMJ crop"
STEP_REGISTER = "Register"
STEPS = (STEP_ORIENT, STEP_RESAMPLE, STEP_APPROXIMATE, STEP_LR_CROP,
         STEP_TMJ_CROP, STEP_REGISTER)

REPORT_NAME = "MRI2CBCT_report.json"

# Upstream's own documented direction matrices, from the comment at the foot of
# MRI2CBCT_ORIENT_CENTER_MRI.py. The argument stays a free string because the
# panel's orientation table can build any of the 48 axis permutations, and
# narrowing it to these two would take that away.
DIRECTION_MRI = "0.0,0.0,-1.0,1.0,0.0,0.0,0.0,-1.0,0.0"
DIRECTION_CBCT = "1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0"

# Which inputs each step cannot run without. Checked before a file is read: a
# request that cannot work has to come back in a second, not after nnUNet has
# loaded a model.
REQUIRED = {
    STEP_ORIENT: ("mri",),
    STEP_RESAMPLE: (),  # any one of mri/cbct/segmentation, checked separately
    STEP_APPROXIMATE: ("cbct", "mri", "condyle_model"),
    STEP_LR_CROP: (),  # same
    STEP_TMJ_CROP: ("cbct", "mri", "segmentation", "condyle_model"),
    STEP_REGISTER: ("mri", "cbct", "segmentation"),
}


def _folder(value, name: str) -> str:
    """Every step here works on FOLDERS, as upstream's CLIs do.

    A single file is accepted and its parent is used, because the client gives
    a packaged tool's path argument a picker that takes either and a user
    registering one case will pick the file.
    """
    path = Path(value)
    if not path.exists():
        raise ToolInputError("{} does not exist: {}".format(name, path))
    return str(path if path.is_dir() else path.parent)


def _numbers(values, name: str, kind) -> str:
    """A list argument in the comma-separated form the ported code parses."""
    if not values:
        return "None"
    try:
        return ",".join(str(kind(value)) for value in values)
    except (TypeError, ValueError) as bad:
        raise ToolInputError("{}: {}".format(name, bad)) from bad


def _check_direction(direction: str) -> str:
    parts = [piece.strip() for piece in str(direction).split(",") if piece.strip()]
    if len(parts) != 9:
        raise ToolInputError(
            "direction must be nine comma-separated numbers, a 3x3 matrix read "
            "row by row. Got {} value(s). The MRI orientation is '{}' and the "
            "CBCT one is '{}'.".format(len(parts), DIRECTION_MRI, DIRECTION_CBCT)
        )
    try:
        [float(piece) for piece in parts]
    except ValueError as bad:
        raise ToolInputError("direction: {}".format(bad)) from bad
    return ",".join(parts)


def _require(step: str, given: dict) -> None:
    missing = [name for name in REQUIRED[step] if not given.get(name)]
    if missing:
        raise ToolInputError(
            "{} needs {}.".format(step, ", ".join(sorted(missing)))
        )
    if step in (STEP_RESAMPLE, STEP_LR_CROP) and not any(
            given.get(name) for name in ("mri", "cbct", "segmentation")):
        raise ToolInputError(
            "{} needs at least one of mri, cbct or segmentation.".format(step))


def _orient(given, output_dir, direction, acquisition_z_spacing):
    from .orient import orient

    destination = output_dir / "MRI"
    destination.mkdir(parents=True, exist_ok=True)
    # Upstream passes the string "None" when the acquisition spacing is not to
    # be touched; there is no nullable type in the schema, so 0 means the same.
    z_spacing = "None" if not acquisition_z_spacing else float(acquisition_z_spacing)
    orient(_folder(given["mri"], "mri"), direction, str(destination), z_spacing)
    return {"MRI": str(destination)}


def _resample(given, output_dir, resample_size, spacing, center):
    from .resample_stage import resample_folder

    size = _numbers(resample_size, "resample_size", int)
    step = _numbers(spacing, "spacing", float)
    written = {}
    # (argument, output folder, is a segmentation, is an MRI). `iso_spacing` is
    # upstream's `isMRI` flag, which is what its `mri=` parameter reads.
    for name, folder, is_seg, is_mri in (
        ("mri", "MRI", False, True),
        ("mri_t2", "MRI_T2", False, True),
        ("cbct", "CBCT", False, False),
        ("cbct_t2", "CBCT_T2", False, False),
        ("segmentation", "Seg", True, False),
        ("segmentation_t2", "Seg_T2", True, False),
    ):
        if not given.get(name):
            continue
        destination = output_dir / folder
        destination.mkdir(parents=True, exist_ok=True)
        resample_folder(
            _folder(given[name], name), str(destination), size, step,
            "True" if center else "False", is_mri, is_seg=is_seg)
        written[folder] = str(destination)
    return written


def _lr_crop(given, output_dir):
    from .lr_crop import crop_folder

    written = {}
    for name, folder, is_cbct in (
        ("cbct", "CBCT", True),
        ("mri", "MRI", False),
        # Upstream crops a segmentation with the CBCT path: it is in CBCT space.
        ("segmentation", "Seg", True),
    ):
        if not given.get(name):
            continue
        destination = output_dir / folder
        crop_folder(_folder(given[name], name), str(destination), is_cbct=is_cbct)
        written[folder] = str(destination)
    return written


def _tmj_crop(given, output_dir, scratch):
    from .tmj_crop import crop_patients

    crop_patients(
        _folder(given["cbct"], "cbct"),
        _folder(given["mri"], "mri"),
        _folder(given["segmentation"], "segmentation"),
        str(output_dir),
        _folder(given["condyle_model"], "condyle_model"),
        str(scratch / "nnunet"),
    )
    return {"output": str(output_dir)}


def _approximate(given, output_dir, scratch):
    from .approx import run_script_first_approximation

    produced = run_script_first_approximation(
        _folder(given["cbct"], "cbct"),
        _folder(given["mri"], "mri"),
        str(output_dir),
        _folder(given["condyle_model"], "condyle_model"),
        str(scratch / "nnunet"),
    )
    return {"first_approximation": str(produced)}


def _register(given, output_dir, normalisation, keep_temporary):
    from .register import register

    register(
        folder_general=str(output_dir),
        mri_folder=_folder(given["mri"], "mri"),
        cbct_folder=_folder(given["cbct"], "cbct"),
        cbct_label2=_folder(given["segmentation"], "segmentation"),
        normalization=str(list(normalisation)),
        keep_temporary=keep_temporary,
    )
    return {"output": str(output_dir)}


def process(step, output_dir, given, direction, acquisition_z_spacing,
            resample_size, spacing, center, normalisation, keep_temporary):
    if step not in STEPS:
        raise ToolInputError(
            "Unknown step '{}'. Available: {}.".format(step, ", ".join(STEPS)))
    _require(step, given)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("MRI2CBCT: running step %s", step)

    with tempfile.TemporaryDirectory(prefix="mri2cbct_") as scratch:
        scratch = Path(scratch)
        try:
            if step == STEP_ORIENT:
                written = _orient(given, output_dir, _check_direction(direction),
                                  acquisition_z_spacing)
            elif step == STEP_RESAMPLE:
                written = _resample(given, output_dir, resample_size, spacing, center)
            elif step == STEP_LR_CROP:
                written = _lr_crop(given, output_dir)
            elif step == STEP_TMJ_CROP:
                written = _tmj_crop(given, output_dir, scratch)
            elif step == STEP_APPROXIMATE:
                written = _approximate(given, output_dir, scratch)
            else:
                written = _register(given, output_dir, normalisation, keep_temporary)
        except ToolInputError:
            raise
        except Exception as failure:  # a step that ran and could not finish
            raise StepFailed("{} failed: {}".format(step, failure)) from failure

    (output_dir / REPORT_NAME).write_text(
        json.dumps({"step": step, "written": written,
                    "inputs": {name: str(value) for name, value in given.items() if value}},
                   indent=2),
        encoding="utf-8")
    return output_dir
