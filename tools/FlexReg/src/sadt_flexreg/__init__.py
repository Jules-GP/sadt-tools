"""FlexReg: build a registration patch on an intraoral surface, and register on it.

Ported from `FlexReg_CLI.py` and the `FlexReg_Method/` package of
SlicerAutomatedDentalTools. Two arches are aligned on a REGION the clinician
chooses rather than on the whole mesh, because teeth move between timepoints and
the palate does not: registering on everything drags the result toward whatever
moved most.

What upstream had that this does not, and why:

- `curve` built a patch from a polyline drawn on the mesh in the 3D view. Its
  input is the stroke itself, so there is no form of it that does not include a
  Slicer scene; sending a mesh across the network per stroke is slower than the
  local pass it replaces.
- `delete` renamed `Butterfly<n+1>` down over `Butterfly<n>` and dropped the
  last. It is array bookkeeping with no computation in it, and a round trip to
  a server costs more than doing it.

Both stay in the Slicer module. What moved here is what needs a GPU or takes
real time: the butterfly patch, and the registration.

The GPU is not optional for the patch. Upstream's propagation calls `.cuda()`
with no availability test and no device argument, which is why the module ships
191 lines of `install_pytorch.py` to get torch into Slicer on a clinician's
laptop. Running it here is the point: the server has the card.
"""

import os
from pathlib import Path
from typing import Literal

import json

from .pipeline import (
    BUTTERFLY_ARRAY,
    MUCOGINGIVAL_ARRAY,
    ToolInputError,
    build_butterfly,
    merge_patches,
    read_surface,
    register,
    surfaces_in,
    write_surface,
    write_transform,
)


def run(
    surfaces: Path,
    output_dir: Path,
    mode: Literal["Patch", "Register", "Patch and register"] = "Patch and register",
    patch: Literal["Palate (butterfly)", "Mucogingival line"] = "Palate (butterfly)",
    reference: Path = "",
    # Universal tooth numbers bounding the patch. The defaults are upstream's
    # form defaults, which are the first molars and canines of an upper arch.
    tooth_anterior_right: int = 6,
    tooth_anterior_left: int = 11,
    tooth_posterior_right: int = 3,
    tooth_posterior_left: int = 14,
    # How far along each tooth the patch boundary sits, 0 to 1.
    ratio_anterior_right: float = 0.5,
    ratio_anterior_left: float = 0.5,
    ratio_posterior_right: float = 0.5,
    ratio_posterior_left: float = 0.5,
    # Millimetres added to each corner after the ratio places it.
    adjust_anterior_right: float = 0.0,
    adjust_anterior_left: float = 0.0,
    adjust_posterior_right: float = 0.0,
    adjust_posterior_left: float = 0.0,
    shift_lr: float = 0.0,
    shift_ap: float = 0.0,
    output_suffix: str = "_Reg",
) -> Path:
    """Build a registration patch on intraoral surfaces, and register them.

    Args:
        surfaces: One labelled intraoral surface (.vtk/.vtp/.stl), or a folder of
            them. A folder is one call rather than one call per patient.
        output_dir: Where the patched or registered surfaces are written.
        mode: Build the patch, register on a patch already there, or both.
        patch: Which region the registration is computed on. The palate is built
            from the four teeth below; the mucogingival line has to be present in
            the mesh already.
        reference: The surface the others are registered onto. Required to
            register, unused when only building a patch.
        tooth_anterior_right: Universal number of the anterior right tooth.
        tooth_anterior_left: Universal number of the anterior left tooth.
        tooth_posterior_right: Universal number of the posterior right tooth.
        tooth_posterior_left: Universal number of the posterior left tooth.
        ratio_anterior_right: Where the boundary sits along that tooth, 0 to 1.
        ratio_anterior_left: Where the boundary sits along that tooth, 0 to 1.
        ratio_posterior_right: Where the boundary sits along that tooth, 0 to 1.
        ratio_posterior_left: Where the boundary sits along that tooth, 0 to 1.
        adjust_anterior_right: Millimetres added to that corner.
        adjust_anterior_left: Millimetres added to that corner.
        adjust_posterior_right: Millimetres added to that corner.
        adjust_posterior_left: Millimetres added to that corner.
        shift_lr: Millimetres the whole patch is moved left or right.
        shift_ap: Millimetres the whole patch is moved forward or back.
        output_suffix: Appended to each written file's name.

    Returns:
        The output directory: the written surfaces, their `.tfm` transforms, and
        `FlexReg_report.json` naming per surface what was built and what it
        registered on.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registering = mode in ("Register", "Patch and register")
    patch_array = BUTTERFLY_ARRAY if patch == "Palate (butterfly)" else MUCOGINGIVAL_ARRAY

    # Checked before a file is read: with no reference there is nothing to
    # register onto whatever the rest of the request says, and reading a cohort
    # first only delays the same answer.
    if registering and not str(reference):
        raise ToolInputError(
            "'{}' registers onto a reference surface: name one in 'reference'.".format(mode)
        )
    target = read_surface(str(reference)) if registering else None

    teeth = {
        "anterior_right": tooth_anterior_right,
        "anterior_left": tooth_anterior_left,
        "posterior_right": tooth_posterior_right,
        "posterior_left": tooth_posterior_left,
    }
    ratios = {
        "anterior_right": ratio_anterior_right,
        "anterior_left": ratio_anterior_left,
        "posterior_right": ratio_posterior_right,
        "posterior_left": ratio_posterior_left,
    }
    adjustments = {
        "anterior_right": adjust_anterior_right,
        "anterior_left": adjust_anterior_left,
        "posterior_right": adjust_posterior_right,
        "posterior_left": adjust_posterior_left,
    }

    root = Path(surfaces)
    report = {"mode": mode, "patch": patch, "surfaces": {}}
    produced = []

    for path in surfaces_in(str(root)):
        # Relative to the input root, so two patients named the same in
        # different folders do not overwrite each other and the output mirrors
        # the tree it came from.
        relative = os.path.relpath(path, str(root)) if root.is_dir() else os.path.basename(path)
        entry = {}
        try:
            surface = read_surface(path)

            if mode in ("Patch", "Patch and register") and patch == "Palate (butterfly)":
                build_butterfly(surface, teeth, ratios, adjustments, 1, shift_lr, shift_ap)
                merge_patches(surface)
                entry["patch"] = BUTTERFLY_ARRAY

            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            stem = destination.stem
            written = destination.with_name(stem + output_suffix + destination.suffix)

            if registering:
                surface, matrix = register(surface, target, patch_array)
                entry["registered_on"] = patch_array
                transform_path = written.with_name(stem + output_suffix + ".tfm")
                write_transform(matrix, str(transform_path))
                entry["transform"] = str(transform_path)
                produced.append(str(transform_path))

            write_surface(surface, str(written))
            entry["status"] = "ok"
            entry["output"] = str(written)
            produced.append(str(written))
        except ToolInputError as refused:
            # One patient the caller has to fix must not cost the other thirty-
            # nine: it is reported and the batch continues.
            entry = {"status": "failed", "error": str(refused)}
        report["surfaces"][relative] = entry

    if not produced:
        raise ToolInputError(
            "None of the {} surface(s) could be processed; the per-surface "
            "reasons are in the report.".format(len(report["surfaces"]))
        )

    report["summary"] = {
        "total": len(report["surfaces"]),
        "processed": sum(1 for e in report["surfaces"].values() if e.get("status") == "ok"),
        "failed": sum(1 for e in report["surfaces"].values() if e.get("status") == "failed"),
    }

    report_path = output_dir / "FlexReg_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_dir
