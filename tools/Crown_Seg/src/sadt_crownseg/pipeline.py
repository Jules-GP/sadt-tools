"""Crown segmentation on intraoral surface scans, via shapeaxi.

Labels every tooth of a mesh with its Universal number, as a point-data array
on a copy of the surface. That array is the precondition for ALI's IOS landmark
identification and for the IOS modes of ASO, AREG and FlexReg, which is why
this is its own tool rather than a helper inside any of them -- and, since the
split, why the server sequences CrownSeg before ALI instead of ALI importing
this module.

Nothing here is ported. The Slicer modules ran the `dentalmodelseg` executable
out of Slicer's own bin directory, and that executable is only the
console-script entry point of the `shapeaxi` PyPI package
(`dentalmodelseg = shapeaxi.dental_model_seg:cml`). There is no Slicer binary
to shell out to, so `shapeaxi.dental_model_seg.main` is called directly with
the namespace its own `cml()` would have built.

What the move out of the server changed: scratch space lives under the caller's
`output_dir`, `segment_crowns()` returns the run report rather than a
`CrownSegRun`, zip extraction is gone (the server unpacks archives before
`run()` is called), the model is a required argument rather than a name looked
up in the data store, and the GPU semaphore is gone because each call is its
own process.
"""

import contextlib
import io
import json
import logging
import os
import shutil
import time
from argparse import Namespace

from .errors import ToolInputError, ToolUnavailableError

logger = logging.getLogger(__name__)

# Extensions shapeaxi's reader handles and this tool therefore discovers.
SURFACE_EXTENSIONS = (".vtk", ".stl")

# Point-data array names that already carry per-tooth labels. A mesh holding
# any of them is segmented and is passed through untouched -- re-running the
# network on it would cost minutes and change nothing.
LABEL_ARRAY_NAMES = ("PredictedID", "UniversalID", "Universal_ID")

DEFAULT_ARRAY_NAME = "Universal_ID"
DEFAULT_SUFFIX = "Seg"

WORK_DIRNAME = ".crownseg_work"

_INSTALL_HINT = (
    "CrownSeg's engine is an optional extra: shapeaxi pulls pytorch3d, which "
    "publishes no usable wheel and is compiled from source. Install it with "
    "`uv sync --extra segmentation` (needs a CUDA toolkit, see README.md)."
)


def _restore_moved_class(saxi_nets) -> bool:
    """Work around an upstream bug: shapeaxi 2.0.x cannot find its own class.

    `DentalModelSeg` moved from `shapeaxi.saxi_nets` to
    `shapeaxi.saxi_nets_lightning`, but `shapeaxi/dental_model_seg.py` -- the
    module behind shapeaxi's own `dentalmodelseg` console script, and the only
    supported way in -- still reads it off `saxi_nets`. Every 2.0.x release
    (2.0.0, 2.0.1, 2.0.2) has this; 1.x referenced the right module. So
    `dental_model_seg.main()` raises AttributeError before touching a mesh, and
    the pre-port tool fails in exactly the same way on the deployed image.

    Two lines rather than reimplementing `main()`: this keeps the entry point
    shapeaxi supports, and the `hasattr` guard makes the workaround vanish by
    itself the day upstream puts the name back. Report it at
    DCBIA-OrthoLab/ShapeAXI; delete this function once a release carries the fix.
    """
    if hasattr(saxi_nets, "DentalModelSeg"):
        return False

    from shapeaxi import saxi_nets_lightning

    target = getattr(saxi_nets_lightning, "DentalModelSeg", None)
    if target is None:
        raise ToolUnavailableError(
            "This shapeaxi exposes DentalModelSeg in neither saxi_nets nor "
            "saxi_nets_lightning; the pinned version is 2.0.2."
        )
    saxi_nets.DentalModelSeg = target
    logger.warning(
        "shapeaxi %s does not expose DentalModelSeg where its own "
        "dental_model_seg module looks for it; pointing it at "
        "saxi_nets_lightning (see _restore_moved_class)",
        _shapeaxi_version(),
    )
    return True


def _shapeaxi_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("shapeaxi")
    except PackageNotFoundError:  # pragma: no cover - installed by definition here
        return "unknown"


def _import_dental_model_seg():
    """Import shapeaxi's segmentation module, lazily.

    Deferred so that importing this package -- which `scripts/describe.py` does
    on every CI run to publish the schema -- costs nothing. It also means a
    venv without the extra still loads and reports the schema; only an actual
    segmentation fails, naming what is missing.
    """
    try:
        from shapeaxi import dental_model_seg, saxi_nets
    except ImportError as exc:
        raise ToolUnavailableError(
            "{} (missing: {})".format(_INSTALL_HINT, exc.name or "shapeaxi")
        ) from exc

    _restore_moved_class(saxi_nets)
    return dental_model_seg


def resolve_device(requested: str = None) -> str:
    """The device string shapeaxi accepts: exactly "cpu" or "cuda:0"."""
    import torch

    wanted = (requested or "cpu").strip().lower()
    if wanted.startswith("cuda") and torch.cuda.is_available():
        return "cuda:0"
    if wanted.startswith("cuda"):
        logger.warning("device=%s requested but CUDA is unavailable; falling back to CPU", wanted)
    return "cpu"


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def is_segmented(mesh_path: str) -> bool:
    """True when the mesh already carries a per-point tooth-label array.

    Reads only the point-data array NAMES, never their values: this decides
    whether to spend minutes of GPU time, and nothing about the patient.
    """
    import vtk

    extension = os.path.splitext(mesh_path)[1].lower()
    if extension == ".stl":
        # An STL has no point data at all by construction, so it can never be
        # segmented. Reading it anyway keeps one code path.
        reader = vtk.vtkSTLReader()
    elif extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        return False

    reader.SetFileName(mesh_path)
    reader.Update()
    point_data = reader.GetOutput().GetPointData()
    names = {point_data.GetArrayName(index) for index in range(point_data.GetNumberOfArrays())}
    return bool(names & set(LABEL_ARRAY_NAMES))


def discover_meshes(input_path: str) -> list:
    """Resolve the `meshes` argument into a list of surface files.

    One mesh, or a folder of them for a batch. Folder scanning is recursive,
    and the tree is what the output mirrors.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path):
        if not input_path.lower().endswith(SURFACE_EXTENSIONS):
            raise ToolInputError(
                f"'{os.path.basename(input_path)}' is not a surface mesh. Expected one of: "
                f"{', '.join(SURFACE_EXTENSIONS)}."
            )
        return [input_path]

    found = [
        os.path.join(root, name)
        for root, _dirs, files in os.walk(input_path)
        for name in sorted(files)
        if name.lower().endswith(SURFACE_EXTENSIONS)
    ]
    if not found:
        raise ToolInputError(
            f"No surface mesh found in the input. Supported extensions: "
            f"{', '.join(SURFACE_EXTENSIONS)}."
        )
    return sorted(found)


def _input_root(input_path: str, meshes: list) -> str:
    """The directory output paths are made relative to, so a batch keeps its
    tree instead of being flattened into one folder where two patients named
    scan.vtk overwrite each other."""
    if os.path.isdir(input_path):
        return input_path
    return os.path.dirname(meshes[0])


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _run_shapeaxi(csv_path: str, output_dir: str, model_path: str, input_root: str,
                  array_name: str, suffix: str, device: str, fdi: bool,
                  num_workers: int) -> None:
    """Call shapeaxi's own `main()` with the namespace its `cml()` would build.

    stdout is swallowed on purpose. shapeaxi prints "Saving results to <path>"
    for every mesh, and that path carries the patient's own file name, which
    must not reach a log. The exception it raises on a real failure is
    untouched.
    """
    dental_model_seg = _import_dental_model_seg()

    args = Namespace(
        surf=None,
        csv=csv_path,
        model=model_path,
        suffix=suffix,
        out=output_dir,
        # Must be >= 1: shapeaxi builds its loader with persistent_workers=True,
        # which PyTorch rejects at 0.
        num_workers=max(1, int(num_workers)),
        crown_segmentation=0,
        array_name=array_name,
        fdi=1 if fdi else 0,
        overwrite=0,
        device=device,
        vtk_folder=input_root,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        dental_model_seg.main(args)


def _predicted_path(output_dir: str, csv_stem: str, suffix: str, mesh: str,
                    input_root: str) -> str:
    """Where shapeaxi's csv branch writes the segmented copy of `mesh`.

    Mirrors `save_data_vtk_from_csv`: the input path minus its extension, plus
    `_<suffix>.vtk`, with `vtk_folder` stripped off the front, filed under
    `<out>/<csv stem>_<suffix>/`. Recomputed rather than parsed back out of
    shapeaxi's stdout, which is where the file names this tool must not log
    would have had to travel.
    """
    without_extension = os.path.splitext(mesh)[0] + f"_{suffix}.vtk"
    relative = without_extension.replace(input_root, "").lstrip(os.sep)
    return os.path.normpath(os.path.join(output_dir, f"{csv_stem}_{suffix}", relative))


def segment_crowns(
    input_path: str,
    model_path: str,
    output_dir: str,
    array_name: str = DEFAULT_ARRAY_NAME,
    suffix: str = DEFAULT_SUFFIX,
    fdi: bool = False,
    skip_segmented: bool = True,
    device: str = "cuda",
    num_workers: int = 2,
) -> dict:
    """Label the teeth of one mesh or a whole folder of them.

    `skip_segmented=True` passes an already-labelled mesh through unchanged
    instead of re-running the network on it, which is what makes a mixed batch
    of raw and pre-segmented meshes one call.
    """
    started_at = time.monotonic()

    array_name = (array_name or DEFAULT_ARRAY_NAME).strip() or DEFAULT_ARRAY_NAME
    suffix = (suffix or DEFAULT_SUFFIX).strip() or DEFAULT_SUFFIX

    output_dir = os.fspath(output_dir)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    if not os.path.isfile(os.fspath(model_path)):
        raise ToolInputError(f"Crown-segmentation checkpoint not found: {model_path}")

    meshes = discover_meshes(os.fspath(input_path))
    input_root = _input_root(os.fspath(input_path), meshes)

    already_segmented, to_segment = [], []
    for mesh in meshes:
        (already_segmented if skip_segmented and is_segmented(mesh) else to_segment).append(mesh)

    records = {}
    produced = []

    # A mesh that already carries labels is copied, not re-predicted. Copied
    # rather than referenced in place because the caller is handed one output
    # tree, and because the input may live on a read-only mount.
    for mesh in already_segmented:
        relative = os.path.relpath(mesh, input_root)
        base = os.path.splitext(os.path.basename(mesh))[0]
        destination = os.path.join(
            output_dir, os.path.dirname(relative), f"{base}_{suffix}.vtk"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        _write_as_vtk(mesh, destination)
        records[relative] = {"status": "already_segmented", "output": destination}
        produced.append(destination)

    if to_segment:
        device = resolve_device(device)

        # Written into the work dir. The Slicer module wrote this csv into the
        # extension's own source folder, which breaks a read-only install and
        # leaves a file behind pointing at the patient's data.
        csv_path = os.path.join(work_dir, "crownseg_input.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("surf\n")
            for mesh in to_segment:
                handle.write(f"{mesh}\n")

        logger.info("CrownSeg: segmenting %d mesh(es) on %s", len(to_segment), device)
        try:
            _run_shapeaxi(
                csv_path=csv_path,
                output_dir=output_dir,
                model_path=os.fspath(model_path),
                input_root=input_root,
                array_name=array_name,
                suffix=suffix,
                device=device,
                fdi=fdi,
                num_workers=num_workers,
            )
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        for mesh in to_segment:
            relative = os.path.relpath(mesh, input_root)
            predicted = _predicted_path(output_dir, csv_stem, suffix, mesh, input_root)
            if os.path.isfile(predicted):
                records[relative] = {"status": "segmented", "output": predicted}
                produced.append(predicted)
            else:
                # One mesh shapeaxi could not write must not cost the batch.
                records[relative] = {
                    "status": "failed",
                    "error": "the segmentation produced no output for this mesh",
                }

    shutil.rmtree(work_dir, ignore_errors=True)

    if not produced:
        raise RuntimeError("CrownSeg produced no segmented mesh for any input.")

    report = {
        "tool": "CrownSeg",
        "array_name": array_name,
        "suffix": suffix,
        "numbering": "FDI" if fdi else "Universal",
        "device": device if to_segment else None,
        "meshes": records,
        # Absolute paths of every mesh that now carries tooth labels, whether
        # this run produced them or found them already labelled. This is what a
        # caller sequencing CrownSeg before ALI reads.
        "segmented_meshes": produced,
        "summary": {
            "total": len(meshes),
            "segmented": sum(1 for r in records.values() if r["status"] == "segmented"),
            "already_segmented": len(already_segmented),
            "failed": sum(1 for r in records.values() if r["status"] == "failed"),
        },
        "duration_seconds": round(time.monotonic() - started_at, 2),
    }

    with open(os.path.join(output_dir, "run_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def _write_as_vtk(source: str, destination: str) -> None:
    """Copy a mesh to `destination` as .vtk polydata.

    A pre-segmented input can be an .stl, and shapeaxi's outputs are always
    .vtk; normalizing here means the caller gets one format whichever branch a
    mesh took, instead of an .stl that every downstream reader then has to
    special-case. (The Slicer module copied it verbatim, and its landmark CLI
    then discovered only .vtk -- so those meshes were silently never used.)
    """
    import vtk

    extension = os.path.splitext(source)[1].lower()
    reader = vtk.vtkSTLReader() if extension == ".stl" else vtk.vtkPolyDataReader()
    reader.SetFileName(source)
    reader.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(destination)
    writer.SetInputData(reader.GetOutput())
    writer.Write()
