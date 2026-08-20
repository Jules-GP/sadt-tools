"""The batch: for every scan, for every matrix that belongs to its patient.

Upstream's `main()` is one loop with three `try/except Exception: continue`
blocks in it, and the exceptions go to a log nobody reads once the Slicer
progress bar has closed. The loop is the same here; what changed is that every
skip is recorded against the file it happened to and comes back in
`AutoMatrix_report.json`, and that a batch which produced NOTHING says so
instead of returning an empty folder and a 200.

That last part is the guard, and it counts files WRITTEN -- not scans walked,
not matrices read. A cohort whose matrices all failed to parse walked every
scan and read every matrix and produced nothing.
"""

import json
import logging
import os
from pathlib import Path

from . import inputs, landmarks, transforms, volumes
from .errors import NothingWritten, ToolInputError

logger = logging.getLogger(__name__)

REPORT_NAME = "AutoMatrix_report.json"

# Where an AREG run leaves the matrix for each region, keyed by the marker that
# appears in the landmark file's name. `_L` -> Maxilla and `_U` -> Mandible
# read backwards and are upstream's, spelled exactly as AREG's output tree
# spells them; changing either would send every landmark set through the wrong
# region's registration.
AREG_LAYOUT = (
    ("_CB", "Cranial Base", "CBReg_matrix.tfm"),
    ("_L", "Maxilla", "MAXReg_matrix.tfm"),
    ("_U", "Mandible", "MANDReg_matrix.tfm"),
)


def _areg_matrix(root, scan):
    """The AREG matrix for one landmark file, or `(None, reason)`.

    Upstream reads this out of an argument the CLI never declares
    (`args.matrix_lineEdit`), so the whole branch raises AttributeError the
    first time a "From AReg" run reaches it. It is wired to the matrix folder
    here, which is what the module's own widget puts in that field.
    """
    name = os.path.basename(str(scan))
    for marker, subdirectory, filename in AREG_LAYOUT:
        if marker not in name:
            continue
        patient = name.split("_")[0]
        # Upstream stops at the FIRST marker present, matched or not, so a name
        # holding both `_CB` and `_L` is a cranial base file. Kept.
        candidate = (Path(root) / subdirectory / "{}_OutReg".format(patient)
                     / "{}_{}".format(patient, filename))
        if candidate.exists():
            return candidate, None
        return None, "no AREG matrix at {}".format(candidate)
    return None, "no region marker ({}) in the name".format(
        ", ".join(marker for marker, _, _ in AREG_LAYOUT))


def _output_path(scan, root, output_dir, extension, out_suffix):
    """Where one result goes: the input's own tree, rebuilt under output_dir.

    Upstream builds this with `str.replace` on the whole path and then splits
    the result on the extension string, so a cohort living under a folder whose
    name contains `.nii` loses part of its path. Same layout, computed from the
    file name.
    """
    relative = Path(scan).relative_to(root)
    stem = relative.name[: len(relative.name) - len(extension)]
    return Path(output_dir) / relative.parent / (stem + out_suffix + extension)


def _read_reference(reference):
    if not str(reference):
        return None
    reference = Path(reference)
    if not reference.is_file():
        raise ToolInputError(
            "Reference volume not found: {}. Leave 'reference' empty to "
            "resample each scan on its own grid.".format(reference)
        )

    import SimpleITK as sitk

    return sitk.ReadImage(str(reference))


def process(scans, matrices, output_dir, suffix, reference, add_matrix_name,
            from_areg, is_segmentation):
    scan_root = Path(scans)
    found = inputs.find_scans(scan_root)
    if found is None:
        raise ToolInputError("Input path does not exist: {}".format(scan_root))
    if not found:
        raise ToolInputError(
            "No scan found at {}. AutoMatrix reads {}.".format(
                scan_root, ", ".join(inputs.SCAN_EXTENSIONS))
        )

    matrix_root = Path(matrices)
    found_matrices, matrices_is_dir = inputs.find_matrices(matrix_root)
    if found_matrices is None:
        raise ToolInputError("Matrix path does not exist: {}".format(matrix_root))
    if not found_matrices:
        raise ToolInputError(
            "No matrix found under {}. AutoMatrix reads {}.".format(
                matrix_root, ", ".join(inputs.MATRIX_EXTENSIONS))
        )

    reference_image = _read_reference(reference)
    root = scan_root if scan_root.is_dir() else scan_root.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "suffix": suffix,
        "add_matrix_name": bool(add_matrix_name),
        "from_areg": bool(from_areg),
        "is_segmentation": bool(is_segmentation),
        "reference": str(reference) or None,
        "cases": [],
    }
    written = 0

    for key, patient_scans, patient_matrices in inputs.group_by_patient(
            found, found_matrices, matrices_is_dir):
        for scan in patient_scans:
            case = {"patient": key, "scan": str(scan), "outputs": [], "skipped": []}
            report["cases"].append(case)
            written += _apply_to_scan(
                case=case,
                scan=scan,
                patient_matrices=patient_matrices,
                matrix_root=matrix_root,
                root=root,
                output_dir=output_dir,
                suffix=suffix,
                add_matrix_name=add_matrix_name,
                from_areg=from_areg,
                is_segmentation=is_segmentation,
                reference_image=reference_image,
            )

    report["written"] = written
    report["skipped"] = sum(len(case["skipped"]) for case in report["cases"])
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    if not written:
        raise NothingWritten(
            "AutoMatrix wrote no file. {} scan(s) and {} matri(x/ces) were read, "
            "and every pairing was skipped: {}".format(
                len(found), len(found_matrices),
                "; ".join(sorted({
                    entry["reason"]
                    for case in report["cases"] for entry in case["skipped"]
                })) or "no matrix matched any patient key",
            )
        )

    logger.info("AutoMatrix: %d file(s) written, %d skipped",
                written, report["skipped"])
    return output_dir


def _apply_to_scan(case, scan, patient_matrices, matrix_root, root, output_dir,
                   suffix, add_matrix_name, from_areg, is_segmentation,
                   reference_image):
    """One scan against every matrix that belongs to it. Returns files written."""
    extension = inputs.extension_of(scan.name, inputs.SCAN_EXTENSIONS)
    is_landmark = extension == inputs.LANDMARK_EXTENSION
    case["kind"] = ("landmarks" if is_landmark else
                    "volume" if extension in inputs.VOLUME_EXTENSIONS else "surface")

    if case["kind"] == "surface":
        # Upstream collects these, hands them to `sitk.ReadImage`, and logs the
        # failure -- so no surface has come out of the CLI since it replaced the
        # in-Slicer implementation. Said plainly here rather than as a stack
        # trace in a log: applying a matrix to a mesh needs a RAS/LPS
        # convention this port has no reference run to fix, and guessing it
        # would return a mirrored mesh that looks right.
        case["skipped"].append({
            "reason": "surface meshes are not supported ({})".format(extension)})
        return 0

    if from_areg and is_landmark:
        matrix, reason = _areg_matrix(matrix_root, scan)
        if matrix is None:
            case["skipped"].append({"reason": reason})
            return 0
        candidates = [matrix]
    else:
        candidates = patient_matrices

    if not candidates:
        case["skipped"].append({"reason": "no matrix matched patient key"})
        return 0

    written = 0
    for matrix in candidates:
        try:
            transform = transforms.read(matrix)
        except Exception as error:  # unreadable, wrong format, truncated
            case["skipped"].append({"matrix": str(matrix), "reason": str(error)})
            continue

        matrix_suffix = "_{}".format(Path(matrix).stem) if add_matrix_name else ""
        output = _output_path(scan, root, output_dir, extension,
                              suffix + matrix_suffix)
        try:
            if is_landmark:
                moved = landmarks.apply(scan, transform, output)
                case["outputs"].append({"matrix": str(matrix),
                                        "output": str(output),
                                        "points_moved": moved})
            else:
                grid = _apply_volume(scan, matrix, transform, output,
                                     reference_image, is_segmentation)
                case["outputs"].append({"matrix": str(matrix),
                                        "output": str(output),
                                        "grid": grid})
        except Exception as error:
            case["skipped"].append({"matrix": str(matrix), "reason": str(error)})
            continue
        written += 1
        logger.info("AutoMatrix: %s -> %s", scan.name, output.name)
    return written


def _apply_volume(scan, matrix, transform, output, reference_image,
                  is_segmentation):
    """Resample one volume, choosing its grid. Returns which grid that was."""
    import SimpleITK as sitk

    image = sitk.ReadImage(str(scan))

    # A mirroring matrix is applied in the scan's OWN space: mirroring onto
    # somebody else's reference grid would move the patient as well as flip
    # them. Upstream keys that on the word "mirror" appearing in the matrix's
    # file name, which is how the module's own Mirror button names the file it
    # downloads.
    mirroring = "mirror" in os.path.basename(str(matrix)).lower()

    grid, neighbour = volumes.composite_reference(transform, matrix)
    if grid == volumes.GRID_COMPOSITE_NEIGHBOUR:
        # A composite overrides the caller's reference entirely: the chain is
        # only valid on the grid it was computed against.
        reference = neighbour
    elif grid == volumes.GRID_COMPOSITE_FALLBACK:
        logger.warning(
            "AutoMatrix: %s is a composite transform with no volume beside it; "
            "resampling %s on its own grid.", os.path.basename(str(matrix)),
            scan.name)
        reference = image
    elif mirroring:
        grid, reference = "mirror_source", image
    elif reference_image is None:
        grid, reference = "scan", image
    else:
        reference = reference_image

    volumes.apply(image, transform, reference, output, is_segmentation)
    return grid
