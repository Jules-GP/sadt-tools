"""ALI's dispatcher: work out what the input is, then run the right engine.

ALI is one tool with two engines that share nothing but their output format.
Which applies is decided HERE, from the data, not from an argument: a folder
can hold either kind, a DICOM series has no extension at all, and a user who
has to declare what they just sent can declare it wrong -- at which point the
error arrives minutes into a GPU run rather than at the door.

Everything before inference lives here (discovery, DICOM conversion, mode
detection, the run report), so `cbct/` and `ios/` only have to know how to
place landmarks.

The schema cannot say "this argument only applies in mode X", so both
selections carry defaults and both are always rendered. Emptying the selection
for the mode that actually ran raises `ToolInputError`, which is how a mode
mismatch explains itself.
"""

import json
import logging
import os
import shutil
import time

from .cbct import catalog as cbct_catalog
from .errors import ToolInputError
from .ios import catalog as ios_catalog

logger = logging.getLogger(__name__)


VOLUME_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")
SURFACE_EXTENSIONS = (".vtk", ".stl")

CBCT = "CBCT"
IOS = "IOS"

REPORT_NAME = "run_report.json"

# Intermediates live here, under the output directory the caller owns, and are
# removed before `identify` returns. A surviving `.ali_work/` means a run
# crashed.
WORK_DIRNAME = ".ali_work"


class Input:
    """What the input turned out to hold, and how to run it.

    `scans` is a list of `(absolute path, key)` pairs. The key is the path
    relative to the input root and is what identifies a scan everywhere
    afterwards -- in the report and in the output tree. Keying by BASE NAME,
    as the original did, meant two patients called `scan.nii.gz` in different
    subfolders silently overwrote each other twice over: once in the working
    dictionary, once in the flat output folder.
    """

    def __init__(self, mode: str, scans: list, converted_dicom: int = 0):
        self.mode = mode
        self.scans = scans
        self.converted_dicom = converted_dicom


def scan_key(path: str, root: str) -> str:
    """A scan's identity: its path relative to the input root."""
    if not root:
        return os.path.basename(path)
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return os.path.basename(path)


# ---------------------------------------------------------------------------
# Discovery and mode detection
# ---------------------------------------------------------------------------

def _dicom_directories(root: str) -> list:
    """Directories under `root` holding a DICOM series.

    Only directories with no volume or surface file of their own are probed:
    a folder that already contains NIfTI is a folder of scans, not a series,
    and asking GDCM about every directory of a large cohort is slow.
    """
    from .cbct import preprocess

    found = []
    for directory, _subdirs, files in os.walk(root):
        if any(name.lower().endswith(VOLUME_EXTENSIONS + SURFACE_EXTENSIONS) for name in files):
            continue
        if not files:
            continue
        if preprocess.is_dicom_series(directory):
            found.append(directory)
    return sorted(found)


def detect(input_path: str, work_dir: str) -> Input:
    """Work out whether this input is CBCT or IOS, and list what to process.

    Raises `ToolInputError` when the input holds neither kind, or both. "Both"
    is refused rather than guessed at: running one engine and silently ignoring
    half the batch is the failure that looks like success.

    No archive is unpacked here. The server extracts a `.zip` before `run()` is
    called -- with the bomb cap and the single-root strip this function used to
    apply itself -- so what arrives is always a real file or directory.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    root = input_path

    if os.path.isfile(root):
        lower = root.lower()
        if lower.endswith(VOLUME_EXTENSIONS):
            return Input(CBCT, [(root, os.path.basename(root))])
        if lower.endswith(SURFACE_EXTENSIONS):
            return Input(IOS, [(root, os.path.basename(root))])
        raise ToolInputError(
            f"'{os.path.basename(root)}' is neither a CBCT volume "
            f"({', '.join(VOLUME_EXTENSIONS)}) nor an intraoral surface "
            f"({', '.join(SURFACE_EXTENSIONS)})."
        )

    volumes, surfaces = [], []
    for directory, _subdirs, files in os.walk(root):
        # Never re-discover our own intermediates as input: `.ali_work/` sits
        # under the output directory, which a caller is free to point at the
        # input tree.
        if WORK_DIRNAME in directory.split(os.sep):
            continue
        for name in sorted(files):
            path = os.path.join(directory, name)
            if name.lower().endswith(VOLUME_EXTENSIONS):
                volumes.append(path)
            elif name.lower().endswith(SURFACE_EXTENSIONS):
                surfaces.append(path)

    dicom_dirs = _dicom_directories(root) if not surfaces else []

    if (volumes or dicom_dirs) and surfaces:
        raise ToolInputError(
            f"This input mixes {len(volumes) + len(dicom_dirs)} CBCT scan(s) and "
            f"{len(surfaces)} intraoral surface(s). ALI runs one engine per call; "
            f"send them as two batches."
        )

    if surfaces:
        return Input(IOS, [(path, scan_key(path, root)) for path in sorted(surfaces)])

    if volumes or dicom_dirs:
        scans = [(path, scan_key(path, root)) for path in sorted(volumes)]
        converted = _convert_dicom(dicom_dirs, root, work_dir)
        scans.extend(converted)
        return Input(CBCT, sorted(scans, key=lambda item: item[1]), converted_dicom=len(converted))

    raise ToolInputError(
        f"No CBCT scan ({', '.join(VOLUME_EXTENSIONS)}), DICOM series, or intraoral surface "
        f"({', '.join(SURFACE_EXTENSIONS)}) found in the input."
    )


def _convert_dicom(directories: list, root: str, work_dir: str) -> list:
    """Convert each DICOM series to NIfTI; return (path, key) pairs.

    Written into the working directory, never into the input. The original
    created `<input>/NIFTI/` inside the folder the user had selected -- so it
    modified their data, and a second run re-discovered its own output as
    input scans.
    """
    if not directories:
        return []

    from .cbct import preprocess

    destination_root = os.path.join(work_dir, "dicom_converted")
    converted = []
    for directory in directories:
        key = scan_key(directory, root)
        destination = os.path.join(destination_root, f"{key.replace(os.sep, '_')}.nii.gz")
        try:
            preprocess.convert_dicom_series(directory, destination)
        except Exception:
            # A folder GDCM claimed as a series but could not read. Skipped
            # with a log line naming nothing: the folder name is the patient.
            logger.exception("Could not convert a DICOM series")
            continue
        converted.append((destination, f"{key}.nii.gz"))
    logger.info("ALI: converted %d DICOM series", len(converted))
    return converted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def identify(
    input_path: str,
    model_path: str,
    output_dir: str,
    cbct_regions=None,
    landmarks=None,
    ios_networks=None,
    prediction_ID: str = "Pred",
    device: str = "cuda",
    search_seconds: float = 0.0,
) -> dict:
    """Place landmarks on whatever this input holds; return the run report.

    Everything is written under `output_dir`: one markups file per scan, in the
    input's own tree, plus `run_report.json`. Intermediates go in
    `<output_dir>/.ali_work/` and are removed before returning.
    """
    started_at = time.monotonic()

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, WORK_DIRNAME)
    os.makedirs(work_dir, exist_ok=True)

    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"

    try:
        # Walking a cohort and converting DICOM are minutes of work on a large
        # batch, and used to happen in complete silence -- so a run looked hung
        # before it had even started. Counts only, never a file name.
        logger.info("ALI: inspecting the input")
        detected = detect(input_path, work_dir)
        logger.info(
            "ALI: %s input, %d scan(s)%s",
            detected.mode,
            len(detected.scans),
            f", {detected.converted_dicom} converted from DICOM"
            if detected.converted_dicom
            else "",
        )

        if detected.mode == CBCT:
            # Imported here, not at module level: the two engines pull torch,
            # monai, itk and (for IOS) pytorch3d, and only one of them runs.
            from .cbct import engine as cbct_engine

            # An explicit landmark list replaces the regions rather than
            # narrowing them -- see engine.requested_landmarks for why, and for
            # the eight-fold cost that motivates it.
            chosen_landmarks = cbct_catalog.landmark_names(landmarks)
            regions = cbct_catalog.region_codes(cbct_regions)
            if not chosen_landmarks and not regions:
                # The cross-argument rule the schema cannot express. Naming the
                # detected mode is the point: it is what tells a caller who
                # emptied the other group's selection what actually happened.
                raise ToolInputError(
                    f"This input is a CBCT batch: select at least one region under "
                    f"'cbct_regions' ({', '.join(cbct_catalog.REGION_NAMES)}), or name the "
                    f"points you want under 'landmarks'."
                )

            report = cbct_engine.predict_landmarks(
                scans=detected.scans,
                model_path=model_path,
                regions=regions,
                landmarks=chosen_landmarks,
                prediction_ID=prediction_ID,
                output_dir=output_dir,
                work_dir=work_dir,
                device=device,
                search_seconds=search_seconds,
            )
            report["dicom_series_converted"] = detected.converted_dicom
        else:
            from .ios import engine as ios_engine

            networks = ios_catalog.network_codes(ios_networks)
            if not networks:
                raise ToolInputError(
                    f"This input is a batch of intraoral surfaces: select at least one landmark "
                    f"family under 'ios_networks' ({', '.join(ios_catalog.NETWORK_NAMES)})."
                )

            report = ios_engine.predict_landmarks(
                meshes=detected.scans,
                model_path=model_path,
                networks=networks,
                prediction_ID=prediction_ID,
                output_dir=output_dir,
                device=device,
            )
    finally:
        # The intermediates are large -- converted DICOM, and every scan
        # preprocessed at two spacings. Removed whether or not the run
        # succeeded, and never from inside the output tree the caller keeps.
        shutil.rmtree(work_dir, ignore_errors=True)

    report["tool"] = "ALI"
    # So the report says which weights ran even when nobody read the argument.
    report["model_bundle"] = os.path.basename(str(model_path).rstrip(os.sep))
    report["output_dir"] = output_dir
    report["duration_seconds"] = round(time.monotonic() - started_at, 2)

    # Named `run_report.json` because that is what the Slicer module reads to
    # tell "the model bundle has no such landmark" from "the agent did not
    # converge on this scan" -- two failures that look identical in the scene
    # and need opposite fixes.
    with open(os.path.join(output_dir, REPORT_NAME), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info(
        "ALI finished in %s mode: %d/%d scan(s) in %.1fs",
        report["mode"],
        report["summary"]["processed"],
        report["summary"]["total"],
        report["duration_seconds"],
    )
    return report
