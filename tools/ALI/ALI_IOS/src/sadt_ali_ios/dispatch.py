"""Everything ALI_IOS does before inference: discovery and the run report.
`engine.py` only has to know how to place landmarks.

ALI used to be one tool choosing an engine from the data. Splitting it in two
moved that question out of the run and into the request: this tool is the
intraoral engine, and an input holding CBCT volumes is refused by name rather
than half-processed.

No DICOM here, unlike ALI_CBCT: a DICOM series is a volume by definition, so
detecting one is that tool's business. It also needs itk, which this
environment does not carry.
"""

import json
import logging
import os
import shutil
import time

from .errors import ToolInputError
from . import catalog as ios_catalog

logger = logging.getLogger(__name__)


REPORT_NAME = "run_report.json"


class Input:
    """What the input turned out to hold, and how to run it.

    `scans` is a list of `(absolute path, key)` pairs. The key is the path
    relative to the input root and is what identifies a scan everywhere
    afterwards -- in the report and in the output tree. Keying by BASE NAME,
    as the original did, meant two patients called `scan.nii.gz` in different
    subfolders silently overwrote each other twice over: once in the working
    dictionary, once in the flat output folder.
    """

    def __init__(self, mode: str, scans: list):
        self.mode = mode
        self.scans = scans



# ---------------------------------------------------------------------------
# Discovery and mode detection
# ---------------------------------------------------------------------------


def detect(input_path: str, work_dir: str) -> Input:
    """List the intraoral surfaces to process, and refuse anything else.

    Before the split this decided WHICH engine ran. It no longer does: this
    tool is the intraoral engine, so the question is only whether the caller
    sent surfaces. Volumes are named rather than ignored -- silently processing
    the meshes of a mixed folder and dropping the scans is the failure that
    looks like success.

    `work_dir` is unused here and kept so both tools' dispatchers have the same
    shape; ALI_CBCT needs it to convert DICOM.

    No archive is unpacked here. The server extracts a `.zip` before `run()`
    is called -- with the bomb cap and the single-root strip this function used
    to apply itself -- so what arrives is always a real file or directory.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    root = input_path

    if os.path.isfile(root):
        lower = root.lower()
        if lower.endswith(SURFACE_EXTENSIONS):
            return Input(IOS, [(root, os.path.basename(root))])
        if lower.endswith(VOLUME_EXTENSIONS):
            raise ToolInputError(
                f"'{os.path.basename(root)}' is a CBCT volume. This tool places "
                f"landmarks on intraoral surfaces; run ALI_CBCT on volumes."
            )
        raise ToolInputError(
            f"'{os.path.basename(root)}' is not an intraoral surface "
            f"({', '.join(SURFACE_EXTENSIONS)})."
        )

    volumes, surfaces = classify(root)

    if volumes and not surfaces:
        raise ToolInputError(
            f"This input holds {len(volumes)} CBCT scan(s) and no intraoral surface. "
            f"Run ALI_CBCT on volumes."
        )
    if volumes:
        raise ToolInputError(
            f"This input mixes {len(volumes)} CBCT scan(s) and {len(surfaces)} intraoral "
            f"surface(s). Send them as two batches, to ALI_CBCT and ALI_IOS respectively."
        )

    if surfaces:
        return Input(IOS, keyed(surfaces, root))

    raise ToolInputError(
        f"No intraoral surface ({', '.join(SURFACE_EXTENSIONS)}) found in the input."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def identify(
    input_path: str,
    model_path: str,
    output_dir: str,
    ios_networks=None,
    prediction_ID: str = "Pred",
    device: str = "cuda",
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
        logger.info("ALI_IOS: inspecting the input")
        detected = detect(input_path, work_dir)
        logger.info("ALI_IOS: %d surface(s)", len(detected.scans))

        # Imported here, not at module level: the engine pulls torch and
        # pytorch3d, and CI imports this package on every PR to publish the
        # schema. That must not cost a CUDA stack.
        from . import engine as ios_engine

        networks = ios_catalog.network_codes(ios_networks)
        if not networks:
            raise ToolInputError(
                f"Select at least one landmark family under 'ios_networks' "
                f"({', '.join(ios_catalog.NETWORK_NAMES)})."
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

    report["tool"] = "ALI_IOS"
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
        "ALI_IOS finished: %d/%d scan(s) in %.1fs",
        report["summary"]["processed"],
        report["summary"]["total"],
        report["duration_seconds"],
    )
    return report
