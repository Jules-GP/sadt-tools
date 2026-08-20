"""GreedyReg -- affine registration of a follow-up CBCT onto its baseline.

Greedy solves the affine; the optional landmark mode brings two scans that are
far apart close enough for Greedy's search to find anything at all. The work is
in pipeline.py; only `run` is public.
"""

from pathlib import Path
from typing import Literal

from .pipeline import process


def run(
    t1: Path,
    t2: Path,
    output_dir: Path,
    mode: Literal["Greedy", "Landmark", "Landmark + Greedy"] = "Greedy",
    metric: Literal["NMI", "NCC", "SSD"] = "NMI",
    transform_type: Literal["Rigid", "Affine"] = "Rigid",
    masks: Path = "",
    init: Path = "",
    # Spelled out because `Literal` takes literals only -- it cannot be built
    # from landmarks.REGION_LANDMARKS. That makes this a second declaration of
    # the same set, which is the thing this contract otherwise avoids, so a test
    # asserts the two agree.
    region: Literal["MANDMASK", "MAXMASK", "CBMASK"] = "MANDMASK",
    landmark_model: Path = "",
    device: Literal["cuda", "cpu"] = "cuda",
    *,
    sup=None,
) -> Path:
    """Register a follow-up CBCT onto its baseline, so the two can be compared.

    Args:
        t1: The baseline (fixed) scans -- one `.nii`/`.nii.gz` volume, or a
            folder of them for a batch. A folder is paired with T2 on the
            leading letters-and-digits of each file name, so `MG01_T1.nii.gz`
            matches `MG01_T2.nii.gz`. Not searched recursively: a previous run's
            output must not come back in as an input.
        t2: The follow-up (moving) scans, paired to T1 by that same key. Give
            two folders for a batch, or two files for a single pair.
        output_dir: Where the registered volumes, their transforms and
            `GreedyReg_report.json` are written. Nothing is written outside it.
        mode: Greedy registers directly, which needs the two timepoints already
            roughly aligned. Landmark only aligns T2 onto T1 from anatomical
            landmarks, writing a repositioned volume without touching a voxel.
            Landmark + Greedy does the second and hands its transform to the
            first, which is the sequence the module instructs you to run by
            hand when the timepoints are far apart.
        metric: The similarity Greedy optimises. NMI tolerates a difference in
            intensity scaling between the timepoints, NCC assumes a linear
            relation, SSD assumes an identical one.
        transform_type: Rigid keeps the follow-up's shape and size (6 degrees of
            freedom); Affine also allows scaling and shear (12).
        masks: Optional folder of T1-space masks, one per patient key, telling
            Greedy which voxels to score. Registering on what has NOT changed
            between the timepoints is what makes the result mean anything.
            Binarised before use, so a multi-label segmentation is accepted.
        init: Optional folder of per-patient `{key}*.mat` starting transforms.
            A case with no matching file starts from identity. Ignored in the
            landmark modes, which compute their own.
        region: Which anatomy the landmark modes align on -- the mandible
            (MANDMASK), the maxilla (MAXMASK) or the cranial base (CBMASK).
            Each names its own landmark set, and at least three of them must be
            found on both scans.
        landmark_model: The ALI landmark model bundle. Required by the landmark
            modes, unused by Greedy mode. Named rather than resolved here: a
            tool does not go looking for weights on the server's disk.
        device: "cuda" or "cpu", passed to the landmark tool. Greedy itself is
            CPU-only and ignores this.

    Returns:
        The output directory, holding one registered volume and transform per
        pair, plus the run report.
    """
    return process(
        t1=t1,
        t2=t2,
        output_dir=output_dir,
        mode=mode,
        masks=masks,
        init=init,
        metric=metric,
        transform_type=transform_type,
        region=region,
        landmark_model=landmark_model,
        device=device,
        sup=sup,
    )
