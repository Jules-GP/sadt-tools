"""AREG -- Automated Registration of a follow-up scan onto its baseline.

Registers every T2 onto its T1 so the two timepoints share one coordinate
system and can be measured against each other. One tool, two modalities:

* **CBCT** -- elastix, rigid, restricted to the anatomy that has not changed:
  the cranial base, the mandible or the maxilla, taken as masks;
* **IOS** -- a patch of the arch that does not move with growth or treatment
  (the palate, or the band around the mucogingival line), matched by ICP.

The pipeline is in dispatch.py; only `run` is public.

AREG does the registration and nothing else. The masks it registers on, the
orientation both timepoints must share, the tooth labels and the mucogingival
line each come from another tool, reached through the supervisor -- see
tools.py. That is what makes this the deepest chain in the family:
`AREG -> ASO -> ALI`, three tools and three virtualenvs.
"""

from pathlib import Path
from typing import Literal

REGIONS = ["Cranial base", "Mandible", "Maxilla"]


def run(
    t1: Path,
    t2: Path,
    output_dir: Path,
    modality: Literal["CBCT", "IOS"] = "CBCT",
    automation: Literal[
        "Semi-Automated", "Fully-Automated", "Oriented + Fully-Automated"
    ] = "Fully-Automated",
    # Spelled out because `Literal` takes literals only -- it cannot be built
    # from catalogs.REGION_CHOICES. That makes this a second declaration of the
    # same set, which is the thing this contract otherwise avoids, so a test
    # asserts the two agree.
    cbct_regions: list[
        Literal["Cranial base", "Mandible", "Maxilla"]
    ] = ["Cranial base"],
    t1_masks: Path = "",
    segmentation_model: Path = "",
    segmentation_label: int = 0,
    cbct_reference: Path = "",
    ios_reference: Path = "",
    ios_patch: Literal[
        "Palate (upper arch)", "Mucogingival line (lower arch)"
    ] = "Palate (upper arch)",
    registration_model: Path = "",
    mgl_landmarks: Path = "",
    mgl_patch_height: float = 0.0,
    dicom_input: bool = False,
    output_suffix: str = "Reg",
    *,
    sup=None,
) -> Path:
    """Register a follow-up scan onto its baseline, so the two can be compared.

    Args:
        t1: The baseline timepoint — a folder of CBCT scans, or of intra-oral
            meshes. Searched recursively; T1 and T2 are paired by patient name.
        t2: The follow-up timepoint, same shape as `t1`.
        output_dir: Where results are written — per patient, the registered T2
            and the transform that produced it — plus `AREG_report.json`.
            Nothing is written outside it.
        modality: CBCT volumes or intra-oral surface scans. Never inferred from
            the file extension: a folder can hold either, and guessing wrong
            registers a patient against the wrong anatomy and calls it success.
        automation: Semi-Automated takes what it needs from you. Fully-Automated
            asks the other tools for it. "Oriented + Fully-Automated" (CBCT
            only) is the middle ground: the scans are already oriented, so the
            orientation step is skipped and the masks are still segmented.
        cbct_regions: CBCT only. The anatomy to register on — pick what has NOT
            changed between the two timepoints. The cranial base is the usual
            choice; the mandible or maxilla suit a patient whose growth is
            elsewhere.
        t1_masks: CBCT Semi-Automated only. Your own T1 segmentation masks, one
            binary file per region, instead of having them segmented.
        segmentation_model: CBCT only. The AMASSS bundle used to segment the T1
            masks when they are not supplied.
        segmentation_label: CBCT only. The label value to read out of a
            multi-label mask file. 0 means "any non-zero voxel", which is what a
            binary mask needs.
        cbct_reference: CBCT only. The reference the scans are oriented onto
            before registering, when the mode orients them.
        ios_reference: IOS only, and the same idea.
        ios_patch: IOS only. Which part of the arch to match on — the palate for
            an upper arch, the band around the mucogingival line for a lower one.
        registration_model: IOS only. The model that finds the patch.
        mgl_landmarks: IOS only, and only for the mucogingival patch. Your own
            13 landmarks per lower scan, instead of having them predicted.
        mgl_patch_height: IOS only. How far the band extends from the
            mucogingival line, in millimetres. 0 uses the model's own default.
        dicom_input: CBCT only. Convert DICOM series found in the input to NIfTI
            before registering.
        output_suffix: Added to each output name, e.g. `patient1_Reg.nii.gz`.

    Returns:
        The output directory, holding the registered cases and the run report.

    A Fully-Automated run needs the tools it drives to be reachable. Without a
    supervisor it says so up front, and names the mode that works instead —
    see tools.py.
    """
    # elastix, torch, monai, pytorch3d, SimpleITK and VTK are all imported
    # inside the pipelines: describe.py imports this module on every CI run to
    # publish the schema, and that must not cost a CUDA stack.
    from .dispatch import main

    output_dir = Path(output_dir)
    main(
        modality=modality,
        automation=automation,
        t1=str(t1),
        t2=str(t2),
        t1_masks=str(t1_masks) if t1_masks else None,
        cbct_regions=list(cbct_regions),
        segmentation_label=segmentation_label,
        segmentation_model=str(segmentation_model) if segmentation_model else None,
        cbct_reference=str(cbct_reference) if cbct_reference else None,
        ios_reference=str(ios_reference) if ios_reference else None,
        registration_model=str(registration_model) if registration_model else None,
        ios_patch=ios_patch,
        mgl_landmarks=str(mgl_landmarks) if mgl_landmarks else None,
        mgl_patch_height=mgl_patch_height or None,
        dicom_input=dicom_input,
        output_suffix=output_suffix,
        output_dir=str(output_dir),
        sup=sup,
    )
    return output_dir
