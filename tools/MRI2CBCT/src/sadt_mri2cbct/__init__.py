"""MRI2CBCT -- bring a TMJ MRI into its CBCT's space, one pipeline step at a time.

The work is in the modules ported from upstream's CLI package; pipeline.py says
which of them a step runs. Only `run` is public.
"""

from pathlib import Path
from typing import Literal

from .pipeline import DIRECTION_MRI, process


def run(
    output_dir: Path,
    # Spelled out because `Literal` takes literals only -- it cannot be built
    # from pipeline.STEPS. That makes this a second declaration of the same
    # set, which is the thing this contract otherwise avoids, so a test asserts
    # the two agree.
    step: Literal[
        "Orient MRI", "Resample", "Approximate", "LR crop", "TMJ crop", "Register"
    ] = "Register",
    mri: Path = "",
    cbct: Path = "",
    segmentation: Path = "",
    mri_t2: Path = "",
    cbct_t2: Path = "",
    segmentation_t2: Path = "",
    condyle_model: Path = "",
    direction: str = DIRECTION_MRI,
    acquisition_z_spacing: float = 0.0,
    resample_size: list[int] = [],
    spacing: list[float] = [],
    center: bool = True,
    mri_min_norm: int = 0,
    mri_max_norm: int = 100,
    mri_lower_percentile: int = 10,
    mri_upper_percentile: int = 95,
    cbct_min_norm: int = 0,
    cbct_max_norm: int = 100,
    cbct_lower_percentile: int = 10,
    cbct_upper_percentile: int = 95,
    keep_temporary: bool = False,
) -> Path:
    """Register a TMJ MRI onto its CBCT, or run one of the steps leading to it.

    Args:
        output_dir: Where this step's results and `MRI2CBCT_report.json` are
            written. Nothing is written outside it.
        step: Which operation to run. The pipeline is Orient MRI, Resample,
            Approximate, a crop, then Register, and it is one call per step on
            purpose: a badly oriented MRI is worth catching before an hour of
            registration, which is why the module puts them on separate tabs.
        mri: The MRI scans, as a folder (a single file is taken with its
            folder). Read by every step.
        cbct: The CBCT scans, paired to the MRI by patient key.
        segmentation: The CBCT segmentations. Register uses them as the mask it
            normalises and registers through; TMJ crop needs them to find the
            joint; LR crop splits them like a CBCT, being in CBCT space.
        mri_t2: A second MRI timepoint, resampled alongside the first. Resample
            only.
        cbct_t2: A second CBCT timepoint. Resample only.
        segmentation_t2: The second timepoint's segmentations. Resample only.
        condyle_model: The nnUNet condyle segmentation model folder, used by
            Approximate and TMJ crop to locate the joint. Named rather than
            resolved here: a tool does not go looking for weights on the
            server's disk.
        direction: The MRI's new direction, nine comma-separated numbers read
            as a 3x3 matrix row by row. Orient MRI only. The default is the
            orientation upstream documents for MRI; a CBCT is
            "1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0".
        acquisition_z_spacing: The slice spacing to write into the MRI header,
            in mm. 0 leaves the acquisition's own spacing alone. Orient MRI
            only.
        resample_size: Target size in voxels, as three numbers. Empty keeps
            each scan's own size. Resample only.
        spacing: Target spacing in mm, as three numbers. Empty keeps each
            scan's own spacing. Resample only.
        center: Centre each resampled volume on its own image centre. Resample
            only.
        mri_min_norm: Lower bound of the MRI intensity range after
            normalisation. Register only.
        mri_max_norm: Upper bound of the MRI intensity range. Register only.
        mri_lower_percentile: Intensity percentile mapped to the lower bound;
            everything below it is clipped. Register only.
        mri_upper_percentile: Intensity percentile mapped to the upper bound.
            Register only.
        cbct_min_norm: Lower bound of the CBCT intensity range. Register only.
        cbct_max_norm: Upper bound of the CBCT intensity range. Register only.
        cbct_lower_percentile: The CBCT's lower percentile. Register only.
        cbct_upper_percentile: The CBCT's upper percentile. Register only.
        keep_temporary: Keep the inverted, normalised and masked volumes the
            registration built on its way. They are what tells you WHICH stage
            went wrong when a registration comes out badly. Register only.

    Returns:
        The output directory, holding this step's results and the run report.
    """
    return process(
        step=step,
        output_dir=output_dir,
        given={
            "mri": mri,
            "cbct": cbct,
            "segmentation": segmentation,
            "mri_t2": mri_t2,
            "cbct_t2": cbct_t2,
            "segmentation_t2": segmentation_t2,
            "condyle_model": condyle_model,
        },
        direction=direction,
        acquisition_z_spacing=acquisition_z_spacing,
        resample_size=resample_size,
        spacing=spacing,
        center=center,
        # The order is `extract_values`': min, max, lower percentile, upper
        # percentile, MRI first. Changing it silently swaps a percentile for a
        # bound, so a test pins it.
        normalisation=(
            mri_min_norm, mri_max_norm, mri_lower_percentile, mri_upper_percentile,
            cbct_min_norm, cbct_max_norm, cbct_lower_percentile, cbct_upper_percentile,
        ),
        keep_temporary=keep_temporary,
    )
