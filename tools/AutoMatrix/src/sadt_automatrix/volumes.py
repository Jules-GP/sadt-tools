"""Resampling one volume through one transform.

A port of `Automatrix_CLI.apply_transform_to_image` and the `ResampleImage`
beside it, including the two decisions that look like details and are not:

* the interpolator is nearest-neighbour for a segmentation and linear for a
  scan, because interpolating label values linearly invents labels that were
  never in the image;
* the grid the result lands on is the REFERENCE image's, not the moving
  image's, so that a cohort resampled against one reference comes out
  voxel-aligned and can be compared.
"""

from pathlib import Path

from . import transforms

# The three grids a case can end up on, named so the report can say which.
GRID_CHOSEN = "chosen"
GRID_COMPOSITE_NEIGHBOUR = "composite_neighbour"
GRID_COMPOSITE_FALLBACK = "composite_fallback"


def composite_reference(transform, matrix):
    """`(grid, image)` for a composite transform; `(GRID_CHOSEN, None)` if not.

    A composite comes out of AREG, which writes `P1_transform.tfm` beside the
    fixed `P1.nii.gz` it was computed against, and the chain inside it is only
    meaningful on that grid -- so the neighbour overrides whatever reference
    the caller chose. Upstream falls back to the moving image when the
    neighbour is missing, with a warning, and that is kept: a wrong grid still
    produces a volume somebody can look at, where refusing produces nothing.
    """
    if not transforms.is_composite(transform):
        return GRID_CHOSEN, None

    import SimpleITK as sitk

    text = str(matrix)
    for extension in (".nii.gz", ".nii"):
        neighbour = Path(text.replace("_transform.tfm", extension))
        if neighbour.exists():
            return GRID_COMPOSITE_NEIGHBOUR, sitk.ReadImage(str(neighbour))
    return GRID_COMPOSITE_FALLBACK, None


def apply(image, transform, reference, output, is_segmentation):
    """Resample `image` onto `reference`'s grid and write it to `output`."""
    import SimpleITK as sitk

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_segmentation else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetReferenceImage(reference)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampler.Execute(image), str(output))
    return output
