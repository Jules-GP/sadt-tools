"""Reading one matrix file, and saying no to the one format that cannot be read.

Every matrix AutoMatrix applies is an ITK transform, read by
`SimpleITK.ReadTransform`, which covers `.tfm`, `.h5`, `.mat` and `.txt`. The
transform's own convention is ITK's: it maps a point of the OUTPUT space back
into the INPUT space, which is what makes it the right thing to hand a
resampler directly and the wrong thing to hand a point -- see landmarks.py.
"""

from pathlib import Path

from .errors import ToolInputError

# What `sitk.ReadTransform` can open. Kept as a list rather than left to the
# reader's exception, because the message the reader raises for an unknown
# format names an ITK factory and helps nobody.
READABLE_EXTENSIONS = (".tfm", ".h5", ".mat", ".txt")

# Collected by upstream's matrix search and readable by nothing in it. The
# in-Slicer AutoMatrix that predates the CLI did read `.npy`, as a raw 4x4 --
# but as a RAS transform-TO-parent applied without inversion, where every other
# format here is an LPS ITK transform applied inverted. The two conventions
# disagree by a coordinate flip and by a direction, so a `.npy` sitting in a
# folder of `.tfm` would move a scan somewhere else entirely while looking like
# it had worked. Refused by name instead of guessing which of the two a given
# file means.
UNREADABLE_EXTENSIONS = (".npy",)


def read(matrix):
    """The ITK transform stored in `matrix`.

    Raises ToolInputError for a format nothing here can read. The caller
    records that against the file and carries on with the rest of the batch:
    one bad matrix must not cost the other 199.
    """
    matrix = Path(matrix)
    if matrix.suffix in UNREADABLE_EXTENSIONS:
        raise ToolInputError(
            "{}: a .npy matrix cannot be read as an ITK transform, and the two "
            "conventions AutoMatrix has used for one (an LPS resampling "
            "transform, and a RAS transform-to-parent) would move a scan to "
            "different places. Convert it to .tfm, stating which it is.".format(
                matrix.name)
        )

    import SimpleITK as sitk

    return sitk.ReadTransform(str(matrix))


def is_composite(transform):
    """Whether this is a chain of transforms rather than a single one.

    A composite comes out of AREG, which writes the deformation it computed
    next to the fixed image it computed it against. That neighbouring image is
    the only grid the chain is valid on, which is why volumes.py goes looking
    for it instead of using the reference the caller passed.
    """
    import SimpleITK as sitk

    return isinstance(transform, sitk.CompositeTransform)
