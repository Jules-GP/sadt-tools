"""What counts as a CBCT volume, what counts as an intraoral surface.

The second of the two things ALI_CBCT and ALI_IOS must not disagree about (the
first is `markups.py`). Both are contracts with the world outside this
repository rather than internal code, which is why they are shared where the
engines themselves are duplicated -- see this package's README.

Concretely: if the two tools drifted on this table, a `.stl` would be a valid
input to one and an unrecognised file to the other, and a caller sending a
mixed folder would be told different things depending on which tool they
asked. The pre-port CLIs had exactly that bug in the other direction -- the UI
counted `.stl` files and the CLI globbed for `.vtk` only, so they were accepted
and then silently ignored.

Deliberately stdlib-only. DICOM detection is NOT here: probing a directory for
a series needs itk, which only ALI_CBCT has, and a DICOM series is a volume by
definition so it is ALI_CBCT's business alone.
"""

import os

VOLUME_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")
SURFACE_EXTENSIONS = (".vtk", ".stl")

CBCT = "CBCT"
IOS = "IOS"

# Intermediates live here, under the output directory the caller owns, and are
# removed before the run returns. A surviving `.ali_work/` means a run crashed.
WORK_DIRNAME = ".ali_work"


def scan_key(path: str, root: str) -> str:
    """A scan's identity: its path relative to the input root.

    Keying by BASE NAME, as the original did, meant two patients called
    `scan.nii.gz` in different subfolders silently overwrote each other twice
    over: once in the working dictionary, once in the flat output folder.
    """
    if not root:
        return os.path.basename(path)
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return os.path.basename(path)


def classify(root: str) -> tuple:
    """Every volume and every surface under `root`, as two sorted lists.

    Returns `(volumes, surfaces)` of absolute paths. It reports what is there
    and decides nothing: refusing the wrong kind, and what "the wrong kind"
    means, is each tool's own business now that they are separate.
    """
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
    return sorted(volumes), sorted(surfaces)


def keyed(paths: list, root: str) -> list:
    """`[(path, key)]`, the pairing every engine consumes."""
    return [(path, scan_key(path, root)) for path in paths]
