"""Which files are scans, which are matrices, and which patient each belongs to.

This is a port of `Automatrix_CLI.GetPatients` and the `search` helper beside
it. The pairing rule is the whole of AutoMatrix's batch behaviour, so it is
reproduced marker for marker rather than rewritten: a folder that pairs today
must pair the same way after the port, and any "tidier" key would silently
re-pair every cohort in the lab.

Two things are deliberately different, and neither changes which file pairs
with which:

* The order is sorted. Upstream walks `glob.iglob`, whose order is whatever the
  filesystem hands back, so two runs on the same folder could process the same
  cohort in different orders and write their report rows in different orders.
* The extension of a file is the longest DECLARED extension it ends with,
  where upstream used `''.join(Path(p).suffixes)`. The two agree on every
  ordinary name and disagree on `P.1_scan.nii.gz`, where `suffixes` returns
  `.1_scan.nii.gz` and the output file would be named from the wrong stem.
"""

import os
from pathlib import Path

# Everything the module's own input check calls a scan, in upstream's order.
LANDMARK_EXTENSION = ".mrk.json"
VOLUME_EXTENSIONS = (".nii", ".nii.gz", ".nrrd")
# Collected, then refused by name. See pipeline.py for why they are not simply
# left out of the search: a user who pointed AutoMatrix at a folder of meshes
# has to be told that nothing came out and why, and an extension nobody
# collects produces "no scan found", which is a different and wrong answer.
SURFACE_EXTENSIONS = (".vtk", ".vtp", ".stl", ".off", ".obj")
SCAN_EXTENSIONS = SURFACE_EXTENSIONS + VOLUME_EXTENSIONS + (LANDMARK_EXTENSION,)

# What upstream's matrix search collects. `.npy` is in the list and is NOT
# readable by `sitk.ReadTransform`; it is refused by name for the reason given
# in transforms.py.
MATRIX_EXTENSIONS = (".npy", ".h5", ".tfm", ".mat", ".txt")

# The markers a patient key is cut at, applied in this order, each to the
# result of the last. Upstream spells this as one 16-call chain of `.split()`
# per file; the order is load-bearing (`_SegOr` before `_Or`, `_MAND` before
# `_MD`) so it is kept exactly, only written as data.
SCAN_KEY_MARKERS = (
    "_Seg", "_seg", "_Scan", "_scan", "_Or", "_OR", "_MAND", "_MD",
    "_MAX", "_MX", "_CB", "_lm", "_T2", "_T1", "_Cl", "_MR",
)

# The matrix chain is NOT the scan chain: it cuts at the side (`_Left`), at the
# mirror markers, and at `_SegOr`, and it does not cut at `_Seg` or `_Scan`.
# That asymmetry is upstream's and it is what makes `P1_MAND_Left_mirror.tfm`
# reach patient `P1`.
MATRIX_KEY_MARKERS = (
    "_SegOr", "_Left", "_left", "_Right", "_right", "_Or", "_OR", "_MAND",
    "_MD", "_MAX", "_MX", "_CB", "_lm", "_T2", "_T1", "_Cl", "_MA", "_Mir",
    "_mir", "_Mirror", "_mirror", "_MR",
)

# Upstream then strips a timepoint suffix for fifty timepoints. `_T1` cuts
# `_T10` as well, because it is a prefix of it -- that is upstream's behaviour
# and the reason the loop is reproduced rather than replaced by a regex.
TIMEPOINTS = 50


def extension_of(name, extensions):
    """The longest of `extensions` that `name` ends with, or `''`.

    Case-sensitive, like upstream's `endswith`: a file named `SCAN.NII.GZ` is
    not a scan here and was not one before the port either.
    """
    found = ""
    for extension in extensions:
        if name.endswith(extension) and len(extension) > len(found):
            found = extension
    return found


def _cut(name, markers):
    for marker in markers:
        name = name.split(marker)[0]
    name = name.split(".")[0]
    for timepoint in range(TIMEPOINTS):
        name = name.split("_T{}".format(timepoint))[0]
    return name


def scan_key(path):
    """The patient a scan belongs to."""
    return _cut(os.path.basename(str(path)), SCAN_KEY_MARKERS)


def matrix_key(path):
    """The patient a matrix belongs to."""
    return _cut(os.path.basename(str(path)), MATRIX_KEY_MARKERS)


def _collect(root, extensions):
    """Every file under `root` ending in one of `extensions`, grouped in order.

    Grouped by extension in the order given, sorted inside each group. That is
    upstream's grouping -- it appends all the `.vtk` it found, then all the
    `.vtp`, and so on -- with the filesystem's arbitrary order replaced by a
    stable one.
    """
    everything = sorted(path for path in root.rglob("*") if path.is_file())
    found = []
    seen = set()
    for extension in extensions:
        for path in everything:
            if path not in seen and path.name.endswith(extension):
                seen.add(path)
                found.append(path)
    return found


def find_scans(scans):
    """Every scan under `scans`, or the single file it names.

    A file that is not one of the declared extensions is not a scan, and a
    `scans` argument naming one is an error rather than an empty batch: it is
    always a mistake, and an empty batch reads as "nothing to do".
    """
    scans = Path(scans)
    if scans.is_dir():
        return _collect(scans, SCAN_EXTENSIONS)
    if scans.is_file():
        if not extension_of(scans.name, SCAN_EXTENSIONS):
            return []
        return [scans]
    return None


def find_matrices(matrices):
    """Every matrix under `matrices`, or the single file it names.

    A single file is handed to every patient, which is how one mirroring matrix
    is applied to a whole cohort. That is upstream's `else` branch and it does
    NOT check the extension -- a named file is taken at its word.
    """
    matrices = Path(matrices)
    if matrices.is_dir():
        return _collect(matrices, MATRIX_EXTENSIONS), True
    if matrices.is_file():
        return [matrices], False
    return None, False


def group_by_patient(scans, matrices, matrices_is_dir):
    """`[(key, [scan, ...], [matrix, ...]), ...]`, in the order the scans came.

    A patient exists because a SCAN was found for it; a matrix whose key
    matches no scan is dropped, exactly as upstream drops it. When `matrices`
    named one file, every patient gets that file.
    """
    patients = {}
    for scan in scans:
        patients.setdefault(scan_key(scan), []).append(scan)

    per_patient = {key: [] for key in patients}
    if matrices_is_dir:
        for matrix in matrices:
            key = matrix_key(matrix)
            if key in per_patient:
                per_patient[key].append(matrix)
    else:
        for key in per_patient:
            per_patient[key] = list(matrices)

    return [(key, patients[key], per_patient[key]) for key in patients]
