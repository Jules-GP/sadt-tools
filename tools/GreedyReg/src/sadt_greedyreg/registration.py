"""Greedy affine registration of one moving volume onto one fixed volume.

Every argument Greedy is given is upstream's, in upstream's order:

    -d 3 -a -m <metric> -i <fixed> <moving> -o <warp>
    -n 100x100x50x25 -e 0.5 -search 100 10 20 -dof <6|12> -ia <init> [-gm <mask>]

then a second pass to resample the moving volume through the transform it just
solved. What changed is only WHO runs it: upstream shells out to an ITK-SNAP
binary it downloaded, this calls the same code through the picsl-greedy wheel.
`Greedy3D.execute` takes the identical command line, so the schedule, the
metric, the search and the degrees of freedom are the ones that produced every
result to date.
"""

import logging
from pathlib import Path

from .errors import RegistrationFailed

logger = logging.getLogger(__name__)

METRICS = {
    "NMI": ["-m", "NMI"],
    # Upstream's window, not a default: NCC without one is not a valid metric
    # for Greedy.
    "NCC": ["-m", "NCC", "4x4x4"],
    "SSD": ["-m", "SSD"],
}

# Upstream's index-to-name tables, kept so the published options and the CLI
# strings cannot drift: ["NMI", "NCC", "SSD"][metricIndex] and
# "Rigid" if dofIndex == 0 else "Affine".
DEGREES_OF_FREEDOM = {"Rigid": "6", "Affine": "12"}


def write_identity_init(path: Path) -> Path:
    """A Greedy-format init matrix holding (almost) identity.

    The 0.001 mm nudge is upstream's and is load-bearing: Greedy treats an exact
    identity initialisation as "no initialisation" and ignores `-ia`.
    """
    import numpy as np

    matrix = np.eye(4)
    matrix[0, 3] = 0.001
    write_matrix(matrix, path)
    return path


def write_matrix(matrix, path: Path) -> Path:
    """Write a 4x4 RAS matrix in the plain-text form Greedy's `-ia` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(" ".join(str(float(value)) for value in row) + "\n")
    return path


def binarize_mask(source: Path, destination: Path) -> Path:
    """Greedy's `-gm` wants a 0/1 float mask; a segmentation is rarely that."""
    import nibabel as nib
    import numpy as np

    mask = nib.load(str(source))
    data = (mask.get_fdata() > 0).astype(np.float32)
    binary = nib.Nifti1Image(data, mask.affine)
    binary.header.set_data_dtype(np.float32)
    nib.save(binary, str(destination))
    return destination


def affine_command(fixed: Path, moving: Path, warp: Path, init: Path,
                   metric: str, transform_type: str, mask: Path = None) -> list:
    command = ["-d", "3", "-a"]
    command += METRICS[metric]
    command += ["-i", str(fixed), str(moving)]
    command += ["-o", str(warp)]
    command += ["-n", "100x100x50x25"]
    command += ["-e", "0.5"]
    command += ["-search", "100", "10", "20"]
    command += ["-dof", DEGREES_OF_FREEDOM[transform_type]]
    command += ["-ia", str(init)]
    if mask:
        command += ["-gm", str(mask)]
    return command


def resample_command(fixed: Path, moving: Path, output: Path, warp: Path) -> list:
    return [
        "-d", "3",
        "-rf", str(fixed),
        "-rm", str(moving), str(output),
        "-r", str(warp),
    ]


def register(fixed: Path, moving: Path, output: Path, warp: Path, init: Path,
             metric: str, transform_type: str, mask: Path = None) -> Path:
    """Solve the affine, then resample `moving` onto `fixed`'s grid through it.

    Returns the resampled volume's path. Raises RegistrationFailed carrying what
    Greedy said -- which is the only diagnosis a caller gets, since a failed
    registration looks exactly like a converged one from the outside.
    """
    from picsl_greedy import Greedy3D

    greedy = Greedy3D()
    for stage, command in (
        ("affine registration", affine_command(
            fixed, moving, warp, init, metric, transform_type, mask)),
        ("resampling", resample_command(fixed, moving, output, warp)),
    ):
        logger.info("greedy %s: %s", stage, " ".join(command))
        try:
            greedy.execute(" ".join(command))
        except RuntimeError as failure:
            # picsl-greedy raises RuntimeError where the binary exited non-zero
            # and upstream read its stderr. Same information, same stage names.
            raise RegistrationFailed(
                "Greedy {} failed: {}".format(stage, failure)
            ) from failure

    if not output.exists():
        raise RegistrationFailed(
            "Greedy reported success but wrote no resampled volume at {}.".format(output)
        )
    return output
