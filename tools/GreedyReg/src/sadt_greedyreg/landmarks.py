"""Landmark-driven rigid alignment: what upstream calls Distant Registration.

Two scans far apart in space defeat Greedy's search window, so a rigid transform
is derived from anatomical landmarks first and used to bring them together. The
landmarks come from ALI_CBCT; everything in this file is what upstream does with
them once it has them.
"""

import json
import logging
from pathlib import Path

from .errors import ToolInputError

logger = logging.getLogger(__name__)

# Upstream: Logic.REGION_CONFIG. The model_dirs it also carries are gone, and
# deliberately: they named subfolders of the ALI release that the in-process CLI
# had to be pointed at one at a time. The packaged ALI_CBCT takes the landmark
# NAMES and resolves its own weights, so a region here is exactly its points.
REGION_LANDMARKS = {
    "MANDMASK": ["RGo", "LGo", "Gn", "Me", "Pog"],
    "MAXMASK": ["A", "ANS", "LOr", "ROr", "PNS"],
    "CBMASK": ["S", "N", "RPo", "LPo"],
}

# Below this a rigid fit is not determined: three points fix a rigid body, fewer
# leave an axis free. Upstream's check, and its failure message.
MINIMUM_LANDMARKS = 3


def read_markups(path: Path) -> dict:
    """Landmark name -> [x, y, z] in RAS, from one Slicer markups file.

    ALI writes LPS and says so in the file. Upstream flips X and Y with the
    conversion hard-coded; here the declared system is READ, because a file
    already in RAS would otherwise be mirrored through two axes in silence.
    """
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)

    found = {}
    for markup in document.get("markups", []):
        system = str(markup.get("coordinateSystem", "LPS")).upper()
        if system not in ("LPS", "RAS"):
            raise ToolInputError(
                "{} declares coordinate system '{}'; expected LPS or RAS.".format(
                    path, system
                )
            )
        for point in markup.get("controlPoints", []):
            label = point.get("label", "")
            position = point.get("position", [0.0, 0.0, 0.0])
            if not label:
                continue
            x, y, z = (float(value) for value in position[:3])
            found[label] = [-x, -y, z] if system == "LPS" else [x, y, z]
    return found


def collect(output_dir: Path, wanted) -> dict:
    """Every wanted landmark ALI left under `output_dir`, from any file there."""
    found = {}
    for markup_file in sorted(Path(output_dir).rglob("*.json")):
        for name, position in read_markups(markup_file).items():
            if name in wanted:
                found[name] = position
    return found


def rigid_from_landmarks(fixed_points, moving_points):
    """The rigid 4x4 RAS transform taking `moving_points` onto `fixed_points`.

    Kabsch/Umeyama by SVD, upstream's implementation including the reflection
    guard: without it a degenerate configuration yields a mirror, which is a
    valid orthogonal matrix and an invalid anatomy.
    """
    import numpy as np

    fixed_centre = fixed_points.mean(axis=0)
    moving_centre = moving_points.mean(axis=0)
    covariance = (moving_points - moving_centre).T @ (fixed_points - fixed_centre)
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = fixed_centre - rotation @ moving_centre
    return transform


def matched(fixed: dict, moving: dict, wanted) -> list:
    """The landmarks BOTH scans got, in the region's own order."""
    return [name for name in wanted if name in fixed and name in moving]


def bake_into_affine(moving_path: Path, transform, output_path: Path) -> Path:
    """Apply a RAS rigid transform by rewriting the image's affine.

    Upstream's choice, and worth keeping: the voxels are untouched, so nothing
    is interpolated and nothing is lost. The image simply says where it is.
    nibabel affines are RAS+, which is the space the transform is already in.
    """
    import nibabel as nib
    import numpy as np

    moving = nib.load(str(moving_path))
    rotation = np.asarray(transform)[:3, :3]
    translation = np.asarray(transform)[:3, 3]

    affine = moving.affine.copy()
    affine[:3, :3] = rotation @ moving.affine[:3, :3]
    affine[:3, 3] = rotation @ moving.affine[:3, 3] + translation

    aligned = nib.Nifti1Image(moving.get_fdata(), affine, moving.header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(aligned, str(output_path))
    return output_path
