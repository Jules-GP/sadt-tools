"""Moving the control points of a Slicer markups file through one transform.

A port of `Automatrix_CLI.apply_transform_to_landmarks`. The one thing worth
stating out loud is the INVERSE: an ITK transform maps a point of the output
space back into the input space, which is what a resampler wants and the
opposite of what moving a point wants. A volume therefore takes the transform
and a landmark takes `GetInverse()`, and the two then agree about where the
patient went.

Everything else in the file is left where it was: the label, the colour, the
display node, the `positionStatus` of every point. A markups file is a scene
object as much as it is coordinates, and rewriting only the positions is what
keeps a transformed landmark set openable next to the original.
"""

import json
from pathlib import Path

DEFINED = "defined"


def apply(landmarks, transform, output):
    """Write `landmarks` again with every defined point moved by `transform`.

    Only the first markups group is touched, which is upstream's behaviour: a
    `.mrk.json` written by Slicer's landmark modules holds exactly one, and a
    hand-made file with two would have the second copied through unmoved.

    Returns the number of points moved, so the caller can record a file that
    parsed, was written, and contained nothing placed.
    """
    landmarks = Path(landmarks)
    with open(landmarks, encoding="utf-8") as handle:
        data = json.load(handle)

    groups = data.get("markups")
    if not groups:
        raise ValueError("{}: no markups in the file.".format(landmarks.name))

    inverted = transform.GetInverse()

    moved = 0
    for point in groups[0].get("controlPoints") or []:
        position = point.get("position")
        if point.get("positionStatus") != DEFINED:
            continue
        if not isinstance(position, list) or len(position) != 3:
            continue
        point["position"] = list(inverted.TransformPoint(position))
        moved += 1

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return moved
