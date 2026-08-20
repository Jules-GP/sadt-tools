"""Matching a baseline scan to its follow-up, by patient key.

The rule is upstream's, character for character: the leading letters-then-digits
of the file name, upper-cased. `C_0001_T1.nii.gz` does NOT match it -- the
pattern wants letters immediately followed by digits -- and that is not an
oversight to fix here. Changing the rule would silently re-pair every folder
that works today, so it stays as it is and the README says what it accepts.
"""

import re
from pathlib import Path

from .errors import ToolInputError

# Upstream: GreedyReg_CLI.ID_PATTERN, and again in Logic.findBatchPairs.
ID_PATTERN = re.compile(r"^([A-Za-z]+\d+)", re.IGNORECASE)

VOLUME_SUFFIXES = (".nii.gz", ".nii")
MATRIX_SUFFIX = ".mat"


def patient_key(name: str) -> str | None:
    """The patient key a file name carries, or None when it carries none."""
    found = ID_PATTERN.match(name)
    return found.group(1).upper() if found else None


def _indexed(folder: Path, suffixes) -> dict:
    """Every file under `folder` whose name yields a patient key.

    Not recursive, because upstream's `os.listdir` is not, and a recursive walk
    would pull a previous run's output back in as an input.
    """
    found = {}
    if not folder or not folder.is_dir():
        return found
    for entry in sorted(folder.iterdir()):
        if not entry.is_file() or not entry.name.endswith(suffixes):
            continue
        key = patient_key(entry.name)
        if key:
            found[key] = entry
    return found


def _one(path: Path, suffixes, role: str) -> Path:
    if not path.exists():
        raise ToolInputError("{} path does not exist: {}".format(role, path))
    if path.is_file() and not path.name.endswith(suffixes):
        raise ToolInputError(
            "{} is not a {} file: {}".format(role, " or ".join(suffixes), path)
        )
    return path


def _for_one(path: Path, key: str, suffixes):
    """The mask or init for a single pair: the file itself, or its key's file."""
    if path is None:
        return None
    if path.is_file():
        return path
    return _indexed(path, suffixes).get(key)


def find_pairs(t1: Path, t2: Path, masks: Path = None, inits: Path = None) -> list:
    """Every (key, fixed, moving, mask, init) case to process, in a stable order.

    Two shapes are accepted, and they are the module's two shapes:

    * two FOLDERS, matched by patient key, which is the batch the CLI runs;
    * two FILES, taken as the one pair, which is what the panel's fixed/moving
      volume selectors are. Upstream can only express that through the scene,
      so there is no file-name rule to be faithful to: the pair is the pair,
      and the key is T1's if it has one and its stem otherwise.
    """
    t1 = _one(Path(t1), VOLUME_SUFFIXES, "t1")
    t2 = _one(Path(t2), VOLUME_SUFFIXES, "t2")

    if t1.is_file() != t2.is_file():
        raise ToolInputError(
            "t1 and t2 must both be folders (a batch) or both be files (one "
            "pair). Got {} and {}.".format(
                "a file" if t1.is_file() else "a folder",
                "a file" if t2.is_file() else "a folder",
            )
        )

    masks = Path(masks) if masks else None
    inits = Path(inits) if inits else None

    if t1.is_file():
        key = patient_key(t1.name) or t1.name.split(".")[0]
        # One pair takes one mask and one init as FILES, not as folders holding
        # a single entry. The client gives every path argument a picker that
        # takes either, so a user registering one pair picks the mask itself --
        # and a folder rule would ignore it in silence.
        return [(
            key, t1, t2,
            _for_one(masks, key, VOLUME_SUFFIXES),
            _for_one(inits, key, (MATRIX_SUFFIX,)),
        )]

    masks_by_key = _indexed(masks, VOLUME_SUFFIXES) if masks else {}
    inits_by_key = _indexed(inits, (MATRIX_SUFFIX,)) if inits else {}

    fixed = _indexed(t1, VOLUME_SUFFIXES)
    moving = _indexed(t2, VOLUME_SUFFIXES)
    shared = sorted(set(fixed) & set(moving))
    if not shared:
        raise ToolInputError(
            "No matching T1/T2 pair between {} and {}. A pair is matched on the "
            "leading letters-and-digits of the file name (e.g. 'MG01_T1.nii.gz' "
            "and 'MG01_T2.nii.gz'); T1 offers {} and T2 offers {}.".format(
                t1, t2,
                sorted(fixed) or "no keyed file",
                sorted(moving) or "no keyed file",
            )
        )
    return [
        (key, fixed[key], moving[key], masks_by_key.get(key), inits_by_key.get(key))
        for key in shared
    ]
