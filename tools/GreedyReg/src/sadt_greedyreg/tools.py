"""The one tool GreedyReg asks for, and what to do when it cannot be reached.

The call goes through the **supervisor**, the object whatever runs GreedyReg
hands it as the keyword-only `sup`. Nothing here imports ALI: the two have
different interpreters and dependency sets that cannot be reconciled, which is
precisely why upstream's in-process version had to install monai, pydicom 2.2.2
and dicom2nifti 2.3.0 into Slicer and break its DICOM modules doing it.

Tools are named by string, never by attribute: `sup.run("ALI_CBCT", ...)`. A
typo in a string is greppable, and `describe.py` publishes the names it finds
here so the SERVER refuses to start on one that names no registered tool --
rather than failing an hour into a batch.
"""

import logging
import os

from .errors import SupervisorRequired

logger = logging.getLogger(__name__)

LANDMARK_TOOL = "ALI_CBCT"


def require(sup, tool: str, mode: str) -> None:
    """Refuse a mode needing `tool` when there is no way to run it.

    Checked before a single file is read: a request that cannot work has to come
    back in a second, not after a folder of scans has been paired and copied.

    The tool is named by a literal at every call site, here and in pipeline.py,
    because `describe.py` reads those call sites to publish what this tool asks
    for -- a name behind a variable is a name the server cannot check at startup.
    A test asserts the two spellings agree.
    """
    if sup is not None:
        return
    raise SupervisorRequired(
        "{} mode needs the '{}' tool, and nothing here can run it: no supervisor "
        "was supplied. Use 'Greedy' mode and align the two timepoints yourself, "
        "or send the alignment in 'init'.".format(mode, tool)
    )


def _output(sup, name: str) -> str:
    """A directory of the supervisor's scratch for one callee's results."""
    destination = os.path.join(str(sup.tmp), "tools", name)
    os.makedirs(destination, exist_ok=True)
    return destination


def _returned(produced) -> str:
    """A tool returns a Path, or a dict of named ones; this wants a directory."""
    if isinstance(produced, dict):
        produced = next(iter(produced.values()))
    return str(produced)


def place_landmarks(sup, scan, model, names, device: str, tag: str) -> str:
    """Ask ALI_CBCT for exactly `names` on one scan; return where it wrote them.

    `landmarks` REPLACES ALI's region selection rather than narrowing it, which
    is what lets this ask for the five points a region needs instead of running
    fifty-eight agents to use five of them. The arguments are ALI_CBCT's
    published schema, not GreedyReg's vocabulary: when it renames one, this file
    is what breaks, in one place, with the name in it.
    """
    logger.info(
        "GreedyReg: asking '%s' for %s on the %s scan", LANDMARK_TOOL,
        ", ".join(names), tag,
    )
    return _returned(sup.run(
        LANDMARK_TOOL,
        input=str(scan),
        model=str(model),
        output_dir=os.path.join(_output(sup, LANDMARK_TOOL), tag),
        landmarks=list(names),
        prediction_ID="GreedyReg",
        device=device,
    ))
