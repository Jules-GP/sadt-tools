"""The failures a caller can do something about.

Anything else raised by this tool is a bug and should surface as one. These
replace the server's `base.ToolArgumentError` / `base.ToolUnavailableError`,
which a tool package cannot import: nothing here knows the server exists.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class ToolUnavailableError(RuntimeError):
    """The segmentation stack is not installed in this venv.

    Unlike every other tool here, CrownSeg's engine is not covered by a plain
    `uv sync`: shapeaxi pulls pytorch3d, which publishes no usable wheel and is
    compiled from source. It is an optional extra, so this is a real state a
    working deployment can be in, and it has to say what to do about it.
    """
