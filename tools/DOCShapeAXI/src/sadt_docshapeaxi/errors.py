"""The failures a caller can do something about.

Anything else raised by this tool is a bug and should surface as one. These
replace the server's `base.ToolArgumentError`, which a tool package cannot
import: nothing here knows the server exists.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class CheckpointNotFound(FileNotFoundError):
    """No checkpoint in the bundle for the requested data type and task."""
