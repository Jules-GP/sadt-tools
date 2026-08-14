"""The two failures a caller can do something about.

Anything else raised by this tool is a bug and should surface as one. These
replace the server's `base.ToolArgumentError` / `base.ToolUnavailableError`,
which a tool package cannot import: nothing here knows the server exists.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class ModelNotFoundError(FileNotFoundError):
    """No usable nnUNet model folder for a requested structure."""
