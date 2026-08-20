"""The failures a caller can do something about.

Anything else raised by this tool is a bug and should surface as one. These
replace the server's `base.ToolArgumentError` / `base.ToolUnavailableError`,
which a tool package cannot import: nothing here knows the server exists.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class NothingWritten(RuntimeError):
    """Files went in, no transformed file came out, and nobody was told why.

    Separate from ToolInputError because the arguments were fine: a matrix was
    unreadable, every scan was a surface, or no patient key matched a matrix.
    The message carries the per-file reasons, which are the only thing that
    tells a user which of those it was.
    """
