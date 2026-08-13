"""The failures a caller can do something about.

Anything else raised by this tool is a bug and should surface as one. These
replace the server's `base.ToolArgumentError`, which a tool package cannot
import: nothing here knows the server exists.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class SupervisorRequired(ToolInputError):
    """Fully-automated mode was asked for with no way to reach the landmark tool.

    Its own class because the fix is unlike any other input error: nothing about
    the request is wrong. Either the caller supplies `landmarks`, or whatever is
    running this tool has to inject a supervisor -- and a plain `uv run` never
    will, which is exactly the standalone case the README documents.
    """
