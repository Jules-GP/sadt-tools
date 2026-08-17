"""The failures a caller can do something about.

Anything else raised by this tool is a bug and should surface as one. These
replace the server's `base.ToolArgumentError` / `base.ToolUnavailableError`,
which a tool package cannot import: nothing here knows the server exists.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class ToolUnavailableError(RuntimeError):
    """An engine's stack is not installed in this venv.

    Real for ALI in a way it is not for most tools: the IOS engine needs
    pytorch3d, which publishes no usable wheel and is compiled from source, so
    a venv where `uv sync` succeeded can still be one where an IOS run cannot
    start. The CBCT engine is unaffected and must keep working, which is why
    this is raised at the point of use rather than at import.
    """
