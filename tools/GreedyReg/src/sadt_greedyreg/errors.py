"""The failures a caller can do something about.

Anything else raised by this tool is a bug and should surface as one. These
replace the server's `base.ToolArgumentError` / `base.ToolUnavailableError`,
which a tool package cannot import: nothing here knows the server exists.
"""


class ToolInputError(ValueError):
    """An argument the tool cannot work with, phrased for whoever sent it."""


class SupervisorRequired(RuntimeError):
    """A mode needing another tool, with no way to reach one.

    Not a bad request: nothing about the caller's arguments is wrong, there is
    simply no supervisor. The message says which mode to use instead, because
    that is a real answer and "deploy a tool" usually is not.
    """


class RegistrationFailed(RuntimeError):
    """Greedy refused a case. Carries what it said, per patient."""
