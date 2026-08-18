"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts — `describe.py` merges these hints
into the published schema and refuses any that name an argument or an option
the signature does not offer. Delete this file and the tool still works; the
panel just gets worse.

Short, because this tool's schema is short. Before the split it shared a panel
with 119 CBCT landmark options and needed `visible_when` on every field to stop
an intraoral user scrolling past them; now the only arguments published are the
ones an intraoral run reads.
"""

_INPUTS = "Inputs"
_LANDMARKS = "Landmarks"
_OUTPUTS = "Outputs"

LAYOUT = {
    "input": {"section": _INPUTS, "label": "Surface or Folder"},
    "model": {"section": _INPUTS, "label": "Model Bundle"},
    "networks": {"section": _LANDMARKS, "label": "Landmark families", "ui": "inline"},
    "prediction_ID": {"section": _OUTPUTS, "label": "Prediction ID"},
}
