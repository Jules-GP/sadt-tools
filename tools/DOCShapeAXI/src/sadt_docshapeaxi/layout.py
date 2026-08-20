"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts -- `describe.py` merges these hints
into the published schema and refuses any that name an argument the signature
does not take, or an option it does not publish.

The one rule worth reading is `options_when` on `task`. Two of the three data
types have a single model, and offering a clinician "binary" for a mandibular
condyle only to refuse the run at the end is the worst of both. The rule is
DERIVED from the catalog rather than restated beside it, which is what stops
the two drifting when a model is added.
"""

from .catalog import AIRWAY, CLEFT, CONDYLE, tasks_for

_INPUTS = "Inputs"
_GRADING = "Grading"
_COMPUTE = "Compute"

LAYOUT = {
    "meshes": {"section": _INPUTS, "label": "Surfaces"},
    "model": {"section": _INPUTS, "label": "Checkpoint bundle"},

    "data_type": {"section": _GRADING, "label": "Anatomy", "ui": "inline"},
    "task": {
        "section": _GRADING,
        "label": "Grade",
        "ui": "inline",
        "options_when": {"data_type": {
            data_type: tasks_for(data_type) for data_type in (CONDYLE, AIRWAY, CLEFT)
        }},
    },
    "explainability": {"section": _GRADING, "label": "Also write GradCAM surfaces"},

    "device": {"section": _COMPUTE, "label": "Device"},
    "num_workers": {"section": _COMPUTE, "label": "Loader workers"},
}
