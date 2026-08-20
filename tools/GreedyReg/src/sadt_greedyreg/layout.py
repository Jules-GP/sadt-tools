"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts -- `describe.py` merges these hints
into the published schema and refuses any that name an argument the signature
does not take. Delete this file and the tool still works; the panel gets worse.

The conditions all key on `mode`, and they mirror upstream's own tabs: the
module puts Automatic Registration and Distant Registration on separate pages
precisely because an argument on one page means nothing on the other.
"""

from .pipeline import MODE_GREEDY, MODE_LANDMARK, MODE_LANDMARK_GREEDY

_INPUTS = "Inputs"
_REGISTRATION = "Registration"
_LANDMARKS = "Landmark alignment"

# Each mirrors a check in pipeline.process. An argument the chosen mode never
# reads is not merely noise: shown as optional beside the ones that matter, it
# reads as something the user chose not to fill, and the refusal then arrives
# at the end of a batch instead of before it.
_USES_GREEDY = {"mode": [MODE_GREEDY, MODE_LANDMARK_GREEDY]}
_USES_LANDMARKS = {"mode": [MODE_LANDMARK, MODE_LANDMARK_GREEDY]}
_PLAIN_GREEDY = {"mode": MODE_GREEDY}

LAYOUT = {
    "t1": {"section": _INPUTS, "label": "T1 (baseline)"},
    "t2": {"section": _INPUTS, "label": "T2 (follow-up)"},
    "mode": {"section": _INPUTS, "label": "Mode"},

    "metric": {"section": _REGISTRATION, "label": "Similarity metric",
               "visible_when": _USES_GREEDY},
    "transform_type": {"section": _REGISTRATION, "label": "Degrees of freedom",
                       "visible_when": _USES_GREEDY},
    "masks": {"section": _REGISTRATION, "label": "T1 masks",
              "visible_when": _USES_GREEDY},
    # Only the plain Greedy mode reads a supplied initialisation: the landmark
    # modes compute their own and would silently ignore this one.
    "init": {"section": _REGISTRATION, "label": "Initial transforms",
             "visible_when": _PLAIN_GREEDY},

    "region": {"section": _LANDMARKS, "label": "Align on",
               "ui": "inline", "visible_when": _USES_LANDMARKS},
    "landmark_model": {"section": _LANDMARKS, "label": "ALI model bundle",
                       "visible_when": _USES_LANDMARKS},
    "device": {"section": _LANDMARKS, "label": "Device",
               "visible_when": _USES_LANDMARKS},
}
