"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts — `describe.py` merges these hints
into the published schema and refuses any that name an argument or an option
the signature does not offer. Delete this file and the tool still works; the
panel just gets worse.

**Everything is DERIVED, never restated.** That is the whole difference from the
`ArgSpec` tables this replaces. Those listed the anatomical tabs by hand, and a
landmark added to the catalog was then reachable through no tab at all — offered
by the schema, invisible in the UI. Here the tabs are computed from
`cbct.catalog.GROUP_LABELS`, so a landmark added there appears in its tab with
no edit here and no client release.

Why it is needed at all: ALI publishes **119** landmark options. A schema that
says "119 strings, pick some" is honest and produces a panel nobody can use.
The client already knows how to render tabs, sections and conditional fields —
it simply stopped being told which.
"""

from .cbct import catalog as cbct_catalog

# The collapsible boxes the panel is laid out in. Both engines' selections are
# always shown -- the schema cannot say "this argument only applies in mode X",
# because the mode is read from the DATA, not from an argument -- so the least a
# panel can do is not interleave them: a CBCT user reads one box and ignores the
# other.
_INPUTS = "Inputs"
_CBCT = "CBCT landmarks"
_IOS = "IOS landmarks"
_OUTPUTS = "Outputs"

LAYOUT = {
    "input": {"section": _INPUTS, "label": "Scan or Folder"},
    "model": {"section": _INPUTS, "label": "Model Bundle"},
    "cbct_regions": {"section": _CBCT, "label": "Regions", "ui": "inline"},
    "landmarks": {
        "section": _CBCT,
        "label": "Individual landmarks",
        # 119 check boxes in one column is a scroll, not a choice. The tabs are
        # the SAME grouping the engine names its output files by, published
        # rather than restated.
        "ui": "tabs",
        "groups": {
            display: list(cbct_catalog.GROUP_LABELS[code])
            for display, code in cbct_catalog.REGION_NAMES.items()
        },
    },
    "ios_networks": {"section": _IOS, "label": "Landmark families", "ui": "inline"},
    "prediction_ID": {"section": _OUTPUTS, "label": "Prediction ID"},
}
