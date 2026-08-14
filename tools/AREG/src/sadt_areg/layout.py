"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts — `describe.py` merges these hints
into the published schema and refuses any that name an argument or an option
the signature does not offer.

AREG has the same problem ASO has, one worse: **two modalities and three
automation modes share one schema**, and most of its arguments apply to exactly
one combination. Without conditions a panel asks a CBCT user about the palate
patch and an IOS user about DICOM. The conditions below are what the Slicer
module expressed as separate pages.

Derived, not restated: the region tabs come from `catalogs`, so a region added
there appears with no edit here.
"""

from . import catalogs

_INPUTS = "Inputs"
_CBCT = "CBCT registration"
_IOS = "IOS registration"
_OUTPUTS = "Outputs"

_CBCT_ONLY = {"modality": catalogs.MODALITY_CBCT}
_IOS_ONLY = {"modality": catalogs.MODALITY_IOS}

# The masks are the one argument a mode makes meaningless rather than merely
# optional: in either automated mode AMASSS produces them, and a field offered
# there reads as something the run needs. Conditions are ANDed, so this is
# "CBCT, and only when you are supplying them yourself" -- the same sentence
# the argument's own description opens with.
_CBCT_SEMI = {"modality": catalogs.MODALITY_CBCT, "automation": catalogs.AUTOMATION_SEMI}

LAYOUT = {
    "t1": {"section": _INPUTS, "label": "T1 (baseline)"},
    "t2": {"section": _INPUTS, "label": "T2 (follow-up)"},
    "modality": {"section": _INPUTS, "label": "Input Type"},
    "automation": {"section": _INPUTS, "label": "Mode"},

    # -- CBCT ---------------------------------------------------------------
    # The one argument a clinician must actually think about: register on what
    # has NOT changed between the two timepoints.
    "cbct_regions": {
        "section": _CBCT,
        "label": "Register on",
        "ui": "inline",
        "visible_when": _CBCT_ONLY,
    },
    "t1_masks": {"section": _CBCT, "label": "T1 masks", "visible_when": _CBCT_SEMI},
    "segmentation_model": {
        "section": _CBCT, "label": "Segmentation model", "visible_when": _CBCT_ONLY,
    },
    "segmentation_label": {
        "section": _CBCT, "label": "Mask label value", "visible_when": _CBCT_ONLY,
    },
    "cbct_reference": {
        "section": _CBCT, "label": "Orientation reference", "visible_when": _CBCT_ONLY,
    },
    "landmark_model": {
        "section": _CBCT, "label": "Landmark model bundle", "visible_when": _CBCT_ONLY,
    },
    "dicom_input": {"section": _INPUTS, "label": "Input is DICOM", "visible_when": _CBCT_ONLY},

    # -- IOS ----------------------------------------------------------------
    "ios_reference": {
        "section": _IOS, "label": "Orientation reference", "visible_when": _IOS_ONLY,
    },
    "ios_patch": {"section": _IOS, "label": "Registration patch", "visible_when": _IOS_ONLY},
    "registration_model": {
        "section": _IOS, "label": "Patch model", "visible_when": _IOS_ONLY,
    },
    "mgl_landmarks": {
        "section": _IOS, "label": "Mucogingival landmarks", "visible_when": _IOS_ONLY,
    },
    "mgl_patch_height": {
        "section": _IOS, "label": "Patch height (mm)", "visible_when": _IOS_ONLY,
    },

    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},
}
