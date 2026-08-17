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

# Narrower still, and each one mirrors a check in dispatch.py. An argument the
# chosen mode never reads is not merely noise: shown as optional beside the
# ones that matter, it reads as something the user chose not to fill, and the
# refusal arrives at the end of a run instead of before it.
_CBCT_SEGMENTED = {  # AMASSS produces the masks; Semi-Automated takes yours
    "modality": catalogs.MODALITY_CBCT,
    "automation": [catalogs.AUTOMATION_FULLY, catalogs.AUTOMATION_ORIENTED],
}
_CBCT_ORIENTED = {  # ASO orients the T1 first, and needs a reference for it
    "modality": catalogs.MODALITY_CBCT,
    "automation": catalogs.AUTOMATION_ORIENTED,
}
_IOS_ORIENTED = {  # the same, for the meshes
    "modality": catalogs.MODALITY_IOS,
    "automation": catalogs.AUTOMATION_FULLY,
}
# The patch decides the rest of the IOS panel: the mucogingival band is built
# from landmarks and involves no network at all, while the palate is predicted
# and involves nothing else. Asking for a checkpoint on the MGL side is how a
# user comes to believe that mode needs one -- dispatch.py says as much where
# it refuses. Listed rather than negated: `visible_when` compares, so "every
# patch but MGL" is written by naming them.
_IOS_MGL = {"modality": catalogs.MODALITY_IOS, "ios_patch": catalogs.PATCH_MGL}
_IOS_PREDICTED = {
    "modality": catalogs.MODALITY_IOS,
    "ios_patch": [p for p in catalogs.PATCH_CHOICES if p != catalogs.PATCH_MGL],
}

LAYOUT = {
    "t1": {"section": _INPUTS, "label": "T1 (baseline)"},
    "t2": {"section": _INPUTS, "label": "T2 (follow-up)"},
    "modality": {"section": _INPUTS, "label": "Input Type"},
    # IOS has no "Oriented + Fully-Automated": there is nothing to orient the
    # meshes onto before registering. Offering it and refusing the run at the
    # end is the worst of both, so the option is not offered at all -- the
    # table is catalogs', so a mode added there needs no edit here.
    "automation": {
        "section": _INPUTS,
        "label": "Mode",
        "options_when": {
            "modality": {
                modality: list(modes)
                for modality, modes in catalogs.AUTOMATION_BY_MODALITY.items()
            }
        },
    },

    # -- CBCT ---------------------------------------------------------------
    # The one argument a clinician must actually think about: register on what
    # has NOT changed between the two timepoints.
    "cbct_regions": {
        "section": _CBCT,
        "label": "Register on",
        "ui": "inline",
        "visible_when": _CBCT_ONLY,
    },
    "t1_masks": {"section": _CBCT, "label": "T1 masks", "visible_when": _CBCT_ONLY},
    "segmentation_model": {
        "section": _CBCT, "label": "Segmentation model", "visible_when": _CBCT_SEGMENTED,
    },
    "segmentation_label": {
        "section": _CBCT, "label": "Mask label value", "visible_when": _CBCT_SEGMENTED,
    },
    "cbct_reference": {
        "section": _CBCT, "label": "Orientation reference", "visible_when": _CBCT_ORIENTED,
    },
    "landmark_model": {
        "section": _CBCT, "label": "Landmark model bundle", "visible_when": _CBCT_ORIENTED,
    },
    "dicom_input": {"section": _INPUTS, "label": "Input is DICOM", "visible_when": _CBCT_ONLY},

    # -- IOS ----------------------------------------------------------------
    "ios_reference": {
        "section": _IOS, "label": "Orientation reference", "visible_when": _IOS_ORIENTED,
    },
    "ios_patch": {"section": _IOS, "label": "Registration patch", "visible_when": _IOS_ONLY},
    "registration_model": {
        "section": _IOS, "label": "Patch model", "visible_when": _IOS_PREDICTED,
    },
    # Only the Fully-Automated mode labels the crowns itself; the others take
    # meshes that already carry the array, whatever the patch.
    "crown_model": {
        "section": _IOS, "label": "Crown segmentation model", "visible_when": _IOS_ORIENTED,
    },
    "mgl_model": {
        "section": _IOS, "label": "Mucogingival landmark bundle", "visible_when": _IOS_MGL,
    },
    "mgl_landmarks": {
        "section": _IOS, "label": "Mucogingival landmarks", "visible_when": _IOS_MGL,
    },
    "mgl_patch_height": {
        "section": _IOS, "label": "Patch height (mm)", "visible_when": _IOS_MGL,
    },

    "output_suffix": {"section": _OUTPUTS, "label": "Output suffix"},
}
