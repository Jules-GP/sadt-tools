"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts -- `describe.py` merges these hints
into the published schema and refuses any that name an argument the signature
does not take. Delete this file and the tool still works; the panel gets worse,
and here that is the difference between five fields and twenty-two.

Every condition keys on `step`, and they mirror upstream's tabs: the module put
these operations on separate pages precisely because an argument on one page
means nothing on another.
"""

from .pipeline import (
    STEP_APPROXIMATE,
    STEP_LR_CROP,
    STEP_ORIENT,
    STEP_REGISTER,
    STEP_RESAMPLE,
    STEP_TMJ_CROP,
)

_INPUTS = "Inputs"
_ORIENT = "Orientation"
_RESAMPLE = "Resampling"
_CONDYLE = "Condyle model"
_NORMALISE = "Normalisation"

_SECOND_TIMEPOINT = {"step": STEP_RESAMPLE}
_ORIENTING = {"step": STEP_ORIENT}
_RESAMPLING = {"step": STEP_RESAMPLE}
_REGISTERING = {"step": STEP_REGISTER}
# The two steps that segment the condyle to find the joint.
_SEGMENTING = {"step": [STEP_APPROXIMATE, STEP_TMJ_CROP]}
# Every step but Orient reads a CBCT; only these read a segmentation.
_USES_CBCT = {"step": [STEP_RESAMPLE, STEP_APPROXIMATE, STEP_LR_CROP,
                       STEP_TMJ_CROP, STEP_REGISTER]}
_USES_SEGMENTATION = {"step": [STEP_RESAMPLE, STEP_LR_CROP, STEP_TMJ_CROP,
                               STEP_REGISTER]}

LAYOUT = {
    "step": {"section": _INPUTS, "label": "Step"},
    "mri": {"section": _INPUTS, "label": "MRI"},
    "cbct": {"section": _INPUTS, "label": "CBCT", "visible_when": _USES_CBCT},
    "segmentation": {"section": _INPUTS, "label": "CBCT segmentation",
                     "visible_when": _USES_SEGMENTATION},

    "mri_t2": {"section": _INPUTS, "label": "MRI (T2)",
               "visible_when": _SECOND_TIMEPOINT},
    "cbct_t2": {"section": _INPUTS, "label": "CBCT (T2)",
                "visible_when": _SECOND_TIMEPOINT},
    "segmentation_t2": {"section": _INPUTS, "label": "Segmentation (T2)",
                        "visible_when": _SECOND_TIMEPOINT},

    "direction": {"section": _ORIENT, "label": "Direction matrix",
                  "visible_when": _ORIENTING},
    "acquisition_z_spacing": {"section": _ORIENT, "label": "Slice spacing (mm)",
                              "visible_when": _ORIENTING},

    "resample_size": {"section": _RESAMPLE, "label": "Size (voxels)",
                      "visible_when": _RESAMPLING},
    "spacing": {"section": _RESAMPLE, "label": "Spacing (mm)",
                "visible_when": _RESAMPLING},
    "center": {"section": _RESAMPLE, "label": "Centre each volume",
               "visible_when": _RESAMPLING},

    "condyle_model": {"section": _CONDYLE, "label": "nnUNet condyle model",
                      "visible_when": _SEGMENTING},

    "mri_min_norm": {"section": _NORMALISE, "label": "MRI min",
                     "visible_when": _REGISTERING},
    "mri_max_norm": {"section": _NORMALISE, "label": "MRI max",
                     "visible_when": _REGISTERING},
    "mri_lower_percentile": {"section": _NORMALISE, "label": "MRI lower percentile",
                             "visible_when": _REGISTERING},
    "mri_upper_percentile": {"section": _NORMALISE, "label": "MRI upper percentile",
                             "visible_when": _REGISTERING},
    "cbct_min_norm": {"section": _NORMALISE, "label": "CBCT min",
                      "visible_when": _REGISTERING},
    "cbct_max_norm": {"section": _NORMALISE, "label": "CBCT max",
                      "visible_when": _REGISTERING},
    "cbct_lower_percentile": {"section": _NORMALISE, "label": "CBCT lower percentile",
                              "visible_when": _REGISTERING},
    "cbct_upper_percentile": {"section": _NORMALISE, "label": "CBCT upper percentile",
                              "visible_when": _REGISTERING},
    "keep_temporary": {"section": _NORMALISE, "label": "Keep intermediate volumes",
                       "visible_when": _REGISTERING},
}
