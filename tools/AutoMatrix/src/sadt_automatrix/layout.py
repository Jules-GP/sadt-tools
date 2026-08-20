"""How a client should lay this tool's panel out. Presentation only.

Nothing here changes what `run()` accepts -- `describe.py` merges these hints
into the published schema and refuses any that name an argument the signature
does not take. Delete this file and the tool still works; the panel gets worse.

There are no `visible_when` conditions, and that is a fact about AutoMatrix
rather than an omission: it has no mode. Every argument is read on every run,
so hiding one would hide something that is about to be used. What the sections
do instead is separate the two paths a user has to fill in (the scans and the
matrices) from the two things they only sometimes change -- how the results are
named, and how they are resampled.
"""

_INPUTS = "Inputs"
_NAMING = "Output names"
_RESAMPLING = "Resampling"

LAYOUT = {
    "scans": {"section": _INPUTS, "label": "Scans or landmarks"},
    "matrices": {"section": _INPUTS, "label": "Matrices"},
    # Upstream keys this off the file names inside an AREG output tree, so it
    # belongs with the matrix folder it reinterprets, not with the options.
    "from_areg": {"section": _INPUTS, "label": "Matrices come from AREG"},

    "suffix": {"section": _NAMING, "label": "Suffix"},
    "add_matrix_name": {"section": _NAMING, "label": "Add the matrix name"},

    "reference": {"section": _RESAMPLING, "label": "Reference volume"},
    "is_segmentation": {"section": _RESAMPLING, "label": "Input is a segmentation"},
}
