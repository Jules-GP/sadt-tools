"""What AMASSS can segment, and how the pieces are labelled.

The single source of truth for structure codes, their label values in a merged
volume, their colours and the order they are painted in. `scripts/describe.py`
publishes only `list[str]`, so the accepted names are documented in `run()`'s
docstring rather than declared as schema choices the way the old `multichoice`
argument did.

The "FIX:" comments record defects of the original Slicer CLI corrected during
the first port; they are kept because the corrections are still load-bearing.
"""

# FIX: "Teeth" (TEETH), "Root canal" (RC) and "Mandibular canal" (MCAN) are
# deliberately ABSENT. They were offered by the Slicer UI while sitting in
# UNAVAILABLE_MODELS, had no entry in the LARGE label table, and produced
# either a KeyError during surface export or a silent collision onto label 1
# (= mandible) during merge. Offering a structure with no model is worse than
# not offering it; they come back here the day a model ships, in one place.
STRUCTURE_GROUPS = {
    "Bones": {
        "Mandible": "MAND",
        "Maxilla": "MAX",
        "Cranial base": "CB",
        "Cervical vertebra": "CV",
    },
    "Soft tissue": {
        "Upper airway": "UAW",
        "Skin": "SKIN",
    },
    "Masks": {
        "Cranial base (Mask)": "CBMASK",
        "Mandible (Mask)": "MANDMASK",
        "Maxilla (Mask)": "MAXMASK",
    },
}

STRUCTURE_CODES = tuple(
    code for group in STRUCTURE_GROUPS.values() for code in group.values()
)

DEFAULT_STRUCTURES = ("MAND", "MAX", "CB", "CV", "UAW")

# Both spellings resolve: the codes `run()` documents, and the display names the
# old schema published, so a client that still sends "Cranial base" keeps
# working.
STRUCTURE_NAMES = {
    display_name: code
    for group in STRUCTURE_GROUPS.values()
    for display_name, code in group.items()
}

# Label values in the merged multi-label volume. Unchanged from the original
# LABELS["LARGE"], because AREG and existing datasets depend on them.
LABELS = {
    "MAND": 1,
    "CB": 2,
    "UAW": 3,
    "MAX": 4,
    "CV": 5,
    "SKIN": 6,
    "CBMASK": 7,
    "MANDMASK": 8,
    "MAXMASK": 9,
}
NAMES_FROM_LABELS = {value: code for code, value in LABELS.items()}

# FIX: the original color table stopped at label 6, so the three mask
# structures fell back to white, and label 3 (upper airway) was pure black --
# invisible against a dark 3D view.
LABEL_COLORS = {
    1: (216, 101, 79),
    2: (128, 174, 128),
    3: (0, 151, 206),
    4: (230, 220, 70),
    5: (111, 184, 210),
    6: (172, 122, 101),
    7: (144, 190, 144),
    8: (230, 130, 110),
    9: (240, 232, 120),
}

# Order in which structures are painted into the merged volume: later entries
# overwrite earlier ones where they overlap.
# FIX: the original list contained "CAN", a code that exists nowhere else
# (the mandibular canal is "MCAN") -- so that entry could never match. Dead
# entries removed rather than left as decoration.
MERGING_ORDER = (
    "SKIN",
    "CV",
    "UAW",
    "CB",
    "MAX",
    "MAND",
    "CBMASK",
    "MANDMASK",
    "MAXMASK",
)

MERGE_MODES = ("MERGED", "SEPARATE")
DEFAULT_MERGE_MODES = ("MERGED",)

MERGE_MODE_NAMES = {
    "One merged segmentation file": "MERGED",
    "Separated segmentation files": "SEPARATE",
}


def _codes_from(selection, name_to_code: dict, fallback: tuple) -> tuple:
    """Turn a `list[str]` argument into tool codes.

    Accepts what legitimately reaches here now that the schema is a plain list:

    * `None` -- an omitted optional argument, which falls back to the defaults;
    * a single name or code as a plain string. Without its own branch a string
      falls through to the iterable case below and is split into a tuple of its
      CHARACTERS, so no code ever matches. `merge` shipped that way once, and
      every run came back as a report with zero segmentation files in it;
    * an iterable of codes or display names, which is the normal path.

    The old `base.Selection` dict `{name: bool}` is still accepted so a caller
    that has not moved off the previous schema keeps working.
    """
    if selection is None:
        return fallback
    if isinstance(selection, str):
        selection = (selection,)
    if isinstance(selection, dict):
        return tuple(
            name_to_code[name]
            for name, enabled in selection.items()
            if enabled and name in name_to_code
        )
    # Display names are translated, codes pass through; anything unknown is
    # kept as-is so the caller sees it named in the error rather than dropped.
    return tuple(name_to_code.get(item, item) for item in selection)


def structure_codes(selection) -> tuple:
    return _codes_from(selection, STRUCTURE_NAMES, DEFAULT_STRUCTURES)


def merge_modes(selection) -> tuple:
    return _codes_from(selection, MERGE_MODE_NAMES, DEFAULT_MERGE_MODES)
