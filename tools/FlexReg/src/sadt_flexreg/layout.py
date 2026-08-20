"""How a client lays FlexReg's panel out.

Nothing here changes what `run()` computes. It says which arguments belong
together, and gives the five corner pairs the 2D pads upstream's Qt widget had:
the knob sits where the point sits on the arch, so the axes have to be described
in the arch's own terms rather than as two numbers.

The ranges are NOT presentation. The server validates against them, so a request
that skips the panel cannot place a corner off the arch and be told it worked.
"""

# The three groups the original panel had, in its order: what goes in, the patch
# you shape, then what comes out. The tooth that bounds a corner sits WITH that
# corner rather than in a section of its own -- upstream put them side by side,
# and a tooth number read three sections away from the pad it drives says
# nothing about which corner it moves.
INPUT = "Inputs"
PATCH = "Patch"
OUTPUT = "Outputs"


# Travel of the antero-posterior axis, in millimetres. Upstream's ADJUST_RANGE:
# the knob saturates there, and a larger number typed into the box still works.
ADJUST_RANGE = 5.0

# Travel of the translation pad, in millimetres on both axes. It moves the four
# centroids together, so it is a rigid shift of the whole patch.
SHIFT_RANGE = 15.0

# The horizontal axis of a corner pad. 0 is mid-arch, 1 lands on the tooth
# itself at the outer edge, which is why both ends are named rather than left as
# bare numbers: "0.8" says nothing about where that is in a mouth.
# Shown only while the palate patch is the one being built. The mucogingival
# line is read off the mesh, not shaped from four teeth, so every control below
# is inert in that mode -- upstream hid them for the same reason.
BUTTERFLY_ONLY = {"patch": "Palate (butterfly)"}

_RATIO = {
    # Two columns, so the four corners read as the 2x2 they are: left column one
    # side of the arch, right column the other, top row anterior. Where a pad
    # sits on screen is then where that corner sits in the mouth, which is what
    # upstream's grid did and what a column of four identical pads loses.
    "section_columns": 2,
    "visible_when": BUTTERFLY_ONLY,
    "x_range": [0.0, 1.0],
    "x_labels": ["mid", "out"],
    "y_range": [-ADJUST_RANGE, ADJUST_RANGE],
    "y_labels": ["POST", "ANT"],
    "ui": "joystick",
    "section": PATCH,
}


def _corner(label):
    hints = dict(_RATIO)
    hints["label"] = label
    return hints


LAYOUT = {
    "anterior_right": _corner("Anterior right"),
    "anterior_left": _corner("Anterior left"),
    "posterior_right": _corner("Posterior right"),
    "posterior_left": _corner("Posterior left"),
    "shift": {
        "ui": "joystick",
        "label": "Move the whole patch",
        "x_range": [-SHIFT_RANGE, SHIFT_RANGE],
        "x_labels": ["L", "R"],
        "y_range": [-SHIFT_RANGE, SHIFT_RANGE],
        "y_labels": ["POST", "ANT"],
        "section": PATCH,
        "visible_when": BUTTERFLY_ONLY,
    },
    "surfaces": {"label": "Arches", "section": INPUT},
    "reference": {"label": "Register onto", "section": INPUT},
    "mode": {"label": "What to do", "section": INPUT},
    "patch": {"label": "Register on", "section": INPUT},
    # Labelled "Teeth", as upstream did: the section and the row it sits in say
    # which corner, so repeating it in the label is noise on four rows.
    "tooth_anterior_right": {"label": "Teeth", "section": PATCH,
                             "visible_when": BUTTERFLY_ONLY},
    "tooth_anterior_left": {"label": "Teeth", "section": PATCH,
                             "visible_when": BUTTERFLY_ONLY},
    "tooth_posterior_right": {"label": "Teeth", "section": PATCH,
                             "visible_when": BUTTERFLY_ONLY},
    "tooth_posterior_left": {"label": "Teeth", "section": PATCH,
                             "visible_when": BUTTERFLY_ONLY},
    "output_suffix": {"label": "Suffix", "section": OUTPUT},
}
