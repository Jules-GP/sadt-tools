"""How a client lays FlexReg's panel out.

Nothing here changes what `run()` computes. It says which arguments belong
together, and gives the five corner pairs the 2D pads upstream's Qt widget had:
the knob sits where the point sits on the arch, so the axes have to be described
in the arch's own terms rather than as two numbers.

The ranges are NOT presentation. The server validates against them, so a request
that skips the panel cannot place a corner off the arch and be told it worked.
"""

# Travel of the antero-posterior axis, in millimetres. Upstream's ADJUST_RANGE:
# the knob saturates there, and a larger number typed into the box still works.
ADJUST_RANGE = 5.0

# Travel of the translation pad, in millimetres on both axes. It moves the four
# centroids together, so it is a rigid shift of the whole patch.
SHIFT_RANGE = 15.0

# The horizontal axis of a corner pad. 0 is mid-arch, 1 lands on the tooth
# itself at the outer edge, which is why both ends are named rather than left as
# bare numbers: "0.8" says nothing about where that is in a mouth.
_RATIO = {
    "x_range": [0.0, 1.0],
    "x_labels": ["mid", "out"],
    "y_range": [-ADJUST_RANGE, ADJUST_RANGE],
    "y_labels": ["POST", "ANT"],
    "ui": "joystick",
    "section": "Patch corners",
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
        "section": "Patch corners",
    },
    "mode": {"label": "What to do", "section": "Registration"},
    "patch": {"label": "Register on", "section": "Registration"},
    "reference": {"label": "Reference surface", "section": "Registration"},
    "surfaces": {"label": "Surfaces", "section": "Input"},
    "tooth_anterior_right": {"label": "Anterior right tooth", "section": "Patch teeth"},
    "tooth_anterior_left": {"label": "Anterior left tooth", "section": "Patch teeth"},
    "tooth_posterior_right": {"label": "Posterior right tooth", "section": "Patch teeth"},
    "tooth_posterior_left": {"label": "Posterior left tooth", "section": "Patch teeth"},
    "output_suffix": {"label": "Output suffix", "section": "Registration"},
}
