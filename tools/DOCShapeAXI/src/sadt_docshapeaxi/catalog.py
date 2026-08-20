"""Which model grades which anatomy, and how many grades it gives.

Upstream spreads this across `find_model_name`, `find_nn_type` and a
`num_classes` attribute set as a side effect of the first. It is one table
here, because the three answers are one fact: pick the anatomy and the task and
everything else follows. `run()` therefore publishes the two questions a
clinician can answer and derives the rest, rather than asking for a checkpoint
name, a network class and a class count that must agree with each other.
"""

from .errors import ToolInputError

# The three data types, spelled as upstream's combo box spells them: the
# strings travel to a client and appear in a panel.
CONDYLE = "Mandibular Condyle"
AIRWAY = "Nasopharynx Airway Obstruction"
CLEFT = "Alveolar Bone Defect in Cleft"
DATA_TYPES = (CONDYLE, AIRWAY, CLEFT)

BINARY = "binary"
SEVERITY = "severity"
REGRESSION = "regression"
TASKS = (BINARY, SEVERITY, REGRESSION)

CLASSIFICATION_NETWORK = "SaxiMHAFBClassification"
REGRESSION_NETWORK = "SaxiMHAFBRegression"

# (data type, task) -> (checkpoint stem, number of classes).
#
# Only the airway has a model per task; the condyle and the cleft have one
# four-class model each, and upstream reaches it whatever `task` says. That is
# NOT reproduced -- a task with no model of its own is refused by name here,
# because "you asked for a binary grade and got a four-class one" is a wrong
# answer rather than a slow one. The panel narrows the options instead
# (layout.py), so the refusal is one a client should never provoke.
MODELS = {
    (CONDYLE, SEVERITY): ("condyles_4_class", 4),
    (CLEFT, SEVERITY): ("clefts_4_class", 4),
    (AIRWAY, BINARY): ("airways_2_class", 2),
    (AIRWAY, SEVERITY): ("airways_4_class", 4),
    (AIRWAY, REGRESSION): ("airways_4_regress", 1),
}


def tasks_for(data_type: str) -> list:
    """The tasks a data type has a model for, in the published order."""
    return [task for task in TASKS if (data_type, task) in MODELS]


def resolve(data_type: str, task: str):
    """(checkpoint stem, network class, number of classes) for one request."""
    if data_type not in DATA_TYPES:
        raise ToolInputError(
            "Unknown data_type '{}'. Available: {}.".format(
                data_type, ", ".join(DATA_TYPES)))
    if task not in TASKS:
        raise ToolInputError(
            "Unknown task '{}'. Available: {}.".format(task, ", ".join(TASKS)))
    if (data_type, task) not in MODELS:
        raise ToolInputError(
            "There is no {} model for {}. It supports: {}.".format(
                task, data_type, ", ".join(tasks_for(data_type))))

    stem, classes = MODELS[(data_type, task)]
    network = REGRESSION_NETWORK if task == REGRESSION else CLASSIFICATION_NETWORK
    return stem, network, classes
