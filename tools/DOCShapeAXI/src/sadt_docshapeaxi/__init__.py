"""DOCShapeAXI -- grade a surface mesh, and show what the model looked at.

A shapeaxi classifier reads a surface and returns a grade; GradCAM writes back
onto that surface which part of it moved the model. The work is in grading.py;
only `run` is public.
"""

from pathlib import Path
from typing import Literal

from .pipeline import process


def run(
    meshes: Path,
    model: Path,
    output_dir: Path,
    # Spelled out because `Literal` takes literals only -- it cannot be built
    # from catalog.DATA_TYPES. That makes this a second declaration of the same
    # set, which is the thing this contract otherwise avoids, so a test asserts
    # the two agree.
    data_type: Literal[
        "Mandibular Condyle",
        "Nasopharynx Airway Obstruction",
        "Alveolar Bone Defect in Cleft",
    ] = "Mandibular Condyle",
    task: Literal["binary", "severity", "regression"] = "severity",
    explainability: bool = True,
    device: Literal["cuda", "cpu"] = "cuda",
    num_workers: int = 4,
) -> Path:
    """Grade a surface mesh, and show which part of it the model graded on.

    Args:
        meshes: One surface (.vtk/.vtp/.stl/.obj), or a folder of them for a
            batch. Folders are searched recursively.
        model: The checkpoint bundle hosted by the server, holding one `.ckpt`
            per anatomy and grade. Which one is used follows from `data_type`
            and `task`, so the bundle is named rather than the file.
        output_dir: Where the prediction table, the GradCAM surfaces and
            `DOCShapeAXI_report.json` are written. Nothing is written outside
            it.
        data_type: The anatomy being graded. It decides which model reads the
            surface, and with `task` which grades that model can give.
        task: What to grade. The airway has a model for each: binary
            (obstructed or not), severity (four grades) and regression (a
            continuous score). The mandibular condyle and the alveolar cleft
            have one four-grade model each, so `severity` is their only task --
            asking for another is refused rather than quietly answered with the
            four-grade model.
        explainability: Also write each surface again with a GradCAM array per
            grade on it, saying which part of the surface moved the model. This
            roughly doubles the run; leaving it off gives the grades alone.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is
            visible, with a warning.
        num_workers: Loader processes used to read and transform surfaces. 0
            reads them in the main process, which is what to use when a machine
            is short of shared memory.

    Returns:
        The output directory, holding the prediction table, the explainability
        surfaces and the run report.
    """
    return process(
        meshes=meshes,
        model=model,
        output_dir=output_dir,
        data_type=data_type,
        task=task,
        explainability=explainability,
        device=device,
        num_workers=num_workers,
    )
