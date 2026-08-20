"""One run: find the surfaces, resolve the checkpoint, grade, then explain.

Upstream downloads the checkpoint from a GitHub release on every run whose
output folder does not already hold one. That is gone: the bundle is named by
the caller and hosted by the deployment, like every other model in this
repository. A tool does not fetch weights over the network -- it cannot know
whether the machine has one, and a run that silently pulls a different revision
is a result nobody can reproduce.
"""

import json
import logging
from pathlib import Path

from . import grading
from .catalog import resolve
from .errors import CheckpointNotFound, ToolInputError
from .meshes import find_surfaces, write_manifest

logger = logging.getLogger(__name__)

REPORT_NAME = "DOCShapeAXI_report.json"
CHECKPOINT_SUFFIX = ".ckpt"


def find_checkpoint(bundle: Path, stem: str) -> Path:
    """The checkpoint for one (data type, task), inside the hosted bundle.

    A file may be given directly, for the case where a deployment hosts one
    checkpoint rather than the set -- but a bundle that simply lacks this
    model has to say which one it was looking for, since the name encodes the
    anatomy and the class count both.
    """
    bundle = Path(bundle)
    if bundle.is_file():
        return bundle
    if not bundle.is_dir():
        raise ToolInputError("Model bundle does not exist: {}".format(bundle))

    wanted = bundle / (stem + CHECKPOINT_SUFFIX)
    if wanted.is_file():
        return wanted

    found = sorted(path.name for path in bundle.rglob("*" + CHECKPOINT_SUFFIX))
    nested = [path for path in bundle.rglob(stem + CHECKPOINT_SUFFIX)]
    if nested:
        return nested[0]
    raise CheckpointNotFound(
        "No '{}{}' in {}. It holds: {}.".format(
            stem, CHECKPOINT_SUFFIX, bundle, ", ".join(found) or "no checkpoint"))


def process(meshes, model, output_dir, data_type, task, explainability,
            device, num_workers):
    stem, network, classes = resolve(data_type, task)
    if num_workers < 0:
        raise ToolInputError("num_workers cannot be negative.")

    meshes = Path(meshes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_checkpoint(model, stem)
    surfaces = find_surfaces(meshes)
    # A single file is given as itself; shapeaxi reads paths relative to a
    # mount point, so the mount point is then the file's folder.
    mount = meshes if meshes.is_dir() else meshes.parent
    manifest = write_manifest(surfaces, output_dir / "files_{}.csv".format(
        data_type.split(" ")[0].lower()))

    logger.info("DOCShapeAXI: %d surface(s), %s / %s, %s",
                len(surfaces), data_type, task, checkpoint.name)

    device = _device(device)
    predictions = grading.predict(
        manifest=manifest, meshes=mount, output_dir=output_dir,
        checkpoint=checkpoint, network=network, task=task, device=device,
        num_workers=num_workers)

    explained = None
    if explainability:
        explained = grading.explain(
            predictions=predictions, meshes=mount, output_dir=output_dir,
            checkpoint=checkpoint, network=network, task=task, classes=classes,
            device=device, num_workers=num_workers)

    (output_dir / REPORT_NAME).write_text(json.dumps({
        "data_type": data_type,
        "task": task,
        "checkpoint": checkpoint.name,
        "network": network,
        "classes": classes,
        "surfaces": len(surfaces),
        "predictions": str(predictions),
        "explainability": str(explained) if explained else None,
    }, indent=2), encoding="utf-8")
    return output_dir


def _device(requested: str) -> str:
    """CUDA when a card is visible, CPU otherwise, with a warning.

    Upstream never asks: it takes `cuda if torch.cuda.is_available() else cpu`
    and gives the caller no say. Asking matters on a shared server, where a
    small run on the CPU beats queueing behind a segmentation for the card.
    """
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("DOCShapeAXI: no CUDA device visible, falling back to CPU.")
        return "cpu"
    return requested
