"""Running the shapeaxi classifier, and the GradCAM that explains it.

Both functions are upstream's `saxi_predict` and `saxi_gradcam`, with the
argparse namespace unpacked into parameters and the progress log file gone --
it existed so the Slicer panel could poll a text file for a percentage, and the
server has its own progress channel. Every torch and shapeaxi import is inside
a function: `describe.py` imports this package on every CI job to publish the
schema, and that must not cost a CUDA stack.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Upstream's, unchanged: the GradCAM attribution is resized to the renderer's
# view size before being written onto the surface.
VIEW_SIZE = (224, 224)
# and clipped to its 1st and 99th percentile, so one hot vertex does not flatten
# the rest of the map.
CLIP_PERCENTILES = (1, 99)


def _scale_cam_image(cam, target_size=None):
    """Upstream's `scale_cam_image`, itself adapted from pytorch-grad-cam."""
    import cv2
    import numpy as np

    result = []
    for image in cam:
        if target_size is not None:
            image = cv2.resize(np.float32(image), target_size)
            upper = np.percentile(image.flatten(), q=CLIP_PERCENTILES[1])
            lower = np.percentile(image.flatten(), q=CLIP_PERCENTILES[0])
            image = np.clip(image, lower, upper)
            image = 2 * ((image - np.min(image)) / (np.max(image) - np.min(image))) - 1
        result.append(image)
    return np.float32(result)


def _load(network: str, checkpoint: Path, device: str):
    from shapeaxi import saxi_nets_lightning

    model = getattr(saxi_nets_lightning, network).load_from_checkpoint(
        str(checkpoint), strict=False)
    model.eval()
    model.to(device)
    return model


def predict(manifest: Path, meshes: Path, output_dir: Path, checkpoint: Path,
            network: str, task: str, device: str, num_workers: int) -> Path:
    """Grade every surface in the manifest; write `<manifest>_prediction.csv`.

    The argmax is applied for a classification and not for a regression, which
    is upstream's `if args.nn == 'SaxiMHAFBClassification'` -- a regression's
    output is the grade itself.
    """
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from shapeaxi.saxi_dataset import SaxiDataset
    from shapeaxi.saxi_transforms import EvalTransform

    model = _load(network, checkpoint, device)
    scale_factor = getattr(model.hparams, "scale_factor", None)

    table = pd.read_csv(manifest)
    dataset = SaxiDataset(
        table, transform=EvalTransform(scale_factor), CN=True,
        surf_column=model.hparams.surf_column, mount_point=str(meshes),
        class_column=None, scalar_column=None)
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers,
                        pin_memory=False)

    softmax = nn.Softmax(dim=1)
    predictions = []
    with torch.no_grad():
        for vertices, faces, normals in loader:
            vertices = vertices.to(device)
            faces = faces.to(device)
            normals = normals.to(device)

            mesh = model.create_mesh(vertices, faces, normals)
            points = model.sample_points_from_meshes(mesh, model.hparams.sample_levels[0])
            views, _faces_per_pixel = model.render(mesh)

            graded = model(points, views)
            if network == "SaxiMHAFBClassification":
                graded = torch.argmax(softmax(graded).detach(), dim=1, keepdim=True)
            predictions.append(graded)

    table["{}_prediction".format(task)] = (
        torch.cat(predictions).cpu().numpy().squeeze())
    written = output_dir / "{}_prediction.csv".format(Path(manifest).stem)
    written.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(written, index=False)
    logger.info("DOCShapeAXI: graded %d surface(s)", len(table))
    return written


def explain(predictions: Path, meshes: Path, output_dir: Path, checkpoint: Path,
            network: str, task: str, classes: int, device: str, num_workers: int) -> Path:
    """Write each surface again with one GradCAM array per class on it.

    This is what makes the grade inspectable: the array says which part of the
    surface moved the model, and a grade nobody can check is a grade nobody
    should act on.
    """
    import pandas as pd
    from torch.utils.data import DataLoader

    from captum.attr import LayerGradCam
    from shapeaxi import post_process as psp
    from shapeaxi import utils
    from shapeaxi.saxi_dataset import SaxiDataset
    from shapeaxi.saxi_gradcam import gradcam_process
    from shapeaxi.saxi_transforms import EvalTransform

    model = _load(network, checkpoint, device)
    table = pd.read_csv(predictions)
    dataset = SaxiDataset(
        table, transform=EvalTransform(), CN=True,
        surf_column=model.hparams.surf_column, mount_point=str(meshes),
        class_column=None, scalar_column=None)
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers,
                        pin_memory=False)

    layer = getattr(model.convnet.module, "_blocks")[-1]
    cam = LayerGradCam(model, layer, device_ids=[0])

    destination = output_dir / "explainability" / task
    destination.mkdir(parents=True, exist_ok=True)

    for index, (vertices, faces, normals) in enumerate(loader):
        vertices = vertices.to(device)
        faces = faces.to(device)
        normals = normals.to(device)

        mesh = model.create_mesh(vertices, faces, normals)
        points = model.sample_points_from_meshes(mesh, model.hparams.sample_levels[0])
        views, faces_per_pixel = model.render(mesh)

        surface = dataset.getSurf(index)
        surface_path = dataset.getSurfPath(index)

        for target in range(classes):
            attribution = cam.attribute(
                inputs=(points, views), target=target, attr_dim_summation=False)
            attribution = attribution.sum(dim=1).cpu().detach()
            upscaled = _scale_cam_image(attribution.numpy(), target_size=VIEW_SIZE)
            # Upstream leaves `target_class` at None for a single-class
            # model and sets it to the index otherwise, while ALWAYS passing
            # the index to captum. The asymmetry is deliberate on its side and
            # is kept: `gradcam_process` reads the first and captum the second.
            upscaled = gradcam_process(
                _GradcamArgs(target if classes > 1 else None),
                upscaled, faces, faces_per_pixel, vertices, device=device)

            surface.GetPointData().AddArray(upscaled)
            psp.MedianFilter(surface, upscaled)

        utils.WriteSurf(surface, os.path.join(destination, os.path.basename(surface_path)))

    logger.info("DOCShapeAXI: wrote explainability for %d surface(s)", len(table))
    return destination


class _GradcamArgs:
    """What `gradcam_process` reads off the namespace it is handed.

    Upstream passes it the whole argparse namespace, of which it uses one
    attribute. Passing a whole namespace to a library is what let
    `args.target_class` be set in a loop and read three frames away; naming the
    one field it needs makes that visible.
    """

    def __init__(self, target_class):
        self.target_class = target_class
