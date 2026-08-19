"""AREG_IOSCBCT's unit tests: no GPU, no weights, no network.

Split out of the single `tools/AREG/tests/test_run.py` AREG had before it
became three tools; see AREG_CBCT/tests/test_run.py for why none of them ran.

The tools this one drives are stood in for by a fake supervisor, which is all a
tool can see of them: five members, duck-typed, nothing imported across venvs.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from sadt_areg_ioscbct import dispatch, run, tools
from sadt_areg_common import catalogs, pairing
from sadt_areg_common.errors import SupervisorRequired, ToolInputError


class FakeSup:
    """A supervisor, as a tool sees one. Records what it was asked for.

    `outputs` maps a tool name to a callable taking the parameters it was sent
    and returning the directory it "produced", so a test can plant results
    without any of the real tools existing.
    """

    def __init__(self, tmp_path, outputs=None):
        self.out = Path(tmp_path) / "out"
        self.tmp = Path(tmp_path) / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.outputs = outputs or {}
        self.calls = []
        self.messages = []

    def run(self, tool, **params):
        self.calls.append((tool, params))
        maker = self.outputs.get(tool)
        if maker is None:
            raise AssertionError(f"nothing planted for {tool!r} in this test")
        return Path(maker(params))

    def progress(self, fraction, message):
        self.messages.append((fraction, message))

    def log(self, message):
        self.messages.append((None, message))


def _phantom(size=48, seed=0, spacing=0.8, origin=(-140.0, -90.0, 60.0)):
    """A textured volume with an origin far from zero.

    Far from zero on purpose: that is the condition under which elastix's
    centre of rotation matters, and a phantom centred on the origin would let
    the bug this suite pins pass unnoticed.
    """
    rng = np.random.default_rng(seed)
    volume = rng.random((size,) * 3).astype(np.float32) * 120
    zz, yy, xx = np.meshgrid(*[np.arange(size)] * 3, indexing="ij")
    half = size // 2
    volume += 1400 * (
        ((zz - half) ** 2 / 180 + (yy - half + 2) ** 2 / 140 + (xx - half - 2) ** 2 / 160) < 1
    )
    volume += 900 * (
        ((zz - half + 12) ** 2 / 40 + (yy - half - 10) ** 2 / 35 + (xx - half + 12) ** 2 / 30) < 1
    )
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((spacing,) * 3)
    image.SetOrigin(origin)
    return image


def _moved(image, rotation=(0.04, -0.025, 0.03), translation=(1.2, -1.6, 0.9)):
    """`image` displaced by a known rigid transform, and that transform."""
    truth = sitk.Euler3DTransform()
    size = np.array(image.GetSize()) / 2.0
    truth.SetCenter(image.TransformContinuousIndexToPhysicalPoint(size.tolist()))
    truth.SetRotation(*rotation)
    truth.SetTranslation(translation)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(truth.GetInverse())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image), truth


def _write(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(image, path, useCompression=True)
    return path


def _full_mask(image):
    mask = sitk.GetImageFromArray(np.ones(sitk.GetArrayViewFromImage(image).shape, np.uint8))
    mask.CopyInformation(image)
    return mask


def _grid_mesh(rows=12, columns=12, spacing=1.0):
    """A flat triangulated grid, the smallest thing with a real adjacency."""
    points = vtk.vtkPoints()
    for row in range(rows):
        for column in range(columns):
            points.InsertNextPoint(column * spacing, row * spacing, 0.0)

    triangles = vtk.vtkCellArray()
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            for corners in ((a, a + 1, a + columns), (a + 1, a + columns + 1, a + columns)):
                triangle = vtk.vtkTriangle()
                for index, corner in enumerate(corners):
                    triangle.GetPointIds().SetId(index, corner)
                triangles.InsertNextCell(triangle)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(triangles)
    return mesh


def test_every_tool_is_named_by_string():
    """`sup.run("ASO", ...)`, never `sup.ASO(...)`. A typo in a string is
    greppable and tools.py is the whole call graph; a typo in an attribute is an
    AttributeError an hour into a job.

    Here rather than in AREG_CBCT, which was where the single pre-split suite
    left it: this is the tool that drives all four, so it is the only one whose
    tools.py can be expected to name all four.
    """
    source = open(tools.__file__, encoding="utf-8").read()
    assert 'sup.run("' in source
    for tool in ("Crown_Seg", "ALI_CBCT", "ALI_IOS", "ASO"):
        assert f'"{tool}"' in source, tool
