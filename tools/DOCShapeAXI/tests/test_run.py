"""Tests for DOCShapeAXI.

The grading itself needs shapeaxi, pytorch3d and a real checkpoint -- the
`grading` extra and the hosted bundle -- so it is marked `models`. Everything
this port actually decides happens before any of that is loaded: which
checkpoint a request resolves to, which tasks an anatomy has a model for, which
surfaces are in a folder and what shapeaxi will be handed to read them with.
That is what is tested here, and it is where a repackaging can go wrong.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import vtk

from sadt_docshapeaxi import run
from sadt_docshapeaxi.catalog import (
    AIRWAY,
    CLASSIFICATION_NETWORK,
    CLEFT,
    CONDYLE,
    DATA_TYPES,
    MODELS,
    REGRESSION_NETWORK,
    TASKS,
    resolve,
    tasks_for,
)
from sadt_docshapeaxi.errors import CheckpointNotFound, ToolInputError
from sadt_docshapeaxi.meshes import find_surfaces, write_manifest
from sadt_docshapeaxi.pipeline import find_checkpoint


def _surface(path: Path):
    """A real .vtk surface: a sphere, written the way a scan would be."""
    source = vtk.vtkSphereSource()
    source.SetThetaResolution(8)
    source.SetPhiResolution(8)
    source.Update()

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(source.GetOutput())
    writer.Write()
    return path


@pytest.fixture
def bundle(tmp_path):
    """A checkpoint bundle holding every model the catalog names."""
    folder = tmp_path / "bundle"
    folder.mkdir()
    for stem, _classes in MODELS.values():
        (folder / f"{stem}.ckpt").write_bytes(b"not a real checkpoint")
    return folder


# ------------------------------------------------------------------- catalog


def test_every_published_pair_resolves_to_a_model():
    for data_type in DATA_TYPES:
        for task in tasks_for(data_type):
            stem, network, classes = resolve(data_type, task)
            assert stem and classes >= 1
            assert network in (CLASSIFICATION_NETWORK, REGRESSION_NETWORK)


def test_a_regression_asks_for_the_regression_network():
    stem, network, classes = resolve(AIRWAY, "regression")
    assert network == REGRESSION_NETWORK
    assert (stem, classes) == ("airways_4_regress", 1)


def test_the_condyle_and_the_cleft_have_one_model_each():
    assert tasks_for(CONDYLE) == ["severity"]
    assert tasks_for(CLEFT) == ["severity"]
    assert tasks_for(AIRWAY) == ["binary", "severity", "regression"]


def test_a_task_with_no_model_is_refused_rather_than_answered_with_another():
    """Upstream reaches the four-class model whatever `task` says.

    Asking for a binary grade and receiving a four-class one is a wrong answer,
    not a slow one, so it is refused by name -- and the panel narrows the
    options so a client never provokes it.
    """
    with pytest.raises(ToolInputError, match="no binary model for Mandibular"):
        resolve(CONDYLE, "binary")


def test_an_unknown_anatomy_or_task_names_what_is_available():
    with pytest.raises(ToolInputError, match="Unknown data_type"):
        resolve("Mandible", "severity")
    with pytest.raises(ToolInputError, match="Unknown task"):
        resolve(AIRWAY, "grading")


def test_the_published_sets_are_the_implemented_ones():
    """The two Literals in run() are second declarations of the catalog."""
    import typing

    hints = typing.get_type_hints(run)
    assert set(typing.get_args(hints["data_type"])) == set(DATA_TYPES)
    assert set(typing.get_args(hints["task"])) == set(TASKS)


def test_the_layout_narrows_the_tasks_from_the_catalog_itself():
    from sadt_docshapeaxi.layout import LAYOUT

    rule = LAYOUT["task"]["options_when"]["data_type"]
    assert rule == {data_type: tasks_for(data_type) for data_type in DATA_TYPES}


# ------------------------------------------------------------------ surfaces


def test_surfaces_are_found_recursively_and_in_a_stable_order(tmp_path):
    _surface(tmp_path / "meshes" / "b.vtk")
    _surface(tmp_path / "meshes" / "a.vtk")
    _surface(tmp_path / "meshes" / "nested" / "c.vtk")
    (tmp_path / "meshes" / "notes.txt").write_text("ignored")

    assert find_surfaces(tmp_path / "meshes") == ["a.vtk", "b.vtk", "nested/c.vtk"]


def test_one_surface_is_named_relative_to_its_own_folder(tmp_path):
    scan = _surface(tmp_path / "meshes" / "a.vtk")

    assert find_surfaces(scan) == ["a.vtk"]


def test_a_folder_with_no_surface_says_what_it_looked_for(tmp_path):
    (tmp_path / "meshes").mkdir()

    with pytest.raises(ToolInputError, match="No surface found"):
        find_surfaces(tmp_path / "meshes")


def test_the_manifest_is_rebuilt_not_appended_to(tmp_path):
    """Upstream opens it with 'a', so a re-run doubles every row."""
    destination = tmp_path / "files.csv"
    write_manifest(["a.vtk", "b.vtk"], destination)
    write_manifest(["a.vtk", "b.vtk"], destination)

    table = pd.read_csv(destination)
    assert list(table["surf"]) == ["a.vtk", "b.vtk"]


# ---------------------------------------------------------------- checkpoint


def test_the_checkpoint_follows_from_the_anatomy_and_the_task(bundle):
    stem, _network, _classes = resolve(AIRWAY, "binary")

    assert find_checkpoint(bundle, stem).name == "airways_2_class.ckpt"


def test_a_checkpoint_may_be_named_directly(tmp_path):
    one = tmp_path / "condyles_4_class.ckpt"
    one.write_bytes(b"x")

    assert find_checkpoint(one, "condyles_4_class") == one


def test_a_nested_bundle_is_searched(tmp_path):
    nested = tmp_path / "bundle" / "v1"
    nested.mkdir(parents=True)
    (nested / "clefts_4_class.ckpt").write_bytes(b"x")

    assert find_checkpoint(tmp_path / "bundle", "clefts_4_class").name == "clefts_4_class.ckpt"


def test_a_bundle_without_the_model_says_which_one_and_what_it_holds(tmp_path):
    folder = tmp_path / "bundle"
    folder.mkdir()
    (folder / "airways_2_class.ckpt").write_bytes(b"x")

    with pytest.raises(CheckpointNotFound, match="condyles_4_class.ckpt"):
        find_checkpoint(folder, "condyles_4_class")


def test_a_missing_bundle_is_reported_as_bad_input(tmp_path):
    with pytest.raises(ToolInputError, match="does not exist"):
        find_checkpoint(tmp_path / "nowhere", "condyles_4_class")


# ------------------------------------------------- refused before any loading


def test_a_bad_request_is_refused_before_shapeaxi_is_imported(tmp_path, bundle):
    """None of these reach a torch import, which is what makes them instant."""
    _surface(tmp_path / "meshes" / "a.vtk")

    with pytest.raises(ToolInputError, match="no binary model"):
        run(meshes=tmp_path / "meshes", model=bundle, output_dir=tmp_path / "out",
            data_type=CONDYLE, task="binary")

    with pytest.raises(ToolInputError, match="num_workers"):
        run(meshes=tmp_path / "meshes", model=bundle, output_dir=tmp_path / "out",
            num_workers=-1)

    with pytest.raises(ToolInputError, match="does not exist"):
        run(meshes=tmp_path / "nowhere", model=bundle, output_dir=tmp_path / "out")


def test_nothing_is_downloaded(tmp_path):
    """Upstream fetches the checkpoint from a GitHub release on every run.

    A tool that pulls weights over the network cannot say which revision
    produced a result, so the bundle is named by the caller instead. This is
    the assertion that the fetch did not come across with the rest.
    """
    source = (Path(__file__).parent.parent / "src" / "sadt_docshapeaxi")
    joined = "\n".join(path.read_text() for path in source.glob("*.py"))
    for forbidden in ("urlretrieve", "requests.get", "download_model"):
        assert forbidden not in joined, forbidden


@pytest.mark.models
def test_grading_a_real_surface():
    pytest.skip(
        "Needs the `grading` extra (shapeaxi, pytorch3d) and the hosted "
        "checkpoint bundle. Run by hand and report in the PR: "
        "uv sync --extra grading, then a run against DATA/DOCShapeAXI/.")
