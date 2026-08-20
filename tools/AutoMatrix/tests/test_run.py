"""End-to-end tests for AutoMatrix.

Every one of these really calls `run()`. The fixtures are synthetic volumes and
landmark files built here, translated by a known number of millimetres, so the
tests can assert WHERE the result landed rather than that a file appeared --
which is the only assertion that would have caught the two things this tool can
get wrong in silence: resampling onto the wrong grid, and applying a transform
in the wrong direction.

The direction is worth stating once, because half the file rests on it. An ITK
transform maps a point of the OUTPUT space back into the INPUT space, so a
resampler is handed it as-is and a landmark is handed its inverse. A volume and
a landmark that came off the same patient therefore have to end up in the same
place, and `test_a_landmark_and_its_volume_land_in_the_same_place` is what says
they do.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from sadt_automatrix import run
from sadt_automatrix.errors import NothingWritten, ToolInputError
from sadt_automatrix.inputs import matrix_key, scan_key
from sadt_automatrix.pipeline import REPORT_NAME

SIZE = 64
CUBE = slice(20, 44)
SHIFT = 4.0  # mm, and with 1 mm voxels also 4 voxels


def _volume(path, size=SIZE, label=100.0, spacing=1.0):
    """A cube of one value in an empty box, saved wherever it is asked for."""
    data = np.zeros((size, size, size), dtype=np.float32)
    data[CUBE, CUBE, CUBE] = label
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((spacing, spacing, spacing))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def _matrix(path, shift=(SHIFT, 0.0, 0.0)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(sitk.TranslationTransform(3, shift), str(path))
    return path


def _landmarks(path, points, status="defined"):
    """A Slicer markups file holding `points`, named L0, L1, ..."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "controlPoints": [
                {"id": str(index), "label": "L{}".format(index),
                 "position": list(point), "positionStatus": status}
                for index, point in enumerate(points)
            ],
        }]
    }), encoding="utf-8")
    return path


def _array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _report(output):
    return json.loads((Path(output) / REPORT_NAME).read_text(encoding="utf-8"))


@pytest.fixture
def cohort(tmp_path):
    """One patient: a scan and the matrix that belongs to it, in two folders."""
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz")
    _matrix(tmp_path / "matrices" / "P1_MAND_Or.tfm")
    return tmp_path / "scans", tmp_path / "matrices"


# ----------------------------------------------------------------------
# Where the result lands
# ----------------------------------------------------------------------

def test_a_volume_moves_by_exactly_what_the_matrix_says(cohort, tmp_path):
    scans, matrices = cohort
    out = run(scans=scans, matrices=matrices, output_dir=tmp_path / "out")

    moved = _array(out / "P1_T1_scan_apply.nii.gz")
    original = _array(scans / "P1_T1_scan.nii.gz")

    # output(x) = input(x + shift), so the content moves by -shift. The array
    # is (z, y, x) and the shift is along x, which is axis 2.
    assert np.allclose(moved, np.roll(original, -int(SHIFT), axis=2), atol=1e-4)

    # Said a second way, on the bounding box rather than the voxels: a cube
    # that occupied [20, 44) now occupies [16, 40).
    occupied = np.nonzero(moved.any(axis=(0, 1)))[0]
    assert (occupied.min(), occupied.max() + 1) == (CUBE.start - int(SHIFT),
                                                    CUBE.stop - int(SHIFT))


def test_a_landmark_moves_by_the_inverse_of_the_matrix(tmp_path):
    _landmarks(tmp_path / "scans" / "P1_CB_lm.mrk.json",
               [(10.0, 20.0, 30.0), (0.0, 0.0, 0.0)])
    _matrix(tmp_path / "matrices" / "P1_CB_Or.tfm")

    out = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
              output_dir=tmp_path / "out")

    moved = json.loads(
        (out / "P1_CB_lm_apply.mrk.json").read_text(encoding="utf-8"))
    positions = [point["position"]
                 for point in moved["markups"][0]["controlPoints"]]
    assert positions == [[10.0 - SHIFT, 20.0, 30.0], [-SHIFT, 0.0, 0.0]]
    assert _report(out)["cases"][0]["outputs"][0]["points_moved"] == 2


def test_a_landmark_and_its_volume_land_in_the_same_place(tmp_path):
    """The one assertion that catches an inverted transform.

    The landmark sits at the centre of the cube in the volume. After the run it
    must still sit at the centre of the cube -- if the volume took the
    transform and the landmark took the same one instead of its inverse, the
    two would end up 8 mm apart rather than 0.
    """
    centre = (CUBE.start + CUBE.stop) / 2 - 0.5
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz")
    _landmarks(tmp_path / "scans" / "P1_T1_lm.mrk.json",
               [(centre, centre, centre)])
    _matrix(tmp_path / "matrices" / "P1_Or.tfm")

    out = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
              output_dir=tmp_path / "out")

    moved = _array(out / "P1_T1_scan_apply.nii.gz")
    # Centre of mass of the moved cube, in voxels, which are 1 mm and start at
    # the origin -- so it reads directly as a physical position.
    indices = np.nonzero(moved)
    cube_centre = [float(axis.mean()) for axis in reversed(indices)]  # x, y, z

    landmark = json.loads(
        (out / "P1_T1_lm_apply.mrk.json").read_text(encoding="utf-8")
    )["markups"][0]["controlPoints"][0]["position"]

    assert np.allclose(landmark, cube_centre, atol=1e-6)


def test_an_undefined_landmark_is_left_where_it_was(tmp_path):
    _landmarks(tmp_path / "scans" / "P1_CB_lm.mrk.json",
               [(1.0, 2.0, 3.0)], status="undefined")
    _matrix(tmp_path / "matrices" / "P1_CB_Or.tfm")

    out = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
              output_dir=tmp_path / "out")

    moved = json.loads(
        (out / "P1_CB_lm_apply.mrk.json").read_text(encoding="utf-8"))
    assert moved["markups"][0]["controlPoints"][0]["position"] == [1.0, 2.0, 3.0]
    assert _report(out)["cases"][0]["outputs"][0]["points_moved"] == 0


# ----------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------

def test_a_segmentation_keeps_the_labels_it_arrived_with(tmp_path):
    """Nearest-neighbour or nothing: a label interpolated is a label invented.

    The shift is deliberately half a voxel, which is where the two
    interpolators differ. Linear resampling of a 0/7 mask produces 3.5.
    """
    _volume(tmp_path / "scans" / "P1_Seg.nii.gz", label=7.0)
    _matrix(tmp_path / "matrices" / "P1_Or.tfm", shift=(2.5, 0.0, 0.0))

    labelled = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
                   output_dir=tmp_path / "seg", is_segmentation=True)
    interpolated = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
                       output_dir=tmp_path / "scan", is_segmentation=False)

    assert set(np.unique(_array(labelled / "P1_Seg_apply.nii.gz"))) == {0.0, 7.0}
    assert set(np.unique(_array(interpolated / "P1_Seg_apply.nii.gz"))) - {0.0, 7.0}


def test_a_reference_volume_is_the_grid_the_result_lands_on(tmp_path):
    """A cohort resampled against one reference comes out voxel-aligned."""
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz", size=SIZE)
    _matrix(tmp_path / "matrices" / "P1_Or.tfm")
    reference = _volume(tmp_path / "reference.nii.gz", size=40)

    out = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
              output_dir=tmp_path / "out", reference=reference)

    assert sitk.ReadImage(
        str(out / "P1_T1_scan_apply.nii.gz")).GetSize() == (40, 40, 40)
    assert _report(out)["cases"][0]["outputs"][0]["grid"] == "chosen"


def test_a_mirroring_matrix_is_applied_in_the_scans_own_space(tmp_path):
    """Mirroring onto somebody else's grid would move the patient as well."""
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz", size=SIZE)
    _matrix(tmp_path / "matrices" / "Matrix_mirror.tfm")
    reference = _volume(tmp_path / "reference.nii.gz", size=40)

    out = run(scans=tmp_path / "scans",
              matrices=tmp_path / "matrices" / "Matrix_mirror.tfm",
              output_dir=tmp_path / "out", reference=reference, suffix="_mir")

    assert sitk.ReadImage(
        str(out / "P1_T1_scan_mir.nii.gz")).GetSize() == (SIZE, SIZE, SIZE)
    assert _report(out)["cases"][0]["outputs"][0]["grid"] == "mirror_source"


def _composite(path):
    """An AREG-shaped transform: a chain, named `<something>_transform.tfm`.

    Named as a FILE in these tests rather than found in a folder, because
    `_transform` is not one of the markers a patient key is cut at -- so
    `P1_transform.tfm` keys as `P1_transform` and pairs with no scan. That is
    upstream's rule, and naming the file is how upstream applies one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(
        sitk.CompositeTransform([sitk.TranslationTransform(3, (SHIFT, 0.0, 0.0))]),
        str(path))
    return path


def test_a_composite_transform_lands_on_the_volume_beside_it(tmp_path):
    """AREG writes `P1_transform.tfm` next to the fixed `P1.nii.gz`.

    The chain is only meaningful on that grid, so it overrides the reference
    the caller chose -- which is the point of the test: the reference here is a
    different size again, and must lose.
    """
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz", size=SIZE)
    _volume(tmp_path / "matrices" / "P1.nii.gz", size=48)
    matrix = _composite(tmp_path / "matrices" / "P1_transform.tfm")

    out = run(scans=tmp_path / "scans", matrices=matrix,
              output_dir=tmp_path / "out",
              reference=_volume(tmp_path / "reference.nii.gz", size=40))

    written = _report(out)["cases"][0]["outputs"][0]
    assert written["grid"] == "composite_neighbour"
    assert sitk.ReadImage(written["output"]).GetSize() == (48, 48, 48)


def test_a_composite_transform_with_no_neighbour_falls_back_to_the_scan(tmp_path):
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz", size=SIZE)
    matrix = _composite(tmp_path / "matrices" / "P1_transform.tfm")

    out = run(scans=tmp_path / "scans", matrices=matrix,
              output_dir=tmp_path / "out",
              reference=_volume(tmp_path / "reference.nii.gz", size=40))

    written = _report(out)["cases"][0]["outputs"][0]
    assert written["grid"] == "composite_fallback"
    assert sitk.ReadImage(written["output"]).GetSize() == (SIZE, SIZE, SIZE)


# ----------------------------------------------------------------------
# Pairing and naming
# ----------------------------------------------------------------------

def test_each_patient_goes_through_its_own_matrix(tmp_path):
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz")
    _volume(tmp_path / "scans" / "P2_T1_scan.nii.gz")
    _matrix(tmp_path / "matrices" / "P1_Or.tfm", shift=(4.0, 0.0, 0.0))
    _matrix(tmp_path / "matrices" / "P2_Or.tfm", shift=(8.0, 0.0, 0.0))
    # A matrix for a patient with no scan is dropped, not an error.
    _matrix(tmp_path / "matrices" / "P9_Or.tfm")

    out = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
              output_dir=tmp_path / "out")

    for name, shift in (("P1", 4), ("P2", 8)):
        occupied = np.nonzero(
            _array(out / "{}_T1_scan_apply.nii.gz".format(name)).any(axis=(0, 1)))[0]
        assert occupied.min() == CUBE.start - shift, name
    assert not (out / "P9_T1_scan_apply.nii.gz").exists()
    assert len(_report(out)["cases"]) == 2


def test_one_named_matrix_serves_every_patient(tmp_path):
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz")
    _volume(tmp_path / "scans" / "P2_T1_scan.nii.gz")
    matrix = _matrix(tmp_path / "Matrix_mirror.tfm")

    out = run(scans=tmp_path / "scans", matrices=matrix,
              output_dir=tmp_path / "out")

    assert (out / "P1_T1_scan_apply.nii.gz").exists()
    assert (out / "P2_T1_scan_apply.nii.gz").exists()


def test_the_matrix_name_is_what_tells_two_results_apart(tmp_path):
    """Two matrices for one patient collide unless their names are added.

    Without `add_matrix_name` both results are written to the same path and the
    second silently replaces the first, which is upstream's behaviour and is
    recorded here rather than quietly fixed: fixing it would rename every file
    a working pipeline already produces.
    """
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz")
    _matrix(tmp_path / "matrices" / "P1_Or.tfm", shift=(4.0, 0.0, 0.0))
    _matrix(tmp_path / "matrices" / "P1_MAND_Or.tfm", shift=(8.0, 0.0, 0.0))

    collided = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
                   output_dir=tmp_path / "collided")
    named = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
                output_dir=tmp_path / "named", add_matrix_name=True)

    assert [entry["output"] for entry in _report(collided)["cases"][0]["outputs"]][0] \
        == [entry["output"] for entry in _report(collided)["cases"][0]["outputs"]][1]
    assert {path.name for path in named.glob("*.nii.gz")} == {
        "P1_T1_scan_apply_P1_Or.nii.gz", "P1_T1_scan_apply_P1_MAND_Or.nii.gz"}


def test_the_input_tree_is_rebuilt_under_the_output_directory(tmp_path):
    _volume(tmp_path / "scans" / "batch1" / "P1_T1_scan.nii.gz")
    _matrix(tmp_path / "matrices" / "P1_Or.tfm")

    out = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
              output_dir=tmp_path / "out")

    assert (out / "batch1" / "P1_T1_scan_apply.nii.gz").exists()


def test_one_file_is_one_run(cohort, tmp_path):
    """The picker offers file-or-folder for every path, so a file must work."""
    scans, matrices = cohort
    out = run(scans=scans / "P1_T1_scan.nii.gz",
              matrices=matrices / "P1_MAND_Or.tfm",
              output_dir=tmp_path / "out")

    assert (out / "P1_T1_scan_apply.nii.gz").exists()


def test_writes_nothing_outside_the_output_directory(cohort, tmp_path):
    scans, matrices = cohort
    before = (sorted(path.name for path in scans.iterdir()),
              sorted(path.name for path in matrices.iterdir()))

    out = run(scans=scans, matrices=matrices, output_dir=tmp_path / "out")

    assert (sorted(path.name for path in scans.iterdir()),
            sorted(path.name for path in matrices.iterdir())) == before
    assert {path.name for path in out.iterdir()} == {
        "P1_T1_scan_apply.nii.gz", REPORT_NAME}


def test_the_patient_keys_are_the_ones_upstream_derives():
    """The pairing rule, pinned. Changing it re-pairs every cohort in the lab."""
    assert scan_key("MG01_T1_scan.nii.gz") == "MG01"
    assert scan_key("MG01_MAND_seg.nii.gz") == "MG01"
    assert scan_key("MG01_CB_lm.mrk.json") == "MG01"
    assert scan_key("MG01_T13_scan.nii.gz") == "MG01"

    assert matrix_key("MG01_MAND_Or.tfm") == "MG01"
    assert matrix_key("MG01_Left_mirror.tfm") == "MG01"
    assert matrix_key("MG01_SegOr_matrix.h5") == "MG01"
    # A name with no marker at all is its own key, which is why a matrix folder
    # of `Matrix_mirror.tfm` pairs with nothing and has to be named as a file.
    assert matrix_key("Matrix_mirror.tfm") == "Matrix"


# ----------------------------------------------------------------------
# From AREG
# ----------------------------------------------------------------------

def _areg_tree(tmp_path):
    _landmarks(tmp_path / "scans" / "P1_CB_lm.mrk.json", [(0.0, 0.0, 0.0)])
    _matrix(tmp_path / "areg" / "Cranial Base" / "P1_OutReg" / "P1_CBReg_matrix.tfm")
    return tmp_path / "scans", tmp_path / "areg"


def test_from_areg_reads_the_region_matrix_out_of_the_tree(tmp_path):
    scans, areg = _areg_tree(tmp_path)

    out = run(scans=scans, matrices=areg, output_dir=tmp_path / "out",
              from_areg=True)

    written = _report(out)["cases"][0]["outputs"][0]
    assert written["matrix"].endswith("P1_OutReg/P1_CBReg_matrix.tfm")
    moved = json.loads(Path(written["output"]).read_text(encoding="utf-8"))
    assert moved["markups"][0]["controlPoints"][0]["position"] == [-SHIFT, 0.0, 0.0]


def test_from_areg_says_which_file_had_no_region_marker(tmp_path):
    scans, areg = _areg_tree(tmp_path)
    _landmarks(scans / "P2_lm.mrk.json", [(0.0, 0.0, 0.0)])

    out = run(scans=scans, matrices=areg, output_dir=tmp_path / "out",
              from_areg=True)

    skipped = [case["skipped"] for case in _report(out)["cases"]
               if case["patient"] == "P2"][0]
    assert "no region marker" in skipped[0]["reason"]


def test_from_areg_says_where_it_looked_when_the_matrix_is_missing(tmp_path):
    _landmarks(tmp_path / "scans" / "P1_L_lm.mrk.json", [(0.0, 0.0, 0.0)])
    # A tree with a matrix for another region, so the folder is not empty.
    _matrix(tmp_path / "areg" / "Cranial Base" / "P1_OutReg" / "P1_CBReg_matrix.tfm")

    with pytest.raises(NothingWritten, match="Maxilla"):
        run(scans=tmp_path / "scans", matrices=tmp_path / "areg",
            output_dir=tmp_path / "out", from_areg=True)


def test_from_areg_leaves_the_volumes_paired_by_name(tmp_path):
    """Upstream's `fromAreg` branch is landmarks-only. Volumes keep pairing."""
    scans, areg = _areg_tree(tmp_path)
    _volume(scans / "P1_T1_scan.nii.gz")

    out = run(scans=scans, matrices=areg, output_dir=tmp_path / "out",
              from_areg=True)

    assert (out / "P1_T1_scan_apply.nii.gz").exists()


# ----------------------------------------------------------------------
# What it refuses, and how loudly
# ----------------------------------------------------------------------

def test_a_surface_mesh_is_refused_by_name(tmp_path):
    """Upstream hands a `.stl` to `sitk.ReadImage` and logs the failure.

    Nothing has come out of the CLI for a mesh since it replaced the in-Slicer
    implementation, so this is not a regression -- it is the same outcome, said
    where somebody reads it.
    """
    (tmp_path / "scans").mkdir()
    (tmp_path / "scans" / "P1_MAND_seg.stl").write_text("solid\nendsolid\n")
    _matrix(tmp_path / "matrices" / "P1_Or.tfm")

    with pytest.raises(NothingWritten, match="surface meshes are not supported"):
        run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
            output_dir=tmp_path / "out")


def test_a_mesh_beside_a_volume_costs_only_itself(tmp_path):
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz")
    (tmp_path / "scans" / "P1_MAND_seg.stl").write_text("solid\nendsolid\n")
    _matrix(tmp_path / "matrices" / "P1_Or.tfm")

    out = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
              output_dir=tmp_path / "out")

    report = _report(out)
    assert report["written"] == 1 and report["skipped"] == 1


def test_a_npy_matrix_is_refused_rather_than_guessed_at(tmp_path):
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz")
    (tmp_path / "matrices").mkdir()
    np.save(tmp_path / "matrices" / "P1_Or.npy", np.eye(4))

    with pytest.raises(NothingWritten, match=r"\.npy"):
        run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
            output_dir=tmp_path / "out")


def test_an_unreadable_matrix_costs_only_its_own_pairing(tmp_path):
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz")
    _matrix(tmp_path / "matrices" / "P1_Or.tfm")
    (tmp_path / "matrices" / "P1_MAND_Or.tfm").write_text("not a transform")

    out = run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
              output_dir=tmp_path / "out", add_matrix_name=True)

    report = _report(out)
    assert report["written"] == 1
    assert report["cases"][0]["skipped"][0]["matrix"].endswith("P1_MAND_Or.tfm")


def test_a_batch_that_produced_nothing_refuses_to_return(tmp_path):
    """The guard counts files WRITTEN, not scans walked or matrices read."""
    _volume(tmp_path / "scans" / "P1_T1_scan.nii.gz")
    _matrix(tmp_path / "matrices" / "P9_Or.tfm")

    with pytest.raises(NothingWritten, match="no matrix matched patient key"):
        run(scans=tmp_path / "scans", matrices=tmp_path / "matrices",
            output_dir=tmp_path / "out")


def test_bad_input_is_reported_as_such(cohort, tmp_path):
    scans, matrices = cohort

    with pytest.raises(ToolInputError, match="Input path does not exist"):
        run(scans=tmp_path / "missing", matrices=matrices,
            output_dir=tmp_path / "out")

    with pytest.raises(ToolInputError, match="Matrix path does not exist"):
        run(scans=scans, matrices=tmp_path / "missing",
            output_dir=tmp_path / "out")

    (tmp_path / "empty").mkdir()
    with pytest.raises(ToolInputError, match="No scan found"):
        run(scans=tmp_path / "empty", matrices=matrices,
            output_dir=tmp_path / "out")

    with pytest.raises(ToolInputError, match="No matrix found"):
        run(scans=scans, matrices=tmp_path / "empty",
            output_dir=tmp_path / "out")

    # A reference that is not there would otherwise resample every scan on its
    # own grid and report success, which is the wrong answer arriving quietly.
    with pytest.raises(ToolInputError, match="Reference volume not found"):
        run(scans=scans, matrices=matrices, output_dir=tmp_path / "out",
            reference=tmp_path / "nowhere.nii.gz")


def test_the_layout_describes_every_argument_the_panel_shows():
    """A hint naming a dead argument is a panel row that never appears."""
    import inspect

    from sadt_automatrix.layout import LAYOUT

    arguments = set(inspect.signature(run).parameters) - {"output_dir"}
    assert set(LAYOUT) == arguments
