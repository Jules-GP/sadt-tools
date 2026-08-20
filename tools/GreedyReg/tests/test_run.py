"""End-to-end tests for GreedyReg.

The registration ones really do run Greedy: the fixtures are synthetic volumes
translated by a known number of millimetres, so the test can assert that what
came back is the shift that went in. An import check would prove none of that.

The landmark ones plant ALI's output through a fake supervisor. What they test
is this tool's half -- which points it asks for, how it fits them, and that the
result is written by moving the image rather than resampling it.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from sadt_greedyreg import run
from sadt_greedyreg.errors import (
    RegistrationFailed,
    SupervisorRequired,
    ToolInputError,
)
from sadt_greedyreg.landmarks import REGION_LANDMARKS
from sadt_greedyreg.pipeline import MODES

SPACING = 0.5  # mm per voxel, so a 4-voxel shift is 2.0 mm

# Landmark positions for the fakes, in LPS. Deliberately NOT collinear: three or
# more points fix a rigid body only when they span the space, and a set on one
# line leaves a rotation about it free -- the SVD then returns some rotation
# rather than the translation that was applied, which is a real property of the
# fit and not a bug to assert around.
_SPREAD = [
    [0.0, 0.0, 0.0],
    [30.0, 2.0, 1.0],
    [1.0, 25.0, 2.0],
    [2.0, 1.0, 20.0],
    [15.0, 14.0, 13.0],
]


def _places(region):
    from sadt_greedyreg.landmarks import REGION_LANDMARKS as _regions
    return dict(zip(_regions[region], _SPREAD))


class FakeSup:
    """A supervisor, as a tool sees one. Records what it was asked for.

    `places` maps a landmark name to its LPS position for the fixed scan; the
    moving scan's are derived by applying `shift`, so the rigid fit has a known
    answer to find.
    """

    def __init__(self, tmp_path, places=None, shift=(0.0, 0.0, 0.0), drop=()):
        self.tmp = Path(tmp_path) / "sup"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.places = places or {}
        self.shift = shift
        self.drop = set(drop)
        self.calls = []

    def run(self, tool, **params):
        self.calls.append((tool, params))
        output = Path(params["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        moving = output.name.endswith("_moving")
        points = []
        for name, position in self.places.items():
            if name in self.drop and moving:
                continue
            if moving:
                position = [p + s for p, s in zip(position, self.shift)]
            points.append({"label": name, "position": list(position)})
        (output / "scan_lm_GreedyReg.mrk.json").write_text(json.dumps({
            "markups": [{"coordinateSystem": "LPS", "controlPoints": points}]
        }), encoding="utf-8")
        return output


def _volume(path, shift_voxels=0, seed=0):
    """A cube in noise, optionally rolled along X, saved as NIfTI."""
    rng = np.random.default_rng(seed)
    data = np.zeros((64, 64, 64), dtype=np.float32)
    data[20:44, 20:44, 20:44] = 100.0
    data += rng.normal(0, 1, data.shape).astype(np.float32)
    if shift_voxels:
        data = np.roll(data, shift=shift_voxels, axis=0)
    affine = np.diag([SPACING, SPACING, SPACING, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


@pytest.fixture
def pair(tmp_path):
    """One patient, T1 and T2 four voxels (2.0 mm) apart, in two folders."""
    _volume(tmp_path / "t1" / "MG01_T1.nii.gz", shift_voxels=0)
    _volume(tmp_path / "t2" / "MG01_T2.nii.gz", shift_voxels=4)
    return tmp_path / "t1", tmp_path / "t2"


def _warp_matrix(path):
    return np.array([[float(v) for v in line.split()]
                     for line in Path(path).read_text().splitlines() if line.strip()])


def test_registration_recovers_a_known_translation(pair, tmp_path):
    t1, t2 = pair
    out = run(t1=t1, t2=t2, output_dir=tmp_path / "out")

    registered = out / "MG01_registered.nii.gz"
    assert registered.exists() and registered.stat().st_size > 0
    assert (out / "MG01_warp.mat").exists()

    # 4 voxels of 0.5 mm. Greedy is an optimiser, not an oracle: the tolerance
    # is a tenth of a voxel, which is far tighter than the shift itself.
    translation = _warp_matrix(out / "MG01_warp.mat")[0, 3]
    assert abs(abs(translation) - 4 * SPACING) < 0.05, translation

    fixed = nib.load(str(t1 / "MG01_T1.nii.gz")).get_fdata()
    moving = nib.load(str(t2 / "MG01_T2.nii.gz")).get_fdata()
    after = nib.load(str(registered)).get_fdata()
    before_correlation = np.corrcoef(fixed.ravel(), moving.ravel())[0, 1]
    after_correlation = np.corrcoef(fixed.ravel(), after.ravel())[0, 1]
    assert after_correlation > 0.99 > before_correlation


def test_a_batch_registers_every_matched_pair(tmp_path):
    for key in ("MG01", "MG02"):
        _volume(tmp_path / "t1" / f"{key}_T1.nii.gz", seed=1)
        _volume(tmp_path / "t2" / f"{key}_T2.nii.gz", shift_voxels=2, seed=1)
    # An unmatched scan is ignored rather than failing the batch.
    _volume(tmp_path / "t1" / "MG09_T1.nii.gz", seed=2)

    out = run(t1=tmp_path / "t1", t2=tmp_path / "t2", output_dir=tmp_path / "out")

    report = json.loads((out / "GreedyReg_report.json").read_text())
    assert [case["patient"] for case in report["cases"]] == ["MG01", "MG02"]
    assert not (out / "MG09_registered.nii.gz").exists()


def test_two_files_are_one_pair(pair, tmp_path):
    t1, t2 = pair
    out = run(t1=t1 / "MG01_T1.nii.gz", t2=t2 / "MG01_T2.nii.gz",
              output_dir=tmp_path / "out")

    assert (out / "MG01_registered.nii.gz").exists()


def test_writes_nothing_outside_the_output_directory(pair, tmp_path):
    t1, t2 = pair
    before = sorted(p.name for p in t1.iterdir()), sorted(p.name for p in t2.iterdir())

    out = run(t1=t1, t2=t2, output_dir=tmp_path / "out")

    assert (sorted(p.name for p in t1.iterdir()),
            sorted(p.name for p in t2.iterdir())) == before
    assert {p.name for p in out.iterdir()} == {
        "MG01_registered.nii.gz", "MG01_warp.mat", "GreedyReg_report.json"}


def test_a_mask_is_binarised_and_scored_on(pair, tmp_path):
    """A multi-label mask is accepted: Greedy's -gm wants 0/1 and gets it."""
    t1, t2 = pair
    mask = np.zeros((64, 64, 64), dtype=np.float32)
    mask[18:46, 18:46, 18:46] = 7.0  # label 7, not 1
    (tmp_path / "masks").mkdir(exist_ok=True)
    nib.save(nib.Nifti1Image(mask, np.diag([SPACING] * 3 + [1.0])),
             str(tmp_path / "masks" / "MG01_mask.nii.gz"))

    out = run(t1=t1, t2=t2, output_dir=tmp_path / "out", masks=tmp_path / "masks")

    assert (out / "MG01_registered.nii.gz").exists()
    assert json.loads((out / "GreedyReg_report.json").read_text())["cases"][0]["mask"]


def test_affine_is_twelve_degrees_of_freedom(pair, tmp_path):
    t1, t2 = pair
    out = run(t1=t1, t2=t2, output_dir=tmp_path / "out", transform_type="Affine")

    matrix = _warp_matrix(out / "MG01_warp.mat")
    assert matrix.shape == (4, 4)
    assert (out / "MG01_registered.nii.gz").exists()


def test_landmark_mode_moves_the_image_without_resampling(pair, tmp_path):
    t1, t2 = pair
    sup = FakeSup(tmp_path, places=_places("MANDMASK"), shift=(3.0, 0.0, 0.0))

    out = run(t1=t1, t2=t2, output_dir=tmp_path / "out", mode="Landmark",
              landmark_model=tmp_path / "bundle", sup=sup)

    aligned = out / "MG01_t2_aligned.nii.gz"
    assert aligned.exists()
    # Voxels untouched, affine moved: that is the whole point of baking it in.
    original = nib.load(str(t2 / "MG01_T2.nii.gz"))
    moved = nib.load(str(aligned))
    assert np.array_equal(original.get_fdata(), moved.get_fdata())
    assert not np.allclose(original.affine, moved.affine)
    # The fit takes moving onto fixed, so a moving scan 3 mm further along
    # LPS x comes back as +3 mm on the RAS x of the affine.
    assert abs(moved.affine[0, 3] - original.affine[0, 3] - 3.0) < 1e-6

    asked, params = sup.calls[0]
    assert asked == "ALI_CBCT"
    assert params["landmarks"] == REGION_LANDMARKS["MANDMASK"]
    assert len(sup.calls) == 2, "one call per scan of the pair"


def test_landmark_mode_asks_only_for_the_region_it_needs(pair, tmp_path):
    t1, t2 = pair
    sup = FakeSup(tmp_path, places=_places("CBMASK"))

    run(t1=t1, t2=t2, output_dir=tmp_path / "out", mode="Landmark",
        region="CBMASK", landmark_model=tmp_path / "bundle", sup=sup)

    assert sup.calls[0][1]["landmarks"] == REGION_LANDMARKS["CBMASK"]
    assert "S" in sup.calls[0][1]["landmarks"]


def test_landmark_and_greedy_initialises_the_registration(pair, tmp_path):
    t1, t2 = pair
    sup = FakeSup(tmp_path, places=_places("MANDMASK"), shift=(1.0, 0.0, 0.0))

    out = run(t1=t1, t2=t2, output_dir=tmp_path / "out", mode="Landmark + Greedy",
              landmark_model=tmp_path / "bundle", sup=sup)

    assert (out / "MG01_registered.nii.gz").exists()
    case = json.loads((out / "GreedyReg_report.json").read_text())["cases"][0]
    assert case["init"].endswith("landmark_init.mat")
    assert case["landmarks_matched"] == REGION_LANDMARKS["MANDMASK"]


def test_too_few_matched_landmarks_is_refused_by_name(pair, tmp_path):
    t1, t2 = pair
    # Three of the five missing from the moving scan leaves two.
    sup = FakeSup(tmp_path, places=_places("MANDMASK"),
                  drop=REGION_LANDMARKS["MANDMASK"][:3])

    with pytest.raises(RegistrationFailed, match="only 2 landmark"):
        run(t1=t1, t2=t2, output_dir=tmp_path / "out", mode="Landmark",
            landmark_model=tmp_path / "bundle", sup=sup)


def test_landmark_mode_without_a_supervisor_says_what_to_use_instead(pair, tmp_path):
    t1, t2 = pair
    with pytest.raises(SupervisorRequired, match="Greedy' mode"):
        run(t1=t1, t2=t2, output_dir=tmp_path / "out", mode="Landmark",
            landmark_model=tmp_path / "bundle")


def test_landmark_mode_without_a_bundle_is_refused_before_anything_runs(pair, tmp_path):
    t1, t2 = pair
    with pytest.raises(ToolInputError, match="landmark_model"):
        run(t1=t1, t2=t2, output_dir=tmp_path / "out", mode="Landmark",
            sup=FakeSup(tmp_path))


def test_bad_input_is_reported_as_such(tmp_path, pair):
    t1, t2 = pair
    with pytest.raises(ToolInputError, match="does not exist"):
        run(t1=tmp_path / "missing", t2=t2, output_dir=tmp_path / "out")

    with pytest.raises(ToolInputError, match="Unknown mode"):
        run(t1=t1, t2=t2, output_dir=tmp_path / "out", mode="Elastix")

    with pytest.raises(ToolInputError, match="Unknown metric"):
        run(t1=t1, t2=t2, output_dir=tmp_path / "out", metric="MI")

    with pytest.raises(ToolInputError, match="both be folders"):
        run(t1=t1, t2=t2 / "MG01_T2.nii.gz", output_dir=tmp_path / "out")


def test_no_matching_pair_names_what_it_found(tmp_path):
    _volume(tmp_path / "t1" / "MG01_T1.nii.gz")
    _volume(tmp_path / "t2" / "XX02_T2.nii.gz")

    with pytest.raises(ToolInputError, match="No matching T1/T2 pair"):
        run(t1=tmp_path / "t1", t2=tmp_path / "t2", output_dir=tmp_path / "out")


def test_the_published_regions_are_the_implemented_ones():
    """The Literal in run() is a second declaration of REGION_LANDMARKS."""
    import inspect
    import typing

    published = typing.get_args(
        typing.get_type_hints(run)["region"]
    )
    assert set(published) == set(REGION_LANDMARKS)
    assert set(typing.get_args(typing.get_type_hints(run)["mode"])) == set(MODES)
    assert "sup" in inspect.signature(run).parameters


def test_the_two_spellings_of_the_landmark_tool_agree():
    """`describe.py` reads the literal at each call site to publish `calls`.

    That is why the name appears both as `tools.LANDMARK_TOOL` (what `sup.run`
    is given) and as a literal in pipeline.py (what `require` is given). Two
    spellings can drift; this is what stops them.
    """
    from pathlib import Path as _Path

    from sadt_greedyreg import pipeline, tools

    source = _Path(pipeline.__file__).read_text(encoding="utf-8")
    assert '"{}"'.format(tools.LANDMARK_TOOL) in source
