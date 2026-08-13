"""Unit tests for the ALI tool.

No GPU, no real model weights and no network inference: the agent that walks
the volume is stubbed, so everything AROUND it -- mode detection, DICOM
recognition, input discovery, weight discovery, the landmark vocabulary,
output naming and tree preservation, the run report, and every cross-argument
rule -- is exercised for real.

The parts that genuinely cannot run here are skipped rather than faked: the
IOS engine needs pytorch3d, which publishes no usable wheel and is compiled
from source behind the `ios` extra. What does not need it (the model naming
rule, the tooth vocabulary, the network selection, the tooth-label check) is
tested regardless, since that is where the original's defects were.

    cd tools/ALI && uv run pytest
"""

import json
import os
import typing

import numpy as np
import pytest
import SimpleITK as sitk

from sadt_ali import dispatch, markups, run
from sadt_ali.cbct import catalog as cbct_catalog
from sadt_ali.errors import ToolInputError, ToolUnavailableError
from sadt_ali.ios import catalog as ios_catalog
from sadt_ali.ios import engine as ios_engine

ALL_REGIONS = list(cbct_catalog.REGION_NAMES)
CRANIAL_BASE_ONLY = ["Cranial base"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write_volume(path, size=(16, 16, 16)):
    """A small synthetic CBCT with a bright cube in it."""
    array = np.zeros(size[::-1], dtype=np.int16)
    array[4:12, 4:12, 4:12] = 800
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 0.5))
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    sitk.WriteImage(image, str(path))
    return str(path)


def write_surface(path, labelled=True):
    """A minimal .vtk polydata, optionally carrying a tooth-label array."""
    vtk = pytest.importorskip("vtk")

    points = vtk.vtkPoints()
    for coordinates in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)):
        points.InsertNextPoint(*coordinates)

    polys = vtk.vtkCellArray()
    for triangle in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)):
        polys.InsertNextCell(3)
        for point_id in triangle:
            polys.InsertCellPoint(point_id)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)

    if labelled:
        labels = vtk.vtkIntArray()
        labels.SetName("Universal_ID")
        for value in (8, 8, 8, 8):
            labels.InsertNextValue(value)
        surface.GetPointData().AddArray(labels)

    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    writer.Write()
    return str(path)


def write_cbct_bundle(root, landmarks_by_region):
    """A CBCT bundle in the <region>/<landmark>/<scale>/*.pth layout."""
    for region, labels in landmarks_by_region.items():
        for label in labels:
            for scale in cbct_catalog.SCALE_KEYS:
                folder = root / region / label / scale
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{label}_Net_{scale}.pth").write_bytes(b"fake checkpoint")
    return str(root)


def write_ios_bundle(root, names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"fake checkpoint")
    return str(root)


@pytest.fixture
def stub_agent(monkeypatch):
    """Replace the deep-RL search with a deterministic voxel position.

    Every landmark whose name starts with "X" is reported as not found, so the
    report's two failure kinds can be told apart in a test.
    """
    from sadt_ali.cbct import agent as agent_module
    from sadt_ali.cbct import engine as cbct_engine

    class StubBrain:
        def __init__(self, scale_keys, device, out_channels=6):
            self.scale_keys = scale_keys

        def load(self, weights_per_scale):
            assert set(weights_per_scale) == set(self.scale_keys)

        def release(self):
            pass

    class StubAgent:
        def __init__(self, target, scale_keys, brain, environment, **_kwargs):
            self.target = target
            self.environment = environment

        def search(self, max_seconds):
            if self.target.startswith("X"):
                raise agent_module.NotFound("stubbed failure")
            return np.array([3.0, 4.0, 5.0])

    monkeypatch.setattr(cbct_engine, "Brain", StubBrain)
    monkeypatch.setattr(cbct_engine, "Agent", StubAgent)
    monkeypatch.setattr(cbct_engine, "resolve_device", lambda requested=None: "cpu")


@pytest.fixture
def cbct_environment():
    pytest.importorskip("monai")
    pytest.importorskip("itk")
    pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# The landmark vocabulary
# ---------------------------------------------------------------------------

def test_impacted_canine_spellings_both_resolve():
    """The defect that lost a whole patient's landmarks.

    The Slicer UI called them UR3OI/UL3OI/UR3RI/UL3RI and the CLI's tables
    called them UR3OIP/.../UL3RIP. `LABEL_GROUPS[landmark]` was then indexed
    with no guard inside the save loop, so one unknown name raised a KeyError
    that was caught far above -- and NOTHING was written for that scan,
    including every landmark that had been found correctly.
    """
    for ui_name, canonical_name in (
        ("UR3OI", "UR3OIP"),
        ("UL3OI", "UL3OIP"),
        ("UR3RI", "UR3RIP"),
        ("UL3RI", "UL3RIP"),
    ):
        assert cbct_catalog.canonical(ui_name) == canonical_name
        assert cbct_catalog.group_of(ui_name) == "CI"
        assert cbct_catalog.group_of(canonical_name) == "CI"


def test_group_of_never_raises_on_an_unknown_name():
    assert cbct_catalog.group_of("NoSuchLandmark") == cbct_catalog.UNGROUPED


def test_scale_keys_match_the_shipped_weight_folders():
    assert cbct_catalog.SCALE_KEYS == ("1", "0-3")


def test_region_codes_from_a_selection():
    assert cbct_catalog.region_codes(["Cranial base", "Lower"]) == ("CB", "L")
    # The codes are accepted too, so a caller driving this package directly
    # need not know the display spellings.
    assert cbct_catalog.region_codes(["CB", "L"]) == ("CB", "L")
    # An omitted optional argument must fall back to every region, not to none.
    assert cbct_catalog.region_codes(None) == cbct_catalog.REGION_CODES
    # Declaration order, not the caller's.
    assert cbct_catalog.region_codes(["Lower", "Cranial base"]) == ("CB", "L")


def test_an_unknown_region_is_refused_not_dropped():
    """`Literal` is published, not enforced -- the runner calls run(**params)
    from a JSON object. A stale client naming a region that no longer exists
    has to be told, not handed a narrower run than it asked for."""
    with pytest.raises(ValueError, match="Cranial base"):
        cbct_catalog.region_codes(["Cranial base", "Sagittal"])


def test_ios_offers_only_landmark_types_a_model_predicts():
    """R, RIP and OIP were selectable in the Slicer UI and predicted by
    nothing: no network produced them and no label table contained them.
    Ticking them did literally nothing."""
    offered = {lm_type for types in ios_catalog.NETWORKS.values() for lm_type in types}
    assert offered == {"O", "MB", "DB", "CL", "CB"}
    assert not offered & {"R", "RIP", "OIP"}


def test_ios_tooth_numbering_matches_the_shipped_label_tables():
    assert ios_catalog.UNIVERSAL_NUMBERS["Upper"]["UL7"] == 15
    assert ios_catalog.UNIVERSAL_NUMBERS["Upper"]["UR7"] == 2
    assert ios_catalog.UNIVERSAL_NUMBERS["Lower"]["LL7"] == 18
    assert ios_catalog.UNIVERSAL_NUMBERS["Lower"]["LR7"] == 31
    # Tooth 8 is UR1: the occlusal network's three channels, in channel order.
    assert ios_catalog.LABELS["O"]["8"] == ["UR1O", "UR1MB", "UR1DB"]
    assert ios_catalog.LABELS["C"]["8"] == ["UR1CL", "UR1CB"]


def test_ios_network_codes_from_a_selection():
    assert ios_catalog.network_codes(["Occlusal"]) == ("O",)
    assert ios_catalog.network_codes(["O"]) == ("O",)
    assert ios_catalog.network_codes(None) == ios_catalog.NETWORK_CODES
    with pytest.raises(ValueError, match="Occlusal"):
        ios_catalog.network_codes(["Buccal"])


# ---------------------------------------------------------------------------
# The published options and the catalogs cannot drift apart
# ---------------------------------------------------------------------------

def _choices(argument):
    """The `Literal` options `run()` publishes for one argument."""
    hint = typing.get_type_hints(run)[argument]
    if typing.get_origin(hint) is list:
        hint = typing.get_args(hint)[0]
    return list(typing.get_args(hint))


def test_the_published_regions_are_the_catalogs_own():
    """`Literal` takes literals only, so it cannot be built from the catalog.
    That makes the signature a second declaration of the same set, and this is
    what keeps the two honest: a region added to one and not the other would be
    unselectable from the client, or offered and then refused."""
    assert _choices("cbct_regions") == list(cbct_catalog.REGION_NAMES)


def test_the_published_landmarks_are_the_catalogs_own():
    assert _choices("landmarks") == list(cbct_catalog.LABELS)


def test_the_published_networks_are_the_catalogs_own():
    assert _choices("ios_networks") == list(ios_catalog.NETWORK_NAMES)


def test_every_published_default_is_one_of_its_own_options():
    """A default outside its option list gives the client a picker that cannot
    produce the value the tool starts from."""
    import inspect

    for argument in ("cbct_regions", "ios_networks", "device"):
        default = inspect.signature(run).parameters[argument].default
        options = _choices(argument)
        for value in (default if isinstance(default, list) else [default]):
            assert value in options, (argument, value)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def test_a_single_volume_is_cbct(tmp_path):
    scan = write_volume(tmp_path / "in" / "patient01.nii.gz")
    detected = dispatch.detect(scan, str(tmp_path / "work"))
    assert detected.mode == dispatch.CBCT
    assert detected.scans == [(scan, "patient01.nii.gz")]


def test_a_single_surface_is_ios(tmp_path):
    mesh = write_surface(tmp_path / "in" / "arch.vtk")
    detected = dispatch.detect(mesh, str(tmp_path / "work"))
    assert detected.mode == dispatch.IOS


def test_a_folder_of_volumes_is_cbct_and_keeps_its_tree(tmp_path):
    """A folder says nothing about which engine applies -- which is exactly why
    there is no `mode` argument for the caller to get wrong.

    The server unpacks a `.zip` before `run()` is called, so what arrives here
    is always a real directory; this is the shape it arrives in.
    """
    write_volume(tmp_path / "cohort" / "siteA" / "patient01.nii.gz")
    write_volume(tmp_path / "cohort" / "siteB" / "patient01.nii.gz")

    detected = dispatch.detect(str(tmp_path / "cohort"), str(tmp_path / "work"))
    assert detected.mode == dispatch.CBCT
    # Two patients with the SAME base name in different folders. Keyed by
    # relative path, they stay distinct; the original keyed by file.name and
    # one silently replaced the other.
    assert sorted(key for _path, key in detected.scans) == [
        os.path.join("siteA", "patient01.nii.gz"),
        os.path.join("siteB", "patient01.nii.gz"),
    ]


def test_a_mixed_input_is_refused_rather_than_guessed(tmp_path):
    write_volume(tmp_path / "mixed" / "patient01.nii.gz")
    write_surface(tmp_path / "mixed" / "arch.vtk")

    with pytest.raises(ToolInputError) as raised:
        dispatch.detect(str(tmp_path / "mixed"), str(tmp_path / "work"))
    assert "mixes" in str(raised.value)


def test_an_input_with_nothing_recognizable_says_what_it_wanted(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("nothing here")

    with pytest.raises(ToolInputError) as raised:
        dispatch.detect(str(tmp_path / "empty"), str(tmp_path / "work"))
    assert ".vtk" in str(raised.value) and ".nii.gz" in str(raised.value)


def test_detection_is_recursive(tmp_path):
    write_volume(tmp_path / "deep" / "a" / "b" / "c" / "patient.nii.gz")
    detected = dispatch.detect(str(tmp_path / "deep"), str(tmp_path / "work"))
    assert len(detected.scans) == 1


def test_the_working_directory_is_not_rediscovered_as_input(tmp_path):
    """`output_dir` may legitimately be the input folder, and `.ali_work/`
    lives inside it. Its converted DICOM must not come back round as scans."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    stale = tmp_path / "cohort" / dispatch.WORK_DIRNAME / "dicom_converted"
    write_volume(stale / "leftover.nii.gz")

    detected = dispatch.detect(str(tmp_path / "cohort"), str(tmp_path / "work"))
    assert [key for _path, key in detected.scans] == ["patient01.nii.gz"]


def test_a_folder_of_dicom_is_not_mistaken_for_an_empty_input(tmp_path, monkeypatch):
    """DICOM slices carry no extension, so only GDCM can recognize them --
    which is why the client offers a folder picker and why detection probes."""
    from sadt_ali.cbct import preprocess

    series = tmp_path / "cohort" / "patient01"
    series.mkdir(parents=True)
    for index in range(3):
        (series / f"IM{index:06d}").write_bytes(b"not really dicom")

    monkeypatch.setattr(preprocess, "is_dicom_series", lambda directory: True)
    monkeypatch.setattr(
        preprocess,
        "convert_dicom_series",
        lambda directory, destination: write_volume(destination),
    )

    detected = dispatch.detect(str(tmp_path / "cohort"), str(tmp_path / "work"))
    assert detected.mode == dispatch.CBCT
    assert detected.converted_dicom == 1
    # Converted into the WORKING dir, never into the user's own folder -- the
    # original wrote <input>/NIFTI/ and then re-ingested it on the next run.
    converted_path = detected.scans[0][0]
    assert str(tmp_path / "work") in converted_path
    assert not (tmp_path / "cohort" / "NIFTI").exists()


def test_a_folder_already_holding_volumes_is_not_probed_for_dicom(tmp_path, monkeypatch):
    from sadt_ali.cbct import preprocess

    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    probed = []
    monkeypatch.setattr(
        preprocess, "is_dicom_series", lambda directory: probed.append(directory) or False
    )

    dispatch.detect(str(tmp_path / "cohort"), str(tmp_path / "work"))
    assert probed == []


# ---------------------------------------------------------------------------
# CBCT model bundles
# ---------------------------------------------------------------------------

def test_weight_discovery_reads_the_folder_tree(tmp_path):
    from sadt_ali.cbct import engine as cbct_engine

    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba", "S"]})
    weights = cbct_engine.discover_weights(bundle)

    assert sorted(weights) == ["Ba", "S"]
    assert sorted(weights["Ba"]) == sorted(cbct_catalog.SCALE_KEYS)


def test_a_landmark_missing_one_scale_is_not_offered(tmp_path):
    """Half a bundle must be reported up front, not fail mid-run: the agent
    walks the coarse scale and then the fine one, and needs both."""
    from sadt_ali.cbct import engine as cbct_engine

    folder = tmp_path / "bundle" / "Ba" / "1"
    folder.mkdir(parents=True)
    (folder / "Ba_Net_1.pth").write_bytes(b"fake")

    assert cbct_engine.discover_weights(str(tmp_path / "bundle")) == {}


def test_requested_landmarks_separates_missing_from_unknown(tmp_path):
    from sadt_ali.cbct import engine as cbct_engine

    weights = {"Ba": {}, "S": {}, "Mystery": {}}
    runnable, without_model, ungrouped = cbct_engine.requested_landmarks(weights, ["CB"])

    assert runnable == ("Ba", "S")
    # Every other cranial-base landmark the catalog knows about: the client
    # renders this as "not in the selected model bundle", i.e. "use another".
    assert "N" in without_model and "Ba" not in without_model
    # Weights for a landmark this catalog has never heard of are surfaced,
    # not silently ignored.
    assert ungrouped == ["Mystery"]


def test_a_bundle_using_the_ui_spelling_resolves(tmp_path):
    from sadt_ali.cbct import engine as cbct_engine

    bundle = write_cbct_bundle(tmp_path / "bundle", {"Canine": ["UR3OI"]})
    weights = cbct_engine.discover_weights(bundle)

    assert list(weights) == ["UR3OIP"]
    runnable, _missing, ungrouped = cbct_engine.requested_landmarks(weights, ["CI"])
    assert runnable == ("UR3OIP",)
    assert ungrouped == []


def test_a_bundle_of_the_wrong_kind_is_an_input_error(tmp_path, cbct_environment):
    """Naming the IOS bundle for a CBCT run (or vice versa) has to answer with
    a message the client shows verbatim, not a stack trace in a log."""
    from sadt_ali.cbct import engine as cbct_engine

    ios_bundle = write_ios_bundle(tmp_path / "ios", ["Upper_O_model.pth"])
    with pytest.raises(ToolInputError, match="No CBCT landmark weights"):
        cbct_engine.predict_landmarks(
            scans=[],
            model_path=ios_bundle,
            regions=None,
            prediction_ID="Pred",
            output_dir=str(tmp_path / "out"),
            work_dir=str(tmp_path / "work"),
        )


# ---------------------------------------------------------------------------
# IOS model bundles and the tooth-label precondition
# ---------------------------------------------------------------------------

def test_ios_weight_discovery_reads_the_published_names(tmp_path):
    """The real ALIDDM bundle: Upper_O_model.pth, Lower_C_model.pth, ..."""
    bundle = write_ios_bundle(
        tmp_path / "bundle",
        ["Upper_O_model.pth", "Lower_O_model.pth", "Upper_C_model.pth", "Lower_C_model.pth"],
    )
    weights, unrecognized = ios_engine.discover_weights(bundle)

    assert weights["O"].keys() == {"Upper", "Lower"}
    assert weights["C"].keys() == {"Upper", "Lower"}
    assert unrecognized == []


def test_an_ios_checkpoint_with_no_jaw_token_is_reported_not_assumed_upper(tmp_path):
    """The original treated every file not containing "Lower" as upper-jaw
    weights, so a bundle missing its mandibular model quietly predicted the
    lower arch with the maxillary one."""
    bundle = write_ios_bundle(tmp_path / "bundle", ["model_O.pth", "Upper_C_model.pth"])
    weights, unrecognized = ios_engine.discover_weights(bundle)

    assert unrecognized == ["model_O.pth"]
    assert "O" not in weights
    assert weights["C"] == {"Upper": str(tmp_path / "bundle" / "Upper_C_model.pth")}


def test_a_mesh_without_tooth_labels_names_the_tool_that_makes_them(tmp_path):
    """The handoff `ALILogic.ensure_segmented()` used to make in-process.

    Tools do not call each other any more, so this cannot segment the mesh
    itself -- but it can say exactly what to run, which is the difference
    between a fixable request and "no known tooth number is present".
    """
    labelled = write_surface(tmp_path / "in" / "good.vtk", labelled=True)
    raw = write_surface(tmp_path / "in" / "raw.vtk", labelled=False)

    with pytest.raises(ToolInputError) as raised:
        ios_engine.require_labels([(labelled, "good.vtk"), (raw, "raw.vtk")])

    message = str(raised.value)
    assert "Crown_Seg" in message
    assert "1 of 2" in message
    # The array names it looked for, so the fix is actionable without reading
    # the source.
    assert "Universal_ID" in message


def test_a_fully_labelled_batch_passes_the_check(tmp_path):
    mesh = write_surface(tmp_path / "in" / "arch.vtk", labelled=True)
    assert ios_engine.require_labels([(mesh, "arch.vtk")]) is None


# ---------------------------------------------------------------------------
# The markups writer
# ---------------------------------------------------------------------------

def test_markups_are_written_as_a_slicer_file(tmp_path):
    path = markups.write(
        {"Ba": np.array([1.5, 2.5, 3.5]), "S": np.array([4.0, 5.0, 6.0])},
        str(tmp_path / "out" / f"scan_lm_Pred{markups.MARKUPS_EXTENSION}"),
    )
    assert path.endswith(".mrk.json")

    content = json.loads(open(path, encoding="utf-8").read())
    node = content["markups"][0]
    assert node["coordinateSystem"] == "LPS"
    assert [point["label"] for point in node["controlPoints"]] == ["Ba", "S"]
    # numpy scalars are cast to float on the way in: json.dump cannot
    # serialize them, and the failure would land after all the inference.
    assert node["controlPoints"][0]["position"] == [1.5, 2.5, 3.5]


def test_both_engines_write_the_same_extension():
    """CBCT wrote .mrk.json and IOS wrote .json for byte-identical content,
    and Slicer only associates the first with a markups node."""
    assert markups.MARKUPS_EXTENSION == ".mrk.json"


def test_the_display_node_is_visible(tmp_path):
    """Regression, and the nastiest kind: the file is perfectly valid, Slicer
    loads it, the node appears in the Markups module -- and nothing is drawn.

    Both original CLIs wrote `display.visibility: false`. That is independent
    of each control point's own `visibility`, so the points below are visible
    inside a node that is not displayed. Dropping a result file into Slicer
    showed an empty scene.
    """
    path = markups.write({"Ba": (1.0, 2.0, 3.0)}, str(tmp_path / "scan.mrk.json"))
    node = json.loads(open(path, encoding="utf-8").read())["markups"][0]

    assert node["display"]["visibility"] is True
    assert node["controlPoints"][0]["visibility"] is True
    # Off by design, and not the same kind of statement as the two above: a
    # point shows on the slice it is on, nothing more. Projecting 113 of them
    # onto neighbouring slices was tried and crowds the view used to judge
    # placement. It stays one checkbox away in the Markups module.
    assert node["display"]["sliceProjection"] is False


def test_positions_are_plain_json_floats(tmp_path):
    """numpy scalars are not JSON-serializable, and both engines produce
    coordinates as numpy arrays. The cast has to happen in the writer."""
    path = markups.write(
        {"Ba": np.array([1.5, 2.5, 3.5], dtype=np.float32)},
        str(tmp_path / "scan.mrk.json"),
    )
    position = json.loads(open(path, encoding="utf-8").read())[
        "markups"
    ][0]["controlPoints"][0]["position"]

    assert all(type(value) is float for value in position)


# ---------------------------------------------------------------------------
# End to end through run(), agent stubbed
# ---------------------------------------------------------------------------

def test_a_cbct_run_writes_one_file_per_scan(tmp_path, stub_agent, cbct_environment):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba", "S", "N"]})

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        cbct_regions=CRANIAL_BASE_ONLY,
    )

    report = json.loads((output_dir / dispatch.REPORT_NAME).read_text())
    assert report["mode"] == "CBCT"
    assert report["summary"] == {"total": 1, "processed": 1, "failed": 0}

    produced = sorted(output_dir.rglob("*.mrk.json"))
    # ONE file holding every region, not one per anatomical group -- which is
    # what every downstream tool (ASO, AREG, AutoMatrix) had to recombine.
    assert len(produced) == 1
    assert produced[0].name == "patient01_lm_Pred.mrk.json"

    content = json.loads(produced[0].read_text())
    assert sorted(point["label"] for point in content["markups"][0]["controlPoints"]) == [
        "Ba", "N", "S"
    ]


def test_run_returns_the_output_directory(tmp_path, stub_agent, cbct_environment):
    """The contract describe.py publishes as `"returns": "path"`."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    returned = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        cbct_regions=CRANIAL_BASE_ONLY,
    )
    assert returned == tmp_path / "out"
    assert returned.is_dir()


def test_nothing_is_written_beside_the_input(tmp_path, stub_agent, cbct_environment):
    """Everything goes under `output_dir`, which the caller owns."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})
    before = sorted(p for p in (tmp_path / "cohort").rglob("*") if p.is_file())

    run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        cbct_regions=CRANIAL_BASE_ONLY,
    )

    assert sorted(p for p in (tmp_path / "cohort").rglob("*") if p.is_file()) == before


def test_the_working_directory_does_not_survive_the_run(tmp_path, stub_agent, cbct_environment):
    """A leftover `.ali_work/` means a run crashed, so it has to be reliable."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        cbct_regions=CRANIAL_BASE_ONLY,
    )
    assert not (output_dir / dispatch.WORK_DIRNAME).exists()


def test_the_working_directory_is_removed_even_when_the_run_fails(tmp_path, cbct_environment):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    empty_bundle = tmp_path / "bundle"
    empty_bundle.mkdir()

    with pytest.raises(ToolInputError):
        run(
            input=tmp_path / "cohort",
            model=empty_bundle,
            output_dir=tmp_path / "out",
            cbct_regions=CRANIAL_BASE_ONLY,
        )
    assert not (tmp_path / "out" / dispatch.WORK_DIRNAME).exists()


def test_a_batch_keeps_its_tree_so_homonyms_cannot_collide(
    tmp_path, stub_agent, cbct_environment
):
    write_volume(tmp_path / "cohort" / "siteA" / "patient01.nii.gz")
    write_volume(tmp_path / "cohort" / "siteB" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        cbct_regions=CRANIAL_BASE_ONLY,
    )

    produced = sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*.mrk.json"))
    assert produced == [
        os.path.join("siteA", "patient01_lm_Pred.mrk.json"),
        os.path.join("siteB", "patient01_lm_Pred.mrk.json"),
    ]


def test_the_report_tells_a_missing_model_from_a_failed_search(
    tmp_path, stub_agent, cbct_environment
):
    """The two failures look identical in the Slicer scene -- a landmark that
    is simply not there -- and need opposite fixes: another bundle, or another
    scan. The report is the only place they are distinguishable."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    # "XBa" is not in the catalog but IS in the bundle, so it is reported as
    # ungrouped and never run; "S" is in the catalog but not in the bundle.
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})
    for scale in cbct_catalog.SCALE_KEYS:
        folder = tmp_path / "bundle" / "Cranial_Base" / "XBa" / scale
        folder.mkdir(parents=True)
        (folder / "x.pth").write_bytes(b"fake")

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        cbct_regions=CRANIAL_BASE_ONLY,
    )
    report = json.loads((output_dir / dispatch.REPORT_NAME).read_text())

    scan = report["scans"]["patient01.nii.gz"]
    assert scan["landmarks_found"] == ["Ba"]
    assert "S" in report["landmarks_without_model"]
    assert report["landmarks_ungrouped"] == ["XBa"]
    assert scan["landmarks_failed"] == {}


def test_a_landmark_that_never_converges_does_not_cost_the_others(
    tmp_path, stub_agent, cbct_environment
):
    """The unguarded `LABEL_GROUPS[landmark]` lookup meant one bad landmark
    lost the whole patient, including the ones already found."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})
    # The stub fails any landmark starting with "X". "XN" is aliased into the
    # catalog below so it is genuinely requested and genuinely runs.
    for scale in cbct_catalog.SCALE_KEYS:
        folder = tmp_path / "bundle" / "Cranial_Base" / "XN" / scale
        folder.mkdir(parents=True)
        (folder / "x.pth").write_bytes(b"fake")
    cbct_catalog.LABEL_GROUPS["XN"] = "CB"
    try:
        output_dir = run(
            input=tmp_path / "cohort",
            model=bundle,
            output_dir=tmp_path / "out",
            cbct_regions=CRANIAL_BASE_ONLY,
        )
    finally:
        del cbct_catalog.LABEL_GROUPS["XN"]

    report = json.loads((output_dir / dispatch.REPORT_NAME).read_text())
    scan = report["scans"]["patient01.nii.gz"]
    assert scan["status"] == "ok"
    assert scan["landmarks_found"] == ["Ba"]
    assert "XN" in scan["landmarks_failed"]
    assert scan["files"]


def test_the_run_report_lands_beside_the_results(tmp_path, stub_agent, cbct_environment):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    output_dir = run(
        input=tmp_path / "cohort", model=bundle, output_dir=tmp_path / "out"
    )

    # The Slicer module reads exactly this file, by this name.
    on_disk = output_dir / dispatch.REPORT_NAME
    assert on_disk.is_file()
    written = json.loads(on_disk.read_text())
    assert written["mode"] == "CBCT"
    assert written["regions"]
    # So a result says which weights produced it.
    assert written["model_bundle"] == "bundle"


def test_prediction_id_reaches_the_file_names(tmp_path, stub_agent, cbct_environment):
    """`SaveId` existed in the Slicer UI and was read by nothing; the suffix
    was hardcoded in both CLIs."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        prediction_ID="T1",
    )
    assert [p.name for p in output_dir.rglob("*.mrk.json")] == ["patient01_lm_T1.mrk.json"]


# ---------------------------------------------------------------------------
# `search_seconds` -- the setting that became an argument
# ---------------------------------------------------------------------------

def test_the_search_budget_defaults_per_device():
    """0 means "not specified": there is no nullable type in the schema, so the
    argument cannot default to None the way the setting it replaces did."""
    from sadt_ali.cbct import engine as cbct_engine

    assert cbct_engine.search_budget("cuda", 0.0) == 15.0
    # CPU inference needs several times longer to reach the same place.
    assert cbct_engine.search_budget("cpu", 0.0) == 60.0
    assert cbct_engine.search_budget("cuda", 2.5) == 2.5


# ---------------------------------------------------------------------------
# A missing dependency belongs to the venv, not to a scan
# ---------------------------------------------------------------------------

def test_a_missing_dependency_fails_before_any_scan_is_touched(tmp_path, monkeypatch):
    """Regression: `itk` absent produced one identical failure PER SCAN -- each
    only after a full histogram correction -- and the run then ended on
    "produced no landmarks for any scan", which buried the one line saying what
    to install. It is a property of the venv; it has to be raised once, before
    the loop."""
    from sadt_ali.cbct import engine as cbct_engine
    from sadt_ali.cbct import preprocess

    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    write_volume(tmp_path / "cohort" / "patient02.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    corrected = []
    monkeypatch.setattr(
        preprocess, "import_itk",
        lambda: (_ for _ in ()).throw(ToolUnavailableError("needs itk (missing: itk)")),
    )
    monkeypatch.setattr(
        preprocess, "correct_histogram",
        lambda *args, **kwargs: corrected.append(args) or args[1],
    )

    with pytest.raises(ToolUnavailableError) as raised:
        cbct_engine.predict_landmarks(
            scans=[(str(tmp_path / "cohort" / "patient01.nii.gz"), "patient01.nii.gz"),
                   (str(tmp_path / "cohort" / "patient02.nii.gz"), "patient02.nii.gz")],
            model_path=bundle,
            regions=("CB",),
            output_dir=str(tmp_path / "out"),
            work_dir=str(tmp_path / "work"),
        )

    # The install message itself, not a summary that hides it.
    assert "missing: itk" in str(raised.value)
    assert "produced no landmarks" not in str(raised.value)
    # And not one scan was preprocessed before finding out.
    assert corrected == []


# ---------------------------------------------------------------------------
# Cross-argument rules the schema cannot express
# ---------------------------------------------------------------------------

def test_an_empty_cbct_selection_on_cbct_input_names_the_argument(tmp_path):
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")

    with pytest.raises(ToolInputError) as raised:
        run(
            input=tmp_path / "cohort",
            model=tmp_path,
            output_dir=tmp_path / "out",
            cbct_regions=[],
        )
    message = str(raised.value)
    assert "CBCT" in message and "cbct_regions" in message
    # The message lists what to tick, so a mode mismatch explains itself.
    for name in cbct_catalog.REGION_NAMES:
        assert name in message


def test_an_empty_ios_selection_on_ios_input_names_the_argument(tmp_path):
    write_surface(tmp_path / "cohort" / "arch.vtk")

    with pytest.raises(ToolInputError) as raised:
        run(
            input=tmp_path / "cohort",
            model=tmp_path,
            output_dir=tmp_path / "out",
            ios_networks=[],
        )
    message = str(raised.value)
    assert "ios_networks" in message and "Occlusal" in message


def test_the_inactive_modes_empty_selection_is_ignored(tmp_path, stub_agent, cbct_environment):
    """Both groups are always rendered by the client and one is always inert.
    Emptying the inert one must not fail the run."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(tmp_path / "bundle", {"Cranial_Base": ["Ba"]})

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        ios_networks=[],
    )
    report = json.loads((output_dir / dispatch.REPORT_NAME).read_text())
    assert report["mode"] == "CBCT"


# ---------------------------------------------------------------------------
# `landmarks` -- asking for named points instead of whole regions
# ---------------------------------------------------------------------------

# What ASO's fully-automated CBCT mode registers on. Straddles two regions,
# which is the whole reason this argument exists.
ASO_LANDMARKS = ("Ba", "S", "N", "RPo", "LPo", "ROr", "LOr")


def _weights_for(labels):
    """A discover_weights() result: every label with both scales present."""
    return {label: {scale: f"/w/{label}/{scale}.pth" for scale in cbct_catalog.SCALE_KEYS}
            for label in labels}


def test_naming_landmarks_replaces_the_region_selection():
    """The point of the argument: 7 agents, not 58.

    ASO's seven points span Cranial base and Upper. Asking by region would run
    every landmark of both (58) to use seven, and one agent is a full two-scale
    walk of the volume.
    """
    from sadt_ali.cbct import engine

    weights = _weights_for(cbct_catalog.LABELS)

    by_region, _missing, _ungrouped = engine.requested_landmarks(
        weights, regions=("CB", "U")
    )
    by_name, missing, _ungrouped = engine.requested_landmarks(
        weights, regions=cbct_catalog.REGION_CODES, landmarks=ASO_LANDMARKS
    )

    assert set(by_name) == set(ASO_LANDMARKS)
    assert not missing
    # The regions were left at their all-on default and still did not widen it.
    assert len(by_region) > 8 * len(by_name)


def test_an_empty_landmark_selection_leaves_the_regions_in_charge():
    """The ordinary case, and the default: an empty list means "not specified"
    and hands the choice back to `cbct_regions`."""
    from sadt_ali.cbct import engine

    weights = _weights_for(cbct_catalog.LABELS)

    without = engine.requested_landmarks(weights, regions=("CB",))
    with_empty = engine.requested_landmarks(weights, regions=("CB",), landmarks=())

    assert without == with_empty
    assert set(without[0]) == set(cbct_catalog.GROUP_LABELS["CB"])


def test_a_named_landmark_the_bundle_lacks_is_reported_not_dropped():
    """Same contract as the region path: "use another bundle" has to be
    distinguishable from "this scan is hard"."""
    from sadt_ali.cbct import engine

    weights = _weights_for(("Ba", "S", "N"))

    runnable, without_model, _ungrouped = engine.requested_landmarks(
        weights, regions=cbct_catalog.REGION_CODES, landmarks=ASO_LANDMARKS
    )

    assert set(runnable) == {"Ba", "S", "N"}
    assert set(without_model) == {"RPo", "LPo", "ROr", "LOr"}


def test_landmark_names_accepts_every_shape_a_caller_uses():
    assert cbct_catalog.landmark_names(["N", "Ba"]) == ("Ba", "N")  # declaration order
    assert cbct_catalog.landmark_names(None) == ()
    assert cbct_catalog.landmark_names([]) == ()
    # Aliases resolve to the spelling the weights are packaged under.
    assert cbct_catalog.landmark_names(["UR3OI"]) == ("UR3OIP",)
    # A name outside the catalog is kept, not refused: the engine runs it if
    # the bundle has weights for it, and reports it as ungrouped otherwise.
    assert cbct_catalog.landmark_names(["Ba", "Mystery"]) == ("Ba", "Mystery")


def test_only_the_named_landmarks_run(tmp_path, stub_agent, cbct_environment):
    """End to end: the seven points ASO needs, from a bundle holding more."""
    write_volume(tmp_path / "cohort" / "patient01.nii.gz")
    bundle = write_cbct_bundle(
        tmp_path / "bundle", {"Cranial_Base": list(ASO_LANDMARKS) + ["C2", "C3"]}
    )

    output_dir = run(
        input=tmp_path / "cohort",
        model=bundle,
        output_dir=tmp_path / "out",
        landmarks=list(ASO_LANDMARKS),
    )
    report = json.loads((output_dir / dispatch.REPORT_NAME).read_text())

    assert set(report["landmarks_requested"]) == set(ASO_LANDMARKS)
    # The regions were left at their all-on default and did not widen it.
    assert "C2" not in report["landmarks_requested"]
    # And the report says the selection came from `landmarks`, not the regions.
    assert report["regions"] == []
