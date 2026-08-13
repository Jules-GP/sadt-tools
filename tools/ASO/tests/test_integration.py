"""ASO as the server actually invokes it: out of process, in its own venv.

Two things are only testable from out here:

* that the tool writes nothing outside `output_dir`. In-process the test shares
  a working directory with the tool, so a stray write lands somewhere the test
  was going to create anyway; `run_tool` runs it from a neutral directory.
* that the published schema and the callable agree. The server reads the schema
  and calls `run(**params)` from it, so an argument renamed in one place and not
  the other breaks the chain there, not here -- unless something checks.

It also pins the **gap**: `sadt_testkit`'s driver is the same contract as the
server's runner, and neither injects a supervisor yet. So fully-automated CBCT
run through it fails with `SupervisorRequired` -- which is the current, correct
state of the world, and this is what will start failing the day that changes.
"""

import json
import os

import numpy as np
import pytest
import SimpleITK as sitk
from sadt_testkit import ToolFailed, is_built, run_tool, tool_schema

TOOL = "ASO"

needs_venv = pytest.mark.skipif(
    not is_built(TOOL),
    reason=f"run `cd tools/{TOOL} && uv sync` first",
)

_REFERENCE_POINTS = {
    "Ba": np.array([0.0, -30.0, -20.0]),
    "S": np.array([0.0, -10.0, 5.0]),
    "N": np.array([0.0, 25.0, 10.0]),
    "RPo": np.array([-35.0, -20.0, -5.0]),
    "LPo": np.array([35.0, -20.0, -5.0]),
    "ROr": np.array([-25.0, 20.0, 0.0]),
    "LOr": np.array([25.0, 20.0, 0.0]),
}


def _write_markups(path, landmarks):
    """A Slicer markups file, written without importing the tool.

    Spelled out rather than imported from `sadt_aso.markups`: this module is
    about the contract seen from OUTSIDE the package, and reaching inside it for
    a helper would quietly make the test depend on what it is checking.
    """
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    content = {
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "controlPoints": [
                    {
                        "id": str(index),
                        "label": label,
                        "position": [float(v) for v in position],
                        "visibility": True,
                        "positionStatus": "defined",
                    }
                    for index, (label, position) in enumerate(landmarks.items(), start=1)
                ],
            }
        ]
    }
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(content, handle)
    return str(path)


def _rotated(points, degrees=17.0, offset=(12.0, -8.0, 5.0)):
    axis = np.array([0.2, 1.0, 0.3])
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(degrees)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    matrix = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    return {name: matrix @ point + np.array(offset) for name, point in points.items()}


@pytest.fixture
def case(tmp_path):
    """One synthetic CBCT with its landmarks beside it, and a reference."""
    scans = tmp_path / "input"
    scans.mkdir()
    array = np.zeros((16, 16, 16), dtype=np.int16)
    array[4:12, 4:12, 4:12] = 800
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(scans / "patient1_scan.nii.gz"), useCompression=True)
    _write_markups(scans / "patient1_lm.mrk.json", _rotated(_REFERENCE_POINTS))

    reference = tmp_path / "gold"
    _write_markups(reference / "reference_lm.mrk.json", _REFERENCE_POINTS)
    return scans, reference


@needs_venv
def test_the_published_schema_matches_the_callable(case, tmp_path):
    """The server drives `run()` from this schema, so it has to be true."""
    schema = tool_schema(TOOL)

    assert schema["name"] == "ASO"
    assert schema["returns"] == "path"
    required = [name for name, spec in schema["arguments"].items() if spec["required"]]
    assert required == ["input", "reference", "output_dir"]

    # The supervisor is not an argument -- a client cannot send one.
    assert "sup" not in schema["arguments"]
    # But the schema says the tool can take one, so a runner that cannot supply
    # one knows to refuse the tool rather than call it and fail halfway.
    assert schema["supervisor"] is True

    # Both modes' arguments are optional, because each is inert in the other.
    for name in ("cbct_landmarks", "ios_teeth", "ios_jaws", "landmarks"):
        assert not schema["arguments"][name]["required"]
    assert schema["arguments"]["modality"]["choices"] == ["CBCT", "IOS"]


@needs_venv
def test_semi_automated_cbct_runs_out_of_process(case, tmp_path):
    """The whole registration, in its own venv, from a neutral directory."""
    scans, reference = case

    output_dir = run_tool(
        TOOL,
        input=scans,
        reference=reference,
        output_dir=tmp_path / "out",
        modality="CBCT",
        automation="Semi-Automated",
    )

    report = json.loads((output_dir / "ASO_report.json").read_text())
    assert report["summary"]["oriented"] == 1
    assert (output_dir / "patient1_Or.nii.gz").is_file()
    assert (output_dir / "patient1_lm_Or.mrk.json").is_file()
    assert (output_dir / "patient1_Or_transform.tfm").is_file()


@needs_venv
def test_nothing_is_written_outside_the_output_directory(case, tmp_path):
    """Run from a neutral cwd, which is what makes this assertion mean anything."""
    scans, reference = case
    before = sorted(path for path in scans.rglob("*") if path.is_file())

    run_tool(
        TOOL, input=scans, reference=reference, output_dir=tmp_path / "out",
        modality="CBCT", automation="Semi-Automated",
    )

    assert sorted(path for path in scans.rglob("*") if path.is_file()) == before


@needs_venv
def test_the_working_directory_does_not_survive(case, tmp_path):
    scans, reference = case

    output_dir = run_tool(
        TOOL, input=scans, reference=reference, output_dir=tmp_path / "out",
        modality="CBCT", automation="Semi-Automated",
    )
    assert not (output_dir / ".aso_work").exists()


@needs_venv
def test_supplied_landmarks_reach_the_registration(case, tmp_path):
    """Fully-automated, standalone: the landmarks are predicted elsewhere and
    passed in, so no supervisor is needed. This is the shape a notebook uses,
    and the shape a server without supervisor support can still drive."""
    scans, reference = case
    supplied = tmp_path / "predicted"
    _write_markups(supplied / "patient1_lm_Pred.mrk.json", _rotated(_REFERENCE_POINTS))

    output_dir = run_tool(
        TOOL,
        input=scans,
        reference=reference,
        output_dir=tmp_path / "out",
        modality="CBCT",
        automation="Fully-Automated",
        landmarks=supplied,
    )

    report = json.loads((output_dir / "ASO_report.json").read_text())
    assert report["landmark_source"] == "supplied"
    assert report["summary"]["oriented"] == 1


@needs_venv
def test_fully_automated_without_landmarks_is_refused_by_this_runner(case, tmp_path):
    """**The gap, pinned.**

    `sadt_testkit`'s driver is the same contract as the server's runner, and
    neither injects a supervisor yet -- so the one mode that needs one cannot
    run through either. The failure is the designed one, naming the way
    forward, rather than a `TypeError` about a missing argument.

    When supervisor support lands on the server side, this test is what says so:
    it will start failing, and the assertion below is where to look.
    """
    scans, reference = case

    with pytest.raises(ToolFailed) as raised:
        run_tool(
            TOOL,
            input=scans,
            reference=reference,
            output_dir=tmp_path / "out",
            modality="CBCT",
            automation="Fully-Automated",
            landmark_models="ALI_CBCT_Models",
        )

    message = str(raised.value)
    assert "SupervisorRequired" in message
    assert "ALI" in message


# ---------------------------------------------------------------------------
# The ALI -> ASO chain, through a real supervisor
# ---------------------------------------------------------------------------

class LocalSup:
    """A supervisor that runs the other tool in its own venv, as a subprocess.

    The same shape the server's runner will build, and the same one the fake in
    `test_run.py` imitates -- five members, duck-typed, nothing imported across
    tools. `sadt_testkit.run_tool` is doing exactly what the server does, so a
    chain that works here works there.

    Worth promoting into `sadt_testkit` once a second tool needs it; it lives
    here while ASO is the only one.
    """

    def __init__(self, out, tmp):
        self.out = out
        self.tmp = tmp

    def run(self, tool, **params):
        return run_tool(tool, **params)

    def progress(self, fraction, message):
        pass

    def log(self, message):
        pass


needs_chain = pytest.mark.skipif(
    not (is_built(TOOL) and is_built("ALI")),
    reason="run `uv sync` in tools/ASO and tools/ALI",
)


@needs_chain
@pytest.mark.models
@pytest.mark.gpu
def test_fully_automated_cbct_drives_ali_through_the_supervisor(case, tmp_path):
    """The real chain: ASO recentres, calls ALI on the centred scans, registers.

    Needs the 4.7 GB `ALI_CBCT_Models` bundle and a card; see
    tests/data/README.md. Everything up to the ALI call is covered by the fake
    supervisor in `test_run.py`, so what this adds is that the real tool accepts
    the arguments ASO sends it and returns landmarks ASO can read.
    """
    from pathlib import Path

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from sadt_aso import run

    scans, reference = case
    bundle = Path("tests/data/ALI_CBCT_Models").resolve()
    if not bundle.is_dir():
        pytest.skip("see tests/data/README.md for the bundle this needs")

    output_dir = run(
        input=scans,
        reference=reference,
        output_dir=tmp_path / "out",
        modality="CBCT",
        automation="Fully-Automated",
        landmark_models=bundle,
        sup=LocalSup(out=tmp_path / "out", tmp=tmp_path / "tmp"),
    )

    report = json.loads((output_dir / "ASO_report.json").read_text())
    assert report["landmark_source"] == "ALI"
    assert report["summary"]["oriented"] == 1
