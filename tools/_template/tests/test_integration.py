"""The tool as the server actually invokes it: out of process, in its own venv.

Copy this file into a new tool and point `run_tool` at whatever produces its
input. `tools/ALI`, for instance, needs meshes that carry tooth labels, so its
version of this runs `Crown_Seg` first and feeds the result straight in — the
real handoff, not a fixture standing in for one.

Two things are only testable from out here:

* that the tool writes nothing outside `output_dir`. In-process the test shares
  a working directory with the tool, so a stray write lands somewhere the test
  was going to create anyway; `run_tool` runs it from a neutral directory.
* that the published schema and the callable agree. The server reads the schema
  and calls `run(**params)` from it, so an argument renamed in one place and not
  the other breaks the chain there, not here — unless something checks.
"""

import numpy as np
import pytest
from sadt_testkit import ToolFailed, is_built, run_tool, tool_schema

TOOL = "_template"

needs_venv = pytest.mark.skipif(
    not is_built(TOOL),
    reason=f"run `cd tools/{TOOL} && uv sync` first",
)


@pytest.fixture
def scans(tmp_path):
    folder = tmp_path / "scans"
    folder.mkdir()
    np.save(folder / "a.npy", np.arange(100, dtype=np.float32))
    return folder


@needs_venv
def test_the_tool_runs_out_of_process_and_returns_its_output(scans, tmp_path):
    summary = run_tool(TOOL, scans=scans, output_dir=tmp_path / "out")

    assert summary.is_file()
    assert "mean=50.0" in summary.read_text()


@needs_venv
def test_nothing_is_written_outside_the_output_directory(scans, tmp_path):
    """Run from a neutral cwd, which is what makes this assertion mean anything."""
    before = sorted(path for path in scans.rglob("*") if path.is_file())

    run_tool(TOOL, scans=scans, output_dir=tmp_path / "out", per_scan_report=True)

    assert sorted(path for path in scans.rglob("*") if path.is_file()) == before


@needs_venv
def test_the_published_schema_matches_the_callable(scans, tmp_path):
    """The server drives `run()` from this schema, so it has to be true."""
    schema = tool_schema(TOOL)

    required = [name for name, spec in schema["arguments"].items() if spec["required"]]
    assert required == ["scans", "output_dir"]

    # Everything the schema calls required, and nothing else, gets the tool to run.
    summary = run_tool(TOOL, **{"scans": scans, "output_dir": tmp_path / "out"})
    assert summary.is_file()

    # And an option the schema publishes is one the tool really accepts.
    for metric in schema["arguments"]["metrics"]["choices"]:
        run_tool(TOOL, scans=scans, output_dir=tmp_path / metric, metrics=[metric])


@needs_venv
def test_a_failure_comes_back_named(scans, tmp_path):
    """A chain that breaks has to say which link broke, and why."""
    with pytest.raises(ToolFailed, match="Unknown metric"):
        run_tool(TOOL, scans=scans, output_dir=tmp_path / "out", metrics=["median"])
