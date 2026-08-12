"""The testkit is what integration tests trust, so it is tested itself.

Everything here runs against `tools/_template`, which is the one tool that is
always cheap to build and always present.
"""

from pathlib import Path

import pytest

from sadt_testkit import (
    ToolFailed,
    ToolNotBuilt,
    is_built,
    repo_root,
    run_tool,
    tool_schema,
    tool_venv_python,
)

TEMPLATE = "_template"

needs_template = pytest.mark.skipif(
    not is_built(TEMPLATE),
    reason="run `cd tools/_template && uv sync` first",
)


@pytest.fixture
def scans(tmp_path):
    """Two `.npy` arrays, written with the caller's numpy, read with the tool's.

    That split is the whole point of the exercise: nothing is shared between
    the two environments except the files on disk.
    """
    numpy = pytest.importorskip("numpy")
    folder = tmp_path / "scans"
    folder.mkdir()
    numpy.save(folder / "a.npy", numpy.arange(100, dtype="float32"))
    numpy.save(folder / "b.npy", numpy.full((10, 10), 7.0, dtype="float32"))
    return folder


def test_repo_root_is_the_checkout_holding_tools_and_scripts():
    root = repo_root()
    assert (root / "tools").is_dir()
    assert (root / "scripts" / "describe.py").is_file()


def test_an_unknown_tool_says_so():
    with pytest.raises(ToolNotBuilt, match="no tool called"):
        tool_venv_python("not-a-tool")


def test_an_unbuilt_tool_names_the_command_to_run(tmp_path, monkeypatch):
    """A contributor working on one tool has not built the others."""
    monkeypatch.setattr("sadt_testkit.repo_root", lambda: tmp_path)
    (tmp_path / "tools" / "ghost").mkdir(parents=True)

    with pytest.raises(ToolNotBuilt, match=r"uv sync"):
        tool_venv_python("ghost")

    assert is_built("ghost") is False


@needs_template
def test_run_tool_returns_the_path_the_tool_returned(scans, tmp_path):
    summary = run_tool(
        TEMPLATE,
        scans=scans,
        output_dir=tmp_path / "out",
        metrics=["mean", "max"],
    )

    assert isinstance(summary, Path)
    assert summary.is_file()
    lines = summary.read_text().splitlines()
    assert len(lines) == 2
    assert "mean=50.0" in lines[0]


@needs_template
def test_the_tool_runs_in_its_own_interpreter(scans, tmp_path):
    """Not this one: that is what makes the isolation real rather than claimed."""
    summary = run_tool(TEMPLATE, scans=scans, output_dir=tmp_path / "out")

    assert summary.is_file()
    assert tool_venv_python(TEMPLATE) != Path(__import__("sys").executable)


@needs_template
def test_paths_survive_the_round_trip(scans, tmp_path):
    """JSON has no path type, so they travel as strings and come back typed."""
    output = tmp_path / "nested" / "out"
    summary = run_tool(TEMPLATE, scans=scans, output_dir=output)

    assert summary.parent == output
    assert summary.is_absolute()


@needs_template
def test_a_single_file_input_works_like_a_folder(scans, tmp_path):
    summary = run_tool(TEMPLATE, scans=scans / "b.npy", output_dir=tmp_path / "out")

    assert summary.read_text().strip() == "scan=b.npy voxels=100 mean=7.0 max=7.0"


@needs_template
def test_optional_arguments_reach_the_tool(scans, tmp_path):
    summary = run_tool(
        TEMPLATE,
        scans=scans,
        output_dir=tmp_path / "out",
        metrics=["min", "std"],
        threshold=50.0,
        per_scan_report=True,
    )

    assert "min=51.0" in summary.read_text().splitlines()[0]
    assert (summary.parent / "a.txt").is_file(), "per_scan_report reached the tool"


@needs_template
def test_a_tool_that_raises_carries_its_stderr_back(tmp_path):
    """A failing chain has to say which link failed and why."""
    with pytest.raises(ToolFailed) as failure:
        run_tool(TEMPLATE, scans=tmp_path / "missing", output_dir=tmp_path / "out")

    message = str(failure.value)
    assert TEMPLATE in message
    assert "ToolInputError" in message and "does not exist" in message


@needs_template
def test_the_tool_writes_only_under_its_output_dir(scans, tmp_path):
    """Run from a neutral cwd, so a tool writing beside itself is visible."""
    before = sorted(path for path in scans.rglob("*") if path.is_file())

    run_tool(TEMPLATE, scans=scans, output_dir=tmp_path / "out", per_scan_report=True)

    assert sorted(path for path in scans.rglob("*") if path.is_file()) == before


@needs_template
def test_tool_schema_reads_what_the_server_would_publish():
    """The other half of a handoff: that the next tool still takes the argument.

    A renamed argument is what breaks a chain silently, and comparing schemas is
    how an integration test catches it before the server does.
    """
    schema = tool_schema(TEMPLATE)

    assert schema["name"] == TEMPLATE
    assert schema["arguments"]["scans"] == {"type": "path", "required": True}
    assert schema["arguments"]["metrics"]["choices"] == ["mean", "max", "min", "std"]
    assert schema["returns"] == "path"
    assert len(schema["source_hash"]) == 64


@needs_template
def test_a_chain_can_be_driven_off_the_published_schema(scans, tmp_path):
    """What the server does: read the schema, then call with what it declares.

    This is the shape every real chain takes -- crownseg -> ali, ali -> aso --
    so it is worth one test on a tool that costs nothing to run.
    """
    schema = tool_schema(TEMPLATE)
    required = [name for name, spec in schema["arguments"].items() if spec["required"]]
    assert required == ["scans", "output_dir"]

    first = run_tool(TEMPLATE, scans=scans, output_dir=tmp_path / "one")
    # The output of one call feeding the next argument of another: here the same
    # tool twice, since the template has no downstream.
    second = run_tool(TEMPLATE, scans=scans, output_dir=first.parent / "two")

    assert first.is_file() and second.is_file()
    assert first.read_text() == second.read_text()


def test_the_driver_leans_on_nothing_but_the_stdlib():
    """It is executed BY a tool's interpreter, which need not have this package.

    `sadt-testkit` does end up in a tool's venv as a dev dependency, so an
    `import sadt_testkit` there would succeed and hide the coupling. The real
    guarantee is in the driver's own imports, so that is what is checked.
    """
    import ast
    import sys

    from sadt_testkit import _driver

    source = Path(_driver.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    assert "sadt_testkit" not in imported
    assert imported <= set(sys.stdlib_module_names), sorted(imported - set(sys.stdlib_module_names))
