"""End-to-end test for the template tool.

A tool is not done until a test has actually run it: an import check proves
nothing about a segmentation pipeline. Every `tools/<name>/tests/test_run.py`
must call `run()` on real input and assert on the files it produced -- they
exist, they are a plausible size, and where a reference output is available,
they match it within a tolerance the README documents.

Fixtures are built here rather than committed. Test data goes in `tests/data/`
only as a download script plus checksums; no large binaries and no patient
data in git, ever.
"""

import numpy as np
import pytest

from sadt_template import run
from sadt_template.pipeline import ToolInputError


@pytest.fixture
def scans(tmp_path):
    """A folder of two arrays with known statistics."""
    folder = tmp_path / "scans"
    folder.mkdir()
    np.save(folder / "a.npy", np.arange(100, dtype=np.float32))
    np.save(folder / "b.npy", np.full((10, 10), 7.0, dtype=np.float32))
    return folder


def test_run_on_a_folder(scans, tmp_path):
    output = run(scans=scans, output_dir=tmp_path / "out")

    assert output.exists() and output.stat().st_size > 0
    lines = output.read_text().splitlines()
    assert len(lines) == 2, "one row per scan, in sorted order"
    assert lines[0].startswith("scan=a.npy")
    # arange(100) above threshold 0 is 1..99: mean 50, max 99.
    assert "mean=50.0" in lines[0] and "max=99.0" in lines[0]


def test_run_on_a_single_file(scans, tmp_path):
    """The same argument takes one file -- batch-capable, not batch-only."""
    output = run(scans=scans / "b.npy", output_dir=tmp_path / "out")

    assert output.read_text().strip() == "scan=b.npy voxels=100 mean=7.0 max=7.0"


def test_writes_nothing_outside_the_output_directory(scans, tmp_path):
    before = sorted(scans.iterdir())
    run(scans=scans, output_dir=tmp_path / "out", per_scan_report=True)

    assert sorted(scans.iterdir()) == before
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "summary.txt",
        "a.txt",
        "b.txt",
    }


def test_optional_arguments_are_honoured(scans, tmp_path):
    output = run(
        scans=scans,
        output_dir=tmp_path / "out",
        metrics=["min", "std"],
        threshold=50.0,
    )

    row = output.read_text().splitlines()[0]
    assert "min=51.0" in row and "std=" in row and "mean=" not in row


def test_bad_input_is_reported_as_such(tmp_path):
    """Bad input raises ToolInputError; a bug raises anything else."""
    with pytest.raises(ToolInputError, match="does not exist"):
        run(scans=tmp_path / "missing", output_dir=tmp_path / "out")

    with pytest.raises(ToolInputError, match="Unknown metric"):
        run(scans=tmp_path, output_dir=tmp_path / "out", metrics=["median"])


@pytest.mark.gpu
def test_runs_on_gpu():
    """Marked tests are skipped in CI and run by hand before opening the PR.

    Delete this in a tool with no GPU path; keep it, and state in the PR that
    you ran `pytest -m gpu` and what came out.
    """
    pytest.skip("the template has no GPU path")


def test_pooled_reduction_collapses_the_batch_to_one_row(scans, tmp_path):
    """The `Literal` single-select: exactly one of a fixed set."""
    output = run(scans=scans, output_dir=tmp_path / "out", reduction="pooled")

    lines = output.read_text().splitlines()
    assert len(lines) == 1
    # arange(100) above 0 is 1..99, plus 100 voxels of 7.0.
    assert lines[0].startswith("scan=all voxels=199")


def test_an_option_outside_the_published_set_is_refused(scans, tmp_path):
    """`Literal` is published as `choices`, not enforced by Python.

    The runner calls run(**params) from a JSON object, so a stale client or a
    direct caller can still send anything. The tool checks.
    """
    with pytest.raises(ToolInputError, match="Unknown reduction"):
        run(scans=scans, output_dir=tmp_path / "out", reduction="median")
