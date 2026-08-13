"""audit.py reports; it must never change anything it reads."""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audit  # noqa: E402


def make_tool(root, name, requires_python, dependencies, locked=()):
    tool = root / "tools" / name
    tool.mkdir(parents=True)
    (tool / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "sadt-{name}"
            version = "0.1.0"
            requires-python = "{requires_python}"
            dependencies = [{dependencies}]
            """
        ).format(
            name=name,
            requires_python=requires_python,
            dependencies=", ".join('"{}"'.format(d) for d in dependencies),
        ),
        encoding="utf-8",
    )
    if locked:
        (tool / "uv.lock").write_text(
            "version = 1\n"
            + "".join(
                '\n[[package]]\nname = "{}"\nversion = "{}"\n'.format(*entry)
                for entry in locked
            ),
            encoding="utf-8",
        )
    return tool


@pytest.fixture
def repo(tmp_path):
    """Three tools on two torch versions, plus one with no torch at all."""
    make_tool(tmp_path, "AMASSS", ">=3.11", ["torch==2.2.0"], [("torch", "2.2.0")])
    make_tool(tmp_path, "ALI", ">=3.9,<3.10", ["torch==2.2.0"], [("torch", "2.2.0")])
    make_tool(
        tmp_path,
        "Crown_Seg",
        ">=3.11",
        ["torch==2.4.1"],
        [("torch", "2.4.1"), ("nvidia-cublas-cu12", "12.4.5.8")],
    )
    make_tool(tmp_path, "surgmovpred", ">=3.11", ["scikit-learn==1.5.2"])
    (tmp_path / "tools" / "aso").mkdir()
    return tmp_path


def run(repo, capsys):
    assert audit.main(["--root", str(repo)]) == 0
    return capsys.readouterr().out


def test_distinct_versions_are_counted(repo, capsys):
    out = run(repo, capsys)

    assert "torch  2.2.0 (2 tools) | 2.4.1 (1) | absent (1)" in out
    assert "python >=3.11 (3 tools) | >=3.9,<3.10 (1)" in out


def test_unpinned_is_reported_as_unpinned(tmp_path, capsys):
    make_tool(tmp_path, "loose", ">=3.11", ["torch"])
    assert "torch  unpinned (1 tool)" in run(tmp_path, capsys)


def test_lockfile_wins_over_the_declared_range(tmp_path, capsys):
    """What gets installed is what the lock resolved, not what pyproject asked."""
    make_tool(tmp_path, "loose", ">=3.11", ["torch>=2.2"], [("torch", "2.2.0")])
    assert "torch  2.2.0 (1 tool)" in run(tmp_path, capsys)


def test_alignment_is_suggested_but_never_applied(repo, capsys):
    out = run(repo, capsys)
    lockfiles = {
        path: path.read_bytes() for path in (repo / "tools").rglob("uv.lock")
    }

    assert "2 distinct torch runtime(s)" in out
    assert "Crown_Seg pins torch 2.4.1; aligning to 2.2.0 would drop one runtime" in out
    # The CUDA wheels in Crown_Seg's lock are what makes it the 4.9 GB estimate
    # rather than the CPU-only 0.9 GB one.
    assert "saving ~4.9 GB" in out
    assert "revalidating" in out
    assert all(path.read_bytes() == content for path, content in lockfiles.items())


def test_tools_without_a_pyproject_are_listed_as_pending(repo, capsys):
    assert "not migrated yet (no pyproject.toml): aso" in run(repo, capsys)


def test_a_single_runtime_gets_no_suggestion(tmp_path, capsys):
    make_tool(tmp_path, "a", ">=3.11", ["torch==2.2.0"], [("torch", "2.2.0")])
    make_tool(tmp_path, "b", ">=3.11", ["torch==2.2.0"], [("torch", "2.2.0")])
    out = run(tmp_path, capsys)

    assert "1 distinct torch runtime(s)" in out
    assert "suggestions" not in out


def test_missing_tools_directory_is_an_error(tmp_path, capsys):
    assert audit.main(["--root", str(tmp_path)]) == 2
