"""describe.py must refuse what it cannot represent, not guess.

Every case here is a schema the client would render wrongly: a dropped argument,
a check box where a number belongs, a required flag that is silently optional.
The script is exercised through its CLI because the exit code is what CI reads.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

DESCRIBE = Path(__file__).resolve().parent.parent / "describe.py"


def make_tool(root, body, package="sadt_fixture"):
    """A minimal tool package on disk: src/<package>/__init__.py holding run()."""
    source = root / "src" / package
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(
        "from pathlib import Path\n\n" + textwrap.dedent(body), encoding="utf-8"
    )
    return root


def describe(tool_dir):
    return subprocess.run(
        [sys.executable, str(DESCRIBE), str(tool_dir)],
        capture_output=True,
        text=True,
    )


GOOD = """
    def run(
        scan: Path,
        output_dir: Path,
        structures: list[str] = ["Mandible", "Maxilla"],
        crop: bool = False,
        margin: float = 1,
    ) -> Path:
        \"\"\"Segment craniofacial structures on a CBCT scan.

        A second paragraph never reaches the client.
        \"\"\"
        return output_dir
"""


def test_schema_matches_the_signature(tmp_path):
    result = describe(make_tool(tmp_path, GOOD))
    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)

    assert schema["name"] == tmp_path.name
    assert schema["description"] == "Segment craniofacial structures on a CBCT scan."
    assert schema["returns"] == "path"
    assert list(schema["arguments"]) == [
        "scan",
        "output_dir",
        "structures",
        "crop",
        "margin",
    ], "argument order follows the signature, so the client's form does too"
    assert schema["arguments"]["scan"] == {"type": "path", "required": True}
    assert schema["arguments"]["structures"] == {
        "type": "list[str]",
        "required": False,
        "default": ["Mandible", "Maxilla"],
    }
    # Declared float, defaulted with an int literal: the client must still get
    # a float widget with a float default.
    assert schema["arguments"]["margin"]["default"] == 1.0
    assert isinstance(schema["arguments"]["margin"]["default"], float)
    assert len(schema["source_hash"]) == 64


def test_source_hash_tracks_the_sources(tmp_path):
    tool = make_tool(tmp_path, GOOD)
    before = json.loads(describe(tool).stdout)["source_hash"]

    helper = tool / "src" / "sadt_fixture" / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    after_add = json.loads(describe(tool).stdout)["source_hash"]
    assert after_add != before

    helper.rename(helper.with_name("renamed.py"))
    assert json.loads(describe(tool).stdout)["source_hash"] != after_add, (
        "a rename changes what gets imported, so it must change the hash"
    )


def test_bytecode_does_not_change_the_hash(tmp_path):
    tool = make_tool(tmp_path, GOOD)
    before = json.loads(describe(tool).stdout)["source_hash"]

    # Already there: the first describe() imported the package and Python wrote
    # its bytecode next to the sources.
    cache = tool / "src" / "sadt_fixture" / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "__init__.cpython-312.pyc").write_bytes(b"\x00\x01")
    assert json.loads(describe(tool).stdout)["source_hash"] == before


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(
            "from typing import Optional\n"
            "def run(scan: Path, n: Optional[int] = None) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "unsupported annotation",
            id="Optional is not a type here, a missing default is",
        ),
        pytest.param(
            "def run(scan: Path, n: int = None) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "defaults to None",
            id="None default would publish a required argument as optional",
        ),
        pytest.param(
            "def run(scan: Path, n: int = True) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "but defaults to True",
            id="bool passes isinstance(int) and would render a check box",
        ),
        pytest.param(
            "def run(scan: Path, opts: dict = {}) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "unsupported annotation",
            id="dict argument",
        ),
        pytest.param(
            "def run(scan: Path, tags: list = []) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "bare list is not supported",
            id="bare list has no element type to build a widget from",
        ),
        pytest.param(
            "def run(scan: Path, **kwargs) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "must be named",
            id="**kwargs cannot come out of a JSON object",
        ),
        pytest.param(
            "def run(scan: Path, n) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "no annotation",
            id="unannotated argument",
        ),
        pytest.param(
            "def run(scan: Path) -> str:\n"
            '    """Do a thing."""\n'
            "    return str(scan)\n",
            "must return a Path",
            id="a tool returns files, not text",
        ),
        pytest.param(
            "def run(scan: Path):\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "no return annotation",
            id="missing return annotation",
        ),
        pytest.param(
            "def run(scan: Path) -> Path:\n    return scan\n",
            "no docstring",
            id="the first line is the description shown to a clinician",
        ),
        pytest.param(
            "run = 3\n",
            "not a function",
            id="run must be callable",
        ),
        pytest.param(
            "def other(scan: Path) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "defines no run()",
            id="no run at all",
        ),
    ],
)
def test_unsupported_signatures_fail_loudly(tmp_path, body, expected):
    result = describe(make_tool(tmp_path, body))

    assert result.returncode == 2
    assert expected in result.stderr
    assert result.stdout == "", "a refused tool must not emit a partial schema"


def test_dict_of_paths_is_accepted_for_several_outputs(tmp_path):
    tool = make_tool(
        tmp_path,
        "def run(scan: Path) -> dict[str, Path]:\n"
        '    """Segment and align a scan."""\n'
        "    return {}\n",
    )
    assert json.loads(describe(tool).stdout)["returns"] == "dict[str, path]"


def test_module_level_heavy_import_is_reported_as_such(tmp_path):
    tool = make_tool(
        tmp_path,
        "import torch\n\n"
        "def run(scan: Path) -> Path:\n"
        '    """Do a thing."""\n'
        "    return scan\n",
    )
    result = describe(tool)

    assert result.returncode == 2
    assert "Move it inside run()" in result.stderr


def test_the_template_describes_itself(tmp_path):
    """The reference package must pass the checker it is the reference for."""
    template = Path(__file__).resolve().parents[2] / "tools" / "_template"
    venv_python = template / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        pytest.skip("run `uv sync` in tools/_template first")

    result = subprocess.run(
        [str(venv_python), str(DESCRIBE), str(template)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    assert schema["arguments"]["scans"]["required"] is True
    assert schema["arguments"]["metrics"]["default"] == ["mean", "max"]
