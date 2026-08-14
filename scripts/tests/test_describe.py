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


# --- Literal: a fixed set of options ---------------------------------------


def test_literal_publishes_choices(tmp_path):
    """`list[Literal[...]]` is several-of, bare `Literal[...]` is exactly-one."""
    tool = make_tool(
        tmp_path,
        "from typing import Literal\n\n"
        "def run(\n"
        "    scan: Path,\n"
        '    structures: list[Literal["MAND", "MAX", "CB"]] = ["MAND", "MAX"],\n'
        '    merge: Literal["MERGED", "SEPARATE"] = "MERGED",\n'
        "    folds: list[Literal[0, 1, 2]] = [0],\n"
        ") -> Path:\n"
        '    """Segment a scan."""\n'
        "    return scan\n",
    )
    schema = json.loads(describe(tool).stdout)

    assert schema["arguments"]["structures"] == {
        "type": "list[str]",
        "required": False,
        "default": ["MAND", "MAX"],
        "choices": ["MAND", "MAX", "CB"],
    }
    assert schema["arguments"]["merge"] == {
        "type": "str",
        "required": False,
        "default": "MERGED",
        "choices": ["MERGED", "SEPARATE"],
    }
    assert schema["arguments"]["folds"]["type"] == "list[int]"
    assert schema["arguments"]["folds"]["choices"] == [0, 1, 2]


def test_choices_come_last_so_keys_stay_stable(tmp_path):
    """Adding options to an argument must not reorder what a reader expects."""
    tool = make_tool(
        tmp_path,
        "from typing import Literal\n\n"
        'def run(scan: Path, mode: Literal["a", "b"] = "a") -> Path:\n'
        '    """Do a thing."""\n'
        "    return scan\n",
    )
    schema = json.loads(describe(tool).stdout)

    assert list(schema["arguments"]["mode"]) == ["type", "required", "default", "choices"]


def test_a_required_argument_may_still_carry_choices(tmp_path):
    tool = make_tool(
        tmp_path,
        "from typing import Literal\n\n"
        'def run(scan: Path, mode: Literal["a", "b"]) -> Path:\n'
        '    """Do a thing."""\n'
        "    return scan\n",
    )
    assert json.loads(describe(tool).stdout)["arguments"]["mode"] == {
        "type": "str",
        "required": True,
        "choices": ["a", "b"],
    }


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(
            "from typing import Literal\n\n"
            'def run(scan: Path, mode: Literal["a", "b"] = "c") -> Path:\n'
            '    """Do a thing."""\n'
            "    return scan\n",
            "not one of its options",
            id="a default the published picker cannot produce",
        ),
        pytest.param(
            "from typing import Literal\n\n"
            'def run(scan: Path, m: list[Literal["a", "b"]] = ["a", "z"]) -> Path:\n'
            '    """Do a thing."""\n'
            "    return scan\n",
            "not one of its options",
            id="one bad element in a multi-select default",
        ),
        pytest.param(
            "from typing import Literal\n\n"
            'def run(scan: Path, mode: Literal["a", 1] = "a") -> Path:\n'
            '    """Do a thing."""\n'
            "    return scan\n",
            "must all be str, or all be int",
            id="mixed option types have no single widget",
        ),
        pytest.param(
            "from typing import Literal\n\n"
            "def run(scan: Path, mode: Literal[True, False] = True) -> Path:\n"
            '    """Do a thing."""\n'
            "    return scan\n",
            "must all be str, or all be int",
            id="bool passes isinstance(int) and would render as a number",
        ),
        pytest.param(
            "from typing import Literal, Optional\n\n"
            'def run(scan: Path, mode: Literal[None] = None) -> Path:\n'
            '    """Do a thing."""\n'
            "    return scan\n",
            "Options must be str or int",
            id="None is not an option a client can render",
        ),
    ],
)
def test_bad_choices_fail_loudly(tmp_path, body, expected):
    result = describe(make_tool(tmp_path, body))

    assert result.returncode == 2
    assert expected in result.stderr
    assert result.stdout == ""


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
    assert schema["arguments"]["metrics"]["choices"] == ["mean", "max", "min", "std"]
    assert schema["arguments"]["reduction"]["choices"] == ["per_scan", "pooled"]


# ---------------------------------------------------------------------------
# The supervisor -- how a tool calls another tool
# ---------------------------------------------------------------------------

SUPERVISED = """
    def run(scan: Path, output_dir: Path, *, sup=None) -> Path:
        \"\"\"Orient a scan onto a reference.\"\"\"
        return output_dir
"""


def test_the_supervisor_is_not_published_as_an_argument(tmp_path):
    """It is not data, so a client must never be asked to send it."""
    result = describe(make_tool(tmp_path, SUPERVISED))

    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    assert "sup" not in schema["arguments"]
    assert list(schema["arguments"]) == ["scan", "output_dir"]


def test_a_tool_that_takes_one_says_so(tmp_path):
    """The server has to know before it accepts a job: a runner that cannot
    inject a supervisor must refuse the tool, not call it and fail halfway."""
    schema = json.loads(describe(make_tool(tmp_path, SUPERVISED)).stdout)
    assert schema["supervisor"] is True


def test_a_tool_without_one_does_not_carry_the_key(tmp_path):
    """Absent, not `false`: every tool written before supervisors existed keeps
    exactly the schema it had."""
    schema = json.loads(describe(make_tool(tmp_path, GOOD)).stdout)
    assert "supervisor" not in schema


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            # Positional: the runner's first argument would land in it.
            """
            def run(scan: Path, sup=None) -> Path:
                \"\"\"Orient a scan.\"\"\"
                return scan
            """,
            "keyword-only",
        ),
        (
            # Annotated: published as a path, so the form grows a file picker
            # for something no client can produce.
            """
            def run(scan: Path, *, sup: Path = None) -> Path:
                \"\"\"Orient a scan.\"\"\"
                return scan
            """,
            "must not be annotated",
        ),
    ],
)
def test_a_near_miss_supervisor_fails_loudly(tmp_path, body, expected):
    """Both of these would otherwise fail a long way from the signature."""
    result = describe(make_tool(tmp_path, body))

    assert result.returncode == 2
    assert expected in result.stderr
    assert result.stdout == ""
