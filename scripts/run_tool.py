#!/usr/bin/env python3
"""Run a tool from the command line, in its own venv, chaining to others.

    scripts/run_tool.py ALI --input scan.nii.gz --model bundle/ --output-dir out/
    scripts/run_tool.py ASO --input cohort/ --reference gold/ --output-dir out/ \
        --automation Fully-Automated --landmark-models ALI_CBCT_Models

One script for every tool: the parser is built from `run()`'s signature, so there
is no per-tool CLI to keep in step with anything. A tool that takes a supervisor
gets one, and its `sup.run("ALI", ...)` re-enters THIS script with ALI's
interpreter -- which is what makes chaining work, and nesting with it:
`AREG -> ASO -> ALI` is three levels of the same recursion, no special case.

**Developer convenience, not the deployment path.** In production the server's
runner invokes tools and sequences them; this exists so the same thing can be
done from a checkout, with no server, no Docker and no network. It is also the
reference a supervisor implementation can be read off.

Two layers, one file:

1. any interpreter -- resolve `tools/<name>/.venv/bin/python` and re-exec;
2. the tool's own venv -- import the package, build the parser, call `run()`.

Stdlib only and Python 3.9 compatible, for the same reason `describe.py` is: it
has to run inside whichever interpreter a tool pinned. The annotation vocabulary
is imported FROM describe.py rather than restated, so the CLI can never offer an
option the schema does not, or refuse one it does.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import typing
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(SCRIPTS))
import describe  # noqa: E402  -- one annotation vocabulary, not two


class ToolNotBuilt(Exception):
    """The tool has no venv yet, and the message says how to make one."""


# ---------------------------------------------------------------------------
# Layer 1: find the tool's interpreter
# ---------------------------------------------------------------------------

def tool_python(name):
    """The interpreter of `tools/<name>/.venv`."""
    root = TOOLS_DIR / name
    if not root.is_dir():
        available = sorted(
            path.name for path in TOOLS_DIR.iterdir()
            if (path / "pyproject.toml").is_file()
        )
        raise ToolNotBuilt(
            "There is no tool called {!r} under tools/. Available: {}.".format(
                name, ", ".join(available) or "none"
            )
        )
    for relative in ("bin/python", "Scripts/python.exe"):
        candidate = root / ".venv" / relative
        if candidate.is_file():
            return candidate
    raise ToolNotBuilt(
        "tools/{0} has no .venv. Build it with `cd tools/{0} && uv sync`.".format(name)
    )


# ---------------------------------------------------------------------------
# Layer 2: the parser, built from run()'s signature
# ---------------------------------------------------------------------------

# Above this many, an argument's options are validated but not spelled into the
# usage line -- ALI's `landmarks` has 119 and argparse prints them all.
_MAX_LISTED_CHOICES = 8


def option_name(argument):
    """`output_dir` -> `--output-dir`."""
    return "--" + argument.replace("_", "-")


def add_argument(parser, name, parameter, hints):
    """One `run()` parameter as one command-line option.

    Everything the schema can express has a command-line spelling, and nothing
    else does: `describe.type_name` decides, so an annotation the server would
    refuse cannot be smuggled in through here.
    """
    kind, choices = describe.type_name(hints[name], "argument '{}'".format(name))
    required = parameter.default is parameter.empty
    default = None if required else parameter.default
    flag = option_name(name)

    if kind == "bool":
        # A flag and its negation, both present, because a tool's default may be
        # either -- `--no-skip-segmented` has to exist for `skip_segmented=True`.
        group = parser.add_mutually_exclusive_group(required=required)
        group.add_argument(flag, dest=name, action="store_true", default=default)
        group.add_argument(
            option_name("no_" + name), dest=name, action="store_false", default=default
        )
        return

    scalar = {"path": str, "str": str, "int": int, "float": float, "bool": str}
    # A long option list is still VALIDATED, just not printed: ALI publishes 119
    # landmarks, and argparse spells every one of them into the usage line.
    metavar = name.upper() if (not choices or len(choices) > _MAX_LISTED_CHOICES) else None
    help_text = (
        "one of: {}".format(", ".join(str(c) for c in choices))
        if choices and metavar
        else None
    )

    if kind.startswith("list["):
        element = kind[len("list["):-1]
        parser.add_argument(
            flag, dest=name, nargs="*", required=required, default=default,
            type=scalar[element], choices=choices, metavar=metavar, help=help_text,
        )
        return

    parser.add_argument(
        flag, dest=name, required=required, default=default,
        type=scalar[kind], choices=choices, metavar=metavar, help=help_text,
    )


def build_parser(run, name):
    signature, hints = describe.inspect.signature(run), typing.get_type_hints(run)
    parser = argparse.ArgumentParser(
        prog="run_tool.py {}".format(name),
        description=(describe.inspect.getdoc(run) or "").strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for argument, parameter in signature.parameters.items():
        if describe.is_supervisor(argument, parameter, hints):
            continue  # injected, never typed by a human
        add_argument(parser, argument, parameter, hints)
    return parser


def takes_supervisor(run):
    """Whether `run()` declares a supervisor, by describe.py's own rule."""
    hints = typing.get_type_hints(run)
    return any(
        describe.is_supervisor(name, parameter, hints)
        for name, parameter in describe.inspect.signature(run).parameters.items()
    )


def coerce(value, annotation):
    """JSON has no path type, so paths travel as strings and become `Path` here.

    The same coercion `sadt_testkit._driver` and the server's runner apply. If
    the three ever disagree, a chain that works from the CLI breaks in
    production, so keep this boring and keep it matching.

    An empty string is ABSENCE and stays a string. `Path("")` is
    `PosixPath(".")` -- the current directory, and truthy -- so coercing the
    "not supplied" default of an optional path turns it into a real directory
    the tool then walks. Calling `run()` directly in Python keeps the `""` the
    signature declares, so coercing it here is also what makes the two paths
    disagree.
    """
    if annotation is Path:
        return Path(value) if value != "" else value
    if typing.get_origin(annotation) is list and typing.get_args(annotation) == (Path,):
        return [Path(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------

class LocalSupervisor:
    """What a tool receives as `sup`, backed by sibling venvs on this machine.

    Five members and nothing more, matching what the server injects and what the
    tools' tests fake. A tool cannot tell the three apart, which is the point:
    nothing here is imported by a tool, and no shared package exists.

    `run()` re-enters this script rather than importing anything, so the callee
    gets its own interpreter and its own dependency set -- and its own
    supervisor, which is how nesting works without a special case.
    """

    def __init__(self, out, tmp, depth=0):
        # Resolved, because `run()` below starts the callee from a NEUTRAL
        # working directory -- which is what makes "writes only under
        # output_dir" testable -- and a relative path handed across that
        # boundary resolves against the wrong place.
        self.out = Path(out).resolve()
        self.tmp = Path(tmp).resolve()
        self._depth = depth

    def run(self, tool, **params):
        self.log("running {}".format(tool))
        scratch = Path(tempfile.mkdtemp(prefix="sup_", dir=str(self.tmp)))
        params_file, result_file = scratch / "params.json", scratch / "result.json"
        params_file.write_text(json.dumps(_jsonable(params)), encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable, str(SCRIPTS / "run_tool.py"), tool,
                "--params-file", str(params_file),
                "--result-file", str(result_file),
                "--depth", str(self._depth + 1),
            ],
            # Not captured: a nested tool's progress is the only sign of life
            # during a run that can take an hour, and swallowing it is how a
            # chain looks hung. Its result never comes back on stdout anyway.
            cwd=str(scratch),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "{} failed (exit {}). Its error is above.".format(tool, completed.returncode)
            )

        result = json.loads(result_file.read_text(encoding="utf-8"))
        if result["kind"] == "path":
            return Path(result["value"])
        return {key: Path(value) for key, value in result["value"].items()}

    def progress(self, fraction, message):
        self.log("{:.0%} {}".format(fraction, message))

    def log(self, message):
        # stderr, never stdout: stdout carries a result when this script is
        # driven by another copy of itself.
        sys.stderr.write("{}[sup] {}\n".format("  " * self._depth, message))
        sys.stderr.flush()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def call(name, params, depth):
    """Import the tool, inject a supervisor if it wants one, call `run()`."""
    src = TOOLS_DIR / name / "src"
    run = describe.load_run(src, describe.find_package(src))
    hints = typing.get_type_hints(run)
    arguments = {
        key: coerce(value, hints[key]) for key, value in params.items() if key in hints
    }

    if not takes_supervisor(run):
        return run(**arguments)

    output_dir = Path(arguments.get("output_dir", Path.cwd()))
    output_dir.mkdir(parents=True, exist_ok=True)
    # OUTSIDE output_dir, and removed afterwards. The supervisor's scratch is
    # not the tool's output, and leaving it there would put files in the one
    # directory the caller keeps -- the tool is held to writing only inside it,
    # so whatever drives the tool should not undo that from the other side.
    scratch = Path(tempfile.mkdtemp(prefix="sadt_run_tool_"))
    try:
        arguments["sup"] = LocalSupervisor(out=output_dir, tmp=scratch, depth=depth)
        return run(**arguments)
    finally:
        shutil.rmtree(str(scratch), ignore_errors=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        sys.stderr.write(
            "usage: run_tool.py <TOOL> [--help]\n\ntools: {}\n".format(
                ", ".join(sorted(
                    path.name for path in TOOLS_DIR.iterdir()
                    if (path / "pyproject.toml").is_file()
                ))
            )
        )
        return 2

    name, rest = argv[0], argv[1:]
    try:
        interpreter = tool_python(name)
    except ToolNotBuilt as error:
        sys.stderr.write("{}\n".format(error))
        return 2

    # Layer 1 -> layer 2. Compared by VENV, not by interpreter: uv's venvs all
    # symlink `bin/python` to the same underlying CPython, so resolving the
    # binary makes every tool look like every other one -- and the callee would
    # then be imported into its CALLER's environment, which is precisely the
    # dependency mixing this repository exists to prevent. `sys.prefix` is the
    # venv root inside a venv, so it is the thing that actually differs.
    if Path(sys.prefix).resolve() != (TOOLS_DIR / name / ".venv").resolve():
        return subprocess.run(
            [str(interpreter), str(Path(__file__).resolve()), name] + rest
        ).returncode

    # -- layer 2, inside the tool's venv ------------------------------------
    machine = argparse.ArgumentParser(add_help=False)
    machine.add_argument("--params-file")
    machine.add_argument("--result-file")
    machine.add_argument("--depth", type=int, default=0)
    known, remaining = machine.parse_known_args(rest)

    src = TOOLS_DIR / name / "src"
    try:
        run = describe.load_run(src, describe.find_package(src))
    except describe.SchemaError as error:
        sys.stderr.write("{}: {}\n".format(name, error))
        return 2

    if known.params_file:
        # Driven by a supervisor: arguments as JSON, result to a file. Never
        # parsed off stdout -- these tools print progress bars and nnUNet
        # banners, and anything scraped from stdout breaks the first time a
        # dependency prints something new.
        params = json.loads(Path(known.params_file).read_text(encoding="utf-8"))
    else:
        try:
            params = vars(build_parser(run, name).parse_args(remaining))
        except describe.SchemaError as error:
            sys.stderr.write("{}: {}\n".format(name, error))
            return 2

    result = call(name, params, known.depth)

    if known.result_file:
        Path(known.result_file).write_text(
            json.dumps(
                {"kind": "path", "value": str(result)}
                if isinstance(result, Path)
                else {"kind": "dict", "value": {k: str(v) for k, v in result.items()}}
            ),
            encoding="utf-8",
        )
    else:
        print(result if isinstance(result, Path) else json.dumps(_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
