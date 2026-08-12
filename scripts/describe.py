#!/usr/bin/env python3
"""Emit the JSON schema the server publishes for a tool, read from its run().

Run it with the tool's OWN interpreter, because importing the tool needs the
tool's dependencies and nothing in this repository shares an environment:

    tools/amasss/.venv/bin/python scripts/describe.py tools/amasss

The signature is the single source of truth. There is no second declaration to
keep in step, which is the whole reason the old `ArgSpec` tables are gone: they
drifted from `run()` and the client rendered widgets the tool did not have.

This script deliberately FAILS (exit 2) on anything it cannot represent rather
than emitting a plausible-looking schema. A wrong schema is worse than no
schema: the client builds its form from it, and a silently dropped argument
becomes a run that succeeds with the wrong parameters.

It stays compatible with Python 3.9 -- a tool may pin an old interpreter, and
this must run inside it -- so no tomllib and no match statement here.
"""

import argparse
import hashlib
import inspect
import json
import sys
import typing
from pathlib import Path

# The whole annotation vocabulary. `Path` means "a file or a directory"; the
# rest are scalars. Nothing else is allowed -- no Optional, no unions, no custom
# marker types, nothing imported from the server.
SCALARS = {
    Path: "path",
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}

# What a default value may be for each type. `float` accepts an int literal
# (`threshold: float = 0`) and is normalised below so the client gets a float.
ACCEPTED_DEFAULTS = {
    "path": (str, Path),
    "str": (str,),
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
}


class SchemaError(Exception):
    """The tool cannot be described. Always actionable, always fatal."""


def type_name(annotation, where):
    """Map an annotation onto its schema name, or refuse it."""
    try:
        if annotation in SCALARS:
            return SCALARS[annotation]
    except TypeError:
        pass  # an unhashable annotation is unsupported by definition

    # Bare `list` is not a generic alias, so it never reaches the branch below.
    # It is a common enough slip to deserve its own message: without an element
    # type the client has nothing to build a widget from.
    if annotation is list:
        raise SchemaError(
            "{}: bare list is not supported. Say what it holds, e.g. "
            "list[str].".format(where)
        )

    if typing.get_origin(annotation) is list:
        args = typing.get_args(annotation)
        if len(args) == 1 and SCALARS.get(args[0]) is not None:
            return "list[{}]".format(SCALARS[args[0]])
        raise SchemaError(
            "{}: list[{}] is not supported. Use list[str], list[int], "
            "list[float], list[bool] or list[Path].".format(
                where, ", ".join(getattr(a, "__name__", str(a)) for a in args) or "..."
            )
        )

    raise SchemaError(
        "{}: unsupported annotation {!r}. Allowed: Path, str, int, float, bool, "
        "list[...] of those. Optional/Union are not supported -- an argument is "
        "optional because it has a default, not because it is typed that way.".format(
            where, annotation
        )
    )


def return_name(annotation):
    """`Path` for one output, `dict[str, Path]` for several named ones."""
    if annotation is inspect.Signature.empty:
        raise SchemaError("run() has no return annotation. Annotate it `-> Path`.")

    if typing.get_origin(annotation) is dict:
        args = typing.get_args(annotation)
        if len(args) == 2 and args[0] is str and args[1] is Path:
            return "dict[str, path]"
        raise SchemaError(
            "return annotation: only dict[str, Path] is supported for multiple "
            "outputs, got {!r}.".format(annotation)
        )

    name = type_name(annotation, "return annotation")
    if name != "path":
        raise SchemaError(
            "run() must return a Path (or dict[str, Path]), not {}. Everything a "
            "tool produces is a file under its output directory.".format(name)
        )
    return name


def check_default(name, kind, default):
    """Validate a default against its annotation and make it JSON-ready."""
    if default is None:
        raise SchemaError(
            "argument '{}' defaults to None. There is no nullable type here: drop "
            "the default to make the argument required, or give it a real one.".format(name)
        )

    if kind.startswith("list["):
        element = kind[len("list[") : -1]
        if not isinstance(default, (list, tuple)):
            raise SchemaError(
                "argument '{}' is {} but defaults to {!r}.".format(name, kind, default)
            )
        return [check_default(name, element, item) for item in default]

    accepted = ACCEPTED_DEFAULTS[kind]
    # bool is a subclass of int, so `count: int = True` passes isinstance and
    # would ship an int argument the client renders as a check box.
    if isinstance(default, bool) != (kind == "bool"):
        raise SchemaError(
            "argument '{}' is {} but defaults to {!r}.".format(name, kind, default)
        )
    if not isinstance(default, accepted):
        raise SchemaError(
            "argument '{}' is {} but defaults to {!r}.".format(name, kind, default)
        )

    if kind == "path":
        return str(default)
    if kind == "float":
        return float(default)
    return default


def describe_run(run):
    """Turn run()'s signature and docstring into the published schema."""
    doc = inspect.getdoc(run)
    if not doc or not doc.strip():
        raise SchemaError(
            "run() has no docstring. Its first line is the tool description the "
            "client shows a clinician, so it cannot be left out."
        )

    signature = inspect.signature(run)
    try:
        hints = typing.get_type_hints(run)
    except Exception as error:  # a forward reference that does not resolve
        raise SchemaError("cannot resolve run()'s annotations: {}".format(error))

    arguments = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise SchemaError(
                "run() takes *{}. The runner calls run(**params) from a JSON "
                "object, so every argument must be named.".format(name)
            )
        if parameter.kind is parameter.POSITIONAL_ONLY:
            raise SchemaError(
                "argument '{}' is positional-only and cannot be passed by "
                "name.".format(name)
            )
        if name not in hints:
            raise SchemaError("argument '{}' has no annotation.".format(name))

        kind = type_name(hints[name], "argument '{}'".format(name))
        # The absence of a default is the ONLY thing that makes an argument
        # required. There is no `required=` to contradict it.
        required = parameter.default is parameter.empty
        arguments[name] = {"type": kind, "required": required}
        if not required:
            arguments[name]["default"] = check_default(name, kind, parameter.default)

    return {
        "description": doc.strip().splitlines()[0].strip(),
        "arguments": arguments,
        "returns": return_name(hints.get("return", signature.return_annotation)),
    }


def source_hash(src_dir):
    """sha256 of src/, so the server can spot a cached schema gone stale.

    The relative path goes into the digest as well as the bytes: renaming a
    module changes what gets imported, so it must change the hash.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        digest.update(path.relative_to(src_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def find_package(src_dir):
    """The one importable package under src/."""
    if not src_dir.is_dir():
        raise SchemaError("no src/ directory in {}.".format(src_dir.parent))

    candidates = sorted(
        p.name for p in src_dir.iterdir() if p.is_dir() and (p / "__init__.py").is_file()
    )
    if len(candidates) != 1:
        raise SchemaError(
            "expected exactly one package under {}, found {}. Pass --module to "
            "choose.".format(src_dir, ", ".join(candidates) or "none")
        )
    return candidates[0]


def load_run(src_dir, module_name):
    """Import the tool and hand back its run().

    src/ goes on the path so this works whether or not the project itself is
    installed in the venv; the venv is still what provides the dependencies.
    """
    sys.path.insert(0, str(src_dir))
    try:
        module = __import__(module_name)
    except ImportError as error:
        raise SchemaError(
            "cannot import {}: {}.\nIf this names a heavy dependency (torch, "
            "monai, nnunetv2...), the tool imports it at module level. Move it "
            "inside run(): schema generation must not cost a CUDA stack.".format(
                module_name, error
            )
        )

    run = getattr(module, "run", None)
    if run is None:
        raise SchemaError("{} defines no run().".format(module_name))
    if not inspect.isfunction(run):
        raise SchemaError("{}.run is {!r}, not a function.".format(module_name, type(run)))
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tool_dir",
        nargs="?",
        default=".",
        type=Path,
        help="the tool package directory, e.g. tools/amasss (default: cwd)",
    )
    parser.add_argument("--module", help="package to import, if src/ holds more than one")
    parser.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    args = parser.parse_args(argv)

    tool_dir = args.tool_dir.resolve()
    src_dir = tool_dir / "src"
    try:
        module_name = args.module or find_package(src_dir)
        schema = {"name": tool_dir.name}
        schema.update(describe_run(load_run(src_dir, module_name)))
        schema["source_hash"] = source_hash(src_dir)
    except SchemaError as error:
        sys.stderr.write("{}: {}\n".format(tool_dir.name, error))
        return 2

    text = json.dumps(schema, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
