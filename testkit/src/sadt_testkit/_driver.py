"""Runs one tool's `run()` inside that tool's own interpreter.

Executed BY the tool's venv python, never imported by it: the tool's
environment does not have sadt_testkit installed and must not need it. So this
file is stdlib-only, takes everything on the command line, and stays 3.9
compatible because a tool may pin an old interpreter.

It is deliberately a miniature of the server's runner -- same job, same shape:
import the tool, coerce a JSON object into the arguments `run()` declares, call
it, hand the paths back. If the two ever disagree about how a parameter is
coerced, an integration test here would pass while the server failed, so keep
this boring and keep it matching.
"""

import argparse
import json
import sys
import typing
from pathlib import Path


def coerce(value, annotation):
    """Turn a JSON value into what the annotation asks for.

    Only paths need it -- JSON has no path type, so they travel as strings.
    Everything else the schema allows (str, int, float, bool, and lists of
    them) is already the right type on arrival.
    """
    if annotation is Path:
        return Path(value)
    if typing.get_origin(annotation) is list and typing.get_args(annotation) == (Path,):
        return [Path(item) for item in value]
    return value


def unwrap(result):
    """`Path` or `dict[str, Path]` -> something JSON can carry."""
    if isinstance(result, Path):
        return {"kind": "path", "value": str(result)}
    if isinstance(result, dict):
        return {"kind": "dict", "value": {k: str(v) for k, v in result.items()}}
    raise TypeError(
        "run() returned {!r}; the contract is a Path or a dict[str, Path].".format(
            type(result).__name__
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="the tool's src/ directory")
    parser.add_argument("--package", required=True, help="package to import")
    parser.add_argument("--params", required=True, help="JSON file of arguments")
    parser.add_argument("--result", required=True, help="JSON file to write the result to")
    args = parser.parse_args(argv)

    sys.path.insert(0, args.src)
    module = __import__(args.package)

    with open(args.params, encoding="utf-8") as handle:
        params = json.load(handle)

    hints = typing.get_type_hints(module.run)
    call = {name: coerce(value, hints.get(name)) for name, value in params.items()}

    result = unwrap(module.run(**call))

    # Written to a file, never printed: these tools put progress bars, nnUNet
    # banners and shapeaxi chatter on stdout, and a result parsed out of that
    # stream would break the first time one of them printed something new.
    with open(args.result, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
