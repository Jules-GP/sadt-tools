#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Report the distinct torch and Python versions across the tools, and their cost.

    uv run scripts/audit.py

This is a DIAGNOSTIC. It reads `pyproject.toml`, `uv.lock` and, when it is
there, `.venv/`; it never writes, never touches a lockfile and never proposes to
merge environments automatically. Aligning two tools onto one torch is a
scientific decision -- the same weights on a different torch can produce
different masks -- so the fix is always a human revalidating outputs against
reference data, not a script rewriting a pin.

The savings figures answer one question: how much disk would drop out if a
version disappeared. Everything else is context for that question.
"""

import argparse
import os
import sys
import tomllib
from pathlib import Path

# Used only when a tool has no .venv to measure. Both come from measuring a
# built deployment image, not from arithmetic: one isolated torch CUDA stack
# installs to 4.9 GB, the CPU-only wheel to roughly 0.9 GB.
ESTIMATED_GB = {"cuda": 4.9, "cpu": 0.9}

GB = 1024**3


class Tool:
    """One tools/<name>/ directory, as far as versions and disk go."""

    def __init__(self, path):
        self.path = path
        self.name = path.name
        self.requires_python = None
        self.declared_torch = None  # what pyproject asks for
        self.locked_torch = None  # what uv.lock resolved to
        self.cuda = False
        self.venv_python = None
        self.venv_bytes = 0
        self.torch_bytes = 0
        self.torch_lib_links = []

    @property
    def torch(self):
        """The version that will actually be installed, or why there is none."""
        if self.locked_torch:
            return self.locked_torch
        if self.declared_torch:
            return "unpinned" if self.declared_torch == "torch" else self.declared_torch
        return "absent"

    @property
    def runtime_gb(self):
        """Disk for this tool's torch runtime: measured if possible."""
        if self.torch_bytes:
            return self.torch_bytes / GB
        return ESTIMATED_GB["cuda" if self.cuda else "cpu"]


def read_pyproject(tool):
    data = tomllib.loads((tool.path / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    tool.requires_python = project.get("requires-python")
    for requirement in project.get("dependencies", []):
        # "torch==2.2.0", "torch>=2.4", "torch" -- keep it verbatim rather than
        # normalising, because the exact spelling is what a reviewer compares
        # against upstream.
        stripped = requirement.replace(" ", "")
        if stripped == "torch" or stripped.startswith(("torch=", "torch>", "torch<", "torch~", "torch!")):
            tool.declared_torch = stripped
            break


def read_lock(tool):
    lock = tool.path / "uv.lock"
    if not lock.is_file():
        return
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    for package in data.get("package", []):
        name = package.get("name", "")
        if name == "torch":
            tool.locked_torch = package.get("version")
        elif name.startswith("nvidia-"):
            # The CUDA runtime ships as its own wheels, so their presence is
            # what separates a 4.9 GB stack from a 0.9 GB one.
            tool.cuda = True


def measure_venv(tool, seen_inodes):
    """Apparent size of .venv, plus the torch runtime inside it.

    `seen_inodes` is shared across tools: a file already counted for another
    tool is a hardlink, so counting it again would report disk that dedup has
    already saved. The per-tool figure stays apparent size (what the tool
    would cost alone); the unique total is what the image actually pays.
    """
    venv = tool.path / ".venv"
    if not venv.is_dir():
        return 0

    config = venv / "pyvenv.cfg"
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            if line.startswith("version"):
                tool.venv_python = line.split("=", 1)[1].strip()

    unique = 0
    for root, _, files in os.walk(venv, followlinks=False):
        in_torch = f"{os.sep}torch" in root or f"{os.sep}nvidia" in root
        for name in files:
            path = os.path.join(root, name)
            try:
                info = os.lstat(path)
            except OSError:
                continue
            tool.venv_bytes += info.st_size
            if in_torch:
                tool.torch_bytes += info.st_size
                if name == "libtorch_cuda.so":
                    tool.torch_lib_links.append(info.st_nlink)
            if info.st_ino not in seen_inodes:
                seen_inodes.add(info.st_ino)
                unique += info.st_size
    return unique


def group(tools, key):
    """Distinct values of `key`, most common first, each with its tools."""
    groups = {}
    for tool in tools:
        groups.setdefault(key(tool), []).append(tool)
    return sorted(groups.items(), key=lambda item: (-len(item[1]), str(item[0])))


def format_counts(groups):
    """`2.2.0 (5 tools) | 2.4.1 (2) | unpinned (3)` -- unit on the first only."""
    parts = []
    for index, (value, members) in enumerate(groups):
        unit = " tool" + ("s" if len(members) > 1 else "") if index == 0 else ""
        parts.append("{} ({}{})".format(value, len(members), unit))
    return " | ".join(parts)


def report(tools, unmigrated, unique_bytes):
    if not tools:
        print("No tool package found. A tool counts once it has a pyproject.toml.")
    else:
        torch_groups = group(tools, lambda t: t.torch)
        python_groups = group(tools, lambda t: t.requires_python or "unspecified")
        print("torch  " + format_counts(torch_groups))
        print("python " + format_counts(python_groups))

        print("\nper tool")
        width = max(len(t.name) for t in tools)
        for tool in sorted(tools, key=lambda t: t.name):
            measured = "measured" if tool.torch_bytes else "estimated"
            disk = (
                "{:>6.1f} GB {}".format(tool.runtime_gb, measured)
                if tool.torch != "absent"
                else "{:>6.2f} GB venv".format(tool.venv_bytes / GB)
            )
            print(
                "  {:<{width}}  python {:<14} torch {:<12} {}".format(
                    tool.name,
                    tool.venv_python or tool.requires_python or "?",
                    tool.torch,
                    disk,
                    width=width,
                )
            )

        report_disk(tools, torch_groups, unique_bytes)

    if unmigrated:
        print("\nnot migrated yet (no pyproject.toml): " + ", ".join(sorted(unmigrated)))
    print("\nRead-only: nothing above was modified. Aligning a pin means revalidating")
    print("the model's outputs against reference data first -- it is not a packaging fix.")


def report_disk(tools, torch_groups, unique_bytes):
    runtimes = [(value, members) for value, members in torch_groups if value != "absent"]
    if not runtimes:
        return

    print("\ndisk")
    measured = [t for t in tools if t.venv_bytes]
    if measured:
        apparent = sum(t.venv_bytes for t in measured) / GB
        print(
            "  {} venv(s) present: {:.1f} GB apparent, {:.1f} GB unique on disk".format(
                len(measured), apparent, unique_bytes / GB
            )
        )
        links = [n for t in tools for n in t.torch_lib_links]
        if len(links) > 1:
            state = "active" if max(links) > 1 else "BROKEN -- every venv holds its own copy"
            print("  libtorch_cuda.so link count {}: hardlink dedup {}".format(links, state))

    per_runtime = {}
    for value, members in runtimes:
        per_runtime[value] = max(t.runtime_gb for t in members)
    print(
        "  {} distinct torch runtime(s), ~{:.1f} GB deduplicated".format(
            len(runtimes), sum(per_runtime.values())
        )
    )

    if len(runtimes) > 1:
        print("\nsuggestions")
        target = runtimes[0][0]
        for value, members in runtimes[1:]:
            names = ", ".join(sorted(t.name for t in members))
            print(
                "  {} pins torch {}; aligning to {} would drop one runtime, "
                "saving ~{:.1f} GB".format(names, value, target, per_runtime[value])
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: the one holding this script)",
    )
    args = parser.parse_args(argv)

    tools_dir = args.root / "tools"
    if not tools_dir.is_dir():
        sys.stderr.write("no tools/ directory under {}\n".format(args.root))
        return 2

    tools, unmigrated, seen_inodes, unique_bytes = [], [], set(), 0
    for path in sorted(p for p in tools_dir.iterdir() if p.is_dir()):
        if not (path / "pyproject.toml").is_file():
            unmigrated.append(path.name)
            continue
        tool = Tool(path)
        read_pyproject(tool)
        read_lock(tool)
        unique_bytes += measure_venv(tool, seen_inodes)
        tools.append(tool)

    report(tools, unmigrated, unique_bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
