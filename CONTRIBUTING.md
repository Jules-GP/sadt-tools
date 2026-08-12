# Adding a tool

Every version choice in this repository is a scientific decision before it is a
packaging one. A tool that imports cleanly is not a working tool, and a pin that
resolves is not necessarily the pin the results were produced with. Most of what
follows exists to keep those two things apart.

## 1. Branch

One branch and one pull request per tool, off `main`, never on `main`:

- `tool/<name>` — migrating or changing a tool
- `infra/<topic>` — repository-level work
- `fix/<topic>` — corrections

Commit messages are a single short sentence prefixed with `ADD :`, `FIX :`,
`CLEAN :` or `UPDATE :`, in English, with no body unless the change genuinely
needs one:

```
ADD : AMASSS tool package with uv lockfile
FIX : describe.py silently accepted unsupported annotations
```

No AI attribution of any kind — no `Co-Authored-By`, no generated-with trailer,
no emoji. Never merge your own PR, and never force-push a branch that has one
open.

Several tools at once is expected. Use `git worktree` rather than stashing:

```bash
git worktree add ../sadt-amasss tool/amasss
```

## 2. Copy the template

```bash
cp -r tools/_template tools/amasss
cd tools/amasss
mv src/sadt_template src/sadt_amasss
```

Then edit `pyproject.toml` (`name`, `requires-python`, `dependencies`), and
delete the placeholder pipeline. The directory name is the tool's slug — it is
what the server puts in a path — so keep it lowercase and stable.

## 3. Write `run()`

One public callable. Nothing else is part of the contract.

**Annotations are stdlib only**: `Path`, `str`, `int`, `float`, `bool`, and
`list[...]` of those. `Path` means "a file or a directory"; everything else is a
scalar. No `Optional`, no unions, no custom marker types, nothing imported from
the server. `scripts/describe.py` refuses anything else rather than publish a
schema the client will render wrongly.

**No default means required.** That is the only thing `required` in the schema
comes from, so there is no second declaration to contradict the signature.
`n: int = None` is refused: it makes a required argument look optional.

**Path arguments take a folder.** The server pays a process start-up cost per
call, so a folder of 40 scans has to be one call, not 40. Upstream is
inconsistent about this; standardise on batch-capable inputs. `iter_scans` in
the template is the shape to copy.

**Heavy imports go inside `run()`**, and that covers the whole import chain, not
just `__init__.py`. CI imports every tool on every PR to generate its schema; a
module-level `import torch` makes that cost a CUDA stack and fail on a machine
without one.

**The first docstring line is the tool description** shown to a clinician in the
client. Write it as a clinical action, not an implementation note. Everything
after it is for maintainers and never reaches the UI.

**Write only under `output_dir`.** Every tool takes it as a required `Path`
argument. `run()` returns a `Path`, or a `dict[str, Path]` when there are
several named outputs, and it must not write beside its inputs, into the working
directory or anywhere else. `output/` at the repository root is gitignored —
point a manual run at it rather than scattering results through the tree.

**`run()` does not read the environment, does not know about `/DATA`, and does
not call another tool.** Path resolution and tool sequencing belong to the
server. Model weights are not packaged either: the server fetches them into
`/DATA/<tool>/models` and passes the path in.

Check the result before going further:

```bash
uv sync
.venv/bin/python ../../scripts/describe.py .
```

## 4. Port the implementation

This is a repackaging exercise, not a rewrite. Keep the upstream algorithm
byte-for-byte identical wherever you can; the Slicer-specific scaffolding around
it (Qt widgets, queue tables, RAM watchdogs, progress dialogs) is not part of
the algorithm and does not come across. If you have to change logic, say so
explicitly in the PR description and in the tool's README.

## 5. Pin what upstream actually used

**Do not bump torch, monai or numpy to "something newer that works".** Changing
them can change model outputs, and outputs must be revalidated against reference
data before any version moves. If upstream's pins are ambiguous or
contradictory, ask — do not pick one yourself.

`requires-python` must be accurate: bound it by what the pins actually support,
or uv will pick an interpreter with no wheels for them and spend a quarter of an
hour building from source.

CUDA-variant wheels need their own index, per tool, and `explicit` is
load-bearing — without it uv looks for *every* package on that index:

```toml
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
```

`pytorch3d` needs torch present at build time and fails under uv's build
isolation. Install it with `--no-build-isolation` and an explicit order, and
write the exact working incantation into the tool's README — deployment
precompiles it once into a local `/wheels` directory. If it will not build and
the only workaround is a different torch, stop and ask.

**Do not create a `[tool.uv.workspace]`.** A workspace forces one shared
lockfile across its members, which is precisely what must not happen here: uv
cannot resolve torch 1.13 and torch 2.4 in one lock. These tools live in one
repository for convenience and share no dependency resolution. For the same
reason there is no shared `sadt-core` package — small helpers like `iter_scans`
are copied between tools on purpose.

Commit `uv.lock`. CI runs `uv sync --frozen`, which fails if it is stale.

## 6. Validate

`tests/test_run.py` calls `run()` end to end on real input and asserts on the
output: the expected files exist, they are a plausible size, and where a
reference output exists, results match within a documented tolerance.

Test data lives in `tools/<name>/tests/data/` as **a download script plus
checksums** — never large binaries, never patient data, and the fixtures must be
anonymised and public-domain. If no suitable public sample exists, ask before
committing anything.

GPU tests carry `@pytest.mark.gpu`. CI skips them (`-m "not gpu"`) because the
runner has no CUDA device, which makes them your responsibility: run them by
hand and state in the PR that you did and what came out.

The tool's README records what you validated against — which input, which model
weights, which reference, what tolerance.

You may use the **Slicer Cloud** application to exercise a tool end to end
against realistic data when you cannot validate locally. Ask for the endpoint
and credentials rather than guessing, and never commit them.

## 7. Record provenance

The tool README opens with the block from `tools/_template/README.md`, filled
in, and the same PR adds the tool's row to [PROVENANCE.md](PROVENANCE.md).
Upstream history is not grafted into this repository — it is one history for
sixteen unrelated modules and the result would be unreadable — so this table is
the only record of where an algorithm came from.

## 8. Open the pull request

State:

- the version pins chosen and why,
- what you validated against, and the result of the GPU tests,
- anything changed from upstream, and why.

Prefer an open question in the PR over a silent decision.

Once the tool is merged here, open a companion PR on
`slicer-remote-tool-server` deleting it there. **Never delete first** — the
server keeps working off its copy until this one is proven.

## Stop and ask when

- upstream's pins are ambiguous or contradictory and you would have to choose,
- porting would mean changing the algorithm rather than repackaging it,
- you cannot find suitable non-patient test data,
- a tool needs model weights you cannot locate,
- `pytorch3d` will not build and the workaround means moving torch.

## Repository checks

```bash
uv run --no-project --python 3.12 --with pytest -- pytest scripts/tests -q
uv run scripts/audit.py
```

`audit.py` is read-only by design: it reports distinct torch and Python versions
and what dropping one would save, and it never edits a lockfile. Aligning two
tools onto one runtime means revalidating both against reference data first.
