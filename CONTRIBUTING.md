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
cp -r tools/_template tools/AMASSS
cd tools/AMASSS
mv src/sadt_template src/sadt_amasss
```

Then edit `pyproject.toml` (`name`, `requires-python`, `dependencies`), and
delete the placeholder pipeline.

**The directory name IS the tool name.** The server looks up a tool's
interpreter at `<TOOLS_DIR>/<tool name>/.venv/bin/python`, so the two cannot
differ. The convention:

- an acronym stays as it is — `ALI`, `ASO`, `AMASSS`;
- anything else is capitalised words joined by underscores —
  `Batch_Dental_Seg`, `Crown_Seg`, `Surg_Mov_Pred`, `Example_Tool`;
- a client renders the name with the underscores as spaces where it can.

The **Python package** under `src/` stays lowercase — `sadt_batch_dental_seg`,
not `sadt_Batch_Dental_Seg` — because it is a Python identifier and PEP 8
applies. `describe.py` finds it as "the single package under src/", so the two
spellings never have to agree.

Renaming a tool is a **breaking change for every client**: the name is what a
client sends, and the Slicer modules hold it in `TOOL_NAME`. Change it in the
same breath as the client, or the tool disappears from the UI.

## 3. Write `run()`

One public callable. Nothing else is part of the contract.

**Annotations are stdlib only**: `Path`, `str`, `int`, `float`, `bool`,
`Literal[...]` and `list[...]` of those. `Path` means "a file or a directory";
everything else is a scalar. No `Optional`, no unions, no custom marker types,
nothing imported from the server. `scripts/describe.py` refuses anything else
rather than publish a schema the client will render wrongly.

**A fixed set of options is a `Literal`.** `list[Literal["MAND", "MAX", ...]]`
is several-of, a bare `Literal["MERGED", "SEPARATE"]` is exactly-one, and
describe.py publishes both as `choices` so the client can render a picker.
Saying it in the annotation is the point — the `ArgSpec.choices` tables this
replaces were a second declaration, and they drifted. The default is checked
against the options, so a picker can always produce the value the tool starts
from.

`Literal` is published, not enforced: the runner calls `run(**params)` from a
JSON object, so a stale client or a direct caller can still send anything.
Validate the value in the tool and raise on a bad one.

**No default means required.** That is the only thing `required` in the schema
comes from, so there is no second declaration to contradict the signature.
`n: int = None` is refused: it makes a required argument look optional.

**Path arguments take a folder.** The server pays a process start-up cost per
call, so a folder of 40 scans has to be one call, not 40. Upstream is
inconsistent about this; standardise on batch-capable inputs. `iter_scans` in
the template is the shape to copy.

**Never unpack a `.zip`.** The server unpacks archives before `run()` is
called, so a tool always receives a real file or directory. The upstream tools
each carried their own extraction, zip-bomb cap and scratch directory for it;
that is exactly the duplication the split removes, and re-adding it in a tool
puts path resolution back on the wrong side of the line.

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

**`run()` does not read the environment and does not know about `/DATA`.** Path
resolution belongs to the server. Model weights are not packaged either: the
server fetches them into `/DATA/<tool>/models` and passes the path in.

**Tool sequencing belongs to the server too — with one exception.** Where one
tool's output is another's input, the server chains them and neither tool knows
the other exists. That covers almost every case: `Crown_Seg → ALI` is two calls
with a folder in between. The exception is a tool that needs another *in the
middle of its own run*, where the output cannot simply be fed in beforehand.
`ASO` is the only one: fully-automated CBCT recentres its scans, predicts
landmarks **on the centred volumes**, and then registers. For that, take a
supervisor:

```python
def run(scans: Path, reference: Path, output_dir: Path, *, sup=None) -> Path:
    ...
    landmarks = sup.run("ALI", input=centered, model=bundle, output_dir=...)
```

- `sup` is **keyword-only and unannotated**, and that shape is the marker:
  `describe.py` keeps it out of the schema and publishes `"supervisor": true`
  instead, so a runner that cannot inject one refuses the tool rather than
  calling it and failing halfway. A `sup` that is positional or annotated is a
  hard error, not a schema entry.
- It is **duck-typed**. Never import a supervisor type — that would need a
  package shared with the server, which is what the split removes.
- Five members, nothing more: `sup.run(tool, **params)`, `sup.out`, `sup.tmp`,
  `sup.progress(fraction, message)`, `sup.log(message)`.
- `sup.run("ALI", ...)`, never `sup.ALI(...)`. A typo in a string is greppable;
  a typo in an attribute is an `AttributeError` fifteen minutes into a job, and
  the call graph stops being inspectable.
- **Give the caller a way in.** A tool that takes `sup` should also accept the
  dependency's output as an ordinary argument (`landmarks: Path = ""`) and skip
  the call when it is supplied. That is what keeps it usable standalone, and it
  is the only way it works with no supervisor at all.

Reach for this only when the ordering genuinely forbids chaining. Two calls with
a folder in between is simpler for everyone, and the server already does it.

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

**When a tool's input is another tool's output**, test against the real thing
rather than a stand-in. `sadt-testkit` runs the other tool through *its* venv as
a subprocess — the way the server does — so nothing is imported across tools:

```python
from sadt_testkit import is_built, run_tool

@pytest.mark.skipif(not is_built("Crown_Seg"), reason="run uv sync in tools/Crown_Seg")
def test_on_freshly_segmented_meshes(tmp_path):
    segmented = run_tool("Crown_Seg", meshes=..., model=..., output_dir=tmp_path / "seg")
    run(scans=segmented, model=..., output_dir=tmp_path / "out")
```

Skip, never fail, when the other tool is not built: CI builds each tool in its
own job. `tools/_template/tests/test_integration.py` is the file to copy, and
[testkit/README.md](testkit/README.md) covers the rest. This is a **development
dependency** and must never appear in `[project] dependencies`.

It is also what a supervisor is built out of in a test: `tools/ASO`'s `LocalSup`
is a dozen lines wrapping `run_tool`, and the tool cannot tell it from the
server's.

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
