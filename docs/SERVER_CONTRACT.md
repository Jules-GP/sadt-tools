# What the server has to do

Written for whoever works on
[slicer-remote-tool-server](https://github.com/Jules-GP/slicer-remote-tool-server).
It is the other half of the split: this repository holds tools that know nothing
about the server, so everything they stopped doing, the server now does.

## Status: re-checked against the server, 2026-08-14

Checked against `slicer-remote-tool-server` on branch **`newArch`** at
`fe40645`, **with uncommitted work in the tree** — `dispatch.py`/`runner.py`
moved under `server/execution/`, `registry.py` and friends under
`server/registry/`. Read-only inspection, so the findings below are about a
moving target; re-check before relying on them.

Two things changed since 2026-08-12, both in the server's favour:

- **`server/deployment.toml` exists now**, and mostly does not need to. The
  commit *"a tool needs no server-side configuration at all"* added
  `server/registry/conventions.py`, which derives from argument **names** what
  that file used to have to state. `deployment.toml` survives as a
  per-argument override for exceptions. The gap this document previously
  reported as blocking is closed.
- **Model arguments are recognised by name.** `model`, `*_model` and
  `*_reference` are published as a name picked from `DATA/<tool>/models/`;
  every other `path` may be uploaded. That is a safety property — a clinician
  must never be able to send weights from a laptop — and it is now the rule
  every tool here has to be named against. See "Naming" below.

**Still missing: the server injects no supervisor.** One occurrence of the word
in the whole repository, and it is documentation:

```
$ grep -rn "supervisor" --include='*.py' --include='*.md' .
ADDING_A_TOOL.md:39:| `*, sup` | — | the supervisor, never published |
```

`server/execution/runner.py` calls `run(**_call_arguments(run, job["params"]))`,
and `_call_arguments` only coerces what the job file carries. So `ASO`'s
fully-automated CBCT mode raises `SupervisorRequired` under this runner —
which is what `test_fully_automated_without_landmarks_is_refused_by_this_runner`
pins, and what will start failing the day this changes.

**The `supervisor` key is ignored, not refused.** `schema_tool._TOP_LEVEL_KEYS`
does not list it, and an unknown key is logged as a warning, so a tool that
declares one still loads — it just loads as though it had not. Adding
`"supervisor"` to that tuple is the one-line change that lets the server refuse
the tool up front instead.

## Naming, because the server reads it

A tool that hosts nothing gets this for free, but two names are load-bearing:

| Argument named | Published as |
|---|---|
| `model`, `*_model`, `*_reference` | a name picked from `DATA/<tool>/models/` |
| any other `path` | a file the caller may upload |

`ASO` shipped `landmark_models` — plural — which misses `*_model` by one letter
and would have put a file picker in front of a 4.7 GB weight bundle. It is
`landmark_model` now, with a test guarding it. Check a new tool's path
arguments against this table before opening the PR; nothing else will.

The rest of this document is the contract itself, kept as the reference the two
repositories are written against.

## 1. Discovery

`scripts/describe.py` in this repository turns a tool's `run()` signature into
the JSON the server publishes. Run it with the **tool's own interpreter** —
importing a tool needs the tool's dependencies:

```bash
/tools/AMASSS/.venv/bin/python /tools/scripts/describe.py /tools/AMASSS
```

It is dependency-free and stays Python 3.9-compatible so it can run inside any
tool venv, including one pinned to an old interpreter. It exits **2** with a
message on stderr for anything it cannot represent, and prints nothing — treat a
non-zero exit as "this tool is not loadable" and say so, rather than serving a
partial schema.

```json
{
  "name": "amasss",
  "description": "Segment craniofacial structures on a CBCT scan.",
  "arguments": {
    "scans":      {"type": "path", "required": true},
    "model":      {"type": "path", "required": true},
    "output_dir": {"type": "path", "required": true},
    "structures": {"type": "list[str]", "required": false,
                   "default": ["MAND", "MAX", "CB", "CV", "UAW"],
                   "choices": ["MAND", "MAX", "CB", "CV", "UAW", "SKIN",
                               "CBMASK", "MANDMASK", "MAXMASK"]},
    "device":     {"type": "str", "required": false, "default": "cuda",
                   "choices": ["cuda", "cpu"]}
  },
  "returns": "path",
  "source_hash": "966bf6d9…"
}
```

Three things to build against:

- **`required` comes only from the absence of a default.** There is no second
  declaration that can disagree with the signature.
- **`choices` is optional and means a fixed set.** `list[...]` with `choices` is
  several-of (the old `multichoice`); a scalar with `choices` is exactly-one (the
  old `choice`). Absent `choices`, the value is free-form.
- **`source_hash` is a sha256 of the tool's `src/`.** Cache schemas against it
  and regenerate when it moves; that is what it is for.
- **`supervisor` is optional and means the tool can call another one.** Absent
  on every tool that cannot, so nothing written before supervisors existed
  changes. See §5.

**Argument order is the signature's order.** Render forms in it.

## 2. Invocation

```
/tools/<name>/.venv/bin/python /opt/sadt/runner.py --job /jobs/<uuid>/job.json
```

`uv sync` installs each tool into its own venv, so `import sadt_<name>` works
directly — no `sys.path` juggling needed, though adding `/tools/<name>/src`
first is harmless and makes the runner work against an unsynced checkout too.

`server/runner.py` implements this, and
[`testkit/src/sadt_testkit/_driver.py`](../testkit/src/sadt_testkit/_driver.py)
in this repository is the same contract, ~60 lines, stdlib-only. The two were
written independently and agree — same coercion by annotation, same result
file, same error-class-name convention. **Keep them in step deliberately**: the
driver is what every tool's integration tests run against, so if they drift, the
tests here pass while production fails.

Two details from it worth carrying over:

- **Coerce by annotation.** JSON has no path type, so paths arrive as strings and
  must become `Path` for parameters annotated `Path` (or `list[Path]`).
  `typing.get_type_hints(run)` is how the driver decides.
- **An empty string is ABSENCE and must stay a string.** `Path("")` is
  `PosixPath(".")` — the current directory, and truthy — so coercing the
  "not supplied" default of an optional path hands the tool a real directory.
  This is not hypothetical: it made `ASO` read an unset `landmarks=""` as a
  supplied landmark folder and walk the entire checkout, `.venv` included.
  Calling `run()` directly in Python keeps the `""` the signature declares, so
  coercing it is also what makes the two call paths disagree. One line:

  ```python
  return Path(value) if value != "" else value
  ```

  `run_tool.py` and the testkit driver both do this; the server's runner must
  too. It is the price of `describe.py` refusing `None` defaults — an optional
  path has no other way to say "unset".
- **Never parse the result off stdout.** These tools print progress bars, nnUNet
  banners and shapeaxi chatter. The driver writes the result to a file whose path
  it is given. Anything scraped from stdout breaks the first time a dependency
  prints something new.

`run()` returns a `Path` or a `dict[str, Path]`; `returns` in the schema says
which. Today: `surgmovpred` returns `dict[str, path]` (`excel` and `csv`), the
others return a `path` — the output directory they were given.

## 3. Work that moved to the server

Each of these used to happen inside a tool and now happens nowhere unless the
server does it. **All of them are already implemented** — the file references
are in the status table above. They are kept here because they are the reasoning
behind the code, and the first person to touch that code will want it.

### Unpacking archives

No tool handles `.zip` any more. The server unpacks before calling `run()` and
passes a real file or directory. Two behaviours have to come with it:

- **the zip-bomb cap.** Tools used to apply `MAX_EXTRACTED_MB` themselves when
  they received the archive directly (AMASSS, ALI, BatchDentalSeg, CrownSeg all
  did). Nothing applies it now.
- **`strip_single_root`.** Zipping `patients/` on any OS produces
  `patients/<files>`, and callers want the files. AMASSS and ALI relied on this.

### Capping GPU work

**Implemented** (`config.py:76`, `MAX_GPU_JOBS`, across tools). Every tool used to hold a
`threading.BoundedSemaphore` — `AMASSS_MAX_GPU_JOBS`, `BATCHDENTALSEG_MAX_GPU_JOBS`,
`CROWNSEG_MAX_GPU_JOBS`, `ALI_MAX_GPU_JOBS`, all defaulting to 1 — because every
tool shared one server process. A tool is now its own process, so an in-process
semaphore would cap nothing and they have all been removed.

Nothing serialises GPU work any more. Two concurrent AMASSS jobs will both take
the card. The server has to hold that limit, and it now has to be a limit
*across* tools, not per tool: an AMASSS run and a CrownSeg run compete for the
same device.

### Creating and owning the output directory

`output_dir` is a required argument on every tool. Create it (or let the tool —
they all `mkdir(parents=True, exist_ok=True)`), pass it, and archive what comes
back. Tools write **only** there; each has a test asserting it.

Intermediates go in a dotted subdirectory that the tool removes before returning
(`.amasss_work/`, `.batchdentalseg_work/`, `.crownseg_work/`). If one survives, a
run crashed.

### Resolving model weights

Tools no longer touch `data_store` or `settings.CROWNSEG_MODEL`. The server
resolves the name and passes a path — but **what kind of path differs per tool**:

| Tool | `model` is | Note |
|---|---|---|
| `surgmovpred` | a folder | every `stacking_package.pkl` under it is loaded, recursively |
| `amasss` | the bundle root | one subfolder per structure code (`MAND/`, `MAX/`, …); a single wrapper folder is descended into |
| `batchdentalseg` | **the bundle folder itself** | its *name* selects the model and its label table — see below |
| `crownseg` | a `.pth` file | not a folder |

**`batchdentalseg` is the sharp edge**: the folder's basename must equal a key in
its `catalogs.MODELS` (`DentalSegmentator`, `PediatricDentalSeg`,
`NasoMaxillaDentSeg`, `UniversalLab`), which must in turn equal the folder
`scripts/data-manifest.yml` downloads that bundle into. The server-side test that
enforced that has no home any more — this repository cannot read the manifest.

**This is the one that needs a `deployment.toml`** — see the status section.
Tool directories here are lowercase
(`amasss`, `batchdentalseg`, `crownseg`, `surgmovpred`) and that is what the
schema's `name` is, while `DATA/` still uses `AMASSS/`, `BatchDentalSeg/`,
`CrownSeg/`, `SurgMovPred/`. Map or rename; do not assume they match.

### Configuring logging

Tools use a plain module logger and attach no handlers — a library that
configures logging takes the decision away from whatever runs it. The runner owns
handlers, levels and formatting. Without that, tool logs go nowhere.

### Settings that became arguments

`run()` must not read the environment, so these moved into the signature with
their previous defaults. The server passes them only to override:

| Was | Now | Default |
|---|---|---|
| `settings.DEVICE` | `device` on every GPU tool | `"cuda"`, falls back to CPU with a warning |
| `settings.AMASSS_TILE_STEP_SIZE` | `tile_step_size` (amasss) | `0.5` |
| `settings.AMASSS_GPU_RESAMPLING` | `gpu_resampling` (amasss) | `true` |
| `settings.BATCHDENTALSEG_TILE_STEP_SIZE` | `tile_step_size` (batchdentalseg) | `0.5` |
| `settings.CROWNSEG_NUM_WORKERS` | `num_workers` (Crown_Seg) | `2` |
| `settings.*_MAX_GPU_JOBS` | — | gone; see "Capping GPU work" |

`tile_step_size` and `gpu_resampling` **change the segmentation**. The tools
record what was used in their run report, so a mask stays reproducible.

## 4. Errors

There is no shared exception type, because there is no shared package. Each tool
defines its own `ToolInputError(ValueError)`, and `surgmovpred` deliberately kept
upstream's `FileNotFoundError` / `RuntimeError` instead. So the server cannot
`isinstance`-check the way `base.ToolArgumentError` allowed.

**Map by exception class name.** The suggested convention, which the runner
should write into its result file:

```json
{"error": {"type": "ToolInputError", "message": "Unknown structure code(s): RC. Known: MAND, MAX, …"}}
```

- `ToolInputError`, `ValueError`, `FileNotFoundError` → **422**, message passed
  through: these always mean the request or the staged data is wrong, and every
  message is written to be read by whoever sent it.
- `ToolUnavailableError` → **503**: the tool is installed but its engine is not.
  Only `crownseg` raises it today, when the `segmentation` extra is missing.
- anything else → **500**, message not passed through.

If you would rather not match on names, the alternative is a `type` field the
tools set explicitly — but that is a shared convention either way, and names are
already the convention.

## 5. Sequencing tools

Almost all sequencing is the server's: where one tool's output is another's
input, the server chains them and neither tool knows the other exists.

| Chain | Why | Handoff |
|---|---|---|
| `Crown_Seg` → `ALI` | ALI's IOS engine needs meshes carrying tooth labels. `ALILogic.ensure_segmented()` used to import CrownSeg directly. | CrownSeg's `run_report.json` has `segmented_meshes`: absolute paths of every mesh that now carries labels, whether this run produced them or found them already labelled. Feed those to ALI. |

CrownSeg passes an already-labelled mesh through untouched, so re-running the
chain on mixed input is cheap and safe. ALI refuses an unlabelled batch up front,
naming `Crown_Seg`, rather than failing per mesh.

### The one chain that cannot be a chain

**`ASO` fully-automated CBCT needs `ALI` from the middle of its own run.** It
recentres each scan, predicts landmarks **on the centred volumes**, then
registers — the order the Slicer chain used (`PRE_ASO_CBCT` before `ALI_CBCT`).
Running ALI first and handing ASO the markups reorders those two steps.

That reordering *ought* to be exact: recentring resamples onto a grid shifted by
the same offset, so the voxel array is untouched and only the origin metadata
moves. But ALI's `physical_position` takes `abs(origin / spacing)`, which does
not commute with moving the origin — [issue #11]. Rather than bet a clinical
result on it, ASO calls ALI where it always ran.

So ASO takes a **supervisor**, and this is the piece the server does not have
yet:

```python
predictions = sup.run("ALI", input=centered_root, model=bundle, output_dir=..., landmarks=[...])
```

What the server has to provide:

- an object with five members — `run(tool, **params)`, `out`, `tmp`,
  `progress(fraction, message)`, `log(message)` — passed as the keyword-only
  `sup`. Nothing is imported across the two repositories; it is duck-typed.
  **[`scripts/run_tool.py`](../scripts/run_tool.py) is a working one**, in about
  fifty lines, and the shortest readable reference for this section.
- **Select the callee by VENV, not by interpreter.** uv's venvs all symlink
  `bin/python` to the same underlying CPython, so dispatching on the resolved
  binary makes every tool look like every other one — and the callee gets
  imported into its *caller's* environment, which is precisely the dependency
  mixing the split exists to prevent. Compare `sys.prefix` against
  `<TOOLS_DIR>/<name>/.venv`. This one cost an hour to find.
- **Absolute paths across the boundary.** The callee should start from a neutral
  working directory — that is what makes "writes only under `output_dir`"
  testable — so every path handed to it must already be absolute.
- **The supervisor's own scratch does not belong in `output_dir`.** The tool is
  held to writing only inside it; whatever drives the tool should not undo that
  from the other side.
- `sup.run` invokes the named tool the way the runner does — its own venv, its
  own interpreter — and returns what that tool returned: a `Path`, or a
  `dict[str, Path]`.
  [`testkit/src/sadt_testkit/_driver.py`](../testkit/src/sadt_testkit/_driver.py)
  already does exactly this, and `tools/ASO/tests/test_integration.py` wraps it
  in a dozen lines to make a working supervisor. That is the reference.
- **a concurrency answer.** A supervised call holds its caller's slot for the
  whole nested run. The old in-process version had the same problem and said so:
  four concurrent ASO runs each waiting on a fifth slot deadlock the server,
  `/health` included. Whatever caps tool concurrency must not count the parent
  while it is blocked on a child.

`describe.py` publishes `"supervisor": true` for a tool that takes one, so **a
runner without supervisor support can refuse the tool at discovery rather than
call it and fail halfway**. Until it lands, ASO's other three modes are servable
and fully-automated CBCT is not — and passing `landmarks` (a folder of
`.mrk.json`) makes even that mode work with no supervisor at all, which is also
what lets ASO be used standalone.

[issue #11]: https://github.com/Jules-GP/sadt-tools/issues/11

## 6. What to delete, and when

Only after the corresponding tool's PR merges *here*, never before — the server
keeps working off its own copy until this one is proven.

Server-side, the whole `server/tools/` tree eventually goes, and with it the
parts of `base.py` that only existed for it: `Tool`, `ArgSpec`, `Selection`,
`ResolvedPath`, `FILE_TYPES`, `ToolArgumentError`, `ToolUnavailableError`, and
`registry.py`'s import-every-tool-at-startup. What stays is `main.py`,
`dispatch.py`, `runner.py`, `data_store.py`, `security.py`, `config.py`.

`requirements.txt` loses every tool dependency — torch, nnunetv2, SimpleITK, vtk,
itk, monai, dicom2nifti, pandas, scikit-learn, lightgbm, joblib, openpyxl, odfpy.
Each now lives in one tool's lockfile. The server keeps only what the server
itself imports.

## 7. Two things to know before you start

- **CrownSeg does not currently run, and did not before the migration.**
  `shapeaxi.dental_model_seg` reads `DentalModelSeg` off `shapeaxi.saxi_nets`,
  but in the whole 2.0.x line the class lives in `shapeaxi.saxi_nets_lightning`.
  The pre-port tool fails the same way on the deployed image. This repository
  carries a two-line workaround (`_restore_moved_class`) guarded so it vanishes
  when upstream fixes it. Anything downstream of crown segmentation — ALI's IOS
  half, the IOS modes of ASO/AREG/FlexReg — has been broken on the current image.
- **Disk.** Each torch venv is 7.2–7.7 GB at cu128. Deduplication across them is
  not automatic and depends on `UV_CACHE_DIR` sitting on the same filesystem as
  the venvs; see the repository README.
