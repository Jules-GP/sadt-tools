# sadt-crownseg

Labels every tooth of an intraoral surface scan with its dental number, as a
point-data array on a copy of the mesh. That array is the precondition for ALI's
IOS landmark identification and for the IOS modes of ASO, AREG and FlexReg,
which is why this is its own tool.

## Provenance

Not ported: written against `shapeaxi` directly. The Slicer modules ran the
`dentalmodelseg` executable out of Slicer's own bin directory, and that
executable is only the console-script entry point of the `shapeaxi` PyPI package
(`dentalmodelseg = shapeaxi.dental_model_seg:cml`). There is no Slicer binary to
shell out to here, so `shapeaxi.dental_model_seg.main` is called directly with
the namespace its own `cml()` would have built.

Carried over from `slicer-remote-tool-server`'s `tools/CrownSeg/`, whose history
this repository holds — `git log --follow` on `src/sadt_crownseg/pipeline.py`
reaches back through it.

Changes made by this port:

- **ALI no longer calls this tool.** The server-side version existed partly so
  `ALILogic.ensure_segmented()` could import it and segment raw meshes inline.
  Tools do not call each other any more, so the server sequences CrownSeg before
  ALI, and the report's `segmented_meshes` list is what it reads to do it.
- **Scratch space lives under `output_dir`** (`.crownseg_work/`, removed before
  returning), and `segment_crowns()` returns the run report rather than a
  `CrownSegRun`.
- **Zip extraction removed** — the server unpacks archives before `run()`.
- **The model is a required argument.** It used to be a name resolved through
  the data store, with `default_model_path()` reading `settings.CROWNSEG_MODEL`;
  weights now arrive as a `Path`, like every other tool.
- **The GPU semaphore is gone** — each call is its own process.

## Working around a shapeaxi bug

**CrownSeg does not currently run in the deployment, and did not before this
port either.** `shapeaxi.dental_model_seg` — the only supported way in — reads
`DentalModelSeg` off `shapeaxi.saxi_nets`, but in the 2.0.x line the class lives
in `shapeaxi.saxi_nets_lightning`. Every 2.0.x release (2.0.0, 2.0.1, 2.0.2) has
this; 1.x referenced the right module. Running the pre-port tool in the deployed
container fails before it touches a mesh:

```
AttributeError: module 'shapeaxi.saxi_nets' has no attribute 'DentalModelSeg'
```

`_restore_moved_class()` in `pipeline.py` points the name back at where the class
went, before calling shapeaxi's `main()`. Two lines rather than reimplementing
`main()`, so the entry point stays the one shapeaxi supports; and it is guarded
by `hasattr`, so it becomes a no-op the day a release puts the name back. **This
should be reported at DCBIA-OrthoLab/ShapeAXI, and the function deleted once a
release carries the fix.** Two tests cover it: that it restores the name, and
that it does nothing when the name is already there.

## What it does

| | |
|---|---|
| Inputs | `meshes`: one `.vtk`/`.stl` surface or a folder of them, searched recursively. `model`: the checkpoint. `output_dir`: where results go. |
| Outputs | One labelled `.vtk` per mesh, mirroring the input tree, plus `run_report.json`. |
| Model files | One checkpoint, `07-21-22_val-loss0.169.pth` (6.6 MB), from the Fly-by-CNN 3.0 release. |
| GPU | Used when available; `device="cpu"` works and is much slower. |

Three things worth knowing before reading a result:

- **An already-labelled mesh is passed through, not re-predicted.** That is what
  makes a mixed batch of raw and pre-segmented meshes one call. Pass
  `skip_segmented=False` to force the network to run anyway.
- **A pre-segmented `.stl` is impossible** — STL carries no point data by
  construction — so every output is `.vtk`, whichever branch a mesh took.
- **`numbering` changes the integers written into the array**, not the mesh.
  Whatever consumes the result has to agree; the report records which was used.

## Versions

torch 2.8.0+cu128, vtk 9.6.2, numpy 2.3.2, Python 3.11 — the deployed stack, as
for AMASSS and BatchDentalSeg. The engine is **not** in that set:

```toml
[project.optional-dependencies]
segmentation = ["shapeaxi==2.0.2", "pytorch3d"]
```

`shapeaxi` requires `pytorch3d`, which publishes no usable wheel: the newest
release on PyPI is 0.7.4 with wheels for cp38–cp310 only, built against a torch
generations older than ours. It is compiled from source, which needs a CUDA
toolkit and tens of minutes — so it sits behind an extra. A plain `uv sync`
therefore stays fast, CI can still import the package and publish its schema, and
only an actual segmentation needs the extra. `pipeline._import_dental_model_seg`
raises `ToolUnavailableError` naming the command to run.

shapeaxi 2.0.2 asks for `torch>=2.8,<2.13`, so the pinned 2.8.0 satisfies it and
**no second torch runtime is needed** — which was the risk that put this tool
last in the migration order.

### The incantation that makes it resolvable

Two non-obvious pieces, both load-bearing:

```toml
[tool.uv.sources]
pytorch3d = { git = "https://github.com/facebookresearch/pytorch3d.git", tag = "v0.7.9" }

[tool.uv.extra-build-dependencies]
pytorch3d = ["torch"]
```

- `pytorch3d` is listed as a **direct** dependency in the extra even though
  shapeaxi pulls it transitively, because `tool.uv.sources` only applies to
  direct dependencies. Left transitive, uv ignores the git source and fails with
  *"no wheels with a matching Python version tag"*.
- `extra-build-dependencies` puts torch into pytorch3d's isolated build
  environment. Its `setup.py` imports torch without declaring it, so without
  this the lock fails with `ModuleNotFoundError: No module named 'torch'`.
  `no-build-isolation-package` does **not** work here — it stops uv from
  providing torch rather than making it available.

With both, `uv lock` resolves 166 packages in about half a minute.
`uv sync --extra segmentation` then compiles pytorch3d; the deployment should do
that once and keep the wheel in a local `/wheels` directory rather than
rebuilding it per image.

## Validated against

- **Input**: `DATA/CrownSeg/testfiles/T1_01_U_segmented.vtk` (294 260 points),
  with `skip_segmented=False` so the network actually runs — the published test
  mesh is already labelled.
- **Weights**: `07-21-22_val-loss0.169.pth`, staged from the release named in the
  server's manifest and checked against its recorded size.
- **Reference**: the pre-port implementation **plus the same shapeaxi
  workaround**, since without it the reference cannot run at all. Both were run
  inside the deployed container, so the engine, the weights and the GPU are
  identical and only the tool code differs.
- **Result**: same geometry (294 260 points, 588 244 cells, points bit-identical)
  and the same two output arrays.

| Array | What it is | Port vs reference |
|---|---|---|
| `PredictedID` | the network's raw per-point prediction | **bit-identical**, 0 of 294 260, in all four runs |
| `Universal_ID` | shapeaxi's post-processed labelling | varies run to run, including between two references |

```
Universal_ID: differing points of 294260
            ref1    ref2    ref3    port
ref1           0    2174    1216     930
ref2        2174       0    2361    2607
ref3        1216    2361       0     826
port         930    2607     826       0
```

- **Tolerance**: none of the difference is attributable to the repackaging.
  `PredictedID` — everything the network produces — is identical every time, so
  the port demonstrably does not touch inference. `Universal_ID` comes out of
  shapeaxi's own "closing operation", which is not deterministic: two
  *reference* runs differ by 1 216–2 361 points, and the port's spread against
  them (826–2 607) is the same range. **Treat a difference in `PredictedID` as a
  regression**; `Universal_ID` alone is this noise floor.

**GPU tests**: `test_real_model_labels_a_real_mesh` is marked `gpu`/`models` and
needs the extra. It was exercised through the run above rather than through
pytest, because pytorch3d cannot be built on this workstation — no CUDA toolkit.
The 23 other tests stub the engine and run anywhere, which is what CI does.

## Working on it

```bash
cd tools/crownseg
uv sync                          # fast: no shapeaxi, no pytorch3d
uv run pytest                    # 23 tests, engine stubbed
uv sync --extra segmentation     # compiles pytorch3d, needs nvcc
uv run pytest -m models          # see tests/data/README.md
```

```bash
# The schema the server publishes
.venv/bin/python ../../scripts/describe.py .
```
