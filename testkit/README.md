# sadt-testkit

Runs one tool from another tool's tests, without coupling them.

**Development dependency only.** It must never appear in a tool's `[project]
dependencies`, and it has no dependencies of its own so that it cannot perturb
the resolution of the tool it is helping to test.

## Why it exists

`tools/ALI` cannot be tested end to end without meshes that carry tooth labels,
and those come out of `tools/Crown_Seg`. `tools/ASO` needs landmarks, which come
out of `tools/ALI`. Before the split each of them simply imported the next
(`ALILogic.py` did `from tools.CrownSeg.src import CrownSegLogic`), and that is
the coupling the split removed: tools are sequenced by the server now.

That leaves a real gap — a tool whose input is another tool's output has nothing
realistic to test against. This closes it **without** putting the coupling back:
the other tool is run as a subprocess, through its own `.venv`, exactly the way
the server runs it.

```
tools/ALI/.venv/bin/python  →  tests
                                 └─ run_tool("Crown_Seg", ...)
                                      └─ tools/Crown_Seg/.venv/bin/python  →  run()
```

Nothing is shared at runtime. The two tools keep different interpreters and
irreconcilable dependency sets, and neither imports the other.

## Using it

```python
from sadt_testkit import is_built, run_tool, tool_schema

@pytest.mark.skipif(not is_built("Crown_Seg"), reason="run uv sync in tools/Crown_Seg")
def test_ali_on_freshly_segmented_meshes(tmp_path):
    segmented = run_tool(
        "Crown_Seg",
        meshes=RAW_SCAN,
        model=CROWNSEG_MODEL,
        output_dir=tmp_path / "segmented",
    )
    run(scans=segmented, model=ALI_MODEL, output_dir=tmp_path / "landmarks")
```

`run_tool` returns what the tool's `run()` returned — a `Path`, or a
`dict[str, Path]` — which is the same value the server's runner would hand to
the next tool.

| | |
|---|---|
| `run_tool(name, **params)` | Call `tools/<name>`'s `run()` in its venv. Raises `ToolFailed` with the tool's stderr when it fails. |
| `tool_schema(name)` | The JSON schema that tool publishes, via `scripts/describe.py`. |
| `is_built(name)` | Whether that tool's `.venv` exists. Skip on `False`; do not fail. |
| `ToolNotBuilt` | Raised with the `uv sync` command to run. |

**Skip, do not fail, when a tool is not built.** A contributor working on one
tool has no reason to have built the others, and CI builds each tool in its own
job, so an integration test that hard-failed there would fail every time.

## What it is also good for

Two properties are only testable from outside the process, so `_template`'s
`tests/test_integration.py` covers both and is the file to copy:

- **that the tool writes nothing outside `output_dir`.** In-process the test
  shares a working directory with the tool; `run_tool` runs it from a neutral
  one, so a stray write shows up.
- **that the published schema and the callable agree.** The server reads the
  schema and calls `run(**params)` from it. An argument renamed in one place and
  not the other breaks the chain in production, not in the tool's own tests —
  unless something compares them.

## Wiring it into a tool

```toml
[dependency-groups]
dev = ["pytest==8.3.4", "sadt-testkit"]

[tool.uv.sources]
sadt-testkit = { path = "../../testkit", editable = true }
```

`_driver.py` is executed **by** the tool's interpreter and never imported by it,
so it is stdlib-only and stays 3.9-compatible — a tool may pin an old Python.
It is deliberately a miniature of the server's runner: same job, same argument
coercion. If the two ever drift, an integration test here would pass while the
server failed, so keep it boring and keep it matching.

## Testing the testkit

```bash
cd testkit
uv sync
uv run pytest        # 13 tests, all against tools/_template
```
