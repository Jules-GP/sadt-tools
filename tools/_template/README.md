# sadt-template

The reference package every tool in this repository is copied from. It is not a
clinical tool: it computes intensity statistics over `.npy` arrays so that the
contract has something real to run, and that placeholder is meant to be deleted.

## Provenance

Every tool README opens with this block, filled in. Keep the wording; it is what
tells someone six months from now whether a result came from upstream code or
from something we changed.

```markdown
## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools, path `AMASSS_CLI/`,
commit `abc1234` (2026-03-14).
Upstream pins kept as-is: torch 2.2.0, monai 1.3.2, nnunetv2 2.8.0.
Changes from upstream: <list, or "none -- repackaging only">.
```

Add the matching row to [`PROVENANCE.md`](../../PROVENANCE.md) in the same PR.

## What it does

| | |
|---|---|
| Inputs | `scans`: one `.npy` array or a folder of them. `output_dir`: where results go. |
| Options | `metrics` (several of mean/max/min/std), `reduction` (one of per_scan/pooled), `threshold`, `per_scan_report`. |
| Outputs | `summary.txt`, one row per scan, plus `<scan>.txt` per scan on request. |
| Model files | None. A real tool lists every weight file it expects here, and where it comes from upstream. |

## The interface

One public callable, `run`. No base class, no registry, no import from the
server:

```python
def run(
    scans: Path,
    output_dir: Path,
    metrics: list[Literal["mean", "max", "min", "std"]] = ["mean", "max"],
    reduction: Literal["per_scan", "pooled"] = "per_scan",
    threshold: float = 0.0,
    per_scan_report: bool = False,
) -> Path:
```

`scripts/describe.py` turns that signature into the JSON schema the server
publishes, so the schema cannot drift from the code:

```console
$ tools/_template/.venv/bin/python scripts/describe.py tools/_template
{
  "name": "_template",
  "description": "Summarise the intensity distribution of a scan or a folder of scans.",
  "arguments": {
    "scans": {"type": "path", "required": true},
    "metrics": {"type": "list[str]", "required": false, "default": ["mean", "max"],
                "choices": ["mean", "max", "min", "std"]},
    ...
  },
  "returns": "path",
  "source_hash": "..."
}
```

The rules behind each line of that signature are in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md); the ones easiest to get wrong are
that heavy imports go inside `run()`, that the absence of a default is the only
thing making an argument required, that a fixed set of options is a `Literal`
rather than a second table, and that a path argument must accept a folder so a
batch is one call rather than forty.

## Working on it

```bash
cd tools/_template
uv sync                 # creates .venv from uv.lock; commit the lock
uv run pytest           # end-to-end, not an import check
uv run pytest -m gpu    # skipped in CI, run by hand before opening a PR
```

## Validated against

Synthetic arrays built in `tests/test_run.py`, asserted exactly (`mean=50.0`,
`max=99.0` over `arange(100)` above threshold 0) -- no tolerance needed because
there is no model.

A real tool records here: which input, which model weights, which reference
output, and what tolerance. "It imported cleanly" is not validation.
