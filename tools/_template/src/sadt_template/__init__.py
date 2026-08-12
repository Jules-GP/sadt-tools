"""Reference tool package: copy this directory to start a new tool.

The whole public surface of a tool is the single callable `run` below. Nothing
else here is part of the contract -- no base class to subclass, no registry to
edit, no import from the server. The server's runner imports this module and
calls `run(**params)`; `scripts/describe.py` reads the same signature to publish
the tool's JSON schema. Signature and schema therefore cannot drift.

What this placeholder computes (intensity statistics over `.npy` arrays) is not
the point and is meant to be deleted. The five rules it demonstrates are:

1. Stdlib annotations only -- `Path`, `str`, `int`, `float`, `bool`,
   `list[...]`. describe.py refuses anything else rather than guess, so an
   `Optional[int]` or a custom marker type fails the build instead of shipping
   a wrong schema to the client.
2. No default means required. That is the *only* place `required` comes from.
3. Batch input. `scans` takes a directory as readily as one file: the server
   pays a process start-up cost per call, so a folder of 40 scans must be one
   call, not 40.
4. Heavy imports live INSIDE run(). describe.py imports this module in every CI
   job; a module-level `import torch` would make schema generation cost a CUDA
   stack and fail on a machine that has none.
5. Everything written goes under `output_dir`, which the caller owns.
"""

from pathlib import Path

from .pipeline import summarise


def run(
    scans: Path,
    output_dir: Path,
    metrics: list[str] = ["mean", "max"],
    threshold: float = 0.0,
    per_scan_report: bool = False,
) -> Path:
    """Summarise the intensity distribution of a scan or a folder of scans.

    This first line is what the client shows a clinician in the tool list, so it
    describes the clinical action, not the implementation. Everything below it
    is for whoever maintains the tool and never reaches the UI.

    Args:
        scans: One `.npy` array, or a folder of them for a batch.
        output_dir: Where results are written. Nothing is written outside it.
        metrics: Statistics to compute, from "mean", "max", "min", "std".
        threshold: Ignore voxels at or below this intensity.
        per_scan_report: Also write one `<scan>.txt` next to the summary.

    Returns:
        The path of the written summary file.
    """
    # The work lives in pipeline.py, which imports numpy inside summarise() --
    # rule 4 covers the whole import chain, not just this file, so `from
    # .pipeline import summarise` at the top must stay free of heavy modules.
    return summarise(
        scans=Path(scans),
        output_dir=Path(output_dir),
        metrics=metrics,
        threshold=threshold,
        per_scan_report=per_scan_report,
    )
