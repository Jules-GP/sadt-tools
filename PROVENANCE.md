# Provenance

Where each tool's algorithm came from. Upstream history is deliberately **not**
grafted into this repository: it is a single history covering sixteen unrelated
modules, and merging it would make neither history readable. This table is the
record instead, and it matters more here than commit history does — it is what
tells you whether a result came from upstream code or from something we changed.

Upstream is
[DCBIA-OrthoLab/SlicerAutomatedDentalTools](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools)
unless a row says otherwise. Each tool's own README carries the same
information in full, including the pins kept and the changes made.

| Tool | Upstream path | Upstream commit | Ported | Algorithm modified |
|---|---|---|---|---|
| [surgmovpred](tools/surgmovpred/) | — | — | not migrated yet | — |
| [amasss](tools/amasss/) | — | — | not migrated yet | — |
| [aso](tools/aso/) | — | — | not migrated yet | — |
| [ali](tools/ali/) | — | — | not migrated yet | — |
| [crownseg](tools/crownseg/) | — | — | not migrated yet | — |
| [batchdentalseg](tools/batchdentalseg/) | — | — | not migrated yet | — |

A row is filled in by the PR that migrates the tool, in the same commit that
adds the package. "Algorithm modified" is `no` for a pure repackaging and
otherwise names what changed and why, matching the tool README.

`tools/_template/` has no row: it is the reference package the others are copied
from, not a port.

**AREG is not in this table on purpose.** It is still under development on an
unmerged `AREG` branch of `slicer-remote-tool-server` and will be migrated once
that branch lands. Its absence is not an oversight.
