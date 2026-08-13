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
| [Surg_Mov_Pred](tools/Surg_Mov_Pred/) | `SurgMovPred_CLI/SurgMovPred_CLI.py` | `d7702ae` (2026-06-24) | 2026-08-12 | no — repackaging only. Model load order sorted for reproducibility, both result tables returned instead of one; predictions bit-identical to the pre-port implementation. |
| [AMASSS](tools/AMASSS/) | `AMASSS_CLI/` | `21a62a8` (2026-05-22) | 2026-08-12 | no — repackaging only. Pinned to the deployed stack (torch 2.8.0+cu128, nnunetv2 2.8.1) rather than upstream's declared torch 2.2.0 / nnunetv2 2.8.0; masks bit-identical to the pre-port implementation, within nnUNet's own CUDA nondeterminism. |
| [ASO](tools/ASO/) | — | — | not migrated yet | — |
| [ALI](tools/ALI/) | — | — | not migrated yet | — |
| [Crown_Seg](tools/Crown_Seg/) | — (written against `shapeaxi` directly) | shapeaxi 2.0.2 | 2026-08-12 | no — the network is untouched and its raw output is bit-identical. Carries a two-line workaround for a shapeaxi 2.0.x bug that breaks the tool upstream and downstream alike. |
| [Batch_Dental_Seg](tools/Batch_Dental_Seg/) | `BATCHDENTALSEG/BATCHDENTALSEGLib/SegmentationWidget.py` | `6df3fab` (2026-08-05) | 2026-08-12 | no — repackaging only. Same stack as AMASSS (torch 2.8.0+cu128, nnunetv2 2.8.1); labels compared against the pre-port implementation. |

A row is filled in by the PR that migrates the tool, in the same commit that
adds the package. "Algorithm modified" is `no` for a pure repackaging and
otherwise names what changed and why, matching the tool README.

`tools/_template/` has no row: it is the reference package the others are copied
from, not a port.

**AREG is not in this table on purpose.** It is still under development on an
unmerged `AREG` branch of `slicer-remote-tool-server` and will be migrated once
that branch lands. Its absence is not an oversight.
