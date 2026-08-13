# Test data

Nothing is committed here. The suite builds its own 16³ synthetic CBCT volumes
with SimpleITK, its own labelled meshes with VTK, and its own landmark sets —
seven points in a plausible skull-ish arrangement, no three collinear, so the
coarse alignment can always find a usable triplet. A known rotation is applied
and the registration has to recover it.

That covers 80 of the 81 tests, including six that run the tool out of process
in its own venv. Only the ALI chain needs real files.

## Staging what the chain test needs

`test_fully_automated_cbct_drives_ali_through_the_supervisor` is marked `gpu`
and `models`. It runs the **real** ALI through the supervisor, so it needs that
tool built and its CBCT bundle staged:

```bash
cd ../ALI && uv sync          # ~8.8 GB, CUDA wheels
scripts/setup-models.sh --tool ALI   # from a slicer-remote-tool-server checkout
```

| File | Layout | Size |
|---|---|---|
| `ALI_CBCT_Models` | `<landmark>/<scale>/*.pth`, scales `1` and `0-3` | 4.7 GB |

Put it (or a symlink to it) at `tests/data/ALI_CBCT_Models`. The test skips
rather than fails when it is absent.

```bash
uv run pytest -m "gpu and models"
```

Only the seven landmarks in `DEFAULT_CBCT_LANDMARKS` are asked for, so a partial
bundle carrying `Ba`, `S`, `N`, `RPo`, `LPo`, `ROr` and `LOr` at both scales is
enough — about 300 MB rather than 4.7 GB.

## Reference bundles

A CBCT *reference* is one already-oriented case's landmark file; it is data, not
weights, and the reference scan is not read. The two published bundles carry
**disjoint** landmark sets:

| Bundle | Landmarks |
|---|---|
| `CBCT_Gold_Frankfurt_Horizontal_Midsagittal_Plane` | Ba, S, N, RPo, LPo, ROr, LOr |
| Occlusal + Midsagittal | ANS, IF, PNS, UL6O, UR1O, UR6O |

`run()`'s defaults are the first set. Choosing the second without changing
`cbct_landmarks` is refused up front with a message naming what the reference
actually offers — the failure that used to be forty identical per-patient ones.

Write manual runs into the repository's gitignored `output/` directory, never
next to the inputs.
