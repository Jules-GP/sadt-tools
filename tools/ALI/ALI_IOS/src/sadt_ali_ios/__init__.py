"""ALI_IOS -- Automatic Landmark Identification on intraoral surface scans.

Per tooth, the mesh is rendered from a dozen viewpoints and a 2D UNet predicts
masks that are projected back onto the surface. Writes one Slicer markups file
per scan.

Split out of the former single `ALI` tool. The reason is concrete: this engine
needs pytorch3d, which ships as a wheel built against one exact torch version
(`+pt2110cu128`), so it is pinned to torch 2.11. The CBCT engine has no reason
to move, and while the two shared a virtualenv, neither could be pinned without
the other. They share their output format and their input vocabulary, both in
`sadt_ali_common`, and nothing else.
"""

from pathlib import Path
from typing import Literal

from .dispatch import identify


def run(
    input: Path,
    model: Path,
    output_dir: Path,
    # Mucogingival is OFF by default: it is one point per lower tooth on the
    # gingival margin, wanted by a mandible registration and by nobody asking
    # for crown landmarks. On by default would add a third pass over every mesh
    # of every existing request.
    networks: list[
        Literal["Occlusal", "Cervical", "Mucogingival"]
    ] = ["Occlusal", "Cervical"],
    prediction_ID: str = "Pred",
    device: Literal["cuda", "cpu"] = "cuda",
) -> Path:
    """Place anatomical landmarks on an intraoral surface scan.

    Args:
        input: One intraoral surface (.vtk/.stl), or a folder of them for a
            batch. Folders are searched recursively. An input holding CBCT
            volumes is refused by name rather than half processed -- run
            ALI_CBCT on those.
        model: The model bundle, holding flat checkpoints named with an 'O' or
            'C' token and an 'Upper' or 'Lower' one, e.g. `Upper_O_model.pth`.
        output_dir: Where results are written -- one `<scan>_lm_<ID>.mrk.json`
            per scan, mirroring the input's own folder tree, plus
            `run_report.json`. Nothing is written outside it.
        networks: Occlusal predicts the occlusal point and the mesio- and
            disto-buccal cusps; Cervical predicts the cervical lingual and
            buccal points; Mucogingival predicts one point per lower tooth on
            the gingival margin rather than on the crown, and runs on the
            mandible only. A point it had to place from a fit of the arch,
            rather than from the render, carries a caveat in its own
            `description` field and in `landmarks_degraded` in the report.
        prediction_ID: Suffix used in output names, e.g. `scan_lm_Pred.mrk.json`.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is
            visible, with a warning.

    Returns:
        The output directory, holding the markups files and the run report.

    The meshes must already carry tooth labels: run `Crown_Seg` over them first
    and pass its output here. This tool does not call another one.
    """
    # torch and pytorch3d are imported inside the engine: CI imports this
    # module on every PR to publish the schema, and that must not cost a CUDA
    # stack.
    output_dir = Path(output_dir)
    identify(
        input_path=str(input),
        model_path=str(model),
        output_dir=str(output_dir),
        ios_networks=networks,
        prediction_ID=prediction_ID,
        device=device,
    )
    return output_dir
