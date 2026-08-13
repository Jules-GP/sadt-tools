"""ALI -- Automatic Landmark Identification, on CBCT scans or intraoral scans.

Places anatomical landmarks and writes Slicer markups files. One tool, two
engines that share nothing but their output format:

* **CBCT** -- one deep-RL agent per landmark walks the volume at 1 mm and then
  at 0.3 mm until it converges on the point;
* **IOS** -- per tooth, the mesh is rendered from a dozen viewpoints and a 2D
  UNet predicts masks that are projected back onto the surface.

The dispatcher in dispatch.py decides which applies, from the data; only `run`
is public.

There is no `mode` argument, on purpose: a folder can hold either kind of data
and a DICOM series has no extension at all, so only the data distinguishes
them. An input holding both kinds is refused rather than guessed at.

The cost is that the schema cannot say "this argument only applies in mode X":
both selections are always published, and one is inert on any given run.
Emptying the selection for the mode that actually ran is an error naming the
argument to fill in (see `dispatch.identify`).
"""

from pathlib import Path
from typing import Literal

from .dispatch import identify


def run(
    input: Path,
    model: Path,
    output_dir: Path,
    # Spelled out because `Literal` takes literals only -- it cannot be built
    # from cbct.catalog.REGION_NAMES. That makes this a second declaration of
    # the same set, which is the thing this contract otherwise avoids, so a
    # test asserts the two agree.
    cbct_regions: list[
        Literal["Cranial base", "Upper", "Lower", "Impacted canine"]
    ] = ["Cranial base", "Upper", "Lower", "Impacted canine"],
    landmarks: list[
        Literal[
            "Ba", "S", "N", "RPo", "LPo", "RFZyg", "LFZyg", "C2", "C3", "C4", "RInfOr",
            "LInfOr", "LMZyg", "RPF", "LPF", "PNS", "ANS", "A", "UR3O", "UR1O", "UL3O",
            "UR6DB", "UR6MB", "UL6MB", "UL6DB", "IF", "ROr", "LOr", "RMZyg", "RNC",
            "LNC", "UR7O", "UR5O", "UR4O", "UR2O", "UL1O", "UL2O", "UL4O", "UL5O",
            "UL7O", "UL7R", "UL5R", "UL4R", "UL2R", "UL1R", "UR2R", "UR4R", "UR5R",
            "UR7R", "UR6MP", "UL6MP", "UL6R", "UR6R", "UR6O", "UL6O", "UL3R", "UR3R",
            "UR1R", "RCo", "RGo", "Me", "Gn", "Pog", "PogL", "B", "LGo", "LCo", "LR1O",
            "LL6MB", "LL6DB", "LR6MB", "LR6DB", "LAF", "LAE", "RAF", "RAE", "LMCo",
            "LLCo", "RMCo", "RLCo", "RMeF", "LMeF", "RSig", "RPRa", "RARa", "LSig",
            "LARa", "LPRa", "LR7R", "LR5R", "LR4R", "LR3R", "LL3R", "LL4R", "LL5R",
            "LL7R", "LL7O", "LL5O", "LL4O", "LL3O", "LL2O", "LL1O", "LR2O", "LR3O",
            "LR4O", "LR5O", "LR7O", "LL6R", "LR6R", "LL6O", "LR6O", "LR1R", "LL1R",
            "LL2R", "LR2R", "UR3OIP", "UL3OIP", "UR3RIP", "UL3RIP",
        ]
    ] = [],
    ios_networks: list[Literal["Occlusal", "Cervical"]] = ["Occlusal", "Cervical"],
    prediction_ID: str = "Pred",
    device: Literal["cuda", "cpu"] = "cuda",
    search_seconds: float = 0.0,
) -> Path:
    """Place anatomical landmarks on a CBCT scan or an intraoral scan.

    Args:
        input: One CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz), one
            intraoral surface (.vtk/.stl), or a folder of either for a batch.
            Folders are searched recursively, and a DICOM series found inside
            one is converted automatically. Which engine runs is decided from
            the data; a folder mixing both kinds is refused rather than half
            processed.
        model: The model bundle. A CBCT bundle holds `<landmark>/<scale>/*.pth`
            folders; an IOS bundle holds flat checkpoints named with an 'O' or
            'C' token and an 'Upper' or 'Lower' one, e.g. `Upper_O_model.pth`.
            The two layouts are mutually exclusive, and a bundle that does not
            match the detected mode is an error naming both.
        output_dir: Where results are written -- one `<scan>_lm_<ID>.mrk.json`
            per scan, mirroring the input's own folder tree, plus
            `run_report.json`. Nothing is written outside it.
        cbct_regions: CBCT only. Anatomical regions to predict. Every region is
            on by default: a landmark whose weights the bundle lacks costs a
            line in the run report, whereas a region left off by default is one
            nobody finds.
        landmarks: CBCT only. Predict exactly these landmarks. Leaving it empty
            is the ordinary case and hands the choice to `cbct_regions`; naming
            any landmark here REPLACES the region selection rather than
            narrowing it, which is what lets a caller ask for the seven points
            it needs instead of running 58 agents to use them.
        ios_networks: IOS only. Occlusal predicts the occlusal point and the
            mesio- and disto-buccal cusps; Cervical predicts the cervical
            lingual and buccal points.
        prediction_ID: Suffix used in output names, e.g. `scan_lm_Pred.mrk.json`.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is
            visible, with a warning.
        search_seconds: CBCT only. Seconds one agent may spend looking for its
            landmark before it is reported as not found. 0 uses the default for
            the device in use -- 15 s on CUDA, 60 s on CPU -- since there is no
            nullable type in the schema to express "unset" with.

    Returns:
        The output directory, holding the markups files and the run report.

    An IOS batch must already carry tooth labels: run `Crown_Seg` over the
    meshes first and pass its output here. This tool does not call another one.
    """
    # torch, monai, itk, VTK, SimpleITK and pytorch3d are all imported inside
    # the engines: CI imports this module on every PR to publish the schema,
    # and that must not cost a CUDA stack.
    output_dir = Path(output_dir)
    identify(
        input_path=str(input),
        model_path=str(model),
        output_dir=str(output_dir),
        cbct_regions=cbct_regions,
        landmarks=landmarks,
        ios_networks=ios_networks,
        prediction_ID=prediction_ID,
        device=device,
        search_seconds=search_seconds,
    )
    return output_dir
