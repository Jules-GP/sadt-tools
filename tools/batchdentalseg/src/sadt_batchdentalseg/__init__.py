"""BatchDentalSeg -- dental CT/CBCT segmentation, DentalSegmentator family.

One nnUNet v2 bundle per model, run over one scan or a whole folder of them.
The pipeline is in pipeline.py; only `run` is public.
"""

from pathlib import Path
from typing import Literal

from .pipeline import segment


def run(
    scans: Path,
    model: Path,
    output_dir: Path,
    separate_segments: bool = False,
    prediction_ID: str = "Seg",
    device: Literal["cuda", "cpu"] = "cuda",
    tile_step_size: float = 0.5,
) -> Path:
    """Segment teeth and jaw structures on a dental CT or CBCT scan.

    Args:
        scans: One scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz), or a
            folder of them for a batch. Folders are searched recursively and
            the output mirrors the input tree, so two patients whose scans
            share a file name stay apart. Files that look like a previous run's
            output are skipped, so a folder can be re-run in place.
        model: The model bundle to use. Its folder name chooses the model and
            with it the label table: DentalSegmentator (adult, five segments),
            PediatricDentalSeg (paediatric, the same five), NasoMaxillaDentSeg
            (six, maxilla split from the upper skull) or UniversalLab (every
            tooth in Universal numbering). Pairing one bundle with another's
            labels is impossible by construction — the choice is one thing.
        output_dir: Where results are written, one file per scan plus
            `BatchDentalSeg_report.json`. Nothing is written outside it.
        separate_segments: Also write one binary file per label the network
            emitted. Only labels PRESENT in the scan are written: a full
            UniversalLab run would otherwise produce 55 mostly-empty files per
            patient, and an empty mask is indistinguishable from a structure
            the model failed on.
        prediction_ID: Suffix used in output names, e.g. `scan_Seg.nii.gz`.
        device: "cuda" or "cpu". CUDA falls back to CPU when no card is
            visible, with a warning.
        tile_step_size: nnUNet's sliding-window overlap; the window advances by
            patch_size times this. It DOES move the segmentation, so it is left
            at nnUNet's own default.

    Returns:
        The output directory, holding the segmentations and the run report. The
        report carries the model's label table — the segmentation is a volume
        of integers, and without that table they mean nothing.
    """
    # torch, nnunetv2, SimpleITK and numpy are imported inside the pipeline: CI
    # imports this module on every PR to publish the schema, and that must not
    # cost a CUDA stack.
    output_dir = Path(output_dir)
    segment(
        input_path=str(scans),
        model_path=str(model),
        output_dir=str(output_dir),
        separate_segments=separate_segments,
        prediction_ID=prediction_ID,
        device=device,
        tile_step_size=tile_step_size,
    )
    return output_dir
