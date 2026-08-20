"""AutoMatrix -- apply a registration matrix to the scans it belongs to.

The tool does no registration of its own. It takes matrices somebody else
computed (AREG, ASO, a mirroring transform) and moves a cohort of volumes and
landmark files through them, pairing each scan with its matrix by patient key.
The work is in pipeline.py; only `run` is public.
"""

from pathlib import Path

from .pipeline import process


def run(
    scans: Path,
    matrices: Path,
    output_dir: Path,
    suffix: str = "_apply",
    reference: Path = "",
    add_matrix_name: bool = False,
    from_areg: bool = False,
    is_segmentation: bool = False,
) -> Path:
    """Apply an existing registration matrix to a cohort of scans or landmarks.

    Args:
        scans: The files to move -- a `.nii`, `.nii.gz` or `.nrrd` volume, a
            Slicer `.mrk.json` landmark file, or a folder of them for a batch.
            A folder is searched recursively and the tree is rebuilt under the
            output directory.
        matrices: The transforms to apply: a `.tfm`, `.h5`, `.mat` or `.txt`
            ITK transform, or a folder of them. A folder is paired with the
            scans by patient key, so `P1_MAND_Or.tfm` is applied to
            `P1_T1_scan.nii.gz`; a single file is applied to every patient,
            which is how one mirroring matrix serves a whole cohort.
        output_dir: Where the moved scans and `AutoMatrix_report.json` are
            written. Nothing is written outside it.
        suffix: Added to every output file name, before the extension, so a
            result never overwrites the scan it came from.
        reference: Optional volume whose grid every resampled result lands on.
            A cohort sharing one reference comes out voxel-aligned and can be
            compared; left empty, each scan is resampled on its own grid.
            Ignored for landmarks, and by a mirroring matrix, which is applied
            in the scan's own space.
        add_matrix_name: Also put the matrix's file name in the output's, which
            is what tells two results apart when a scan is moved through
            several matrices in one run.
        from_areg: Read the matrix for each LANDMARK file out of an AREG output
            tree instead of pairing by name: the `_CB`, `_L` or `_U` marker in
            the file's name picks the region, and the matrix is taken from
            `<matrices>/<region>/<patient>_OutReg/`. Volumes are still paired
            by name.
        is_segmentation: Resample with nearest-neighbour instead of linear
            interpolation. A segmentation interpolated linearly comes back with
            label values that were never in it.

    Returns:
        The output directory, holding one moved file per scan and matrix, plus
        the run report.
    """
    return process(
        scans=Path(scans),
        matrices=Path(matrices),
        output_dir=Path(output_dir),
        suffix=suffix,
        reference=reference,
        add_matrix_name=add_matrix_name,
        from_areg=from_areg,
        is_segmentation=is_segmentation,
    )
