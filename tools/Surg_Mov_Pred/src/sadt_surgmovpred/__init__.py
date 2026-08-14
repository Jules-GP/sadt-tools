"""SurgMovPred -- predicted surgical movements from pre-operative cephalometrics.

One stacking regressor per predicted measurement, each shipped with the scaler
it was trained against. A run loads every model found under `model`, resolves
each model's expected features against the input table's columns, and skips any
model whose features the table does not carry -- so an input with fewer
measurements yields fewer predicted targets rather than an error.

The pipeline is in pipeline.py. Only `run` is public.
"""

from pathlib import Path

from .pipeline import predict


def run(
    measurements: Path,
    model: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Predict post-surgical skeletal movements from cephalometric measurements.

    Args:
        measurements: One CSV/XLSX/ODS table of pre-operative measurements, one
            row per patient, or a folder of them concatenated into a batch. A
            patient identifier column ('#', 'ID', 'PatientID'...) is detected
            and carried into the output; without one the identifiers are blank.
        model: Folder of model packages, one subfolder per predicted
            measurement, each holding a `stacking_package.pkl`.
        output_dir: Where the two result tables are written.

    Returns:
        The Excel and CSV predictions tables, one row per patient and one column
        per predicted measurement.
    """
    # sklearn, pandas, joblib and lightgbm are imported inside the pipeline:
    # loading them costs seconds, and generating this tool's schema must not
    # pay for it.
    return predict(
        measurements=Path(measurements),
        model=Path(model),
        output_dir=Path(output_dir),
    )
