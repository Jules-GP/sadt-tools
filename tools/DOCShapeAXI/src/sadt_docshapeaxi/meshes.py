"""Finding the surfaces to grade, and the manifest shapeaxi reads them from.

shapeaxi's `SaxiDataset` takes a dataframe with a `surf` column of paths
relative to a mount point, so a run starts by writing one. Upstream builds it
by appending to a CSV file it may have written on an earlier run (`open(...,
'a')`), which silently doubles every row when an output folder is reused; it is
rebuilt from scratch here.
"""

from pathlib import Path

from .errors import ToolInputError

SURFACE_SUFFIXES = (".vtk", ".vtp", ".stl", ".obj")

# The column shapeaxi's dataset keys on, and the one upstream writes.
SURFACE_COLUMN = "surf"


def find_surfaces(meshes: Path) -> list:
    """Every surface to grade, relative to `meshes`, in a stable order.

    Recursive, unlike upstream's single `os.listdir`: a cohort arrives as a
    folder of folders as often as a flat one, and the server pays a process
    start-up cost per call, so a folder of forty has to be one run.

    Sorting matters -- readdir order varies between filesystems, and an
    unordered batch makes two runs on the same folder produce differently
    ordered prediction tables.
    """
    meshes = Path(meshes)
    if not meshes.exists():
        raise ToolInputError("Input path does not exist: {}".format(meshes))
    if meshes.is_file():
        if meshes.suffix.lower() not in SURFACE_SUFFIXES:
            raise ToolInputError(
                "{} is not a surface ({}).".format(meshes, ", ".join(SURFACE_SUFFIXES)))
        return [meshes.name]

    found = sorted(
        str(path.relative_to(meshes))
        for path in meshes.rglob("*")
        if path.is_file() and path.suffix.lower() in SURFACE_SUFFIXES
    )
    if not found:
        raise ToolInputError(
            "No surface found under {}. Looked for {}.".format(
                meshes, ", ".join(SURFACE_SUFFIXES)))
    return found


def write_manifest(surfaces: list, destination: Path) -> Path:
    """The one-column table shapeaxi's dataset reads. Rebuilt, never appended."""
    import pandas as pd

    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({SURFACE_COLUMN: surfaces}).to_csv(destination, index=False)
    return destination
