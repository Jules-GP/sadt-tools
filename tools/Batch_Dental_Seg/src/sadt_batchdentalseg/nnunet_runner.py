"""nnUNet v2 inference for BatchDentalSeg, isolated from the pipeline.

The upstream widget drove nnUNet through `SlicerNNUNetLib.Parameter` and a
`QProcess`, which is why most of that file is process management: killing a
crashed inference tree, reclaiming stray workers, a RAM watchdog. None of it
applies here -- the Python API returns when it is done, and the server bounds
concurrency.

AMASSS carries a near-identical module and they stay separate copies. In the
server that was because `registry.py` imported every tool at startup, so one
tool's missing dependency would take both out of the registry; here it is
because the two packages share no environment at all and may pin different
nnUNet versions. See CONTRIBUTING.md.

torch and nnunetv2 are imported lazily even though the lockfile guarantees
them: `scripts/describe.py` imports this package on every CI run to publish the
schema, and that must not pay for a CUDA stack.
"""

import inspect
import logging
import os

from .errors import ModelNotFoundError

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint_final.pth"

# A bundle is the directory holding these three, whatever it is called and
# however deeply the archive nested it. Discovered rather than assumed: the
# four bundles do not share one layout -- three are three flat files plus a
# fold_0/, and DentalSegmentator arrives as a zip with its own Dataset<n>/
# tree inside.
_REQUIRED_FILES = ("dataset.json", "plans.json")
_FOLD_DIR = "fold_0"


def resolve_device(requested: str = None) -> str:
    """The device to actually use, falling back to CPU when CUDA is absent."""
    import torch

    wanted = (requested or "cpu").strip().lower()
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("device=%s requested but CUDA is unavailable; falling back to CPU", wanted)
        return "cpu"
    return wanted


def find_model_folder(model_root: str):
    """The nnUNet bundle inside `model_root`, or None.

    Walks for the directory holding dataset.json, plans.json and
    fold_0/checkpoint_final.pth, all three confirmed before a candidate is
    accepted. A half-downloaded bundle therefore reports "this model is not
    installed" rather than failing inside nnUNet's loader.
    """
    if not os.path.isdir(model_root):
        return None

    for directory, _dirs, _files in os.walk(model_root):
        if not all(os.path.isfile(os.path.join(directory, name)) for name in _REQUIRED_FILES):
            continue
        if os.path.isfile(os.path.join(directory, _FOLD_DIR, CHECKPOINT_NAME)):
            return directory
    return None


def _build_predictor(device: str, tile_step_size: float):
    """Instantiate an nnUNetPredictor, tolerating nnUNet's renamed kwargs.

    nnUNet 2.x renamed `perform_everything_on_gpu` to
    `perform_everything_on_device` mid-series; passing whichever the installed
    version declares keeps this working across the range.
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    options = {
        "tile_step_size": float(tile_step_size),
        "use_gaussian": True,
        # No test-time mirroring, matching upstream's inference settings.
        "use_mirroring": False,
        "device": torch.device(device),
        "verbose": False,
        "verbose_preprocessing": False,
        "allow_tqdm": False,
    }
    accepted = set(inspect.signature(nnUNetPredictor.__init__).parameters)
    for name in ("perform_everything_on_device", "perform_everything_on_gpu"):
        if name in accepted:
            options[name] = device.startswith("cuda")
            break

    return nnUNetPredictor(**{key: value for key, value in options.items() if key in accepted})


def predict_folder(model_folder: str, input_dir: str, output_dir: str, device: str,
                   tile_step_size: float = 0.5) -> None:
    """Segment every `*_0000.nii.gz` in `input_dir`, writing masks to `output_dir`.

    A whole folder per call, so the checkpoint is loaded once for the batch
    rather than once per scan.

    AMASSS additionally redirects nnUNet's resamplers to the GPU, which is
    worth ~2.5x there. It is deliberately not done here yet: it drops the input
    resampling from spline order 3 to order 1, and nothing has measured what
    that costs THESE models.
    """
    os.makedirs(output_dir, exist_ok=True)

    predictor = _build_predictor(device, tile_step_size)
    # An explicit path, never nnUNet's `nnUNet_results` environment variable:
    # os.environ is process-global and the variable would outlive the call.
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name=CHECKPOINT_NAME,
    )
    predictor.predict_from_files(
        input_dir,
        output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )


__all__ = [
    "CHECKPOINT_NAME",
    "ModelNotFoundError",
    "find_model_folder",
    "predict_folder",
    "resolve_device",
]
