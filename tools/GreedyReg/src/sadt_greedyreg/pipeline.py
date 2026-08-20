"""The batch: pair the timepoints, then run each case through the chosen mode.

Upstream halts the whole batch on the first case that fails (`sys.exit(1)` in
GreedyReg_CLI, and the same in the Batch Distant queue), leaving the cases
already done on disk. That is kept: a folder half registered is a fact worth
knowing about, and continuing past a failure has never been this tool's
behaviour, so it is not quietly introduced by the port.
"""

import json
import logging
import tempfile
from pathlib import Path

from . import landmarks as landmark_fitting
from . import registration, tools
from .errors import RegistrationFailed, ToolInputError
from .pairs import find_pairs

logger = logging.getLogger(__name__)

MODE_GREEDY = "Greedy"
MODE_LANDMARK = "Landmark"
MODE_LANDMARK_GREEDY = "Landmark + Greedy"
MODES = (MODE_GREEDY, MODE_LANDMARK, MODE_LANDMARK_GREEDY)

REPORT_NAME = "GreedyReg_report.json"


def _align_by_landmarks(sup, key, fixed, moving, region, model, device):
    """The rigid transform ALI's landmarks put between one pair of scans."""
    wanted = landmark_fitting.REGION_LANDMARKS[region]
    places = {}
    for tag, scan in (("fixed", fixed), ("moving", moving)):
        written = tools.place_landmarks(sup, scan, model, wanted, device, f"{key}_{tag}")
        places[tag] = landmark_fitting.collect(written, wanted)

    common = landmark_fitting.matched(places["fixed"], places["moving"], wanted)
    if len(common) < landmark_fitting.MINIMUM_LANDMARKS:
        raise RegistrationFailed(
            "{}: only {} landmark(s) found on both scans ({}), and a rigid fit "
            "needs {}. The region asked for {}.".format(
                key, len(common), ", ".join(common) or "none",
                landmark_fitting.MINIMUM_LANDMARKS, ", ".join(wanted),
            )
        )

    import numpy as np

    transform = landmark_fitting.rigid_from_landmarks(
        np.array([places["fixed"][name] for name in common]),
        np.array([places["moving"][name] for name in common]),
    )
    logger.info("GreedyReg %s: rigid fit on %d landmark(s)", key, len(common))
    return transform, common


def process(t1, t2, output_dir, mode, masks, init, metric, transform_type,
            region, landmark_model, device, sup):
    if mode not in MODES:
        raise ToolInputError(
            "Unknown mode '{}'. Available: {}.".format(mode, ", ".join(MODES))
        )
    if metric not in registration.METRICS:
        raise ToolInputError(
            "Unknown metric '{}'. Available: {}.".format(
                metric, ", ".join(registration.METRICS)
            )
        )
    if transform_type not in registration.DEGREES_OF_FREEDOM:
        raise ToolInputError(
            "Unknown transform type '{}'. Available: {}.".format(
                transform_type, ", ".join(registration.DEGREES_OF_FREEDOM)
            )
        )
    if region not in landmark_fitting.REGION_LANDMARKS:
        raise ToolInputError(
            "Unknown region '{}'. Available: {}.".format(
                region, ", ".join(landmark_fitting.REGION_LANDMARKS)
            )
        )

    uses_landmarks = mode in (MODE_LANDMARK, MODE_LANDMARK_GREEDY)
    if uses_landmarks:
        tools.require(sup, "ALI_CBCT", mode)
        if not landmark_model:
            raise ToolInputError(
                "{} mode needs the ALI landmark model bundle named in "
                "'landmark_model'.".format(mode)
            )

    cases = find_pairs(t1, t2, masks or None, init or None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("GreedyReg: %d pair(s) in %s mode: %s", len(cases), mode,
                ", ".join(case[0] for case in cases))

    report = {"mode": mode, "metric": metric, "transform_type": transform_type,
              "cases": []}
    for key, fixed, moving, mask, supplied_init in cases:
        entry = {"patient": key, "t1": str(fixed), "t2": str(moving), "outputs": {}}
        with tempfile.TemporaryDirectory(prefix=f"greedyreg_{key}_") as scratch:
            scratch = Path(scratch)
            registered_from = moving

            if uses_landmarks:
                transform, common = _align_by_landmarks(
                    sup, key, fixed, moving, region, landmark_model, device)
                entry["landmarks_matched"] = common
                if mode == MODE_LANDMARK:
                    aligned = landmark_fitting.bake_into_affine(
                        moving, transform, output_dir / f"{key}_t2_aligned.nii.gz")
                    entry["outputs"]["aligned"] = str(aligned)
                    report["cases"].append(entry)
                    continue
                # Landmark + Greedy: the fit becomes Greedy's initialisation,
                # which is what `-ia` and upstream's initFolder exist for. The
                # panel says as much after a Distant run: "Now run Automatic
                # Registration to refine."
                resolved_init = registration.write_matrix(
                    transform, scratch / "landmark_init.mat")
            elif supplied_init:
                resolved_init = supplied_init
            else:
                resolved_init = registration.write_identity_init(scratch / "init.mat")
            entry["init"] = str(resolved_init)

            resolved_mask = None
            if mask:
                resolved_mask = registration.binarize_mask(
                    mask, scratch / "mask.nii.gz")
                entry["mask"] = str(mask)

            registered = registration.register(
                fixed=fixed,
                moving=registered_from,
                output=output_dir / f"{key}_registered.nii.gz",
                warp=output_dir / f"{key}_warp.mat",
                init=resolved_init,
                metric=metric,
                transform_type=transform_type,
                mask=resolved_mask,
            )
            entry["outputs"]["registered"] = str(registered)
            entry["outputs"]["transform"] = str(output_dir / f"{key}_warp.mat")

        report["cases"].append(entry)
        logger.info("GreedyReg %s done", key)

    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return output_dir
