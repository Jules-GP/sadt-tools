# Test data

Nothing is committed here. The bundles are 250 MB (PediatricDentalSeg) to
820 MB (UniversalLab), and the CBCTs available to test with are patient data.

Everything the CI suite needs, it builds: `test_run.py` writes 8³ synthetic
volumes and stubs `nnunet_runner.predict_folder`, so bundle discovery, the label
tables, the output tree, geometry matching, segment splitting and the report are
all exercised without a checkpoint or a GPU.

## Staging a bundle

The weights are public, on the upstream project's own release pages, and the
server's `scripts/data-manifest.yml` holds the URLs, sizes and (for most files)
sha256s. Stage them with the server's script:

```bash
scripts/setup-models.sh --tool BatchDentalSeg
```

Or fetch one bundle by hand — PediatricDentalSeg is the smallest at 250 MB, and
the layout matters: the checkpoint goes under `fold_0/`, not beside the two JSON
files, because that is the tree nnUNet's `initialize_from_trained_model_folder`
walks.

```
PediatricDentalSeg/
├── dataset.json
├── plans.json
└── fold_0/checkpoint_final.pth
```

The **folder name selects the model**, and with it the label table, so it has to
match a key in `catalogs.MODELS` exactly.

## Running the real-model test

`test_real_model_segments_a_real_scan` is marked `gpu` and `models` and skips
unless both paths are given:

```bash
SADT_BATCHDENTALSEG_MODEL=/path/to/DATA/BatchDentalSeg/models/PediatricDentalSeg \
SADT_BATCHDENTALSEG_SCAN=/path/to/a/dental/CBCT.nii.gz \
uv run pytest -m models
```

Write manual runs into the repository's gitignored `output/` directory, never
next to the inputs. If an anonymised, publishable dental CBCT ever exists, it
belongs here as a download script plus checksums. Ask before committing
anything.
