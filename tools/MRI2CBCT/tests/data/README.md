# Test data

Nothing is committed here. The Orient, Resample and LR-crop tests build their
own fixtures -- synthetic volumes with a known size, spacing and origin --
because that is what lets them assert the geometry that came back rather than
merely that a file appeared.

Three steps cannot be tested that way and are marked `models`:

* **Approximate** and **TMJ crop** segment the mandibular condyle with an nnUNet
  model. The bundle is deployment state (`DATA/MRI2CBCT/`), not a fixture.
* **Register** drives elastix across two modalities. On synthetic cubes it
  converges to nothing meaningful, so the only useful reference is a clinical
  pair and the result a clinician accepted. Run it by hand and report what came
  out in the PR.
