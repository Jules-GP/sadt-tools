# Test data

Nothing is committed here. `test_run.py` builds its own fixtures -- a cube in
an empty box, a translation written as a `.tfm`, a landmark file holding a
handful of points -- because that is what lets the tests assert where the
result landed rather than merely that a file appeared.

The whole of AutoMatrix is geometry, so a synthetic fixture is not a stand-in
for real data here in the way a synthetic scan is for a segmentation network:
a 4 mm translation applied to a cube has one right answer, and the tests check
that answer to a tenth of a voxel. What a real patient adds is oblique
direction matrices, anisotropic spacing and the transform formats AREG and ASO
actually emit, and that run is reported in the PR rather than automated here --
patient scans never go in git.
