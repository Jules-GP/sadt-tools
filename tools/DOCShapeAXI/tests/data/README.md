# Test data

Nothing is committed here. The surfaces the tests read are written by the tests
themselves with VTK -- a sphere is as good as a condyle for asserting which
files are found, in what order, and what shapeaxi is handed to read them with.

Grading a surface for real needs two things this repository does not hold: the
`grading` extra (`uv sync --extra grading`, which pulls shapeaxi and a prebuilt
pytorch3d wheel) and the checkpoint bundle the deployment hosts under
`DATA/DOCShapeAXI/`. That run is the last verification step and is reported in
the PR.
