# sadt-ali-common

The one thing `ALI_CBCT` and `ALI_IOS` must not disagree about: the Slicer
markups file they both write.

## Why this is shared rather than duplicated

The two engines share nothing else. They have different dependencies, different
inference, different inputs, and — since the split — different virtualenvs and
different torch versions. Duplicating 129 lines between them would cost
nothing in maintenance and would follow this repository's usual instinct, which
is that a second copy beats a coupling (`nnunet_runner.py` is deliberately
duplicated between AMASSS and BatchDentalSeg for exactly that reason).

`markups.py` is the exception because it is not internal code. It is a
**contract with a third party**: the `.mrk.json` file Slicer opens. Its
schema URL, its `LPS` coordinate system, its display block and its control-point
structure are all things Slicer reads, and a divergence between the two engines
does not fail — it produces a file that opens for one modality and not for the
other.

That is not hypothetical here. The pre-port CLIs both set
`display.visibility: false`, which switches the markups display node off: Slicer
loads the file, builds the node, and draws nothing. The bug was in **both**
copies, invisible for as long as nobody opened a result outside the module. Two
copies is how one mistake becomes two.

The rule this encodes: **duplicate implementation, share formats.** Anything
that defines how bytes leaving this repository are shaped belongs here; anything
that decides what to compute belongs to its engine.

## What is deliberately NOT here

- **`errors.py`** stays duplicated in both tools. Errors cross the process
  boundary by exception class *name* — the runner records the name and the
  server maps it to an HTTP status — so a shared base class is not merely
  unnecessary, it is not the mechanism. Twenty-one lines, and sharing them
  would couple two virtualenvs for nothing.
- **Anything with a dependency.** `dependencies` is empty and must stay so:
  this package installs into both environments, and their pins are
  incompatible by design.
