# Test data

Nothing is committed here. The registration tests build their own fixtures --
synthetic volumes translated by a known number of millimetres -- because that is
what lets them assert the answer rather than merely that a file appeared.

The landmark modes need no data either: ALI's output is planted through a fake
supervisor, so what is tested is this tool's half of the exchange.

A run against real data is the last verification step and is reported in the
PR, not automated here: it needs the ALI model bundle, which is deployment
state, and a patient scan, which never goes in git.
