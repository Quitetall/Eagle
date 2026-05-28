# LamQuant internal (vendor) introspection benchmark namespace.
#
# This package marks the boundary between Eagle's EXTERNAL, codec-agnostic
# LamQuant Standard (LQS) suite and the INTERNAL LamQuant-vendor benchmarks
# that introspect neural codec internals.
#
# The introspection benchmarks themselves still physically live under
# tests/benchmarks/ (they are KEPT, not moved/deleted). They are tagged with
# `pytestmark = pytest.mark.internal` so that:
#
#   pytest -m "not internal"   # external LQS suite (default external CI)
#   pytest -m internal         # internal LamQuant dev suite
#
# See README.md in this directory for the dependency contract.
