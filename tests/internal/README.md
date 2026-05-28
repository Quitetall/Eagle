# Eagle — Internal LamQuant Introspection Benchmarks

These are **LamQuant-vendor introspection benchmarks**. They are a *dev
dependency for LamQuant*, **not** part of the external LamQuant Standard
(LQS). They are KEPT (never deleted) but clearly namespaced and gated so the
external, codec-agnostic suite can run without them.

## What they introspect

Unlike the external LQS benches (which treat the codec as an opaque
compress/decompress box and only measure agnostic quantities — compression
ratio, round-trip fidelity, latency), these benches reach *inside* LamQuant's
neural codec internals:

- FSQ token entropy vs activity level
- FSQ validation against reference vector-quantize implementations
- Latent-space utilization (per-group symbol usage, mutual information)
- Cayley rotation effectiveness (dead-code / entropy delta)
- Residual (multi-stage) FSQ rate-distortion
- Subband boundary leakage (Le Gall 5/3 stopband)
- TNN encoder memory audit (RP2350 SRAM budget)
- XNOR + cpop MAC kernel throughput (Hazard3 stub)
- C-vs-Python parity (ternary weight extraction + forward-pass drift)
- Ablation matrix across pipeline configurations

## Why they are Python-only

They require the full LamQuant **neural / torch stack** — `torch`, the
`lamquant_codec.models.encoder` (`TernaryMobileNetV5_Subband`) module, and
trained checkpoints. They cannot be expressed in the external Rust `lqs/`
crate or the codec-agnostic Python suites. They stay in Python by design.

## Dependency contract

To run these you need the **sibling LamQuant-Neural** source tree (for
`lamquant_codec.models.*`) *and* the **LamQuant-Lossless wheel** (the codec
under test). Install per the Eagle root README "Sibling-clone" path, then add
the neural extra:

```bash
pip install -e '.[neural]'   # pulls lamquant-neural alongside lamquant-codec
```

Without that stack the imports at the top of each benchmark module will fail.

## How they are gated

Each introspection benchmark module under `tests/benchmarks/` carries:

```python
import pytest
pytestmark = pytest.mark.internal
```

The `internal` marker is registered in the root `pyproject.toml` under
`[tool.pytest.ini_options] markers`.

```bash
# External LQS suite — default external CI. Skips the internal benches.
pytest -m "not internal"

# Internal LamQuant dev suite — requires the neural + lossless stack.
pytest -m internal
```

External CI runs `pytest -m "not internal"` and therefore does **not** require
the neural stack. The internal suite is run only in LamQuant-vendor dev
environments that have the sibling neural tree checked out.
