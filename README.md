# TPU-Style Mesh Planning Study

Independent GlacierEQ portfolio work exploring deterministic mesh communication and compute-overlap arithmetic for TPU-style accelerator scenarios.

**Status:** local scenario model + reference RTL/Mojo artifacts.  
**Evidence token:** `MODELED_MESH_SCENARIO_NOT_TPU_MEASUREMENT`

This repository is **not affiliated with, endorsed by, or operated by Google or Google DeepMind**. It does not claim proprietary TPU access, Google Cloud deployment, measured inter-chip performance, or production accelerator control.

## Verified capability

The canonical runnable surface is `src/tpu_mesh_optimizer.py` plus the Python tests in `tests/`.

It deterministically models:

- activation sharding across a configured mesh;
- transfer time from an explicit link-bandwidth assumption;
- compute time from an explicit FLOP-rate assumption;
- modeled communication/compute overlap;
- a simple multimodal token-balance suggestion.

These outputs are **scenario arithmetic**, not measurements from TPU v4/v5/v6 hardware.

## Engineering anatomy

| Surface | Evidence-bound role |
|---|---|
| `src/tpu_mesh_optimizer.py` | Canonical tested mesh timing/planning model |
| `tests/test_mesh.py` | Mesh validation and arithmetic regression |
| `tests/test_tpu.py` | Ring-attention and multimodal scenario regression |
| `tests/test_tpu_matmul.py` | Local Python accumulator simulation; does not execute RTL |
| `hdl/tpu_matmul.v` | Verilog systolic-array reference artifact; not exercised by the Python truth gate |
| `src/mojo_tensor_kernel.mojo` | Mojo source example; not a current compiled-performance receipt |
| `mastermind_sidecar.py` | Local status helper; not proof of APEX/Mastermind runtime integration |

## Native proof

```bash
PYTHONPATH=src python -m pytest -q
```

The repository-owned Public Truth Gate runs the Python proof on Python 3.11 and 3.13 and verifies that the public surface retains its modeled-evidence and non-affiliation boundaries.

## Explicit nonclaims

Current evidence does **not** establish:

- execution on Google Cloud TPU hardware;
- TPU v4/v5p/v6e performance;
- measured ICI latency or bandwidth;
- near-C Mojo throughput;
- compiled custom TPU kernels;
- Verilog synthesis, timing closure, or silicon behavior;
- MCP tool registration;
- live APEX/AKOS/Mastermind connectivity;
- Google or Google DeepMind employment, endorsement, affiliation, or proprietary access.

Those are higher evidence states and require separate hardware/runtime receipts.

## Why the capability matters

The engineering value is the explicit separation of **topology and communication assumptions from hardware claims**. The model gives a reproducible way to reason about sharding and overlap before a future accelerator experiment exists, while keeping the public portfolio honest about what has actually been measured.
