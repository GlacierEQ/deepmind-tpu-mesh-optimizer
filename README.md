# DeepMind TPU Mesh Optimizer — TPU Hardware Accelerator 🧠

> **Systolic array matrix multiplication and Mojo SIMD tensor vectorization for Google Cloud TPU meshes.**

[![Mojo](https://img.shields.io/badge/Mojo-SIMD-FF6B6B)]()
[![Verilog](https://img.shields.io/badge/Verilog-RTL-green)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![Domain](https://img.shields.io/badge/Domain-TPU%20Acceleration-purple)]()

---

## 🎯 For Recruiters & Hiring Managers

This repository implements a **TPU Mesh & Hardware Matrix Accelerator** — optimizing tensor operations on Google Cloud TPU (v4/v5p/v6e) topologies. It demonstrates:

- **Mojo SIMD vectorization** achieving high-throughput tensor math at near-C speeds
- **Verilog RTL systolic array design** for TPU hardware matrix multiplication logic
- **Topological mesh optimization** reducing inter-chip interconnect latency across TPU pods
- **Custom kernel compilation** bypassing high-level framework overhead

**Why this matters**: Scaling large AI models requires understanding chip architectures down to systolic array dataflow and SIMD vector lanes.

---

## 🔬 For Engineers & Technical Reviewers

### Architecture

```
High-Level Model ──→ Mojo SIMD Compiler ──→ TPU Mesh Topology Mapper
                                                    │
                                           Verilog Systolic Array
                                            (Weight-Stationary)
```

### Core Components

| Component | Language | Purpose |
|---|---|---|
| `mojo/tpu_kernel.mojo` | Mojo | High-performance SIMD tensor vectorization routines |
| `hdl/tpu_matmul.v` | Verilog | Hardware systolic array RTL implementation |
| `src/tpu_optimizer.py` | Python | Mesh topology optimizer and placement planner |
| `tests/` | Python | Matrix multiplication verification test suite |

---

## 🤖 ML/AI & Programmatic Mesh Integration

- **MCP Tool**: `tpu_mesh_status()` — TPU pod health and topology queryable by agents
- **Mastermind Sidecar**: Telemetry bridge to APEX Highway mesh
- **SHA-256 Integrity**: Tracked in `.integrity/file_hashes.json`

---

## ⚡ Quick Start

```bash
python3 src/tpu_optimizer.py
python3 tests/test_tpu.py
```
