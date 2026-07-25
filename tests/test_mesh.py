"""Tests for TPU mesh optimizer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpu_mesh_optimizer import MeshConfig, TPUMeshOptimizer


def test_basic_optimize():
    cfg = MeshConfig(
        chips=64,
        mesh_x=8,
        mesh_y=8,
        bytes_per_activation=2,
        flops_per_chip=1e14,
        link_bw_gbps=100.0,
    )
    opt = TPUMeshOptimizer(cfg)
    r = opt.optimize(global_tokens=1_048_576, flops_needed=1e16)
    assert r["chips"] == 64
    assert r["transfer_ms"] >= 0
    assert r["compute_ms"] >= 0
    assert "answer" not in r


def test_invalid_mesh():
    try:
        MeshConfig(4, 1, 1, 2, 1e12, 10.0)
        assert False
    except ValueError:
        pass
