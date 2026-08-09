"""Tests for the deterministic TPU-style mesh model."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpu_mesh_optimizer import MeshConfig, TPUMeshOptimizer


def make_optimizer(chips: int = 64) -> TPUMeshOptimizer:
    return TPUMeshOptimizer(
        MeshConfig(
            chips=chips,
            mesh_x=8,
            mesh_y=8,
            bytes_per_activation=2,
            flops_per_chip=1e14,
            link_bw_gbps=100.0,
        )
    )


def test_basic_optimize():
    result = make_optimizer().optimize(global_tokens=1_048_576, flops_needed=1e16)
    assert result["chips"] == 64
    assert result["transfer_ms"] >= 0
    assert result["compute_ms"] >= 0
    assert "answer" not in result


def test_activation_sharding_preserves_global_token_count():
    optimizer = make_optimizer()
    for tokens in (1, 63, 64, 65, 129):
        per_chip_bytes = optimizer.shard_activation_bytes(tokens)
        reconstructed_tokens = per_chip_bytes * optimizer.cfg.chips / 2
        assert math.isclose(reconstructed_tokens, tokens)


def test_invalid_mesh_and_numeric_inputs_fail_closed():
    with pytest.raises(ValueError):
        MeshConfig(4, 1, 1, 2, 1e12, 10.0)
    with pytest.raises(ValueError):
        MeshConfig(4.5, 2, 3, 2, 1e12, 10.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MeshConfig(4, 2, 2, 2, float("nan"), 10.0)
    with pytest.raises(ValueError):
        MeshConfig(4, 2, 2, 2, 1e12, float("inf"))

    optimizer = make_optimizer()
    with pytest.raises(ValueError):
        optimizer.optimize(global_tokens=1.5, flops_needed=1e9)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        optimizer.optimize(global_tokens=1, flops_needed=float("nan"))
    with pytest.raises(ValueError):
        optimizer.optimize(global_tokens=1, flops_needed=float("inf"))
