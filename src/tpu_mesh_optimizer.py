"""Deterministic TPU-style mesh planning and communication timing model.

The module performs local arithmetic over explicit mesh, bandwidth, activation,
and compute assumptions. It does not query Google Cloud TPU hardware, measure ICI,
compile TPU kernels, or establish production accelerator performance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EVIDENCE_STATE = "MODELED_MESH_SCENARIO_NOT_TPU_MEASUREMENT"


def _require_int(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite(value: object, name: str, minimum: float, inclusive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric input")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    invalid = numeric < minimum if inclusive else numeric <= minimum
    if invalid:
        operator = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be {operator} {minimum}")
    return numeric


@dataclass(frozen=True)
class MeshConfig:
    chips: int
    mesh_x: int
    mesh_y: int
    bytes_per_activation: int
    flops_per_chip: float
    link_bw_gbps: float

    def __post_init__(self) -> None:
        _require_int(self.chips, "chips", 1)
        _require_int(self.mesh_x, "mesh_x", 1)
        _require_int(self.mesh_y, "mesh_y", 1)
        _require_int(self.bytes_per_activation, "bytes_per_activation", 1)
        if self.mesh_x * self.mesh_y < self.chips:
            raise ValueError("mesh_x * mesh_y must cover chips")
        _require_finite(self.flops_per_chip, "flops_per_chip", 0.0, inclusive=False)
        _require_finite(self.link_bw_gbps, "link_bw_gbps", 0.0, inclusive=False)


class TPUMeshOptimizer:
    """Evaluate a modeled mesh step from explicit arithmetic assumptions."""

    def __init__(self, cfg: MeshConfig) -> None:
        self.cfg = cfg

    def shard_activation_bytes(self, global_tokens: int) -> float:
        tokens = _require_int(global_tokens, "global_tokens", 1)
        # Exact average preserves the configured global token count. Padding is not modeled.
        per_chip_tokens = tokens / self.cfg.chips
        return per_chip_tokens * self.cfg.bytes_per_activation

    def transfer_ms(self, global_tokens: int) -> float:
        per_chip = self.shard_activation_bytes(global_tokens)
        wire = per_chip * (self.cfg.chips - 1) / self.cfg.chips
        bytes_per_ms = self.cfg.link_bw_gbps * 125_000.0
        return wire / bytes_per_ms

    def compute_tick_ms(self, flops_needed: float) -> float:
        flops = _require_finite(flops_needed, "flops_needed", 0.0, inclusive=True)
        total_flops = self.cfg.flops_per_chip * self.cfg.chips
        return (flops / total_flops) * 1000.0

    def optimize(self, global_tokens: int, flops_needed: float) -> dict:
        t_tx = self.transfer_ms(global_tokens)
        t_cmp = self.compute_tick_ms(flops_needed)
        exposed = max(0.0, t_tx - t_cmp)
        total = t_cmp + exposed
        hide_pct = 0.0 if t_tx <= 0 else min(100.0, 100.0 * (1.0 - exposed / t_tx))
        status = "MODELED_HIGH_OVERLAP" if exposed < t_tx * 0.25 else "MODELED_ICI_BOUND"
        return {
            "chips": self.cfg.chips,
            "mesh": [self.cfg.mesh_x, self.cfg.mesh_y],
            "transfer_ms": round(t_tx, 4),
            "compute_ms": round(t_cmp, 4),
            "exposed_ici_ms": round(exposed, 4),
            "step_ms": round(total, 4),
            "ici_hide_percent": round(hide_pct, 2),
            "latency_hidden_percent": round(hide_pct, 2),
            "status": status,
            "evidence_state": EVIDENCE_STATE,
        }


class TPUMeshRingOptimizer:
    """Ring-attention-oriented wrapper over the local mesh model."""

    def __init__(self, tpu_slices: int = 64, ici_bandwidth_gbps: float = 4800.0) -> None:
        slices = _require_int(tpu_slices, "tpu_slices", 1)
        side = int(max(1, slices**0.5))
        while side * side < slices:
            side += 1
        self._opt = TPUMeshOptimizer(
            MeshConfig(
                chips=slices,
                mesh_x=side,
                mesh_y=side,
                bytes_per_activation=2,
                flops_per_chip=1e14,
                link_bw_gbps=ici_bandwidth_gbps,
            )
        )

    def optimize_ring_attention(self, sequence_length: int) -> dict:
        sequence = _require_int(sequence_length, "sequence_length", 1)
        # Explicit heuristic for a local scenario model, not measured TPU FLOPs.
        flops = float(sequence) * float(sequence) * 64.0
        result = self._opt.optimize(global_tokens=sequence, flops_needed=flops)
        result["sequence_length"] = sequence
        result["tpu_slices"] = self._opt.cfg.chips
        return result


class MultimodalPipelineBalancer:
    """Return a deterministic token-balance suggestion for a modeled pipeline."""

    def balance_multimodal_batch(
        self, video_frames: int, text_tokens: int, tokens_per_frame: int = 256
    ) -> dict:
        frames = _require_int(video_frames, "video_frames", 0)
        text = _require_int(text_tokens, "text_tokens", 0)
        per_frame = _require_int(tokens_per_frame, "tokens_per_frame", 1)
        vision_tokens = frames * per_frame
        total = vision_tokens + text
        if total == 0:
            return {
                "balance_status": "EMPTY",
                "total_tokens": 0,
                "vision_tokens": 0,
                "text_tokens": 0,
                "evidence_state": EVIDENCE_STATE,
            }
        ratio = vision_tokens / total
        stall_risk = abs(ratio - 0.5)
        status = "MODELED_BALANCED" if stall_risk < 0.35 else "REBALANCE_SUGGESTED"
        return {
            "balance_status": status,
            "total_tokens": total,
            "vision_tokens": vision_tokens,
            "text_tokens": text,
            "vision_fraction": round(ratio, 4),
            "suggested_text_tokens": vision_tokens,
            "evidence_state": EVIDENCE_STATE,
        }
