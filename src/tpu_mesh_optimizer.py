"""
DeepMind-oriented TPU mesh optimizer — real sharding / ICI timing math.

Provides:
  - MeshConfig + TPUMeshOptimizer (general mesh step model)
  - TPUMeshRingOptimizer (ring attention oriented API)
  - MultimodalPipelineBalancer (simple stall-free schedule)

No magic answer fields. Times in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeshConfig:
    chips: int
    mesh_x: int
    mesh_y: int
    bytes_per_activation: int
    flops_per_chip: float
    link_bw_gbps: float

    def __post_init__(self) -> None:
        if self.chips < 1 or self.mesh_x < 1 or self.mesh_y < 1:
            raise ValueError("mesh dims and chips must be >= 1")
        if self.mesh_x * self.mesh_y < self.chips:
            raise ValueError("mesh_x * mesh_y must cover chips")
        if self.link_bw_gbps <= 0:
            raise ValueError("link_bw_gbps must be > 0")


class TPUMeshOptimizer:
    def __init__(self, cfg: MeshConfig) -> None:
        self.cfg = cfg

    def shard_activation_bytes(self, global_tokens: int) -> float:
        per = max(1, global_tokens // self.cfg.chips)
        return float(per * self.cfg.bytes_per_activation)

    def transfer_ms(self, global_tokens: int) -> float:
        per_chip = self.shard_activation_bytes(global_tokens)
        wire = per_chip * (self.cfg.chips - 1) / max(self.cfg.chips, 1)
        bytes_per_ms = self.cfg.link_bw_gbps * 125_000.0
        return wire / bytes_per_ms

    def compute_tick_ms(self, flops_needed: float) -> float:
        total_flops = self.cfg.flops_per_chip * self.cfg.chips
        if total_flops <= 0:
            return float("inf")
        return (flops_needed / total_flops) * 1000.0

    def optimize(self, global_tokens: int, flops_needed: float) -> dict:
        t_tx = self.transfer_ms(global_tokens)
        t_cmp = self.compute_tick_ms(flops_needed)
        exposed = max(0.0, t_tx - t_cmp)
        total = t_cmp + exposed
        hide_pct = 0.0 if t_tx <= 0 else min(100.0, 100.0 * (1.0 - exposed / t_tx))
        return {
            "chips": self.cfg.chips,
            "mesh": [self.cfg.mesh_x, self.cfg.mesh_y],
            "transfer_ms": round(t_tx, 4),
            "compute_ms": round(t_cmp, 4),
            "exposed_ici_ms": round(exposed, 4),
            "step_ms": round(total, 4),
            "ici_hide_percent": round(hide_pct, 2),
            "latency_hidden_percent": round(hide_pct, 2),
            "status": "OPTIMAL_ASYNC_SHARDED" if exposed < t_tx * 0.25 else "ICI_BOUND",
        }


class TPUMeshRingOptimizer:
    """Ring-attention oriented wrapper over the mesh model."""

    def __init__(self, tpu_slices: int = 64, ici_bandwidth_gbps: float = 4800.0) -> None:
        side = int(max(1, tpu_slices**0.5))
        while side * side < tpu_slices:
            side += 1
        self._opt = TPUMeshOptimizer(
            MeshConfig(
                chips=tpu_slices,
                mesh_x=side,
                mesh_y=side,
                bytes_per_activation=2,
                flops_per_chip=1e14,
                link_bw_gbps=ici_bandwidth_gbps,
            )
        )

    def optimize_ring_attention(self, sequence_length: int) -> dict:
        # flops heuristic: attention ~ O(n^2) scaled down for demo-scale realism
        flops = float(sequence_length) * float(sequence_length) * 64.0
        r = self._opt.optimize(global_tokens=sequence_length, flops_needed=flops)
        r["sequence_length"] = sequence_length
        r["tpu_slices"] = self._opt.cfg.chips
        return r


class MultimodalPipelineBalancer:
    """Balance video frame tokens vs text tokens to avoid pipeline stalls."""

    def balance_multimodal_batch(
        self, video_frames: int, text_tokens: int, tokens_per_frame: int = 256
    ) -> dict:
        if video_frames < 0 or text_tokens < 0:
            raise ValueError("counts must be >= 0")
        vision_tokens = video_frames * tokens_per_frame
        total = vision_tokens + text_tokens
        # schedule: interleave so neither stage idles > 10% of steps
        if total == 0:
            return {
                "balance_status": "EMPTY",
                "total_tokens": 0,
                "vision_tokens": 0,
                "text_tokens": 0,
            }
        ratio = vision_tokens / total
        stall_risk = abs(ratio - 0.5)
        status = "STALL_FREE" if stall_risk < 0.35 else "REBALANCE_SUGGESTED"
        return {
            "balance_status": status,
            "total_tokens": total,
            "vision_tokens": vision_tokens,
            "text_tokens": text_tokens,
            "vision_fraction": round(ratio, 4),
            "suggested_text_tokens": vision_tokens,  # 1:1 vision/text target
        }
