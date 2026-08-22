#!/usr/bin/env python3
"""Generate a modeled mesh frontier selection and content-hashed receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mesh_frontier import (  # noqa: E402
    MeshConstraints,
    MeshFrontierPlanner,
    MeshHardwareModel,
    MeshWorkload,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--chips", type=int, default=16)
    parser.add_argument("--reserve-chips", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--flops-per-chip", type=float, default=1e14)
    parser.add_argument("--link-bandwidth-gbps", type=float, default=800.0)
    parser.add_argument(
        "--preference",
        choices=("balanced", "latency", "memory", "communication"),
        default="balanced",
    )
    args = parser.parse_args()

    workload = MeshWorkload(
        tokens=args.tokens,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        batch_size=args.batch_size,
        flops_per_token=5e8,
    )
    hardware = MeshHardwareModel(
        chips=args.chips,
        reserve_chips=args.reserve_chips,
        flops_per_chip=args.flops_per_chip,
        link_bandwidth_gbps=args.link_bandwidth_gbps,
    )
    planner = MeshFrontierPlanner(workload, hardware)
    frontier = planner.frontier()
    selected = planner.choose(MeshConstraints(), preference=args.preference)

    payload = {
        "schema": "glaciereq.deepmind-mesh-frontier-plan.v1",
        "evidence_state": selected.evidence_state,
        "workload": {
            "tokens": workload.tokens,
            "hidden_dim": workload.hidden_dim,
            "layers": workload.layers,
            "batch_size": workload.batch_size,
            "bytes_per_element": workload.bytes_per_element,
            "flops_per_token": workload.flops_per_token,
        },
        "hardware_model": {
            "chips": hardware.chips,
            "active_chips": hardware.active_chips,
            "reserve_chips": hardware.reserve_chips,
            "flops_per_chip": hardware.flops_per_chip,
            "link_bandwidth_gbps": hardware.link_bandwidth_gbps,
        },
        "preference": args.preference,
        "candidate_count": len(planner.candidates()),
        "frontier_count": len(frontier),
        "selected": selected.as_dict(),
        "selected_fingerprint": selected.fingerprint,
        "frontier": [plan.as_dict() for plan in frontier],
        "claims_not_established": [
            "Google DeepMind or Google affiliation",
            "TPU hardware execution",
            "ICI fabric measurement",
            "XLA or JAX compilation",
            "production accelerator performance",
        ],
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()

    receipt = {
        "schema": "glaciereq.deepmind-mesh-frontier-receipt.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": os.environ.get(
            "GITHUB_REPOSITORY", "GlacierEQ/deepmind-tpu-mesh-optimizer"
        ),
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "artifact": str(args.output),
        "artifact_sha256": digest,
        "verified_state": "DETERMINISTIC_MESH_FRONTIER_EXECUTED",
        "selected_fingerprint": selected.fingerprint,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
