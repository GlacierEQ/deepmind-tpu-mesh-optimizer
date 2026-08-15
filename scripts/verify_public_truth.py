#!/usr/bin/env python3
"""Fail-closed truth checks for the TPU-style mesh planning study."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"PUBLIC_TRUTH_FAIL: {message}")


def main() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = readme.replace("**", "").replace("`", "")
    caps = json.loads((ROOT / "machine/capabilities.json").read_text(encoding="utf-8"))
    state = json.loads((ROOT / "machine/excellence-state.json").read_text(encoding="utf-8"))

    require(
        "MODELED_MESH_SCENARIO_NOT_TPU_MEASUREMENT" in readme,
        "modeled-mesh evidence token missing",
    )
    require(
        "not affiliated with, endorsed by, or operated by Google or Google DeepMind"
        in normalized,
        "Google/DeepMind non-affiliation boundary missing",
    )
    require(
        "scenario arithmetic, not measurements from TPU" in normalized,
        "hardware-measurement boundary missing",
    )
    require(
        "not proof of APEX/Mastermind runtime integration" in normalized,
        "mesh/runtime boundary missing",
    )

    allowed = {
        "deterministic-mesh-activation-sharding-model",
        "explicit-link-bandwidth-transfer-time-estimation",
        "explicit-flop-rate-compute-time-estimation",
        "modeled-communication-compute-overlap",
        "deterministic-multimodal-token-balance-suggestion",
    }
    require(set(caps.get("capabilities", [])) == allowed, "capability allowlist drift")
    require(caps.get("operational_authority") is False, "operational authority must be false")
    require(caps.get("google_tpu_hardware_measurement") is False, "TPU measurement claim must be false")
    require(caps.get("tpu_kernel_execution") is False, "TPU kernel execution claim must be false")
    require(caps.get("verilog_synthesis_or_silicon_proven") is False, "RTL hardware claim must be false")
    require(caps.get("mojo_performance_proven") is False, "Mojo performance claim must be false")
    require(
        caps.get("live_mcp_apex_mastermind_integration") is False,
        "live mesh claim must be false",
    )

    require(state.get("principal_state") == "FUNCTIONAL_CANDIDATE", "stale promotion restored")
    require(state.get("operational_authority") is False, "state grants operational authority")
    proof = state.get("gates", {}).get("DETERMINISTIC_PROOF_GREEN", {})
    require(proof.get("status") == "PENDING_CANONICAL_CI", "fresh exact-head proof gate missing")

    print("PUBLIC_TRUTH_PASS")


if __name__ == "__main__":
    main()
