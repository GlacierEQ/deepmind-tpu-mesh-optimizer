from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_public_surface_uses_real_paths_and_modeled_evidence_token() -> None:
    text = README.read_text(encoding="utf-8")
    assert "src/tpu_mesh_optimizer.py" in text
    assert "src/mojo_tensor_kernel.mojo" in text
    assert "hdl/tpu_matmul.v" in text
    assert "MODELED_MESH_SCENARIO_NOT_TPU_MEASUREMENT" in text


def test_public_surface_excludes_stale_or_unverified_claims() -> None:
    text = README.read_text(encoding="utf-8").casefold()
    forbidden = (
        "src/tpu_optimizer.py",
        "mojo/tpu_kernel.mojo",
        "near-c speeds",
        "mcp tool: `tpu_mesh_status()`",
        "fully integrated with apex highway",
    )
    assert all(marker not in text for marker in forbidden)


def test_public_surface_declares_non_affiliation_and_hardware_boundary() -> None:
    text = README.read_text(encoding="utf-8").casefold()
    assert "not affiliated with, endorsed by, or operated by google or google deepmind" in text
    assert "does not claim proprietary tpu access" in text
    assert "not measurements from tpu v4/v5/v6 hardware" in text
