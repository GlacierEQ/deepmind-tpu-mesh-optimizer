from __future__ import annotations

import pytest

from src.experiment_promotion import (
    PROMOTION_EVIDENCE_STATE,
    EvidenceBundle,
    EvidenceLevel,
    evaluate_promotion,
)


def test_modeled_work_cannot_self_promote_to_compiled_or_hardware() -> None:
    evidence = EvidenceBundle(source_sha="a" * 40)
    compiled = evaluate_promotion(evidence, EvidenceLevel.COMPILED)
    hardware = evaluate_promotion(evidence, EvidenceLevel.HARDWARE_MEASURED)
    assert compiled["decision"] == "REFUSE_PROMOTION"
    assert hardware["decision"] == "REFUSE_PROMOTION"
    assert compiled["maximum_supported_level"] == "MODELED"
    assert compiled["evidence_state"] == PROMOTION_EVIDENCE_STATE
    assert compiled["operational_authority"] is False


def test_test_receipt_promotes_only_through_local_tested() -> None:
    evidence = EvidenceBundle(
        source_sha="a" * 40,
        test_receipt_sha256="b" * 64,
    )
    tested = evaluate_promotion(evidence, EvidenceLevel.LOCAL_TESTED)
    compiled = evaluate_promotion(evidence, EvidenceLevel.COMPILED)
    assert tested["decision"] == "PROMOTE"
    assert compiled["decision"] == "REFUSE_PROMOTION"
    assert tested["maximum_supported_level"] == "LOCAL_TESTED"


def test_receipt_chain_unlocks_only_matching_evidence_levels() -> None:
    compiled = EvidenceBundle(
        source_sha="a" * 40,
        test_receipt_sha256="b" * 64,
        compile_receipt_sha256="c" * 64,
    )
    assert evaluate_promotion(compiled, EvidenceLevel.COMPILED)["decision"] == "PROMOTE"
    assert (
        evaluate_promotion(compiled, EvidenceLevel.HARDWARE_MEASURED)["decision"]
        == "REFUSE_PROMOTION"
    )

    hardware = EvidenceBundle(
        source_sha="a" * 40,
        test_receipt_sha256="b" * 64,
        compile_receipt_sha256="c" * 64,
        hardware_receipt_sha256="d" * 64,
    )
    result = evaluate_promotion(hardware, EvidenceLevel.HARDWARE_MEASURED)
    assert result["decision"] == "PROMOTE"
    assert result["maximum_supported_level"] == "HARDWARE_MEASURED"
    assert len(result["receipt_sha256"]) == 64


def test_incomplete_or_malformed_evidence_chain_fails_closed() -> None:
    with pytest.raises(ValueError, match="compiled evidence requires"):
        EvidenceBundle(
            source_sha="a" * 40,
            compile_receipt_sha256="c" * 64,
        ).validate()
    with pytest.raises(ValueError, match="hardware evidence requires"):
        EvidenceBundle(
            source_sha="a" * 40,
            test_receipt_sha256="b" * 64,
            hardware_receipt_sha256="d" * 64,
        ).validate()
    with pytest.raises(ValueError):
        EvidenceBundle(source_sha="not-a-sha").validate()
