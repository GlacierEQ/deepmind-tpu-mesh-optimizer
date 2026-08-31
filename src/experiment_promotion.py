"""Evidence-level promotion fence for modeled accelerator experiments.

The fence prevents modeled or locally tested work from being labeled as compiled
or hardware-measured without the corresponding receipt class. It grants no TPU,
Google Cloud, or deployment authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import IntEnum

PROMOTION_SCHEMA = "glaciereq.tpu-study.experiment-promotion.v1"
PROMOTION_EVIDENCE_STATE = "LOCAL_EXPERIMENT_PROMOTION_FENCE_NOT_TPU_AUTHORITY"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_RE = re.compile(r"^[0-9a-f]{40,64}$")


class EvidenceLevel(IntEnum):
    MODELED = 0
    LOCAL_TESTED = 1
    COMPILED = 2
    HARDWARE_MEASURED = 3


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    source_sha: str
    test_receipt_sha256: str | None = None
    compile_receipt_sha256: str | None = None
    hardware_receipt_sha256: str | None = None

    def validate(self) -> None:
        if not _SOURCE_RE.fullmatch(self.source_sha):
            raise ValueError(
                "source_sha must be lowercase hexadecimal with length 40-64"
            )
        for name in (
            "test_receipt_sha256",
            "compile_receipt_sha256",
            "hardware_receipt_sha256",
        ):
            value = getattr(self, name)
            if value is not None and not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.compile_receipt_sha256 is not None and self.test_receipt_sha256 is None:
            raise ValueError("compiled evidence requires test evidence")
        if self.hardware_receipt_sha256 is not None and self.compile_receipt_sha256 is None:
            raise ValueError("hardware evidence requires compiled evidence")

    @property
    def maximum_level(self) -> EvidenceLevel:
        self.validate()
        if self.hardware_receipt_sha256 is not None:
            return EvidenceLevel.HARDWARE_MEASURED
        if self.compile_receipt_sha256 is not None:
            return EvidenceLevel.COMPILED
        if self.test_receipt_sha256 is not None:
            return EvidenceLevel.LOCAL_TESTED
        return EvidenceLevel.MODELED


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def evaluate_promotion(
    evidence: EvidenceBundle,
    requested_level: EvidenceLevel | int,
) -> dict[str, object]:
    """Return an explicit promotion or refusal receipt."""

    evidence.validate()
    try:
        requested = EvidenceLevel(requested_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported requested evidence level") from exc
    maximum = evidence.maximum_level
    allowed = requested <= maximum
    body: dict[str, object] = {
        "schema": PROMOTION_SCHEMA,
        "source_sha": evidence.source_sha,
        "requested_level": requested.name,
        "maximum_supported_level": maximum.name,
        "allowed": allowed,
        "decision": "PROMOTE" if allowed else "REFUSE_PROMOTION",
        "evidence_state": PROMOTION_EVIDENCE_STATE,
        "operational_authority": False,
        "tpu_hardware_authority": False,
    }
    body["receipt_sha256"] = _digest(body)
    return body
