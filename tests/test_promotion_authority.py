from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from src.promotion_authority import (
    LOCAL_OPERATOR_SECRET,
    PromotionAuthority,
    verify_bound_grant,
)

ROOT = Path(__file__).resolve().parents[1]


class PromotionAuthTests(unittest.TestCase):
    def test_issue_verify(self) -> None:
        authority = PromotionAuthority(b"test-secret", ttl_s=60)
        grant = authority.issue("GlacierEQ/x", "abc", "def", now=1000.0)
        ok, reason = authority.verify(grant, now=1001.0)
        self.assertTrue(ok, reason)

    def test_expired(self) -> None:
        authority = PromotionAuthority(b"test-secret", ttl_s=10)
        grant = authority.issue("GlacierEQ/x", "abc", "def", now=1000.0)
        ok, reason = authority.verify(grant, now=2000.0)
        self.assertFalse(ok)
        self.assertEqual(reason, "GRANT_EXPIRED")

    def test_historical_machine_grant_integrity_is_reproducible(self) -> None:
        grant_path = ROOT / "machine" / "promotion_authority.json"
        proof_path = ROOT / "machine" / "proof_receipt.json"
        if not grant_path.is_file() or not proof_path.is_file():
            self.skipTest("historical receipts not present")

        grant = json.loads(grant_path.read_text(encoding="utf-8"))
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        file_digest = hashlib.sha256(proof_path.read_bytes()).hexdigest()
        self.assertEqual(grant["proof_receipt_digest"], file_digest)
        self.assertEqual(grant["source_sha"], proof["source_sha"])

        historical_time = float(grant["not_after"]) - 1.0
        ok, reason = verify_bound_grant(
            grant,
            proof_path,
            secret=LOCAL_OPERATOR_SECRET,
            now=historical_time,
        )
        self.assertTrue(ok, reason)

        state = json.loads(
            (ROOT / "machine" / "excellence-state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(state["principal_state"], "FUNCTIONAL_CANDIDATE")
        self.assertFalse(state["operational_authority"])


if __name__ == "__main__":
    unittest.main()
