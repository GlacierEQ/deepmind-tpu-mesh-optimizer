"""Regression tests for the deterministic TPU-style mesh scenario model."""
import unittest

from tpu_mesh_optimizer import (
    EVIDENCE_STATE,
    MultimodalPipelineBalancer,
    TPUMeshRingOptimizer,
)


class TestTPUMeshOptimizer(unittest.TestCase):
    def test_ring_attention_scenario(self):
        optimizer = TPUMeshRingOptimizer(tpu_slices=64, ici_bandwidth_gbps=4800.0)
        result = optimizer.optimize_ring_attention(sequence_length=1_048_576)

        self.assertIn(result["status"], {"MODELED_HIGH_OVERLAP", "MODELED_ICI_BOUND"})
        self.assertGreaterEqual(result["latency_hidden_percent"], 0)
        self.assertEqual(result["evidence_state"], EVIDENCE_STATE)

    def test_multimodal_pipeline_scenario(self):
        balancer = MultimodalPipelineBalancer()
        result = balancer.balance_multimodal_batch(video_frames=120, text_tokens=32_000)

        self.assertIn(
            result["balance_status"],
            {"MODELED_BALANCED", "REBALANCE_SUGGESTED"},
        )
        self.assertGreater(result["total_tokens"], 0)
        self.assertEqual(result["evidence_state"], EVIDENCE_STATE)

    def test_invalid_scenario_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            TPUMeshRingOptimizer(tpu_slices=0)
        with self.assertRaises(ValueError):
            TPUMeshRingOptimizer().optimize_ring_attention(sequence_length=0)
        with self.assertRaises(ValueError):
            MultimodalPipelineBalancer().balance_multimodal_batch(
                video_frames=1,
                text_tokens=1,
                tokens_per_frame=0,
            )


if __name__ == "__main__":
    unittest.main()
