import unittest

from src.mesh_frontier import (
    MeshConstraints,
    MeshFrontierPlanner,
    MeshHardwareModel,
    MeshWorkload,
    NoFeasibleMesh,
    ShardingMode,
)


def planner(reserve=0):
    return MeshFrontierPlanner(
        MeshWorkload(
            tokens=8192,
            hidden_dim=4096,
            layers=32,
            batch_size=2,
            flops_per_token=5e8,
        ),
        MeshHardwareModel(
            chips=16,
            reserve_chips=reserve,
            flops_per_chip=1e14,
            link_bandwidth_gbps=800,
        ),
    )


class MeshFrontierTests(unittest.TestCase):
    def test_enumerates_factor_topologies_and_modes(self):
        plans = planner().candidates()
        shapes = {(plan.mesh_x, plan.mesh_y) for plan in plans}
        self.assertIn((4, 4), shapes)
        self.assertIn((1, 16), shapes)
        self.assertEqual(len(plans), len(shapes) * len(ShardingMode))

    def test_frontier_is_non_dominated(self):
        mesh = planner()
        frontier = mesh.frontier()
        self.assertGreater(len(frontier), 0)
        for candidate in frontier:
            for other in frontier:
                if candidate is other:
                    continue
                self.assertFalse(mesh._dominates(other, candidate))

    def test_reserve_chips_changes_active_topology(self):
        plans = planner(reserve=4).candidates()
        self.assertTrue(all(plan.active_chips == 12 for plan in plans))
        self.assertTrue(all(plan.reserve_chips == 4 for plan in plans))
        self.assertIn((3, 4), {(plan.mesh_x, plan.mesh_y) for plan in plans})

    def test_memory_preference_selects_low_memory_candidate(self):
        mesh = planner()
        selected = mesh.choose(preference="memory")
        minimum = min(plan.memory_mib_per_chip for plan in mesh.frontier())
        self.assertEqual(selected.memory_mib_per_chip, minimum)

    def test_constraints_fail_closed_when_impossible(self):
        with self.assertRaises(NoFeasibleMesh):
            planner().choose(MeshConstraints(max_step_ms=0.000001))

    def test_plan_is_evidence_bound_and_fingerprinted(self):
        selected = planner().choose(preference="balanced")
        self.assertEqual(
            selected.evidence_state,
            "MODELED_MESH_FRONTIER_NOT_TPU_MEASUREMENT",
        )
        self.assertEqual(len(selected.fingerprint), 64)
        self.assertGreater(selected.step_ms, 0)

    def test_invalid_hardware_or_workload_refuses(self):
        with self.assertRaises(ValueError):
            MeshHardwareModel(
                chips=4,
                reserve_chips=4,
                flops_per_chip=1,
                link_bandwidth_gbps=1,
            )
        with self.assertRaises(ValueError):
            MeshWorkload(tokens=0, hidden_dim=1, layers=1)


if __name__ == "__main__":
    unittest.main()
