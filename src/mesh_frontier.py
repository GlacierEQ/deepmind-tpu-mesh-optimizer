"""Pareto mesh-topology planner over explicit accelerator modeling assumptions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from math import isfinite

EVIDENCE_STATE = "MODELED_MESH_FRONTIER_NOT_TPU_MEASUREMENT"


class ShardingMode(str, Enum):
    DATA = "DATA"
    TENSOR = "TENSOR"
    SEQUENCE = "SEQUENCE"
    HYBRID = "HYBRID"


@dataclass(frozen=True)
class MeshWorkload:
    tokens: int
    hidden_dim: int
    layers: int
    batch_size: int = 1
    bytes_per_element: int = 2
    flops_per_token: float = 1e9

    def __post_init__(self) -> None:
        for name in ("tokens", "hidden_dim", "layers", "batch_size", "bytes_per_element"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not isfinite(self.flops_per_token) or self.flops_per_token <= 0:
            raise ValueError("flops_per_token must be finite and positive")

    @property
    def activation_bytes(self) -> int:
        return self.tokens * self.hidden_dim * self.batch_size * self.bytes_per_element

    @property
    def total_flops(self) -> float:
        return self.tokens * self.batch_size * self.layers * self.flops_per_token


@dataclass(frozen=True)
class MeshHardwareModel:
    chips: int
    flops_per_chip: float
    link_bandwidth_gbps: float
    reserve_chips: int = 0

    def __post_init__(self) -> None:
        if self.chips <= 0:
            raise ValueError("chips must be positive")
        if self.reserve_chips < 0 or self.reserve_chips >= self.chips:
            raise ValueError("reserve_chips must be within [0, chips)")
        for name in ("flops_per_chip", "link_bandwidth_gbps"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def active_chips(self) -> int:
        return self.chips - self.reserve_chips


@dataclass(frozen=True)
class MeshConstraints:
    max_memory_mib_per_chip: float | None = None
    max_step_ms: float | None = None
    max_exposed_comm_ms: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_memory_mib_per_chip", "max_step_ms", "max_exposed_comm_ms"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when provided")


@dataclass(frozen=True)
class MeshPlan:
    mesh_x: int
    mesh_y: int
    mode: ShardingMode
    active_chips: int
    reserve_chips: int
    memory_mib_per_chip: float
    communication_ms: float
    compute_ms: float
    exposed_comm_ms: float
    step_ms: float
    topology_penalty: float
    evidence_state: str = EVIDENCE_STATE

    def as_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["mode"] = self.mode.value
        return row

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class NoFeasibleMesh(RuntimeError):
    pass


class MeshFrontierPlanner:
    """Enumerate topology/sharding candidates and select non-dominated plans."""

    _COMM_FACTORS = {
        ShardingMode.DATA: 1.00,
        ShardingMode.TENSOR: 1.35,
        ShardingMode.SEQUENCE: 0.85,
        ShardingMode.HYBRID: 0.70,
    }
    _MEMORY_FACTORS = {
        ShardingMode.DATA: 1.00,
        ShardingMode.TENSOR: 0.62,
        ShardingMode.SEQUENCE: 0.72,
        ShardingMode.HYBRID: 0.48,
    }

    def __init__(self, workload: MeshWorkload, hardware: MeshHardwareModel) -> None:
        self.workload = workload
        self.hardware = hardware

    @staticmethod
    def _factor_pairs(n: int) -> tuple[tuple[int, int], ...]:
        pairs = set()
        for x in range(1, int(n**0.5) + 1):
            if n % x == 0:
                y = n // x
                pairs.add((x, y))
                pairs.add((y, x))
        return tuple(sorted(pairs))

    def candidates(self) -> tuple[MeshPlan, ...]:
        active = self.hardware.active_chips
        plans: list[MeshPlan] = []
        for mesh_x, mesh_y in self._factor_pairs(active):
            aspect_penalty = 1.0 + abs(mesh_x - mesh_y) / (mesh_x + mesh_y)
            for mode in ShardingMode:
                memory_bytes = (
                    self.workload.activation_bytes
                    * self.workload.layers
                    * self._MEMORY_FACTORS[mode]
                    / active
                )
                wire_bytes = (
                    self.workload.activation_bytes
                    * self._COMM_FACTORS[mode]
                    * (active - 1)
                    / active
                    * aspect_penalty
                )
                bytes_per_ms = self.hardware.link_bandwidth_gbps * 125_000.0
                communication_ms = wire_bytes / bytes_per_ms
                compute_ms = (
                    self.workload.total_flops
                    / (self.hardware.flops_per_chip * active)
                    * 1000.0
                )
                overlap_fraction = {
                    ShardingMode.DATA: 0.50,
                    ShardingMode.TENSOR: 0.62,
                    ShardingMode.SEQUENCE: 0.72,
                    ShardingMode.HYBRID: 0.80,
                }[mode]
                hidden_comm = min(communication_ms, compute_ms * overlap_fraction)
                exposed_comm = max(0.0, communication_ms - hidden_comm)
                step_ms = compute_ms + exposed_comm
                plans.append(
                    MeshPlan(
                        mesh_x=mesh_x,
                        mesh_y=mesh_y,
                        mode=mode,
                        active_chips=active,
                        reserve_chips=self.hardware.reserve_chips,
                        memory_mib_per_chip=memory_bytes / (1024 * 1024),
                        communication_ms=communication_ms,
                        compute_ms=compute_ms,
                        exposed_comm_ms=exposed_comm,
                        step_ms=step_ms,
                        topology_penalty=aspect_penalty,
                    )
                )
        return tuple(plans)

    @staticmethod
    def _dominates(left: MeshPlan, right: MeshPlan) -> bool:
        lhs = (
            left.memory_mib_per_chip,
            left.exposed_comm_ms,
            left.step_ms,
            left.topology_penalty,
        )
        rhs = (
            right.memory_mib_per_chip,
            right.exposed_comm_ms,
            right.step_ms,
            right.topology_penalty,
        )
        return all(a <= b for a, b in zip(lhs, rhs)) and any(
            a < b for a, b in zip(lhs, rhs)
        )

    def frontier(self) -> tuple[MeshPlan, ...]:
        plans = self.candidates()
        frontier = [
            candidate
            for candidate in plans
            if not any(
                self._dominates(other, candidate)
                for other in plans
                if other is not candidate
            )
        ]
        return tuple(
            sorted(
                frontier,
                key=lambda plan: (
                    plan.step_ms,
                    plan.memory_mib_per_chip,
                    plan.exposed_comm_ms,
                    plan.topology_penalty,
                    plan.mode.value,
                    plan.mesh_x,
                    plan.mesh_y,
                ),
            )
        )

    @staticmethod
    def _feasible(plan: MeshPlan, constraints: MeshConstraints) -> bool:
        return (
            (
                constraints.max_memory_mib_per_chip is None
                or plan.memory_mib_per_chip <= constraints.max_memory_mib_per_chip
            )
            and (
                constraints.max_step_ms is None
                or plan.step_ms <= constraints.max_step_ms
            )
            and (
                constraints.max_exposed_comm_ms is None
                or plan.exposed_comm_ms <= constraints.max_exposed_comm_ms
            )
        )

    def choose(
        self,
        constraints: MeshConstraints = MeshConstraints(),
        *,
        preference: str = "balanced",
    ) -> MeshPlan:
        feasible = [plan for plan in self.frontier() if self._feasible(plan, constraints)]
        if not feasible:
            raise NoFeasibleMesh("no modeled mesh satisfies the supplied constraints")
        keys = {
            "balanced": lambda plan: (
                plan.step_ms * (1.0 + 0.15 * (plan.topology_penalty - 1.0)),
                plan.memory_mib_per_chip,
                plan.exposed_comm_ms,
            ),
            "latency": lambda plan: (
                plan.step_ms,
                plan.exposed_comm_ms,
                plan.memory_mib_per_chip,
            ),
            "memory": lambda plan: (
                plan.memory_mib_per_chip,
                plan.step_ms,
                plan.exposed_comm_ms,
            ),
            "communication": lambda plan: (
                plan.exposed_comm_ms,
                plan.communication_ms,
                plan.step_ms,
            ),
        }
        if preference not in keys:
            raise ValueError(f"unknown preference: {preference}")
        return min(feasible, key=keys[preference])
