from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ExpertLoad(BaseModel):
    expert_id: int
    token_count: int
    activation_fraction: float


class LayerReport(BaseModel):
    layer_id: int
    imbalance_ratio: float
    max_expert_id: int
    min_expert_id: int
    cross_numa_rate: float | None = None
    experts: list[ExpertLoad]


class GpuStatsReport(BaseModel):
    per_gpu_sm_utilisation: dict[str, float]
    imbalance_confirmed: bool
    sm_samples: dict[str, list[float]] = Field(default_factory=dict)


class PlacementReport(BaseModel):
    method: str
    expert_placement: dict[str, list[int]]
    estimated_improvement_pct_min: float
    estimated_improvement_pct_max: float
    estimated_improvement_pct: float


class CommunicationCostReport(BaseModel):
    current_overhead_us: float
    recommended_overhead_us: float
    delta_us: float
    caveat: str | None = None


class CoactivationPairReport(BaseModel):
    expert_a: int
    expert_b: int
    coactivation_count: int
    cross_gpu: bool


class CoactivationLayerReport(BaseModel):
    layer_id: int
    cross_gpu_coactivation_rate: float
    estimated_extra_hops_per_pass: float
    total_coactivation_count: int
    top_misplaced_pairs: list[CoactivationPairReport]


class CoactivationReport(BaseModel):
    layers: list[CoactivationLayerReport]
    total_cross_gpu_coactivation_rate: float


class GpuCapabilityReport(BaseModel):
    index: int
    name: str
    memory_total_mib: int
    memory_free_mib: int
    compute_cap: str
    max_sm_clock_mhz: int
    max_mem_clock_mhz: int
    relative_compute_score: float


class HeterogeneousTopologyReport(BaseModel):
    gpus: list[GpuCapabilityReport]
    is_homogeneous: bool
    mixed_vendor: bool
    compute_score_range: tuple[float, float]


class PlacementViolationReport(BaseModel):
    layer_id: int
    expert_id: int
    assigned_gpu: int
    fastest_gpu: int


class HeterogeneousPlacementReport(BaseModel):
    gpu_expert_counts: dict[str, int]
    violations: list[PlacementViolationReport]


class ProfileReport(BaseModel):
    model_name: str
    hardware_config: str
    profiling_duration_seconds: float
    total_forward_passes: int
    layers: list[LayerReport]
    placement: PlacementReport | None = None
    gpu_to_numa: dict[int, int] | None = None
    gpu_stats: GpuStatsReport | None = None
    communication_cost: CommunicationCostReport | None = None
    coactivation: CoactivationReport | None = None
    heterogeneous_topology: HeterogeneousTopologyReport | None = None
    heterogeneous_placement: HeterogeneousPlacementReport | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def summary(self) -> dict[str, Any]:
        if not self.layers:
            return {}
        ratios = [la.imbalance_ratio for la in self.layers]
        worst = max(self.layers, key=lambda la: la.imbalance_ratio)
        return {
            "num_layers_profiled": len(self.layers),
            "mean_imbalance_ratio": round(sum(ratios) / len(ratios), 3),
            "max_imbalance_ratio": round(max(ratios), 3),
            "worst_layer_id": worst.layer_id,
            "total_forward_passes": self.total_forward_passes,
        }
