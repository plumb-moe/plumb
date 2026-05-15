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


class PlacementReport(BaseModel):
    method: str
    # JSON-safe key: "layer_id:expert_id"
    expert_placement: dict[str, int]
    estimated_improvement_pct_min: float
    estimated_improvement_pct_max: float
    estimated_improvement_pct: float


class ProfileReport(BaseModel):
    model_name: str
    hardware_config: str
    profiling_duration_seconds: float
    total_forward_passes: int
    layers: list[LayerReport]
    placement: PlacementReport | None = None
    gpu_to_numa: dict[int, int] | None = None
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
