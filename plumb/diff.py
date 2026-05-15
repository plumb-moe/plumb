from __future__ import annotations

from dataclasses import dataclass, field

from .report.schema import ProfileReport


@dataclass
class ExpertDelta:
    layer_id: int
    expert_id: int
    token_count_before: int
    token_count_after: int
    delta: int
    delta_pct: float


@dataclass
class DiffResult:
    model_name_a: str
    model_name_b: str
    mean_imbalance_before: float
    mean_imbalance_after: float
    max_imbalance_before: float
    max_imbalance_after: float
    ttft_est_before: float | None
    ttft_est_after: float | None
    expert_deltas: list[ExpertDelta] = field(default_factory=list)


def compute_diff(report_a: ProfileReport, report_b: ProfileReport) -> DiffResult:
    def _mean_imbalance(r: ProfileReport) -> float:
        if not r.layers:
            return 0.0
        return sum(la.imbalance_ratio for la in r.layers) / len(r.layers)

    def _max_imbalance(r: ProfileReport) -> float:
        if not r.layers:
            return 0.0
        return max(la.imbalance_ratio for la in r.layers)

    layers_b = {la.layer_id: la for la in report_b.layers}

    deltas: list[ExpertDelta] = []
    for layer_a in report_a.layers:
        layer_b = layers_b.get(layer_a.layer_id)
        experts_b = {e.expert_id: e.token_count for e in layer_b.experts} if layer_b else {}

        for exp_a in layer_a.experts:
            before = exp_a.token_count
            after = experts_b.get(exp_a.expert_id, 0)
            delta = after - before
            delta_pct = (delta / before * 100.0) if before else 0.0
            deltas.append(ExpertDelta(
                layer_id=layer_a.layer_id,
                expert_id=exp_a.expert_id,
                token_count_before=before,
                token_count_after=after,
                delta=delta,
                delta_pct=round(delta_pct, 2),
            ))

    ttft_a = report_a.placement.estimated_improvement_pct if report_a.placement else None
    ttft_b = report_b.placement.estimated_improvement_pct if report_b.placement else None

    return DiffResult(
        model_name_a=report_a.model_name,
        model_name_b=report_b.model_name,
        mean_imbalance_before=round(_mean_imbalance(report_a), 4),
        mean_imbalance_after=round(_mean_imbalance(report_b), 4),
        max_imbalance_before=round(_max_imbalance(report_a), 4),
        max_imbalance_after=round(_max_imbalance(report_b), 4),
        ttft_est_before=ttft_a,
        ttft_est_after=ttft_b,
        expert_deltas=deltas,
    )
