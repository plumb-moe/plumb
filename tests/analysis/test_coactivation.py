from plumb.analysis.coactivation import (
    build_coactivation_matrix,
    compute_cross_gpu_coactivation,
)
from plumb.topology import Topology


def _flat4() -> Topology:
    return Topology.flat(4)


def _dual_numa() -> Topology:
    return Topology({0: 0, 1: 0, 2: 1, 3: 1})


# ---------------------------------------------------------------------------
# build_coactivation_matrix
# ---------------------------------------------------------------------------

def test_matrix_keys_are_ordered_pairs():
    snapshot = {(0, 0): 100, (0, 1): 80, (0, 2): 50}
    matrix = build_coactivation_matrix(snapshot)
    for (a, b) in matrix[0]:
        assert a < b


def test_matrix_value_is_product_of_loads():
    snapshot = {(0, 0): 100, (0, 1): 80}
    matrix = build_coactivation_matrix(snapshot)
    assert matrix[0][(0, 1)] == 100 * 80


def test_matrix_per_layer_isolation():
    snapshot = {(0, 0): 10, (0, 1): 20, (1, 0): 5, (1, 1): 15}
    matrix = build_coactivation_matrix(snapshot)
    assert 0 in matrix and 1 in matrix
    assert matrix[0][(0, 1)] == 10 * 20
    assert matrix[1][(0, 1)] == 5 * 15


def test_matrix_single_expert_no_pairs():
    snapshot = {(0, 0): 100}
    matrix = build_coactivation_matrix(snapshot)
    assert matrix[0] == {}


def test_matrix_empty_snapshot():
    assert build_coactivation_matrix({}) == {}


# ---------------------------------------------------------------------------
# compute_cross_gpu_coactivation — cross_gpu_rate = 0 when all same GPU
# ---------------------------------------------------------------------------

def test_cross_gpu_rate_zero_all_same_gpu():
    snapshot = {(0, e): 100 for e in range(4)}
    matrix = build_coactivation_matrix(snapshot)
    placement = {(0, e): [0] for e in range(4)}
    results = compute_cross_gpu_coactivation(matrix, placement, snapshot, _flat4())
    assert len(results) == 1
    assert results[0].cross_gpu_coactivation_rate == 0.0
    assert results[0].top_misplaced_pairs == []


def test_cross_gpu_rate_one_all_different_gpus():
    snapshot = {(0, 0): 100, (0, 1): 100}
    matrix = build_coactivation_matrix(snapshot)
    # Experts on different GPUs
    placement = {(0, 0): [0], (0, 1): [1]}
    results = compute_cross_gpu_coactivation(matrix, placement, snapshot, _flat4())
    assert results[0].cross_gpu_coactivation_rate == 1.0


def test_top_misplaced_pairs_sorted_descending():
    # Expert 0 very hot, expert 1 moderate, expert 2 cold
    snapshot = {(0, 0): 1000, (0, 1): 500, (0, 2): 50}
    matrix = build_coactivation_matrix(snapshot)
    # All on different GPUs
    placement = {(0, 0): [0], (0, 1): [1], (0, 2): [2]}
    results = compute_cross_gpu_coactivation(matrix, placement, snapshot, _flat4())
    pairs = results[0].top_misplaced_pairs
    counts = [p.coactivation_count for p in pairs]
    assert counts == sorted(counts, reverse=True)


def test_estimated_extra_hops_positive_when_cross_gpu():
    snapshot = {(0, 0): 100, (0, 1): 100}
    matrix = build_coactivation_matrix(snapshot)
    placement = {(0, 0): [0], (0, 1): [1]}
    results = compute_cross_gpu_coactivation(matrix, placement, snapshot, _flat4())
    assert results[0].estimated_extra_hops_per_pass > 0.0


def test_estimated_extra_hops_zero_when_same_gpu():
    snapshot = {(0, 0): 100, (0, 1): 100}
    matrix = build_coactivation_matrix(snapshot)
    placement = {(0, 0): [0], (0, 1): [0]}
    results = compute_cross_gpu_coactivation(matrix, placement, snapshot, _flat4())
    assert results[0].estimated_extra_hops_per_pass == 0.0


def test_missing_placement_defaults_to_same_gpu_zero_cross():
    snapshot = {(0, 0): 100, (0, 1): 80}
    matrix = build_coactivation_matrix(snapshot)
    results = compute_cross_gpu_coactivation(matrix, {}, snapshot, _flat4())
    assert results[0].cross_gpu_coactivation_rate == 0.0


def test_multi_layer_results():
    snapshot = {(0, 0): 100, (0, 1): 100, (1, 0): 50, (1, 1): 50}
    matrix = build_coactivation_matrix(snapshot)
    placement = {(0, 0): [0], (0, 1): [1], (1, 0): [0], (1, 1): [0]}
    results = compute_cross_gpu_coactivation(matrix, placement, snapshot, _flat4())
    by_layer = {r.layer_id: r for r in results}
    assert by_layer[0].cross_gpu_coactivation_rate == 1.0
    assert by_layer[1].cross_gpu_coactivation_rate == 0.0
