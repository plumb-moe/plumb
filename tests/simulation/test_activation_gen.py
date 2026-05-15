"""Tests for the synthetic MoE activation generator.

Gate G0-A: generator output must match MoE-CAP reference distributions within 10%.
Reference statistics are stored in tests/fixtures/activation-reference/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plumb.simulation.activation_gen import (
    GeneratorConfig,
    _sigma_for_target,
    from_profile,
    generate,
    imbalance_ratios,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "activation-reference"


# ---------------------------------------------------------------------------
# Gate G0-A: reference fixture validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_file,profile_name", [
    ("deepseek-v3.json", "deepseek_v3"),
    ("mixtral-8x7b.json", "mixtral_8x7b"),
])
def test_matches_reference_within_tolerance(fixture_file: str, profile_name: str) -> None:
    """Gate G0-A: mean imbalance ratio must match reference within tolerance_pct."""
    with open(FIXTURES / fixture_file) as f:
        ref = json.load(f)

    load = from_profile(profile_name, seed=42)
    ratios = imbalance_ratios(load)

    tol = ref["tolerance_pct"] / 100.0
    ref_mean = ref["statistics"]["mean_imbalance_ratio"]
    actual_mean = float(ratios.mean())
    rel_error = abs(actual_mean - ref_mean) / ref_mean

    assert rel_error <= tol, (
        f"{profile_name}: mean imbalance {actual_mean:.3f} deviates "
        f"{rel_error*100:.1f}% from reference {ref_mean} "
        f"(tolerance {tol*100:.0f}%)"
    )


def test_deepseek_v3_shape_matches_reference() -> None:
    with open(FIXTURES / "deepseek-v3.json") as f:
        ref = json.load(f)
    load = from_profile("deepseek_v3", seed=0)
    assert load.shape == (ref["model"]["num_layers"], ref["model"]["num_experts"])


def test_mixtral_shape_matches_reference() -> None:
    with open(FIXTURES / "mixtral-8x7b.json") as f:
        ref = json.load(f)
    load = from_profile("mixtral_8x7b", seed=0)
    assert load.shape == (ref["model"]["num_layers"], ref["model"]["num_experts"])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_output() -> None:
    cfg = GeneratorConfig(num_experts=64, num_layers=16, seed=7)
    a = generate(cfg)
    b = generate(cfg)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ() -> None:
    cfg_a = GeneratorConfig(num_experts=64, num_layers=16, seed=1)
    cfg_b = GeneratorConfig(num_experts=64, num_layers=16, seed=2)
    assert not np.array_equal(generate(cfg_a), generate(cfg_b))


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------

def test_output_shape() -> None:
    cfg = GeneratorConfig(num_experts=32, num_layers=12, active_k=2)
    load = generate(cfg)
    assert load.shape == (12, 32)


def test_output_dtype_float32() -> None:
    cfg = GeneratorConfig(num_experts=16, num_layers=4)
    assert generate(cfg).dtype == np.float32


def test_all_non_negative() -> None:
    cfg = GeneratorConfig(num_experts=64, num_layers=8)
    assert (generate(cfg) >= 0).all()


# ---------------------------------------------------------------------------
# Token conservation
# ---------------------------------------------------------------------------

def test_token_conservation_per_layer() -> None:
    """Each layer must route exactly batch_size * seq_len * active_k tokens."""
    cfg = GeneratorConfig(
        num_experts=16, num_layers=8,
        active_k=3, batch_size=16, seq_len=128,
    )
    expected_tokens = cfg.batch_size * cfg.seq_len * cfg.active_k
    load = generate(cfg)
    for layer_idx in range(cfg.num_layers):
        total = int(load[layer_idx].sum())
        assert total == expected_tokens, (
            f"Layer {layer_idx}: {total} tokens, expected {expected_tokens}"
        )


# ---------------------------------------------------------------------------
# Imbalance properties
# ---------------------------------------------------------------------------

def test_imbalance_ratio_at_least_one() -> None:
    """max/mean is always >= 1."""
    load = from_profile("mixtral_8x7b", seed=0)
    assert (imbalance_ratios(load) >= 1.0).all()


def test_higher_target_gives_higher_imbalance() -> None:
    """Increasing mean_imbalance should increase the observed ratio."""
    base = dict(num_experts=64, num_layers=24, active_k=4, batch_size=64, seq_len=512, seed=0)
    low = generate(GeneratorConfig(**base, mean_imbalance=1.3, std_imbalance=0.05))
    high = generate(GeneratorConfig(**base, mean_imbalance=3.5, std_imbalance=0.05))
    assert imbalance_ratios(high).mean() > imbalance_ratios(low).mean()


def test_imbalance_ratios_shape() -> None:
    load = np.ones((6, 32), dtype=np.float32)
    ratios = imbalance_ratios(load)
    assert ratios.shape == (6,)
    np.testing.assert_allclose(ratios, 1.0, rtol=1e-5)


def test_imbalance_ratios_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        imbalance_ratios(np.ones(8))  # 1-D not allowed


# ---------------------------------------------------------------------------
# EPLB compatibility
# ---------------------------------------------------------------------------

def test_eplb_compatible_numpy() -> None:
    """Output format must be float32 array passable to torch.tensor()."""
    load = from_profile("deepseek_v3", seed=1)
    assert load.dtype == np.float32
    assert load.ndim == 2


def test_eplb_compatible_torch() -> None:
    """If torch is available, tensor conversion must succeed."""
    pytest.importorskip("torch")
    import torch

    load = from_profile("mixtral_8x7b", seed=0)
    t = torch.tensor(load)
    assert t.dtype == torch.float32
    assert t.shape == load.shape


# ---------------------------------------------------------------------------
# from_profile API
# ---------------------------------------------------------------------------

def test_from_profile_known_profiles() -> None:
    for name in ("deepseek_v3", "mixtral_8x7b", "qwen_moe"):
        load = from_profile(name, seed=0)
        assert load.ndim == 2
        assert load.dtype == np.float32


def test_from_profile_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        from_profile("gpt4_moe")


def test_from_profile_override_num_layers() -> None:
    load = from_profile("deepseek_v3", seed=0, num_layers=4)
    assert load.shape[0] == 4


def test_from_profile_override_batch_size() -> None:
    cfg_base = dict(num_experts=8, num_layers=4, active_k=2, seed=0)
    small = generate(GeneratorConfig(**cfg_base, batch_size=4,  seq_len=64))
    large = generate(GeneratorConfig(**cfg_base, batch_size=32, seq_len=64))
    # Larger batch → more total tokens per layer
    assert large.sum() > small.sum()


# ---------------------------------------------------------------------------
# Sigma calibration
# ---------------------------------------------------------------------------

def test_sigma_increases_with_target() -> None:
    assert _sigma_for_target(64, 1.5) < _sigma_for_target(64, 3.0)


def test_sigma_decreases_with_num_experts() -> None:
    """More experts → lower sigma needed for same imbalance (more draws dilute max)."""
    assert _sigma_for_target(256, 2.0) < _sigma_for_target(8, 2.0)


def test_sigma_calibration_matches_target_within_5pct() -> None:
    """Calibration formula must be within 5% for standard batch sizes."""
    rng = np.random.default_rng(99)
    for n, total_tokens, target in [
        (256, 32 * 512 * 8, 2.24),
        (8,   32 * 512 * 2, 1.72),
        (64,  32 * 512 * 4, 2.05),
    ]:
        sigma = _sigma_for_target(n, target)
        observed = []
        for _ in range(200):
            w = rng.lognormal(0.0, sigma, n)
            w /= w.sum()
            c = rng.multinomial(total_tokens, w).astype(np.float32)
            observed.append(c.max() / c.mean())
        err = abs(np.mean(observed) - target) / target
        assert err < 0.05, (
            f"n={n}, target={target}: sigma={sigma:.4f} gives "
            f"observed={np.mean(observed):.3f}, error={err*100:.1f}%"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_expert() -> None:
    cfg = GeneratorConfig(num_experts=1, num_layers=4, active_k=1, batch_size=8, seq_len=16)
    load = generate(cfg)
    assert load.shape == (4, 1)
    assert (load == 8 * 16 * 1).all()


def test_single_layer() -> None:
    cfg = GeneratorConfig(num_experts=8, num_layers=1)
    load = generate(cfg)
    assert load.shape == (1, 8)
