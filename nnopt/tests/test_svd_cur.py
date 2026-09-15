"""Tests for nnopt.cur.svd_cur (README Sec 2.3 / Sec 3.2-C)."""

from __future__ import annotations

import numpy as np
import pytest

from nnopt.cur.svd_cur import (
    analyze_spectrum,
    build_cur,
    column_leverage_scores,
    compression_ratio,
    cur_matmul_flops,
    cur_param_count,
    original_matmul_flops,
    original_param_count,
    row_leverage_scores,
    select_cur_columns,
    select_cur_columns_by_leverage,
    select_cur_rows,
    truncated_svd_baseline,
)


def _make_low_rank(m: int, n: int, rank: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, rank))
    b = rng.standard_normal((rank, n))
    w = a @ b
    if noise > 0:
        w = w + noise * rng.standard_normal((m, n))
    return w


def test_spectrum_rank_for_energy_recovers_true_rank():
    w = _make_low_rank(20, 15, rank=3, noise=1e-6, seed=0)
    spectrum = analyze_spectrum(w)
    r99 = spectrum.rank_for_energy(0.999999)
    assert r99 <= 4  # true rank 3 (+/-1 slack for the tiny noise floor)
    r_full = spectrum.rank_for_energy(1.0)
    assert r_full == min(20, 15)


def test_row_leverage_scores_sum_to_one_over_full_rank_subspace():
    w = np.random.default_rng(1).standard_normal((12, 9))
    spectrum = analyze_spectrum(w)
    full_rank = min(w.shape)
    scores = row_leverage_scores(spectrum, rank=full_rank)
    assert scores.sum() == pytest.approx(1.0, abs=1e-8)


def test_select_cur_columns_respects_priority_and_budget():
    priorities = {0: 5.0, 1: 1.0, 2: 9.0, 3: 3.0}
    chosen = select_cur_columns(list(priorities.keys()), priorities, c=2)
    assert chosen == [2, 0]  # highest priority first


def test_build_cur_exactly_reconstructs_true_low_rank_matrix():
    m, n, rank = 10, 8, 3
    w = _make_low_rank(m, n, rank=rank, noise=0.0, seed=2)
    spectrum = analyze_spectrum(w)
    row_idx = select_cur_rows(spectrum, rank=rank, r=5)
    col_priorities = {j: 1.0 for j in range(n)}  # arbitrary; all columns candidate
    col_idx = select_cur_columns(list(range(n)), col_priorities, c=5)

    result = build_cur(w, col_idx, row_idx)
    assert result.frobenius_error_relative < 1e-8, "exact rank-r matrix must reconstruct essentially exactly"


def test_cur_error_is_in_the_right_ballpark_versus_svd_optimum():
    m, n, true_rank = 30, 25, 4
    w = _make_low_rank(m, n, rank=true_rank, noise=0.05, seed=3)
    spectrum = analyze_spectrum(w)
    r = true_rank
    row_idx = select_cur_rows(spectrum, rank=r, r=8)
    col_priorities = {j: 1.0 for j in range(n)}
    col_idx = select_cur_columns(list(range(n)), col_priorities, c=8)

    cur_result = build_cur(w, col_idx, row_idx)
    _, svd_err = truncated_svd_baseline(w, rank=r)

    assert svd_err < 0.5  # sanity: the low-rank signal really is recoverable
    # CUR is not optimal but should not be wildly worse than the SVD floor
    # for a well-conditioned synthetic case with slack (c, r > true_rank).
    assert cur_result.frobenius_error_relative < 5.0 * svd_err + 1e-6


def test_resource_accounting_matches_hand_computation():
    m, n, c, r = 100, 100, 10, 8
    assert cur_param_count(m, n, c, r) == m * c + c * r + r * n
    assert original_param_count(m, n) == m * n
    ratio = compression_ratio(m, n, c, r)
    assert ratio == pytest.approx((m * n) / (m * c + c * r + r * n))
    assert ratio > 1.0  # this configuration genuinely compresses


def test_column_leverage_scores_sum_to_one_over_full_rank_subspace():
    """Symmetric property to row_leverage_scores (already tested): column
    leverage scores over the FULL right-singular subspace must sum to 1."""
    w = np.random.default_rng(6).standard_normal((12, 9))
    spectrum = analyze_spectrum(w)
    full_rank = min(w.shape)
    scores = column_leverage_scores(spectrum, rank=full_rank)
    assert scores.sum() == pytest.approx(1.0, abs=1e-8)


def test_select_cur_columns_by_leverage_matches_top_scores():
    w = np.random.default_rng(7).standard_normal((20, 15))
    spectrum = analyze_spectrum(w)
    scores = column_leverage_scores(spectrum, rank=5)
    chosen = select_cur_columns_by_leverage(spectrum, rank=5, c=4)
    expected = list(np.argsort(-scores)[:4])
    assert chosen == expected


def test_cur_can_be_worse_than_original_for_small_operators():
    """README Sec 3.2-C: CUR turns 1 GEMM into 3; for small matrices with
    generously sized c/r, the 3-matmul FLOPs total can exceed the original
    single matmul -- this is precisely why acceptance must be gated on
    measured latency (nnopt.bench.latency), not FLOPs/param counts alone."""
    m, n, c, r, batch = 64, 64, 32, 32, 1
    orig_flops = original_matmul_flops(m, n, batch)
    cur_flops = cur_matmul_flops(m, n, c, r, batch)
    assert cur_flops > orig_flops
