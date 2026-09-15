import numpy as np
import pytest

from nnopt.cur.lowrank_baselines import (
    activation_aware_svd,
    output_relative_error,
    rank_for_param_budget_cur,
    rank_for_param_budget_svd,
    truncated_svd,
    weight_relative_error,
)


def test_truncated_svd_is_exact_at_full_rank():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((12, 20))
    assert np.allclose(truncated_svd(w, 12), w)


def test_truncated_svd_recovers_exact_low_rank_matrix():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((30, 4)), rng.standard_normal((4, 25))
    w = a @ b
    assert weight_relative_error(w, truncated_svd(w, 4)) < 1e-10


def test_truncated_svd_beats_any_other_rank_r_on_weight_error():
    """Eckart-Young: nothing may beat truncated SVD in Frobenius weight
    error at the same rank -- including our activation-aware variant."""
    rng = np.random.default_rng(2)
    w = rng.standard_normal((24, 30))
    x = rng.standard_normal((60, 30))
    r = 6
    e_svd = weight_relative_error(w, truncated_svd(w, r))
    e_aa = weight_relative_error(w, activation_aware_svd(w, x, r))
    assert e_svd <= e_aa + 1e-9


def test_activation_aware_svd_beats_plain_svd_on_output_error():
    """The whole point: when calibration activations are anisotropic, the
    calibration-optimal factorization must win on OUTPUT error."""
    rng = np.random.default_rng(3)
    w = rng.standard_normal((24, 30))
    # strongly anisotropic activations: a few input directions dominate
    z = rng.standard_normal((200, 30))
    z[:, :5] *= 30.0
    r = 6
    e_svd = output_relative_error(w, truncated_svd(w, r), z)
    e_aa = output_relative_error(w, activation_aware_svd(w, z, r), z)
    assert e_aa < e_svd, f"activation-aware {e_aa:.5f} should beat plain {e_svd:.5f}"


def test_activation_aware_svd_exact_at_full_rank():
    rng = np.random.default_rng(4)
    w = rng.standard_normal((10, 14))
    x = rng.standard_normal((40, 14))
    assert output_relative_error(w, activation_aware_svd(w, x, 10), x) < 1e-6


def test_activation_aware_handles_rank_deficient_calibration():
    """Fewer calibration rows than input dims -> singular Gram matrix; the
    ridge must keep this solvable rather than raising."""
    rng = np.random.default_rng(5)
    w = rng.standard_normal((8, 20))
    x = rng.standard_normal((5, 20))  # rank <= 5 << 20
    out = activation_aware_svd(w, x, 3)
    assert out.shape == w.shape
    assert np.all(np.isfinite(out))


def test_param_budget_ranks_respect_their_formulas():
    m, n = 1024, 4096
    budget = 2_000_000
    r_svd = rank_for_param_budget_svd(m, n, budget)
    r_cur = rank_for_param_budget_cur(m, n, budget)
    assert r_svd * (m + n) <= budget
    assert (r_svd + 1) * (m + n) > budget
    assert r_cur * (m + n) + r_cur**2 <= budget
    assert (r_cur + 1) * (m + n) + (r_cur + 1) ** 2 > budget


def test_cur_affords_lower_rank_than_svd_at_equal_budget():
    """CUR carries an extra r^2 block, so at a fixed parameter budget it
    must settle for a rank no higher than SVD's -- the reason matched-rank
    comparisons flatter CUR unfairly."""
    m, n, budget = 1024, 1024, 1_000_000
    assert rank_for_param_budget_cur(m, n, budget) <= rank_for_param_budget_svd(m, n, budget)


@pytest.mark.parametrize("rank", [1, 3, 8])
def test_output_error_is_monotone_nonincreasing_in_rank(rank):
    rng = np.random.default_rng(6)
    w = rng.standard_normal((16, 20))
    x = rng.standard_normal((50, 20))
    e_here = output_relative_error(w, truncated_svd(w, rank), x)
    e_more = output_relative_error(w, truncated_svd(w, rank + 4), x)
    assert e_more <= e_here + 1e-9
