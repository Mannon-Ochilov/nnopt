"""Tests for nnopt.cascade.budget (README Sec 1.1 -- cross-stage compensation)."""

from __future__ import annotations

import pytest

from nnopt.cascade.budget import (
    check_feasibility,
    closed_form_rank_for_target,
    discount_target_for_future_stage,
    max_cur_compression_factor,
    quant_max_factor,
    required_reduction_factor,
    residual_after_stage,
)
from nnopt.cur.svd_cur import cur_param_count, original_param_count


def test_quant_max_factor_known_transitions():
    assert quant_max_factor(32, 16) == pytest.approx(2.0)
    assert quant_max_factor(32, 8) == pytest.approx(4.0)
    assert quant_max_factor(32, 4) == pytest.approx(8.0)
    assert quant_max_factor(32, 32) == pytest.approx(1.0)  # no-op


def test_required_reduction_factor_already_fits():
    assert required_reduction_factor(1_000_000, 2_000_000) == pytest.approx(1.0)


def test_required_reduction_factor_needs_shrinking():
    assert required_reduction_factor(10_000_000, 2_000_000) == pytest.approx(5.0)


def test_worked_example_from_readme_sec_1_1():
    """operator needs 5x; INT8 gives its max 4x; residual must be exactly 1.25x."""
    required = required_reduction_factor(10.0, 2.0)  # 10MB -> 2MB budget
    assert required == pytest.approx(5.0)

    int8_max = quant_max_factor(32, 8)
    assert int8_max == pytest.approx(4.0)

    residual = residual_after_stage(required, int8_max)
    assert residual == pytest.approx(1.25)

    discounted = discount_target_for_future_stage(required, int8_max)
    assert discounted == pytest.approx(residual)  # the two framings must agree


def test_discount_never_goes_below_one():
    # future stage alone can already cover everything -> this stage owes nothing
    assert discount_target_for_future_stage(3.0, 10.0) == pytest.approx(1.0)


def test_check_feasibility_exact_boundary_and_shortfall():
    ok = check_feasibility(required_factor=5.0, stage_max_factors=[4.0, 1.25])
    assert ok.feasible
    assert ok.shortfall_factor == pytest.approx(1.0)

    bad = check_feasibility(required_factor=6.0, stage_max_factors=[4.0, 1.25])
    assert not bad.feasible
    assert bad.shortfall_factor == pytest.approx(6.0 / 5.0)


def test_residual_after_softer_than_max_achieved():
    # if the stage used a milder format than its ceiling, more is owed downstream
    required = 5.0
    residual_full = residual_after_stage(required, achieved_factor=4.0)
    residual_mild = residual_after_stage(required, achieved_factor=2.0)
    assert residual_mild > residual_full


@pytest.mark.parametrize("m,n,target", [(64, 64, 1.25), (256, 128, 3.0), (100, 100, 1.0)])
def test_closed_form_rank_is_the_largest_feasible_integer(m, n, target):
    r = closed_form_rank_for_target(m, n, target)
    assert 1 <= r <= min(m, n)

    achieved = original_param_count(m, n) / cur_param_count(m, n, r, r)
    if target <= 1.0:
        assert r == min(m, n)
        return
    # r must actually satisfy the target...
    assert achieved >= target - 1e-6, f"rank {r} does not reach target {target} (achieved {achieved})"
    # ...and r+1 must NOT (otherwise r wasn't the largest/softest feasible choice)
    if r < min(m, n):
        achieved_plus_one = original_param_count(m, n) / cur_param_count(m, n, r + 1, r + 1)
        assert achieved_plus_one < target, (
            f"rank {r+1} still reaches target {target} (achieved {achieved_plus_one}) "
            f"-- closed_form_rank_for_target under-shot the softest sufficient rank"
        )


def test_max_cur_compression_factor_matches_the_r_equals_one_ceiling():
    m, n = 100, 100
    ceiling = max_cur_compression_factor(m, n)
    achieved_at_r1 = original_param_count(m, n) / cur_param_count(m, n, 1, 1)
    assert ceiling == pytest.approx(achieved_at_r1)


def test_closed_form_rank_best_effort_when_target_exceeds_ceiling():
    """target_factor above max_cur_compression_factor cannot be reached by
    ANY integer rank (even the most aggressive r=1); the function must
    still return a valid rank (best-effort r=1) rather than raise or return
    something out of range -- feasibility itself is the caller's job
    (check_feasibility / max_cur_compression_factor)."""
    m, n = 100, 100
    ceiling = max_cur_compression_factor(m, n)
    unreachable_target = ceiling * 1.5
    r = closed_form_rank_for_target(m, n, unreachable_target)
    assert r == 1
    fc = check_feasibility(unreachable_target, [ceiling])
    assert not fc.feasible


def test_end_to_end_worked_example_produces_a_soft_rank():
    """Full README Sec 1.1 story: 5x needed, INT8 covers 4x, CUR only needs
    to cover the 1.25x residual -- this should yield a *large* (soft) rank,
    not an aggressively small one, and its own compression ratio should be
    close to 1.25x, not wastefully higher."""
    m, n = 512, 512
    required = required_reduction_factor(10.0, 2.0)  # 5x
    residual = residual_after_stage(required, quant_max_factor(32, 8))  # 1.25x
    r = closed_form_rank_for_target(m, n, residual)

    achieved = original_param_count(m, n) / cur_param_count(m, n, r, r)
    assert achieved >= residual - 1e-6
    # "soft": should not overshoot the 1.25x target by a large margin
    assert achieved < 1.5 * residual
    # "soft" in absolute terms too: rank should be a large fraction of min(m,n)
    assert r > 0.3 * min(m, n)
