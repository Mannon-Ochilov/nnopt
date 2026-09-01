import numpy as np
import pytest

from nnopt.cascade.rank_allocation import OperatorCurve, allocate_greedy, uniform_allocation


def curve(name, cost, ranks, errs):
    return OperatorCurve(name=name, param_cost_per_rank=cost,
                         errors={r: e for r, e in zip(ranks, errs)})


def test_error_at_clamps_outside_measured_range():
    c = curve("a", 10, [8, 64], [0.5, 0.1])
    assert c.error_at(1) == 0.5
    assert c.error_at(1000) == 0.1


def test_error_at_interpolates_between_points():
    c = curve("a", 10, [0, 100], [1.0, 0.0])
    assert c.error_at(50) == pytest.approx(0.5)


def test_allocation_respects_budget():
    cs = [curve(f"op{i}", 100, [1, 50, 100], [1.0, 0.3, 0.1]) for i in range(4)]
    budget = 20_000
    for alloc in (allocate_greedy(cs, budget), uniform_allocation(cs, budget)):
        assert alloc.total_params <= budget


def test_greedy_beats_uniform_when_sensitivities_differ():
    """The whole point (README Sec 8.3.10): measured per-operator errors
    spanned 0.02-0.20 under a uniform rank, so budget moved from tolerant
    to sensitive operators must reduce total error."""
    # one very sensitive operator, three tolerant ones, equal cost
    sensitive = curve("sens", 100, [1, 25, 50, 100, 200], [2.0, 1.2, 0.7, 0.25, 0.05])
    tolerant = [curve(f"tol{i}", 100, [1, 25, 50, 100, 200], [0.30, 0.12, 0.09, 0.08, 0.075])
                for i in range(3)]
    cs = [sensitive] + tolerant
    budget = 4 * 100 * 60  # average rank 60 if split evenly

    g = allocate_greedy(cs, budget, step=5)
    u = uniform_allocation(cs, budget)
    assert g.total_error < u.total_error
    # and it must do so by giving the sensitive operator more rank
    assert g.ranks["sens"] > u.ranks["sens"]


def test_greedy_gives_more_rank_to_steeper_curve():
    steep = curve("steep", 10, [1, 100], [1.0, 0.0])
    flat = curve("flat", 10, [1, 100], [0.5, 0.45])
    alloc = allocate_greedy([steep, flat], budget=10 * 120, step=1)
    assert alloc.ranks["steep"] > alloc.ranks["flat"]


def test_expensive_operator_gets_less_rank_at_equal_gain():
    """Cost per rank must matter, not just error gain."""
    cheap = curve("cheap", 10, [1, 100], [1.0, 0.0])
    pricey = curve("pricey", 1000, [1, 100], [1.0, 0.0])
    alloc = allocate_greedy([cheap, pricey], budget=20_000, step=1)
    assert alloc.ranks["cheap"] > alloc.ranks["pricey"]


def test_all_ranks_at_least_one():
    cs = [curve(f"op{i}", 10_000, [1, 100], [1.0, 0.1]) for i in range(5)]
    alloc = allocate_greedy(cs, budget=1)  # budget below even the floor
    assert all(r >= 1 for r in alloc.ranks.values())


def test_greedy_actually_spends_the_budget_when_curve_starts_above_one():
    """Regression: curves measured from rank 64 upward left the allocator
    stuck at rank 1 with the budget untouched, because every step below the
    first probe point clamps to the same error and so shows zero gain."""
    cs = [curve(f"op{i}", 5120, [64, 128, 200, 300, 409, 550],
                [0.20, 0.15, 0.12, 0.09, 0.07, 0.05]) for i in range(48)]
    budget = sum(c.param_cost_per_rank * 409 for c in cs)
    alloc = allocate_greedy(cs, budget, step=8)
    assert alloc.total_params > 0.8 * budget, (
        f"only spent {alloc.total_params:,} of {budget:,}")
    assert min(alloc.ranks.values()) >= 64


def test_greedy_falls_back_when_even_the_floor_exceeds_budget():
    cs = [curve(f"op{i}", 10_000, [64, 128], [1.0, 0.5]) for i in range(10)]
    alloc = allocate_greedy(cs, budget=100, step=8)
    assert all(r >= 1 for r in alloc.ranks.values())


def test_greedy_never_exceeds_measured_max_rank():
    cs = [curve("a", 10, [1, 32], [1.0, 0.2])]
    alloc = allocate_greedy(cs, budget=10**9, step=1)
    assert alloc.ranks["a"] == 32


def test_total_error_matches_sum_of_curve_values():
    cs = [curve("a", 10, [1, 50], [1.0, 0.2]), curve("b", 10, [1, 50], [0.8, 0.1])]
    alloc = allocate_greedy(cs, budget=1000, step=1)
    expected = sum(c.error_at(alloc.ranks[c.name]) for c in cs)
    assert alloc.total_error == pytest.approx(expected)
