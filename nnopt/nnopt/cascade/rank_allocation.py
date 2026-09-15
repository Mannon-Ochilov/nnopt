"""Global rank allocation across operators under one shared budget.

Motivation from measurement (README Sec 8.3.10): giving every FFN operator
the same rank produced held-out errors ranging from 0.02 to 0.20 -- a 10x
spread. Uniform rank is therefore provably wasteful in both directions: the
sensitive operators are damaged while the tolerant ones give back capacity
they never needed.

The cascade so far decided each operator in isolation (greedy, README Sec
2.5). This module states the cross-operator problem properly and solves it:

    minimize    sum_i  E_i(r_i)              (total output error)
    subject to  sum_i  c_i * r_i  <=  B      (shared parameter budget)
                r_i in {1, ..., r_max_i}

where E_i(r) is operator i's measured error curve and c_i = m_i + n_i is
its per-rank parameter cost.

Solution. Each E_i is non-increasing and, for spectral truncation,
convex in r over the region that matters (successive singular values are
non-increasing, so each extra rank unit buys less than the previous one).
Under convexity the continuous relaxation is solved exactly by equalizing
the marginal error reduction per parameter across operators -- the standard
water-filling / Lagrangian condition

    -dE_i/dr_i / c_i  =  lambda   for all i

and the integer solution is obtained by the greedy that repeatedly spends
the next unit of budget wherever it buys the largest error reduction per
parameter. That greedy is optimal for separable convex objectives, which is
why `allocate_greedy` is exact here rather than a heuristic.

`E_i` is supplied by the caller as a measured curve (rank -> error), so the
allocator stays independent of which factorization produced it (SVD, CUR,
activation-aware SVD).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass


@dataclass
class OperatorCurve:
    """One operator's measured error-vs-rank curve."""

    name: str
    param_cost_per_rank: int          # c_i = m_i + n_i
    errors: dict[int, float]          # rank -> measured error (held-out)

    def ranks(self) -> list[int]:
        return sorted(self.errors)

    def error_at(self, rank: int) -> float:
        """Error at `rank`, interpolating linearly between measured points
        and clamping outside the measured range."""
        ks = self.ranks()
        if rank <= ks[0]:
            return self.errors[ks[0]]
        if rank >= ks[-1]:
            return self.errors[ks[-1]]
        for a, b in zip(ks, ks[1:]):
            if a <= rank <= b:
                if b == a:
                    return self.errors[a]
                t = (rank - a) / (b - a)
                return self.errors[a] * (1 - t) + self.errors[b] * t
        return self.errors[ks[-1]]


@dataclass
class Allocation:
    ranks: dict[str, int]
    total_params: int
    total_error: float
    budget: int


def uniform_allocation(curves: list[OperatorCurve], budget: int) -> Allocation:
    """Baseline: the same rank everywhere, scaled to fit the budget."""
    total_cost = sum(c.param_cost_per_rank for c in curves)
    r = max(1, budget // max(total_cost, 1))
    ranks = {c.name: min(r, max(c.ranks())) for c in curves}
    return _finish(curves, ranks, budget)


def allocate_greedy(curves: list[OperatorCurve], budget: int, step: int = 8) -> Allocation:
    """Spend the budget where each parameter buys the most error reduction.

    Exact for separable convex objectives (see module docstring). `step` is
    the rank granularity; smaller is finer but slower.
    """
    # Start at each curve's SMALLEST MEASURED rank, not at 1. error_at()
    # clamps below the measured range, so a run starting at rank 1 sees
    # zero gain for every step that stays under the first probe point,
    # never pushes a candidate, and allocates nothing -- the budget is left
    # untouched. Falling back to 1 only when even the floor does not fit.
    floor = {c.name: min(c.ranks()) for c in curves}
    if sum(floor[c.name] * c.param_cost_per_rank for c in curves) > budget:
        floor = {c.name: 1 for c in curves}
    ranks = dict(floor)
    spent = sum(ranks[c.name] * c.param_cost_per_rank for c in curves)
    by_name = {c.name: c for c in curves}

    # max-heap keyed by error reduction per parameter for the next step
    heap: list[tuple[float, str]] = []

    def push(c: OperatorCurve):
        cur = ranks[c.name]
        nxt = min(cur + step, max(c.ranks()))
        if nxt <= cur:
            return
        gain = c.error_at(cur) - c.error_at(nxt)
        cost = (nxt - cur) * c.param_cost_per_rank
        if cost <= 0 or gain <= 0:
            return
        heapq.heappush(heap, (-gain / cost, c.name, nxt, cost))

    for c in curves:
        push(c)

    while heap:
        neg_ratio, name, nxt, cost = heapq.heappop(heap)
        if spent + cost > budget:
            continue
        c = by_name[name]
        ranks[name] = nxt
        spent += cost
        push(c)

    return _finish(curves, ranks, budget)


def _finish(curves: list[OperatorCurve], ranks: dict[str, int], budget: int) -> Allocation:
    total_params = sum(ranks[c.name] * c.param_cost_per_rank for c in curves)
    total_error = sum(c.error_at(ranks[c.name]) for c in curves)
    return Allocation(ranks=ranks, total_params=total_params,
                      total_error=total_error, budget=budget)
