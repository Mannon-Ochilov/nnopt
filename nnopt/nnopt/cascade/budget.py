"""Cross-stage compensation budget math -- README.md Sec 1.1, the
methodology's stated centerpiece ("markaziy g'oya"): every compression
stage has a maximum achievable reduction factor; when one stage cannot
close the full gap on its own, the *residual* is handed to the next
stage, and -- symmetrically -- a stage's own target is *discounted* by
whatever a later, already-known stage can still contribute, so no stage
over-compresses when a milder choice plus the next stage's guaranteed
contribution would already suffice.

Worked example from README Sec 1.1:
    operator needs 5x reduction to fit cache.
    INT8 quantization's max is 4x -> residual = 5/4 = 1.25x.
    CUR structural rank is then chosen to close *only* that 1.25x
    (the largest/softest rank that reaches >=1.25x), not to squeeze out
    the maximum possible compression on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Maximum achievable byte-reduction factor for a bit-width transition
# (bits_from / bits_to); this is a hard ceiling -- actual achieved
# reduction from quantizer.scale_refine can only be <= this.
QUANT_MAX_FACTOR = {
    (32, 16): 2.0,
    (32, 8): 4.0,
    (32, 4): 8.0,
    (16, 8): 2.0,
    (16, 4): 4.0,
    (8, 4): 2.0,
}


def quant_max_factor(bits_from: int, bits_to: int) -> float:
    if bits_to >= bits_from:
        return 1.0
    key = (bits_from, bits_to)
    if key in QUANT_MAX_FACTOR:
        return QUANT_MAX_FACTOR[key]
    return bits_from / bits_to  # generic fallback, still a valid hard ceiling


def required_reduction_factor(m_eff_current: float, m_cache_eff: float) -> float:
    """README (2.6)-adjacent: how many-fold must the current footprint
    shrink to fit the effective cache budget. <=1.0 means it already fits."""
    if m_cache_eff <= 0:
        return math.inf
    return max(1.0, m_eff_current / m_cache_eff)


def discount_target_for_future_stage(required_factor: float, future_max_factor: float) -> float:
    """README Sec 1.1 'orqaga rejalashtirish' (backward scheduling): if a
    later stage is guaranteed to contribute up to `future_max_factor` more
    reduction, THIS stage only needs to achieve the residual on its own.
    Never returns less than 1.0 (a stage is never asked to *expand* the
    operator)."""
    if future_max_factor <= 0:
        raise ValueError("future_max_factor must be positive")
    return max(1.0, required_factor / future_max_factor)


@dataclass
class FeasibilityCheck:
    required_factor: float
    max_achievable_factor: float
    feasible: bool
    shortfall_factor: float  # how much further reduction is unreachable if infeasible (1.0 if feasible)


def check_feasibility(required_factor: float, stage_max_factors: list[float]) -> FeasibilityCheck:
    """README Sec 1.1 point 3: before running any stage, check whether the
    *product* of every stage's maximum achievable factor can even reach the
    required reduction. If not, the operator cannot be fit into this cache
    on this hardware no matter how the cascade is tuned -- report this
    upfront instead of grinding through stages that cannot possibly help.
    """
    max_total = 1.0
    for f in stage_max_factors:
        max_total *= max(1.0, f)
    feasible = max_total >= required_factor - 1e-9
    shortfall = 1.0 if feasible else required_factor / max_total
    return FeasibilityCheck(
        required_factor=required_factor,
        max_achievable_factor=max_total,
        feasible=feasible,
        shortfall_factor=shortfall,
    )


def residual_after_stage(required_factor: float, achieved_factor: float) -> float:
    """README Sec 1.1 point 1 (forward compensation): after a stage
    actually achieves `achieved_factor` (<= its max, possibly less if a
    milder/quality-preserving choice was picked), how much reduction is
    still owed to subsequent stages."""
    if achieved_factor <= 0:
        raise ValueError("achieved_factor must be positive")
    return max(1.0, required_factor / achieved_factor)


def max_cur_compression_factor(m: int, n: int) -> float:
    """Ceiling of the (c=r) CUR parametrization: the most aggressive valid
    choice is r=1 (c=1), giving param count m+n+1. No target above this is
    reachable by any integer rank -- callers should check this (or rely on
    check_feasibility) before calling closed_form_rank_for_target with a
    target that cannot be met."""
    return (m * n) / (m + n + 1)


def closed_form_rank_for_target(m: int, n: int, target_factor: float, c_equals_r: bool = True) -> int:
    """Largest rank r (with c == r, the simplifying tie used by this
    toolkit -- README leaves c and r as independently tunable, but ties
    them 1:1 by default for a tractable closed-form budget search) such
    that the CUR parameter count m*c + c*r + r*n does not exceed
    (m*n) / target_factor. Solved exactly via the quadratic formula
    instead of iterative search:

        r^2 + r*(m+n) - (m*n / target_factor) <= 0
        r <= [ -(m+n) + sqrt((m+n)^2 + 4*m*n/target_factor) ] / 2

    Returns an integer r clamped to [1, min(m, n)]. If target_factor <= 1,
    no compression is required and min(m, n) is returned (unconstrained).

    If target_factor exceeds max_cur_compression_factor(m, n) -- i.e. no
    integer rank can reach it even at the most aggressive r=1 -- this
    returns 1 as a best-effort floor; the achieved ratio will fall short
    of target_factor and the caller MUST separately verify feasibility
    (e.g. via check_feasibility with max_cur_compression_factor(m, n) as
    that stage's max factor) before relying on this rank alone to close
    the residual.
    """
    if not c_equals_r:
        raise NotImplementedError("independent c,r budget solve not implemented; tie c=r or size c separately")
    if target_factor <= 1.0:
        return min(m, n)
    budget_params = (m * n) / target_factor
    a, b, c = 1.0, float(m + n), -budget_params
    disc = b * b - 4 * a * c
    if disc < 0:
        return 1  # no positive real root: target is unreachable by any rank; caller checks feasibility separately
    r = (-b + math.sqrt(disc)) / (2 * a)
    r_int = int(math.floor(r))
    return max(1, min(r_int, min(m, n)))
