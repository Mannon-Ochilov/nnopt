"""Quantization scale refinement -- README.md Sec 2.4 / Sec 4.2.

Implements the *corrected* mechanism agreed with the author (README Sec 4.2):
the dissertation text describes "gradient descent" for the scale, but since
the quantized codes q = round(clip(W/s)) make L(s) piecewise-constant, plain
gradient descent is not well-founded there. This module instead uses:

  Phase 1 (L_W, primary):  alternating minimization.
      q  <- round(clip(W / s, -q_max, q_max))         # codes step
      s  <- <W, q> / <q, q>                            # closed-form scale step
                                                        # (least squares, q fixed)
      repeats until convergence; L_W(s) is monotonically non-increasing
      at each step (each half-step is an exact minimizer given the other
      variable held fixed).

  Phase 2 (L_calib, secondary): local grid search around the Phase-1
      optimum. L_calib(s) is *also* piecewise-constant in s (it depends on
      s only through the quantized codes), so it is scanned, not
      differentiated. A candidate scale is accepted only if it does not
      increase L_W by more than a factor (1+beta) -- the overfitting guard
      from README Sec 2.4 / 4.2: "kalibrlashdagi kichik yaxshilanish
      vaznlarning umumiy tiklash xatoligini sezilarli oshirish hisobiga
      yuzaga kelsa, bunday masshtab qabul qilinmaydi."

Only the scale is ever updated; model weights are left untouched (pure
post-training scope, README Sec 2.4 point: "Gradient tushish modelning
asosiy vazn parametrlariga nisbatan bajarilmaydi").

This module is model-agnostic: the calibration term is wired in via a
caller-supplied `layer_response_fn(W_dequantized) -> Y` callable, so it can
be unit-tested with a synthetic linear layer without any ONNX/calibration
plumbing (that live wiring belongs in nnopt.calibrator, built once a real
model + calibration set are available).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

XI = 1e-12  # guards divide-by-zero, consistent with README's xi


def quantize_codes(w: np.ndarray, scale: float, q_max: int) -> np.ndarray:
    """q = round(clip(W/s, -q_max, q_max)) -- symmetric quantization (z=0),
    matching the scalar formulas actually used in README Sec 2.4 / 4."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return np.clip(np.round(w / scale), -q_max, q_max)


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return scale * q


def reconstruction_loss(w: np.ndarray, scale: float, q_max: int) -> float:
    """L_W(s) = ||W - s*round(clip(W/s))||^2 -- README (2.4) L_W term."""
    q = quantize_codes(w, scale, q_max)
    diff = w - dequantize(q, scale)
    return float(np.sum(diff * diff))


def initial_scale_minmax(w: np.ndarray, q_max: int) -> float:
    """README Sec 2.4: s0 = max|M_q| / q_max (symmetric min/max)."""
    max_abs = float(np.max(np.abs(w)))
    if max_abs == 0.0:
        return 1.0  # degenerate all-zero tensor; any positive scale is fine
    return max_abs / q_max


@dataclass
class AlternatingResult:
    scale: float
    initial_scale: float
    loss_initial: float
    loss_final: float
    iterations: int
    converged: bool
    history: list[float] = field(repr=False, default_factory=list)


def refine_scale_alternating(
    w: np.ndarray,
    q_max: int,
    s0: float | None = None,
    max_iter: int = 20,
    rel_tol: float = 1e-6,
) -> AlternatingResult:
    """Phase 1: alternating minimization of L_W(s) = ||W - s*q(s)||^2.

    Each iteration performs an *exact* minimization of one variable with
    the other fixed:
        q_t   = argmin_q ||W - s_t * q||^2   s.t. q integer in [-q_max,q_max]
              = round(clip(W / s_t))
        s_{t+1} = argmin_s ||W - s * q_t||^2  (q_t fixed, continuous s)
                = <W, q_t> / <q_t, q_t>

    L_W is therefore non-increasing at every half-step; the loop stops when
    the scale itself stops moving (relative change < rel_tol) or max_iter
    is reached. No learning rate, no gradient -- see module docstring.
    """
    s = initial_scale_minmax(w, q_max) if s0 is None else float(s0)
    s_initial = s
    loss_initial = reconstruction_loss(w, s, q_max)
    history = [loss_initial]

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        q = quantize_codes(w, s, q_max)
        denom = float(np.sum(q * q))
        if denom < XI:
            # All codes collapsed to zero (s far too large) -- cannot
            # improve via the closed-form step; keep current scale.
            break
        s_new = float(np.sum(w * q)) / denom
        if s_new <= 0:
            # Should not happen for a non-degenerate W with a sane s0, but
            # guard against a pathological sign flip.
            break
        loss_new = reconstruction_loss(w, s_new, q_max)
        history.append(loss_new)
        rel_change = abs(s_new - s) / (abs(s) + XI)
        s = s_new
        if rel_change < rel_tol:
            converged = True
            it += 0
            break
    return AlternatingResult(
        scale=s,
        initial_scale=s_initial,
        loss_initial=loss_initial,
        loss_final=history[-1],
        iterations=it,
        converged=converged,
        history=history,
    )


@dataclass
class CalibrationRefinementResult:
    scale: float
    scale_before_calibration: float
    initial_scale_minmax: float
    L_W_final: float
    L_calib_final: float | None
    candidates_evaluated: int
    candidates_accepted: int
    calibration_applied: bool  # False if no layer_response_fn / y_reference given


def refine_scale(
    w: np.ndarray,
    q_max: int,
    layer_response_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    y_reference: np.ndarray | None = None,
    s0: float | None = None,
    max_iter: int = 20,
    rel_tol: float = 1e-6,
    grid_points: int = 41,
    grid_half_range: float = 0.2,
    lam: float = 0.2,
    beta: float = 0.05,
) -> CalibrationRefinementResult:
    """Full README Sec 2.4 pipeline: min/max init -> alternating minimization
    (L_W) -> local grid search around the result, guided by L_calib with an
    overfitting-reject guard (README Sec 4.2, point B).

    If `layer_response_fn`/`y_reference` are omitted, Phase 2 is skipped and
    the pure-weights optimum from Phase 1 is returned (this matches README
    Sec 2.5 step 2: quantization is meaningful even without a downstream
    CUR/graph context wired in yet).
    """
    phase1 = refine_scale_alternating(w, q_max, s0=s0, max_iter=max_iter, rel_tol=rel_tol)
    s_w = phase1.scale
    l_w_w = phase1.loss_final

    if layer_response_fn is None or y_reference is None:
        return CalibrationRefinementResult(
            scale=s_w,
            scale_before_calibration=s_w,
            initial_scale_minmax=phase1.initial_scale,
            L_W_final=l_w_w,
            L_calib_final=None,
            candidates_evaluated=0,
            candidates_accepted=0,
            calibration_applied=False,
        )

    lo = s_w * (1.0 - grid_half_range)
    hi = s_w * (1.0 + grid_half_range)
    candidates = np.linspace(max(lo, XI), hi, grid_points)

    best_scale = s_w
    best_total = l_w_w + lam * float(
        np.mean((layer_response_fn(dequantize(quantize_codes(w, s_w, q_max), s_w)) - y_reference) ** 2)
    )
    best_l_calib = (best_total - l_w_w) / lam if lam > 0 else None
    accepted = 1  # s_w itself always satisfies the guard trivially

    for s_cand in candidates:
        s_cand = float(s_cand)
        if s_cand == s_w:
            continue
        l_w_cand = reconstruction_loss(w, s_cand, q_max)
        if l_w_cand > (1.0 + beta) * l_w_w:
            continue  # overfitting guard (README Sec 2.4 / 4.2)
        accepted += 1
        w_deq = dequantize(quantize_codes(w, s_cand, q_max), s_cand)
        y_cand = layer_response_fn(w_deq)
        l_calib_cand = float(np.mean((y_cand - y_reference) ** 2))
        total = l_w_cand + lam * l_calib_cand
        if total < best_total:
            best_total = total
            best_scale = s_cand
            best_l_calib = l_calib_cand

    return CalibrationRefinementResult(
        scale=best_scale,
        scale_before_calibration=s_w,
        initial_scale_minmax=phase1.initial_scale,
        L_W_final=reconstruction_loss(w, best_scale, q_max),
        L_calib_final=best_l_calib,
        candidates_evaluated=len(candidates),
        candidates_accepted=accepted,
        calibration_applied=True,
    )
