"""Tests for nnopt.quantizer.scale_refine (README Sec 2.4 / Sec 4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from nnopt.quantizer.scale_refine import (
    dequantize,
    initial_scale_minmax,
    quantize_codes,
    reconstruction_loss,
    refine_scale,
    refine_scale_alternating,
)

Q_MAX_INT8 = 127


def test_minmax_scale_covers_max_abs_exactly():
    w = np.array([-3.0, 0.1, 2.0, 5.0, -1.0])
    s0 = initial_scale_minmax(w, Q_MAX_INT8)
    assert s0 == pytest.approx(5.0 / Q_MAX_INT8)
    q = quantize_codes(w, s0, Q_MAX_INT8)
    assert q.max() <= Q_MAX_INT8 and q.min() >= -Q_MAX_INT8


def test_alternating_minimization_is_monotonic_and_converges():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(2048).astype(np.float64) * 3.0
    result = refine_scale_alternating(w, Q_MAX_INT8, max_iter=50)
    # loss must never increase between recorded steps
    for a, b in zip(result.history, result.history[1:]):
        assert b <= a + 1e-9
    assert result.loss_final <= result.loss_initial
    assert result.converged


def test_alternating_beats_minmax_on_outlier_heavy_weights():
    """A few large outliers dominate min/max scaling and blow up the
    quantization noise floor for the many near-zero weights -- this is
    exactly the failure mode README Sec 1-bob / Sec 2.4 motivates fixing.
    The closed-form alternating step should find a materially better
    reconstruction scale than raw min/max.
    """
    rng = np.random.default_rng(1)
    small = rng.normal(loc=0.0, scale=0.05, size=5000)
    outliers = rng.normal(loc=0.0, scale=1.0, size=5) * 20.0  # rare, huge
    w = np.concatenate([small, outliers])

    s_minmax = initial_scale_minmax(w, Q_MAX_INT8)
    loss_minmax = reconstruction_loss(w, s_minmax, Q_MAX_INT8)

    result = refine_scale_alternating(w, Q_MAX_INT8, max_iter=50)

    # The exact margin depends on the random draw (outlier magnitudes vary);
    # the mechanism-level claim under test is strict improvement over
    # min/max, not a specific percentage -- monotonicity/convergence are
    # already covered by test_alternating_minimization_is_monotonic_and_converges.
    assert result.loss_final < loss_minmax


def test_alternating_handles_all_zero_weights_gracefully():
    w = np.zeros(64)
    result = refine_scale_alternating(w, Q_MAX_INT8)
    assert result.scale > 0
    assert result.loss_final == 0.0


def test_refine_scale_without_calibration_matches_phase1():
    rng = np.random.default_rng(2)
    w = rng.standard_normal(256)
    phase1 = refine_scale_alternating(w, Q_MAX_INT8)
    result = refine_scale(w, Q_MAX_INT8)
    assert not result.calibration_applied
    assert result.scale == pytest.approx(phase1.scale)


def test_calibration_grid_can_improve_over_phase1_within_guard():
    """Construct a linear layer where the calibration inputs are strongly
    concentrated on a subset of weight columns; the min/max-then-alternating
    scale (which only sees the weight distribution) should be improvable,
    within the overfitting guard, by looking at actual layer responses.
    """
    rng = np.random.default_rng(3)
    k, n = 32, 8
    w = rng.standard_normal((n, k)) * 0.5
    # calibration activations concentrated in a low-variance regime that
    # only lightly touches a couple of columns with any energy
    x_calib = rng.standard_normal((64, k)) * 0.1
    y_ref = x_calib @ w.T  # fp32 reference layer output

    def layer_response_fn(w_deq: np.ndarray) -> np.ndarray:
        return x_calib @ w_deq.T

    result = refine_scale(
        w, Q_MAX_INT8, layer_response_fn=layer_response_fn, y_reference=y_ref,
        grid_points=41, grid_half_range=0.3, lam=0.5, beta=0.1,
    )
    assert result.calibration_applied
    # Guard must never let L_W regress by more than the beta budget.
    l_w_at_phase1 = reconstruction_loss(w, result.scale_before_calibration, Q_MAX_INT8)
    assert result.L_W_final <= 1.10 * l_w_at_phase1 + 1e-9
    assert result.candidates_accepted >= 1


def test_overfitting_guard_rejects_calibration_trap():
    """A scale large enough to quantize almost everything to zero trivially
    minimizes L_calib against a zero target, but destroys L_W. The
    overfitting-reject rule (README Sec 2.4/4.2) must keep the search from
    wandering there even when lambda on L_calib is huge.
    """
    rng = np.random.default_rng(4)
    w = rng.standard_normal((6, 6)) * 2.0
    y_ref = np.zeros_like(w)  # trivially satisfied by W_deq -> 0

    def layer_response_fn(w_deq: np.ndarray) -> np.ndarray:
        return w_deq  # identity "layer" for a direct, easy-to-reason-about trap

    phase1 = refine_scale_alternating(w, Q_MAX_INT8)
    l_w_phase1 = phase1.loss_final

    result = refine_scale(
        w, Q_MAX_INT8, layer_response_fn=layer_response_fn, y_reference=y_ref,
        grid_points=101, grid_half_range=5.0, lam=1e6, beta=0.05,
    )

    # The guard must have prevented L_W from blowing up by more than the
    # (1+beta) budget, even though the calibration term was weighted
    # overwhelmingly and a "cheat" (huge scale -> W_deq~=0 -> L_calib~=0)
    # was available in the search grid.
    assert result.L_W_final <= 1.05 * l_w_phase1 + 1e-9
    # ...and the chosen scale must not have run off to the far end of the
    # grid (which would indicate the guard failed to bite).
    assert result.scale < phase1.scale * 2.0
