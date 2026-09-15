"""Validate the GPTQ / AWQ reimplementations by their own invariants.

These are reference implementations of published algorithms, so they cannot
be checked against official outputs on this machine. What CAN be checked is
that each one satisfies the properties its paper claims: GPTQ must beat
round-to-nearest on calibration output error by exploiting the Hessian, AWQ
must protect high-magnitude input channels, and both must reduce to sensible
behaviour in degenerate cases.
"""

import numpy as np
import pytest

from nnopt.quantizer.baselines import (
    awq_quantize,
    gptq_quantize,
    output_relative_error,
    rtn_quantize,
)


def make_case(seed=0, m=32, n=64, rows=256, anisotropy=1.0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((m, n))
    x = rng.standard_normal((rows, n))
    if anisotropy != 1.0:
        # make a few input channels dominate
        scale = np.ones(n)
        scale[: n // 8] = anisotropy
        x = x * scale[None, :]
    return w, x


def test_rtn_is_close_for_fine_grids():
    w, x = make_case()
    err = output_relative_error(w, rtn_quantize(w, q_max=127), x)
    assert err < 0.05, f"INT8 RTN should be reasonably accurate, got {err}"


def test_gptq_preserves_shape_and_is_finite():
    w, x = make_case()
    q = gptq_quantize(w, x)
    assert q.shape == w.shape
    assert np.all(np.isfinite(q))


def test_gptq_beats_rtn_on_calibration_error():
    """The whole point of GPTQ: Hessian-guided error compensation should
    reduce output error relative to plain rounding."""
    wins = 0
    for seed in range(5):
        w, x = make_case(seed=seed, anisotropy=6.0)
        e_rtn = output_relative_error(w, rtn_quantize(w), x)
        e_gptq = output_relative_error(w, gptq_quantize(w, x), x)
        wins += int(e_gptq <= e_rtn)
    assert wins >= 4, f"GPTQ should beat RTN on most cases, won {wins}/5"


def test_gptq_handles_dead_channels():
    """Columns with zero activation give a singular Hessian block."""
    w, x = make_case()
    x[:, 5] = 0.0
    x[:, 17] = 0.0
    q = gptq_quantize(w, x)
    assert np.all(np.isfinite(q))
    assert np.allclose(q[:, 5], 0.0) and np.allclose(q[:, 17], 0.0)


def test_awq_returns_alpha_in_range():
    w, x = make_case()
    _, alpha = awq_quantize(w, x)
    assert 0.0 <= alpha <= 1.0


def test_awq_beats_rtn_under_anisotropic_activations():
    """AWQ's premise: when some input channels are far more active, scaling
    them before quantization protects the output."""
    w, x = make_case(seed=3, anisotropy=20.0)
    e_rtn = output_relative_error(w, rtn_quantize(w), x)
    w_awq, _ = awq_quantize(w, x)
    e_awq = output_relative_error(w, w_awq, x)
    assert e_awq < e_rtn, f"AWQ {e_awq:.5f} should beat RTN {e_rtn:.5f}"


def test_awq_alpha_zero_reduces_to_rtn():
    """With alpha = 0 the scaling is uniform, so AWQ must equal plain RTN;
    the grid search may still pick something better, but never worse."""
    w, x = make_case(seed=7)
    w_awq, _ = awq_quantize(w, x)
    e_rtn = output_relative_error(w, rtn_quantize(w), x)
    e_awq = output_relative_error(w, w_awq, x)
    assert e_awq <= e_rtn + 1e-9


def test_awq_selects_larger_alpha_when_anisotropy_is_stronger():
    w, x_flat = make_case(seed=11, anisotropy=1.0)
    _, x_spiky = make_case(seed=11, anisotropy=50.0)
    _, a_flat = awq_quantize(w, x_flat)
    _, a_spiky = awq_quantize(w, x_spiky)
    assert a_spiky >= a_flat


@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_methods_do_not_change_weight_magnitude_wildly(method):
    w, x = make_case()
    out = gptq_quantize(w, x) if method == "gptq" else awq_quantize(w, x)[0]
    ratio = np.linalg.norm(out) / np.linalg.norm(w)
    assert 0.8 < ratio < 1.2, f"{method} changed weight norm by {ratio:.3f}x"


def test_gptq_block_size_does_not_change_result_much():
    """Block size is an implementation detail for speed, not a knob that
    should move the answer."""
    w, x = make_case(seed=5)
    e_small = output_relative_error(w, gptq_quantize(w, x, block_size=16), x)
    e_large = output_relative_error(w, gptq_quantize(w, x, block_size=128), x)
    assert abs(e_small - e_large) < 0.02
