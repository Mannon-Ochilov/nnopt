import numpy as np
import pytest

from nnopt.quantizer.per_channel import bias_correction as _bias_correction


def test_bias_correction_removes_the_constant_output_offset():
    """The corrected output must have the same mean as the original."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 24)) + 3.0        # non-zero mean activations
    w = rng.normal(size=(8, 24))
    w_hat = w + 0.05 * rng.normal(size=w.shape)  # stand-in for quantization

    off = _bias_correction(w, w_hat, x)
    mean_before = (x @ w_hat.T).mean(axis=0)
    mean_after = (x @ w_hat.T + off).mean(axis=0)
    mean_ref = (x @ w.T).mean(axis=0)
    assert np.abs(mean_after - mean_ref).max() < 1e-10
    assert np.abs(mean_before - mean_ref).max() > 1e-6   # there was an offset


def test_bias_correction_is_zero_for_an_exact_weight():
    rng = np.random.default_rng(1)
    x, w = rng.normal(size=(50, 10)), rng.normal(size=(4, 10))
    assert np.allclose(_bias_correction(w, w, x), 0.0)


def test_bias_correction_vanishes_for_zero_mean_activations():
    """Nothing to correct when the activations average to zero: the offset is
    a projection of the error onto the mean."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4000, 12))
    x -= x.mean(axis=0)
    w = rng.normal(size=(5, 12))
    off = _bias_correction(w, w + 0.1 * rng.normal(size=w.shape), x)
    assert np.abs(off).max() < 1e-10


def test_bias_correction_rejects_mismatched_shapes():
    rng = np.random.default_rng(3)
    w, x = rng.normal(size=(4, 10)), rng.normal(size=(20, 7))
    with pytest.raises(ValueError, match="mos emas"):
        _bias_correction(w, w, x)

from nnopt.quantizer.per_channel import (
    dequantize_pc,
    gram_factor,
    initial_scales_minmax,
    quantize_codes_pc,
    quantize_weight_per_channel,
    reconstruction_loss_pc,
    refine_scales_alternating_pc,
    refine_scales_per_channel,
    rescale_output_domain,
)
from nnopt.quantizer.scale_refine import refine_scale

Q8 = 127


def test_initial_scales_are_per_row():
    w = np.array([[1.0, -2.0], [100.0, 50.0]])
    s = initial_scales_minmax(w, Q8)
    assert s.shape == (2, 1)
    assert s[0, 0] == pytest.approx(2.0 / Q8)
    assert s[1, 0] == pytest.approx(100.0 / Q8)


def test_all_zero_channel_gets_positive_scale():
    w = np.array([[0.0, 0.0], [3.0, -3.0]])
    s = initial_scales_minmax(w, Q8)
    assert s[0, 0] > 0  # must stay usable as a divisor
    assert np.all(np.isfinite(quantize_codes_pc(w, s, Q8)))


def test_alternating_never_increases_per_channel_loss():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 64)) * rng.uniform(0.1, 10.0, size=(16, 1))
    s0 = initial_scales_minmax(w, Q8)
    s1 = refine_scales_alternating_pc(w, Q8)
    l0 = reconstruction_loss_pc(w, s0, Q8)
    l1 = reconstruction_loss_pc(w, s1, Q8)
    assert np.all(l1 <= l0 + 1e-9)


def test_codes_stay_in_range():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((8, 32)) * 100
    s = refine_scales_alternating_pc(w, Q8)
    q = quantize_codes_pc(w, s, Q8)
    assert q.min() >= -Q8 and q.max() <= Q8
    assert np.allclose(q, np.round(q))


def test_per_channel_beats_per_tensor_when_channel_scales_differ():
    """The whole point of README Sec 8.3.7's granularity fix: when output
    channels have wildly different magnitudes, one tensor-wide scale wastes
    resolution on the small channels."""
    rng = np.random.default_rng(2)
    base = rng.standard_normal((32, 128))
    # channel magnitudes spanning three orders of magnitude
    w = base * np.logspace(-2, 1, 32).reshape(-1, 1)

    w_pc, _ = quantize_weight_per_channel(w, Q8)
    err_pc = np.linalg.norm(w - w_pc)

    res_pt = refine_scale(w, Q8)
    from nnopt.quantizer.scale_refine import dequantize, quantize_codes

    w_pt = dequantize(quantize_codes(w, res_pt.scale, Q8), res_pt.scale)
    err_pt = np.linalg.norm(w - w_pt)

    assert err_pc < err_pt * 0.5, f"per-channel {err_pc:.5f} vs per-tensor {err_pt:.5f}"


def test_uniform_channels_per_channel_is_not_worse():
    """Sanity: when all channels share a scale, per-channel must not lose."""
    rng = np.random.default_rng(3)
    w = rng.standard_normal((16, 64))
    w_pc, _ = quantize_weight_per_channel(w, Q8)
    res_pt = refine_scale(w, Q8)
    from nnopt.quantizer.scale_refine import dequantize, quantize_codes

    w_pt = dequantize(quantize_codes(w, res_pt.scale, Q8), res_pt.scale)
    assert np.linalg.norm(w - w_pc) <= np.linalg.norm(w - w_pt) + 1e-9


def test_calibration_phase_respects_overfitting_guard():
    """No channel may end up with an L_W more than (1+beta) worse than its
    Phase-1 optimum -- README Sec 2.4 / 4.2 guard, enforced per channel."""
    rng = np.random.default_rng(4)
    w = rng.standard_normal((12, 48))
    x = rng.standard_normal((64, 48))
    beta = 0.05
    res = refine_scales_per_channel(w, Q8, x_calib=x, beta=beta)
    l_phase1 = reconstruction_loss_pc(w, res.scales_before_calibration, Q8)
    l_final = reconstruction_loss_pc(w, res.scales, Q8)
    assert np.all(l_final <= (1.0 + beta) * l_phase1 + 1e-9)
    assert res.calibration_applied


def test_gram_shortcut_matches_direct_calibration_error():
    """The quadratic-form shortcut d^T (X^T X) d must equal ||X d||^2 --
    this is what makes the per-channel grid search affordable."""
    rng = np.random.default_rng(5)
    w = rng.standard_normal((10, 32))
    x = rng.standard_normal((50, 32))
    res = refine_scales_per_channel(w, Q8, x_calib=x)
    w_deq = dequantize_pc(quantize_codes_pc(w, res.scales, Q8), res.scales)
    d = w_deq - w
    direct = np.sum((x @ d.T) ** 2, axis=0)          # ||X d_i||^2 per channel
    g = x.T @ x
    viagram = np.sum((d @ g) * d, axis=1)
    assert np.allclose(direct, viagram, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    "n_rows, n_cols",
    [
        (50, 32),   # B > n : Gram is formed and factored
        (24, 64),   # B <= n: X is used as the factor directly
    ],
)
def test_calibration_error_is_exact_in_both_shape_regimes(n_rows, n_cols):
    """The factor F with F^T F = X^T X is chosen by shape purely for speed,
    so the reported calibration error must equal ||X D||_F^2 either way.

    The wide case (B <= n) is the one that matters in practice: Llama's
    down_proj has n = 8640 inputs against 4096 calibration rows, where
    forming the 8640x8640 Gram costs more than never forming it at all.
    """
    rng = np.random.default_rng(7)
    w = rng.standard_normal((10, n_cols))
    x = rng.standard_normal((n_rows, n_cols))
    res = refine_scales_per_channel(w, Q8, x_calib=x)
    d = dequantize_pc(quantize_codes_pc(w, res.scales, Q8), res.scales) - w
    direct = float(np.sum((x @ d.T) ** 2))
    assert res.l_calib_total == pytest.approx(direct, rel=1e-8, abs=1e-8)


def test_wide_and_tall_calibration_agree_on_the_same_gram():
    """The same Gram reached through both branches must select the same
    scales -- otherwise the speed optimization would silently be a method
    change."""
    rng = np.random.default_rng(8)
    n = 40
    w = rng.standard_normal((12, n))
    x_tall = rng.standard_normal((80, n))          # B > n  -> factored branch
    # An equivalent narrow representation of the SAME Gram: the symmetric
    # square root has n rows, so B == n <= n takes the direct branch.
    g = x_tall.T @ x_tall
    evals, evecs = np.linalg.eigh(g)
    x_wide = (evecs * np.sqrt(np.maximum(evals, 0.0))) @ evecs.T

    a = refine_scales_per_channel(w, Q8, x_calib=x_tall)
    b = refine_scales_per_channel(w, Q8, x_calib=x_wide)
    assert np.allclose(a.scales, b.scales, rtol=1e-9, atol=1e-12)
    assert a.l_calib_total == pytest.approx(b.l_calib_total, rel=1e-6)


def _gain(w, w_hat, x):
    """Per-channel least-squares slope of approximated output on true output."""
    y, y_hat = x @ w.T, x @ w_hat.T
    return np.sum(y * y_hat, axis=0) / np.sum(y * y, axis=0)


@pytest.mark.parametrize("n_rows, n_cols", [(80, 40), (30, 60)])
def test_gram_factor_reproduces_the_gram_on_both_sides(n_rows, n_cols):
    rng = np.random.default_rng(9)
    x = rng.standard_normal((n_rows, n_cols))
    f = gram_factor(x)
    assert np.allclose(f.T @ f, x.T @ x, rtol=1e-8, atol=1e-8)


def test_unbiased_rescale_gives_gain_exactly_one():
    """The point of the "unbiased" mode: <y, yhat> = <y, y> per channel, so
    nothing is attenuated and the error cannot compound with depth."""
    rng = np.random.default_rng(10)
    w = rng.standard_normal((12, 40))
    x = rng.standard_normal((90, 40))
    res = refine_scales_per_channel(w, 7, x_calib=x)
    codes = quantize_codes_pc(w, res.scales, 7)
    s = rescale_output_domain(w, codes, gram_factor(x), mode="unbiased")
    assert np.allclose(_gain(w, codes * s, x), 1.0, rtol=1e-8, atol=1e-8)


def test_ls_rescale_minimizes_output_error_among_all_scales():
    """The "ls" mode must beat every other multiplier of the same codes."""
    rng = np.random.default_rng(11)
    w = rng.standard_normal((10, 32))
    x = rng.standard_normal((70, 32))
    res = refine_scales_per_channel(w, 7, x_calib=x)
    codes = quantize_codes_pc(w, res.scales, 7)
    f = gram_factor(x)
    s_ls = rescale_output_domain(w, codes, f, mode="ls")

    def out_err(scales):
        return np.linalg.norm(x @ w.T - x @ (codes * scales).T)

    best = out_err(s_ls)
    for mult in (0.85, 0.95, 1.0, 1.05, 1.15):
        assert best <= out_err(s_ls * mult) + 1e-9
    assert best <= out_err(res.scales) + 1e-9


def test_ls_rescale_attenuates_less_than_the_weight_domain_scale():
    """Both fixes must move the gain toward 1 relative to the weight-domain
    scale -- that is the whole reason they exist (README Sec 8.3.17)."""
    rng = np.random.default_rng(12)
    # heavy-tailed weights: the regime where a min/max scale wastes levels
    # and a fitted scale starts clipping
    w = rng.standard_normal((16, 64)) ** 3
    x = rng.standard_normal((100, 64))
    base, _ = quantize_weight_per_channel(w, 7, x_calib=x)
    ls, _ = quantize_weight_per_channel(w, 7, x_calib=x, output_scale="ls")
    ub, _ = quantize_weight_per_channel(w, 7, x_calib=x, output_scale="unbiased")

    g_base = float(np.mean(_gain(w, base, x)))
    assert g_base < 1.0, "weight-domain scale is expected to attenuate"
    assert abs(float(np.mean(_gain(w, ls, x))) - 1.0) < abs(g_base - 1.0)
    assert abs(float(np.mean(_gain(w, ub, x))) - 1.0) < abs(g_base - 1.0)


def test_output_scale_keeps_the_integer_codes_untouched():
    """The rescale must be a storage-free change: same codes, same bit width."""
    rng = np.random.default_rng(13)
    w = rng.standard_normal((10, 32))
    x = rng.standard_normal((70, 32))
    res = refine_scales_per_channel(w, 7, x_calib=x)
    codes = quantize_codes_pc(w, res.scales, 7)
    for mode in ("ls", "unbiased"):
        deq, s = quantize_weight_per_channel(w, 7, x_calib=x, output_scale=mode)
        assert np.allclose(deq / s, codes)
        assert np.abs(codes).max() <= 7


def test_output_scale_requires_calibration():
    rng = np.random.default_rng(14)
    with pytest.raises(ValueError):
        quantize_weight_per_channel(rng.standard_normal((4, 8)), 7,
                                    x_calib=None, output_scale="ls")


def test_calibration_without_x_skips_phase_two():
    rng = np.random.default_rng(6)
    w = rng.standard_normal((8, 16))
    res = refine_scales_per_channel(w, Q8, x_calib=None)
    assert not res.calibration_applied
    assert res.l_calib_total is None
    assert np.allclose(res.scales, res.scales_before_calibration)
