"""Per-output-channel quantization scales -- the granularity fix motivated
by README Sec 8.3.7: whole-network per-tensor INT8 gave E_glob = 0.2275,
far worse than INT8 post-training quantization should cost, and the
per-operator spread (0.0045 on fc1 vs 0.0576 on encoder_attn.out_proj,
a 13x range) is the signature of per-channel weight-scale variance being
forced through a single tensor-wide scale.

Everything here keeps README Sec 2.4's two-phase structure -- alternating
minimization for L_W, then a calibration-guided local grid search with the
overfitting-reject guard -- but runs it INDEPENDENTLY PER OUTPUT CHANNEL.

Why that is exact rather than a heuristic: for Y = X @ W^T, output column
i depends only on weight row i,

    Y[:, i] = X @ W[i, :]

so the calibration objective separates exactly across output channels.
Choosing each channel's scale independently is therefore optimal for the
separable objective, not a greedy approximation.

Cost control: the calibration error for channel i at weight perturbation
d_i = W_deq[i, :] - W[i, :] is the quadratic form

    ||X d_i||^2 = d_i^T (X^T X) d_i = ||F d_i||^2   for any F with F^T F = G,

so a factor is built ONCE per operator and every candidate is then one
(m, n) @ (n, k) product plus a row-wise dot. Which factor is cheapest
depends on the operator's shape -- see refine_scales_per_channel; both
choices give the same value, they differ only in arithmetic cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

XI = 1e-12


@dataclass
class PerChannelResult:
    scales: np.ndarray            # (m, 1) one scale per output channel
    scales_before_calibration: np.ndarray
    l_w_total: float              # sum over channels of ||W - s*q||^2
    l_calib_total: float | None   # sum over channels of ||X (W_deq - W)||^2
    channels_moved_by_calibration: int
    calibration_applied: bool


def initial_scales_minmax(w: np.ndarray, q_max: int) -> np.ndarray:
    """s0[i] = max_j |W[i, j]| / q_max -- README Sec 2.4 min/max init, per row."""
    max_abs = np.max(np.abs(w), axis=1, keepdims=True)
    scales = max_abs / q_max
    # degenerate all-zero channels: any positive scale is fine, 1.0 keeps
    # the later divisions well-defined.
    scales[scales <= 0] = 1.0
    return scales


def quantize_codes_pc(w: np.ndarray, scales: np.ndarray, q_max: int) -> np.ndarray:
    return np.clip(np.round(w / scales), -q_max, q_max)


def dequantize_pc(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q * scales


def reconstruction_loss_pc(w: np.ndarray, scales: np.ndarray, q_max: int) -> np.ndarray:
    """Per-channel L_W, shape (m, 1)."""
    d = w - dequantize_pc(quantize_codes_pc(w, scales, q_max), scales)
    return np.sum(d * d, axis=1, keepdims=True)


def gram_factor(x: np.ndarray) -> np.ndarray:
    """F with F^T F = X^T X, built on whichever side is cheaper.

    Each use costs m*n*rows(F), so the side matters:
      B <= n : return X itself. Forming G would cost n^2*B up front and
               leave n rows anyway -- strictly worse. This is the wide case
               (Llama down_proj: n = 8640 inputs, B = 4096 rows).
      B >  n : form G once and factor it, so the cost stops growing with
               the calibration set.
    Both give identical values; only the arithmetic differs.

    Shared by the scale search and the output-domain rescale so the two
    always agree on the same Gram.
    """
    b, n = x.shape
    if b <= n:
        return np.ascontiguousarray(x)
    g = x.T @ x
    try:
        return np.linalg.cholesky(g + XI * np.trace(g) / max(n, 1) * np.eye(n)).T
    except np.linalg.LinAlgError:
        # Rank-deficient Gram: the symmetric eigen-factor always exists.
        evals, evecs = np.linalg.eigh(g)
        return (evecs * np.sqrt(np.maximum(evals, 0.0))).T


def refine_scales_alternating_pc(
    w: np.ndarray,
    q_max: int,
    max_iter: int = 20,
    rel_tol: float = 1e-6,
) -> np.ndarray:
    """Phase 1 per channel: the same exact alternating minimization as
    scale_refine.refine_scale_alternating, vectorized over rows.

        q_i     <- round(clip(W_i / s_i))
        s_i     <- <W_i, q_i> / <q_i, q_i>

    Each half-step is an exact minimizer of that channel's L_W with the
    other variable fixed, so per-channel L_W is non-increasing.
    """
    s = initial_scales_minmax(w, q_max)
    for _ in range(max_iter):
        q = quantize_codes_pc(w, s, q_max)
        den = np.sum(q * q, axis=1, keepdims=True)
        num = np.sum(w * q, axis=1, keepdims=True)
        s_new = np.where(den > XI, num / np.maximum(den, XI), s)
        s_new = np.where(s_new > 0, s_new, s)  # guard pathological sign flip
        rel = np.abs(s_new - s) / (np.abs(s) + XI)
        s = s_new
        if float(np.max(rel)) < rel_tol:
            break
    return s


def refine_scales_per_channel(
    w: np.ndarray,
    q_max: int,
    x_calib: np.ndarray | None = None,
    grid_points: int = 21,
    grid_half_range: float = 0.2,
    lam: float = 0.2,
    beta: float = 0.05,
) -> PerChannelResult:
    """Full README Sec 2.4 pipeline at per-channel granularity.

    `x_calib` is (batch, n) real calibration activations; when omitted the
    calibration phase is skipped and the pure-weights optimum is returned.

    The Phase-2 grid is a shared set of RELATIVE multipliers around each
    channel's own Phase-1 scale, so one candidate sweep evaluates all
    channels at once; each channel then independently keeps the candidate
    minimizing L_W + lam * L_calib, subject to the README Sec 4.2
    overfitting guard L_W(cand) <= (1+beta) * L_W(phase1).
    """
    s_w = refine_scales_alternating_pc(w, q_max)
    l_w_base = reconstruction_loss_pc(w, s_w, q_max)

    if x_calib is None:
        return PerChannelResult(
            scales=s_w,
            scales_before_calibration=s_w.copy(),
            l_w_total=float(np.sum(l_w_base)),
            l_calib_total=None,
            channels_moved_by_calibration=0,
            calibration_applied=False,
        )

    # The calibration error of channel i is ||X d_i||^2 = d_i^T G d_i with
    # G = X^T X, which through any factor F with G = F^T F is just ||F d_i||^2.
    f = gram_factor(x_calib)

    def calib_err(scales: np.ndarray) -> np.ndarray:
        d = dequantize_pc(quantize_codes_pc(w, scales, q_max), scales) - w  # (m, n)
        fd = d @ f.T  # (m, k)
        return np.sum(fd * fd, axis=1, keepdims=True)

    best_scales = s_w.copy()
    best_l_w = l_w_base.copy()
    best_calib = calib_err(s_w)
    best_total = best_l_w + lam * best_calib

    for mult in np.linspace(1.0 - grid_half_range, 1.0 + grid_half_range, grid_points):
        if abs(mult - 1.0) < 1e-12:
            continue
        cand = s_w * float(mult)
        l_w_cand = reconstruction_loss_pc(w, cand, q_max)
        allowed = l_w_cand <= (1.0 + beta) * l_w_base  # overfitting guard, per channel
        total = l_w_cand + lam * calib_err(cand)
        take = allowed & (total < best_total)
        best_scales = np.where(take, cand, best_scales)
        best_l_w = np.where(take, l_w_cand, best_l_w)
        best_total = np.where(take, total, best_total)

    moved = int(np.sum(best_scales != s_w))
    return PerChannelResult(
        scales=best_scales,
        scales_before_calibration=s_w,
        l_w_total=float(np.sum(best_l_w)),
        l_calib_total=float(np.sum(calib_err(best_scales))),
        channels_moved_by_calibration=moved,
        calibration_applied=True,
    )


def rescale_output_domain(
    w: np.ndarray, codes: np.ndarray, f: np.ndarray, mode: str = "ls"
) -> np.ndarray:
    """Re-pick the per-channel scale against the OUTPUT, codes held fixed.

    Motivated by README Sec 8.3.17. Minimizing weight error per operator is
    locally right and globally wrong: it shrinks each channel slightly, and
    since every operator shrinks in the SAME direction the attenuation
    compounds with depth (measured: gain 0.9896 per operator, 0.444 over the
    78 feed-forward operators of open_llama_3b) even though the per-operator
    error is 26% BELOW round-to-nearest.

    Both modes keep the integer codes and the storage format untouched --
    only the scalar multiplier per output channel changes.

        "ls"       s = (q^T G w) / (q^T G q)
                   minimizes ||X(w - s q)||^2. Least error, but the residual
                   is then orthogonal to the approximation, which pins the
                   gain at ||yhat||^2/||y||^2 <= 1: attenuation is reduced,
                   not removed.

        "unbiased" s = (w^T G w) / (q^T G w)
                   sets <y, yhat> = <y, y>, i.e. gain exactly 1, paying
                   error for it. This is the mode that kills the compounding
                   term outright.
    """
    fq = codes @ f.T
    fw = w @ f.T
    q_g_w = np.sum(fq * fw, axis=1)
    if mode == "ls":
        num, den = q_g_w, np.sum(fq * fq, axis=1)
    elif mode == "unbiased":
        num, den = np.sum(fw * fw, axis=1), q_g_w
    else:
        raise ValueError(f"unknown mode: {mode}")
    s = np.where(np.abs(den) > XI, num / np.where(np.abs(den) > XI, den, 1.0), 1.0)
    return s.reshape(-1, 1)


def bias_correction(w: np.ndarray, w_hat: np.ndarray,
                    x_calib: np.ndarray) -> np.ndarray:
    """The constant part of the quantization error, ready to add to a bias.

    Quantization error need not average to zero over the calibration
    distribution, and the part that does not is a fixed offset on the output:

        offset = (W - W_hat) @ mean(X)

    Adding it to the operator's existing bias removes that offset at no cost
    -- no integer code moves, the bit width and the memory layout are
    untouched, and the bias vector is already there. It is the same identity
    the structural stage uses when it folds a discarded channel's mean into
    the bias, applied to a different source of error.

    Measured on open_llama_3b feed-forward operators with held-out
    activations, the offset is worth removing for scale-based quantizers and
    not for GPTQ, whose error compensation already drives it to nearly zero:

        method   relative offset   held-out error improvement
        RTN            0.0836            4.8% (INT4), 3.4% (INT8)
        ours           0.0554            6.0% (INT4), 1.3% (INT8)
        GPTQ           0.0036            0.0%

    Returns a vector of length w.shape[0]; an operator with no bias term has
    to gain one before this can be applied.
    """
    if w.shape != w_hat.shape:
        raise ValueError("w va w_hat shakllari mos emas")
    if x_calib.shape[1] != w.shape[1]:
        raise ValueError(f"x_calib ustunlari ({x_calib.shape[1]}) w ning "
                         f"kirish o'lchamiga ({w.shape[1]}) mos emas")
    mu = np.asarray(x_calib, dtype=np.float64).mean(axis=0)
    return (np.asarray(w, dtype=np.float64)
            - np.asarray(w_hat, dtype=np.float64)) @ mu


def quantize_weight_per_channel(
    w: np.ndarray,
    q_max: int,
    x_calib: np.ndarray | None = None,
    output_scale: str | None = None,
    with_bias_correction: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper: returns (dequantized_weight, scales).

    `output_scale` in {None, "ls", "unbiased"} selects the optional
    output-domain rescale described in rescale_output_domain. It requires
    `x_calib` and never changes the integer codes.

    `with_bias_correction` appends the offset from `bias_correction` to the
    returned tuple. It is opt-in rather than automatic because the caller has
    to have somewhere to put it: the value is only useful if it actually
    reaches the operator's bias, and silently returning a corrected weight
    would be wrong -- the correction is additive on the OUTPUT, not on W.
    """
    res = refine_scales_per_channel(w, q_max, x_calib=x_calib, **kwargs)
    q = quantize_codes_pc(w, res.scales, q_max)
    if output_scale is None:
        w_hat, scales = dequantize_pc(q, res.scales), res.scales
    else:
        if x_calib is None:
            raise ValueError("output_scale requires x_calib")
        scales = rescale_output_domain(w, q, gram_factor(x_calib),
                                       mode=output_scale)
        w_hat = dequantize_pc(q, scales)
    if not with_bias_correction:
        return w_hat, scales
    if x_calib is None:
        raise ValueError("with_bias_correction requires x_calib")
    return w_hat, scales, bias_correction(w, w_hat, x_calib)
