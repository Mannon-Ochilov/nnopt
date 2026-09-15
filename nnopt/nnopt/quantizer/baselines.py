"""Faithful CPU reimplementations of GPTQ and AWQ weight quantization.

Why reimplement. auto-gptq and autoawq ship CUDA kernels and will not run on
a CPU-only machine, so the published implementations cannot be executed
here. Both algorithms are, however, pure linear algebra and fully specified
in their papers, so a reference implementation is straightforward and can be
validated by its own invariants (see tests/test_baselines.py). Any results
obtained here must be reported as REIMPLEMENTATIONS, not as runs of the
official code -- numbers may differ from the papers, which use per-group
asymmetric quantization, different calibration corpora and different bit
widths.

To keep the comparison about ALGORITHMS rather than about bit width or
granularity, every method here quantizes to symmetric INT8 with
per-output-channel scales, matching nnopt.quantizer.per_channel.

GPTQ (Frantar et al., ICLR 2023): quantizes columns left to right, and after
each column pushes the induced error into the not-yet-quantized columns using
the inverse Hessian of the layer's calibration Gram matrix,

    H = 2 X^T X + lambda I ,

so later columns compensate for the damage done by earlier ones.

AWQ (Lin et al., MLSys 2024): observes that channels with large activation
magnitude matter disproportionately, and rescales input channels by
s = s_x^alpha before quantization (dividing back afterwards), with alpha
chosen by grid search on the calibration output error.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-10


def _quantize_per_channel(w: np.ndarray, q_max: int = 127) -> np.ndarray:
    """Symmetric per-output-channel (per-row) quantize-dequantize."""
    scale = np.max(np.abs(w), axis=1, keepdims=True) / q_max
    scale = np.where(scale > 0, scale, 1.0)
    q = np.clip(np.round(w / scale), -q_max, q_max)
    return q * scale


def _quantize_col(col: np.ndarray, scale_1d: np.ndarray, q_max: int) -> np.ndarray:
    """Quantize one column (m,) against fixed per-output-channel scales (m,)."""
    q = np.clip(np.round(col / scale_1d), -q_max, q_max)
    return q * scale_1d


def gptq_quantize(
    w: np.ndarray,
    x_calib: np.ndarray,
    q_max: int = 127,
    block_size: int = 128,
    damping: float = 0.01,
) -> np.ndarray:
    """GPTQ weight quantization.

    w: (m, n) with Y = X W^T, so columns index input features.
    x_calib: (B, n) calibration activations.
    Returns the dequantized weight of the same shape.
    """
    w = np.array(w, dtype=np.float64, copy=True)
    m, n = w.shape

    h = 2.0 * (x_calib.T @ x_calib)
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h += np.eye(n) * (damping * np.mean(np.diag(h)))

    # Upper-triangular Cholesky factor of H^-1, as the paper prescribes.
    h_inv = np.linalg.inv(h)
    h_inv = np.linalg.cholesky(h_inv).T

    # Scales are fixed up front from the original weights, so that error
    # compensation cannot drift the grid mid-pass.
    scale = np.max(np.abs(w), axis=1) / q_max
    scale = np.where(scale > 0, scale, 1.0)

    q_out = np.zeros_like(w)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        err_block = np.zeros((m, end - start), dtype=np.float64)

        for j in range(start, end):
            col = w[:, j]
            q_col = _quantize_col(col, scale, q_max)
            q_out[:, j] = q_col
            d = h_inv[j, j]
            err = (col - q_col) / (d if abs(d) > EPS else EPS)
            err_block[:, j - start] = err
            if j + 1 < end:
                w[:, j + 1:end] -= np.outer(err, h_inv[j, j + 1:end])

        if end < n:
            w[:, end:] -= err_block @ h_inv[start:end, end:]

    return q_out


def awq_quantize(
    w: np.ndarray,
    x_calib: np.ndarray,
    q_max: int = 127,
    n_grid: int = 20,
    search_rows: int | None = 512,
) -> tuple[np.ndarray, float]:
    """AWQ activation-aware weight quantization.

    Searches the scaling exponent alpha so that input channels with large
    activation magnitude are protected. Returns (dequantized weight, alpha).

    `search_rows` subsamples the calibration rows used to SCORE grid
    candidates. The channel magnitudes s_x are still computed from all rows;
    only the candidate ranking uses a subset. Scoring every candidate on the
    full set dominates the runtime on large operators (each candidate costs a
    full X @ W^T), and alpha is a single scalar chosen from a coarse grid, so
    a few hundred rows rank the candidates just as well.
    """
    w = np.asarray(w, dtype=np.float64)
    s_x = np.mean(np.abs(x_calib), axis=0)
    s_x = np.where(s_x > 0, s_x, EPS)

    if search_rows is not None and x_calib.shape[0] > search_rows:
        step = x_calib.shape[0] // search_rows
        x_score = np.ascontiguousarray(x_calib[::step][:search_rows])
    else:
        x_score = x_calib

    y_ref = x_score @ w.T
    y_norm = np.linalg.norm(y_ref) + EPS
    x_calib = x_score

    best_w, best_alpha, best_err = None, 0.0, np.inf
    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = s_x ** alpha
        s = s / (np.sqrt(np.max(s) * np.min(s)) + EPS)  # keep the scale centred
        s = np.where(s > EPS, s, EPS)

        w_scaled = w * s[None, :]
        w_q = _quantize_per_channel(w_scaled, q_max)
        w_eff = w_q / s[None, :]

        err = np.linalg.norm(y_ref - x_calib @ w_eff.T) / y_norm
        if err < best_err:
            best_err, best_alpha, best_w = err, alpha, w_eff

    return best_w, best_alpha


def rtn_quantize(w: np.ndarray, q_max: int = 127) -> np.ndarray:
    """Round-to-nearest per-output-channel baseline (no calibration)."""
    return _quantize_per_channel(np.asarray(w, dtype=np.float64), q_max)


def output_relative_error(w: np.ndarray, w_hat: np.ndarray, x: np.ndarray) -> float:
    y = x @ w.T
    return float(np.linalg.norm(y - x @ w_hat.T) / (np.linalg.norm(y) + EPS))
