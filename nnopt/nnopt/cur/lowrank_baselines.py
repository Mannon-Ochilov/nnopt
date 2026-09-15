"""Low-rank baselines that CUR must be measured against -- README Sec 7.4
ablation, extended after Sec 8.3.7.

Two baselines matter, and they are NOT the same strength:

1. `truncated_svd` -- plain Eckart-Young-Mirsky optimum. Provably the best
   rank-r approximation of W in Frobenius norm. Calibration-free.

2. `activation_aware_svd` -- the strong competitor, and the honest threat
   to a CUR-based thesis. Instead of minimizing ||W - W'||_F it minimizes
   the quantity we actually care about,

       ||X (W - W')^T||_F     (the OUTPUT error on calibration data)

   subject to rank(W') = r. This has a closed-form optimum: with the
   Cholesky factor L of the Gram matrix G = X^T X = L L^T,

       ||X (W - W')^T||_F = ||(W - W') L||_F

   so truncating the SVD of (W L) and mapping back through L^{-1} is
   optimal. Any claim that calibration-guided CUR beats "SVD" must be
   made against THIS, not against plain truncated SVD -- otherwise the
   comparison is against a baseline that was never trying to solve the
   same problem.

Parameter accounting (why matched-rank comparisons are misleading):

    truncated / activation-aware SVD : r * (m + n)
    CUR with c = r                   : r * (m + n) + r^2

CUR is strictly more expensive at equal rank, so a fair comparison fixes
the PARAMETER BUDGET and lets each method pick the rank it can afford --
see `rank_for_param_budget_svd` / `rank_for_param_budget_cur`.
"""

from __future__ import annotations

import numpy as np

XI = 1e-9


def truncated_svd(w: np.ndarray, rank: int) -> np.ndarray:
    """Eckart-Young-Mirsky optimal rank-`rank` approximation of W."""
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    rank = max(1, min(rank, len(s)))
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def activation_aware_svd(w: np.ndarray, x_calib: np.ndarray, rank: int, ridge: float = 1e-8) -> np.ndarray:
    """Rank-`rank` W' minimizing ||X (W - W')^T||_F (output error), the
    calibration-optimal low-rank approximation.

    G = X^T X = L L^T  (Cholesky);  minimize ||(W - W') L||_F
      -> W' = trunc_svd(W L, rank) @ L^{-1}

    `ridge` regularizes G so the Cholesky factor exists when the
    calibration activations do not span the full input space (common: far
    fewer calibration rows than input dimensions, or dead channels).
    """
    n = w.shape[1]
    g = x_calib.T @ x_calib
    scale = float(np.trace(g)) / max(n, 1)
    g = g + np.eye(n) * (ridge * max(scale, XI))
    try:
        l_mat = np.linalg.cholesky(g)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        evals, evecs = np.linalg.eigh(g)
        evals = np.clip(evals, XI, None)
        l_mat = evecs * np.sqrt(evals)

    wl = w @ l_mat
    wl_approx = truncated_svd(wl, rank)
    # Solve W' L = wl_approx  ->  W' = wl_approx L^{-1}
    return np.linalg.solve(l_mat.T, wl_approx.T).T


def rank_for_param_budget_svd(m: int, n: int, budget_params: float) -> int:
    """Largest r with r*(m+n) <= budget_params."""
    return max(1, min(int(budget_params // (m + n)), min(m, n)))


def rank_for_param_budget_cur(m: int, n: int, budget_params: float) -> int:
    """Largest r with r*(m+n) + r^2 <= budget_params (c = r tie)."""
    a, b, c = 1.0, float(m + n), -float(budget_params)
    disc = b * b - 4 * a * c
    if disc <= 0:
        return 1
    r = int((-b + np.sqrt(disc)) / (2 * a))
    return max(1, min(r, min(m, n)))


def svd_param_count(m: int, n: int, rank: int) -> int:
    return rank * (m + n)


def output_relative_error(w: np.ndarray, w_approx: np.ndarray, x_calib: np.ndarray) -> float:
    """E_loc: ||X W^T - X W'^T|| / ||X W^T||."""
    y_ref = x_calib @ w.T
    y_hat = x_calib @ w_approx.T
    return float(np.linalg.norm(y_ref - y_hat) / (np.linalg.norm(y_ref) + XI))


def weight_relative_error(w: np.ndarray, w_approx: np.ndarray) -> float:
    return float(np.linalg.norm(w - w_approx) / (np.linalg.norm(w) + XI))
