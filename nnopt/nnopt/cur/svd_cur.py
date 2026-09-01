"""SVD-guided CUR decomposition -- README.md Sec 2.3 (CUR construction) and
Sec 3.2-C (the 1-GEMM-vs-3-GEMM latency risk that makes acceptance
conditional on *measured* latency, not just the Frobenius error here).

Pipeline (README Sec 2.3 points 6-7 / Sec 1.3.2):

  1. Column candidates C come from the representative (non-zero) columns
     produced by functional grouping (nnopt.grouping.functional_grouping);
     this module only requires a candidate pool + priority scores, so it
     can also be driven with "all columns" for the leverage-score-CUR /
     truncated-SVD ablation baselines required by README Sec 7.4.
  2. Rank r and row set R come from an SVD spectral analysis of W~: the
     singular-value energy spectrum gives a rank search space; rows are
     picked by their leverage score against the top-r left-singular
     subspace (standard CUR row-selection, Drineas/Mahoney).
  3. U = C^+ W~ R^+ via Moore-Penrose pseudo-inverses (README (1.29)).

SVD here is an *analysis tool* (spectrum -> rank/row choice), not the final
approximation -- consistent with README Sec 2.3: "SVD bu yerda yakuniy
yaqinlashtirish vositasi sifatida emas, balki CUR ranki va muhim satrlarni
tanlashni yo'naltiruvchi tahliliy vosita sifatida qo'llanadi."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

XI = 1e-9


# --------------------------------------------------------------------------
# Rank search space (SVD spectrum) and row selection (leverage scores)
# --------------------------------------------------------------------------

@dataclass
class SpectralAnalysis:
    singular_values: np.ndarray
    energy_cumulative_fraction: np.ndarray  # cumsum(S^2)/sum(S^2)
    left_singular_vectors: np.ndarray  # U from full SVD (m x k)
    right_singular_vectors_t: np.ndarray  # Vt from full SVD (k x n)

    def rank_for_energy(self, energy_fraction: float) -> int:
        """README Sec 2.3: singular-value spectrum gives the initial rank
        search space. Smallest r such that the top-r singular values retain
        at least `energy_fraction` of sum(S^2)."""
        idx = int(np.searchsorted(self.energy_cumulative_fraction, energy_fraction))
        return min(idx + 1, len(self.singular_values))


def analyze_spectrum(w_tilde: np.ndarray) -> SpectralAnalysis:
    u, s, vt = np.linalg.svd(w_tilde, full_matrices=False)
    energy = s * s
    total = float(np.sum(energy)) + XI
    cum_frac = np.cumsum(energy) / total
    return SpectralAnalysis(s, cum_frac, u, vt)


def row_leverage_scores(spectrum: SpectralAnalysis, rank: int) -> np.ndarray:
    """Standard CUR row-leverage score restricted to the top-`rank`
    left-singular subspace (Drineas & Mahoney): for row i,
        score_i = (1/rank) * sum_{k=1}^{rank} U[i, k]^2
    Rows with high score contribute most to the dominant spectral
    directions -- README Sec 2.3: "muhim spektral yo'nalishlarga katta
    hissa qo'shuvchi satrlar CURning satrlar matritsasiga kiritiladi."
    """
    rank = max(1, min(rank, spectrum.left_singular_vectors.shape[1]))
    u_r = spectrum.left_singular_vectors[:, :rank]
    return np.sum(u_r * u_r, axis=1) / rank


def select_cur_rows(spectrum: SpectralAnalysis, rank: int, r: int) -> np.ndarray:
    """Top-r row indices by leverage score against the rank-`rank` subspace."""
    scores = row_leverage_scores(spectrum, rank)
    r = max(1, min(r, len(scores)))
    return np.argsort(-scores)[:r]


def select_cur_columns(candidate_indices: list[int], priority_scores: dict[int, float], c: int) -> list[int]:
    """README Sec 2.3 point 6: columns are chosen from among the functional-
    grouping representative columns, ranked by group size / calibration
    activity / residual mismatch -- `priority_scores` bundles whatever
    composite ranking the caller wants (higher = picked first)."""
    ranked = sorted(candidate_indices, key=lambda idx: -priority_scores.get(idx, 0.0))
    c = max(1, min(c, len(ranked)))
    return ranked[:c]


def column_leverage_scores(spectrum: SpectralAnalysis, rank: int) -> np.ndarray:
    """Standard (Drineas & Mahoney) CUR COLUMN-leverage score, restricted to
    the top-`rank` right-singular subspace: for column j,
        score_j = (1/rank) * sum_{k=1}^{rank} V[j, k]^2 = (1/rank) * sum_k Vt[k, j]^2
    This is the generic/baseline column-selection criterion -- calibration-
    and functional-clustering-FREE -- used as the comparison point for
    README Sec 7.4's ablation (dissertation's core claim: functional-
    clustering-guided column selection, Fig 2.7/2.8, should beat this).
    """
    rank = max(1, min(rank, spectrum.right_singular_vectors_t.shape[0]))
    v_r = spectrum.right_singular_vectors_t[:rank, :]  # (rank, n)
    return np.sum(v_r * v_r, axis=0) / rank


def select_cur_columns_by_leverage(spectrum: SpectralAnalysis, rank: int, c: int) -> list[int]:
    """Baseline column selection for the Sec 7.4 ablation: top-c columns by
    raw leverage score, with NO functional grouping / calibration input at
    all -- the "generic CUR" this dissertation's method is compared against."""
    scores = column_leverage_scores(spectrum, rank)
    c = max(1, min(c, len(scores)))
    return list(np.argsort(-scores)[:c])


# --------------------------------------------------------------------------
# CUR assembly
# --------------------------------------------------------------------------

@dataclass
class CURResult:
    col_indices: list[int]
    row_indices: np.ndarray
    C: np.ndarray
    U: np.ndarray
    R: np.ndarray
    frobenius_error_relative: float  # ||W~ - CUR|| / ||W~||

    def reconstruct(self) -> np.ndarray:
        return self.C @ self.U @ self.R

    @property
    def rank(self) -> int:
        return self.U.shape[0]


def build_cur(w_tilde: np.ndarray, col_indices: list[int], row_indices: np.ndarray) -> CURResult:
    """README (1.28)-(1.29): W~ ~= C U R, U = C^+ W~ R^+."""
    c_mat = w_tilde[:, col_indices]
    r_mat = w_tilde[row_indices, :]
    c_pinv = np.linalg.pinv(c_mat)
    r_pinv = np.linalg.pinv(r_mat)
    u_mat = c_pinv @ w_tilde @ r_pinv
    reconstruction = c_mat @ u_mat @ r_mat
    err = np.linalg.norm(w_tilde - reconstruction, "fro") / (np.linalg.norm(w_tilde, "fro") + XI)
    return CURResult(
        col_indices=list(col_indices),
        row_indices=np.asarray(row_indices),
        C=c_mat,
        U=u_mat,
        R=r_mat,
        frobenius_error_relative=float(err),
    )


def truncated_svd_baseline(w_tilde: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    """Eckart-Young-Mirsky optimal rank-`rank` approximation -- the
    theoretical lower bound on Frobenius error for ANY rank-`rank`
    approximation, used as the ablation reference (README Sec 7.4:
    functional-CUR vs leverage-score-CUR vs truncated-SVD)."""
    u, s, vt = np.linalg.svd(w_tilde, full_matrices=False)
    rank = max(1, min(rank, len(s)))
    approx = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    err = np.linalg.norm(w_tilde - approx, "fro") / (np.linalg.norm(w_tilde, "fro") + XI)
    return approx, float(err)


# --------------------------------------------------------------------------
# Resource accounting -- README Sec 3.2-C (1 GEMM -> 3 GEMM latency risk)
# --------------------------------------------------------------------------

def cur_param_count(m: int, n: int, c: int, r: int) -> int:
    """README (1.25)-style: m*c + c*r + r*n vs original m*n."""
    return m * c + c * r + r * n


def original_param_count(m: int, n: int) -> int:
    return m * n


def cur_matmul_flops(m: int, n: int, c: int, r: int, batch: int = 1) -> int:
    """Y = C @ (U @ (R @ X)),  X: (n, batch). Three sequential matmuls
    instead of one -- this FLOPs count is a *necessary* (not sufficient)
    pre-filter; README Sec 3.2-C requires the CUR candidate to also pass
    a measured-latency check (nnopt.bench.latency) before acceptance,
    because kernel-launch and intermediate-tensor overhead are not
    captured by FLOPs alone.
    """
    return 2 * batch * (r * n + c * r + m * c)


def original_matmul_flops(m: int, n: int, batch: int = 1) -> int:
    return 2 * batch * m * n


def compression_ratio(m: int, n: int, c: int, r: int) -> float:
    orig = original_param_count(m, n)
    cur = cur_param_count(m, n, c, r)
    return orig / cur if cur > 0 else float("inf")
