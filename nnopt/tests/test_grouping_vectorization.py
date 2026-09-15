"""Equivalence check for the vectorized greedy_group inner loop.

greedy_group's per-anchor candidate scan was rewritten from a Python loop
into one matrix-vector product per anchor (a 4096-channel FFN needed ~17M
Python iterations and took hours). The rewrite is an optimization only, so
it has to reproduce the original algorithm exactly -- these tests pin that
by re-implementing the original loop here and comparing group membership.
"""

import numpy as np
import pytest

from nnopt.grouping.functional_grouping import (
    EPS_NUM,
    compensation_gamma,
    cosine_similarity,
    greedy_group,
    output_weighted_residual,
    sin_theta_residual,
)


def reference_greedy(h_vectors, w_col_norms, y_norm, tau, eps_threshold,
                     zero_activity_eps=1e-6, metric="output_weighted"):
    """The pre-optimization implementation, kept verbatim as the oracle."""
    num_nodes = h_vectors.shape[0]
    norms = np.linalg.norm(h_vectors, axis=1)
    zero_set = {i for i in range(num_nodes) if norms[i] < zero_activity_eps}
    ungrouped = set(range(num_nodes)) - zero_set
    groups = []
    while ungrouped:
        anchor = max(ungrouped, key=lambda i: norms[i])
        ungrouped.discard(anchor)
        members = [anchor]
        h_anchor = h_vectors[anchor]
        for j in sorted(ungrouped):
            h_j = h_vectors[j]
            cos_theta = cosine_similarity(h_j, h_anchor)
            if metric == "sin_theta":
                eps_val = sin_theta_residual(cos_theta)
            else:
                eps_val, _, _ = output_weighted_residual(
                    h_j, h_anchor, float(w_col_norms[j]), y_norm)
            if cos_theta >= tau and eps_val <= eps_threshold:
                members.append(j)
                ungrouped.discard(j)
        groups.append(sorted(members))
    return sorted(groups)


def as_membership(result):
    return sorted(sorted(g.members) for g in result.groups)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("tau", [0.99, 0.9, 0.7])
def test_vectorized_matches_reference(seed, tau):
    rng = np.random.default_rng(seed)
    n, d = 40, 24
    base = rng.standard_normal((8, d))
    # build channels as noisy copies of a few prototypes so groups exist
    h = np.vstack([base[rng.integers(0, 8)] + 0.15 * rng.standard_normal(d) for _ in range(n)])
    w_norms = rng.uniform(0.5, 2.0, size=n)
    y_norm = float(np.linalg.norm(h) * 2)

    got = as_membership(greedy_group(h, w_norms, y_norm, tau=tau, eps_threshold=0.5))
    want = reference_greedy(h, w_norms, y_norm, tau=tau, eps_threshold=0.5)
    assert got == want


def test_matches_reference_with_sin_theta_metric():
    rng = np.random.default_rng(7)
    n, d = 30, 16
    h = rng.standard_normal((n, d))
    w_norms = np.ones(n)
    y_norm = 1.0
    got = as_membership(greedy_group(h, w_norms, y_norm, tau=0.5,
                                     eps_threshold=0.9, metric="sin_theta"))
    want = reference_greedy(h, w_norms, y_norm, tau=0.5,
                            eps_threshold=0.9, metric="sin_theta")
    assert got == want


def test_gammas_match_closed_form():
    """Vectorized gamma must equal the per-pair least-squares coefficient."""
    rng = np.random.default_rng(11)
    n, d = 25, 12
    proto = rng.standard_normal(d)
    h = np.vstack([proto * rng.uniform(0.8, 1.2) + 0.02 * rng.standard_normal(d)
                   for _ in range(n)])
    res = greedy_group(h, np.ones(n), 100.0, tau=0.5, eps_threshold=10.0)
    for g in res.groups:
        for member, gamma in g.gammas.items():
            expected = compensation_gamma(h[member], h[g.representative])
            assert gamma == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_zero_activity_channels_still_folded():
    rng = np.random.default_rng(3)
    h = rng.standard_normal((12, 8))
    h[4] = 0.0
    h[9] = 0.0
    res = greedy_group(h, np.ones(12), 10.0, tau=0.9, eps_threshold=0.5)
    everyone = sorted(m for g in res.groups for m in g.members)
    assert everyone == list(range(12)), "no channel may be dropped"
    for g in res.groups:
        for z in (4, 9):
            if z in g.members and z != g.representative:
                assert g.gammas[z] == 0.0


def test_large_input_completes_quickly():
    """The reason for the rewrite: this shape must not take minutes."""
    import time

    rng = np.random.default_rng(5)
    h = rng.standard_normal((2048, 64))
    t0 = time.perf_counter()
    greedy_group(h, np.ones(2048), 100.0, tau=0.95, eps_threshold=0.5)
    assert time.perf_counter() - t0 < 20.0
