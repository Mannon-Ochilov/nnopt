"""Tests for nnopt.grouping.functional_grouping (README Sec 2.3 / Sec 3.2-A)."""

from __future__ import annotations

import numpy as np
import pytest

from nnopt.grouping.functional_grouping import (
    build_compensated_weight,
    compensation_gamma,
    cosine_similarity,
    greedy_group,
    output_weighted_residual,
    sin_theta_residual,
    trim_to_budget,
)


def _grouped(n=64, dim=32, seed=0, tau=0.9):
    """A layer with real merges in it, for the budget-trimming tests."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n // 4, dim))
    h = np.repeat(base, 4, axis=0) + 0.01 * rng.normal(size=(n, dim))
    w_norms = np.abs(rng.normal(size=n)) + 0.1
    return h, w_norms, greedy_group(h, w_norms, float(np.linalg.norm(h)),
                                    tau=tau, eps_threshold=1e9)


def test_anti_collinear_pairs_merge_only_when_the_sign_is_dropped():
    """An anti-collinear pair is exactly as mergeable as a collinear one: the
    gamma that reproduces it is simply negative, and compensation applies
    gamma signed. The default gate is signed and refuses it -- measured to
    cost nothing on Whisper and real channels on Llama -- so `abs_cosine`
    exists to accept it."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(4, 32))
    h = np.vstack([base, -1.7 * base])            # each pair at cos = -1
    w = np.ones(len(h))
    y = float(np.linalg.norm(h))

    signed = greedy_group(h, w, y, tau=0.9, eps_threshold=1e9)
    absolute = greedy_group(h, w, y, tau=0.9, eps_threshold=1e9,
                            abs_cosine=True)
    assert len(signed.groups) == 8                # nothing merged
    assert len(absolute.groups) == 4              # each pair collapsed


def test_abs_cosine_keeps_the_negative_gamma_that_reproduces_the_pair():
    """Merging is only correct if the compensation carries the sign; a
    positive gamma would add the mirrored contribution instead of cancelling
    it."""
    rng = np.random.default_rng(4)
    base = rng.normal(size=(1, 16))
    h = np.vstack([base, -2.0 * base])
    g = greedy_group(h, np.ones(2), float(np.linalg.norm(h)), tau=0.9,
                     eps_threshold=1e9, abs_cosine=True)
    gammas = [gm for grp in g.groups for gm in grp.gammas.values()]
    assert len(gammas) == 1
    assert gammas[0] < 0


def test_abs_cosine_does_not_disturb_a_purely_collinear_layer():
    """Where no anti-collinear pair exists the two gates must agree exactly,
    which is what makes the option safe to turn on for a model whose
    activations sit in a one-signed cone."""
    h, w, _ = _grouped()
    y = float(np.linalg.norm(h))
    a = greedy_group(h, w, y, tau=0.9, eps_threshold=1e9)
    b = greedy_group(h, w, y, tau=0.9, eps_threshold=1e9, abs_cosine=True)
    assert [sorted(g.members) for g in a.groups] == \
           [sorted(g.members) for g in b.groups]


def test_cosine_similarity_basic_cases():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-9)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_sin_theta_residual_matches_the_dissertation_derivation():
    """README Sec 3.2-A claims sin_theta_residual(cos) == ||h_j - gamma*h_p|| / ||h_j||
    exactly, under the least-squares-optimal gamma. Verify numerically on
    several random vector pairs -- this is the load-bearing proof that the
    original two-criteria formulation is redundant."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        h_j = rng.standard_normal(16)
        h_p = rng.standard_normal(16)
        cos_theta = cosine_similarity(h_j, h_p)
        gamma = compensation_gamma(h_j, h_p)
        direct_residual = np.linalg.norm(h_j - gamma * h_p) / (np.linalg.norm(h_j) + 1e-9)
        derived_residual = sin_theta_residual(cos_theta)
        assert direct_residual == pytest.approx(derived_residual, abs=1e-6)


def test_output_weighted_metric_is_not_redundant_with_cosine():
    """Two candidate nodes with the *same* direction (identical cosine, gamma)
    but different downstream weight-column norms must get different eps_j --
    proving the corrected metric carries information beyond cosine alone
    (unlike sin_theta_residual, which by construction cannot)."""
    rng = np.random.default_rng(1)
    h_p = rng.standard_normal(32)
    h_j = 0.9 * h_p + 0.1 * rng.standard_normal(32)  # same direction for both trials

    eps_small_w, cos_a, gamma_a = output_weighted_residual(h_j, h_p, w_col_j_norm=0.01, y_norm=10.0)
    eps_large_w, cos_b, gamma_b = output_weighted_residual(h_j, h_p, w_col_j_norm=5.0, y_norm=10.0)

    assert cos_a == pytest.approx(cos_b)
    assert gamma_a == pytest.approx(gamma_b)
    assert eps_small_w < eps_large_w  # same direction, different downstream impact


def test_greedy_group_merges_near_parallel_nodes():
    rng = np.random.default_rng(2)
    dim = 64
    anchor_dir = rng.standard_normal(dim)
    anchor_dir /= np.linalg.norm(anchor_dir)

    # 3 nodes nearly parallel to the anchor direction, 1 clearly orthogonal
    h = np.zeros((4, dim))
    h[0] = 5.0 * anchor_dir  # strongest -> becomes anchor
    h[1] = 4.0 * anchor_dir + 0.01 * rng.standard_normal(dim)
    h[2] = 3.0 * anchor_dir + 0.01 * rng.standard_normal(dim)
    orthogonal = rng.standard_normal(dim)
    orthogonal -= np.dot(orthogonal, anchor_dir) * anchor_dir  # project out anchor component
    h[3] = 2.0 * orthogonal / np.linalg.norm(orthogonal)

    w_col_norms = np.ones(4) * 0.5
    y_norm = 10.0

    result = greedy_group(h, w_col_norms, y_norm, tau=0.99, eps_threshold=0.5)
    n2g = result.node_to_group()
    assert n2g[0] == n2g[1] == n2g[2], "near-parallel nodes should share a group"
    assert n2g[3] != n2g[0], "orthogonal node must be in its own group"


def test_representative_is_chosen_by_cosine_not_euclidean_distance():
    """Dissertation Fig 2.7 box 3: j*_t = argmax_j sim(h_j, mu_t) -- cosine
    similarity to the group mean, NOT Euclidean distance to it.

    The two criteria are put in direct conflict: node 0 is perfectly
    aligned with the group mean but has a large magnitude (far from the
    mean in Euclidean terms), while node 2 is closer to the mean in
    absolute distance but worse aligned in direction. Amplitude is already
    compensated by gamma, so the correct representative is node 0.
    """
    dim = 24
    base = np.zeros(dim)
    base[0] = 1.0
    off = np.zeros(dim)
    off[1] = 1.0

    h = np.stack([
        6.0 * base,            # node 0: perfect direction, large magnitude
        1.0 * base,            # node 1: perfect direction, small magnitude
        1.6 * base + 0.9 * off,  # node 2: near the mean in Euclidean terms, worse aligned
    ])
    mean_h = h.mean(axis=0)

    cos_scores = [cosine_similarity(h[i], mean_h) for i in range(3)]
    euclid_dists = [float(np.linalg.norm(h[i] - mean_h)) for i in range(3)]
    # sanity: the two criteria really do disagree on this construction
    assert int(np.argmax(cos_scores)) != int(np.argmin(euclid_dists)), (
        "test construction failed: cosine and Euclidean agree here, so the "
        "test cannot discriminate between the two selection rules"
    )
    expected_rep = int(np.argmax(cos_scores))

    w_col_norms = np.ones(3)
    # tau/eps permissive enough to keep all three nodes in one group
    result = greedy_group(h, w_col_norms, y_norm=10.0, tau=0.5, eps_threshold=10.0)
    assert len(result.groups) == 1, f"expected a single group, got {len(result.groups)}"
    assert result.groups[0].representative == expected_rep, (
        f"representative must be argmax cosine-similarity to the group mean "
        f"(node {expected_rep}), got node {result.groups[0].representative}; "
        f"cos={np.round(cos_scores, 4).tolist()} euclid={np.round(euclid_dists, 4).tolist()}"
    )


def test_zero_activity_node_is_folded_not_blocking():
    rng = np.random.default_rng(3)
    dim = 32
    h = rng.standard_normal((5, dim))
    h[2] = np.zeros(dim)  # zero-activity node
    w_col_norms = np.ones(5)
    result = greedy_group(h, w_col_norms, y_norm=5.0, tau=0.9, eps_threshold=0.5)

    assert 2 in result.zero_activity_nodes
    n2g = result.node_to_group()
    assert 2 in n2g  # it must have ended up in *some* group, not orphaned
    group = result.groups[n2g[2]]
    assert group.gammas[2] == 0.0


def test_build_compensated_weight_preserves_shape_and_masks_non_representatives():
    rng = np.random.default_rng(4)
    dim = 16
    # Force a 2-node group: h1 exactly parallel to h0 (gamma easy to check).
    h0 = rng.standard_normal(dim)
    h1 = 2.0 * h0  # exactly parallel, gamma should be 2.0 (since h1 = 2*h0 => LS gamma of h1 wrt h0 is 2.0... check direction)
    h_vectors = np.stack([h0, h1])
    w_col_norms = np.array([1.0, 1.0])

    result = greedy_group(h_vectors, w_col_norms, y_norm=1.0, tau=0.99, eps_threshold=10.0)
    assert len(result.groups) == 1
    group = result.groups[0]
    assert group.size == 2

    out_features, in_features = 6, 2
    w = rng.standard_normal((out_features, in_features))
    w_orig = w.copy()
    w_tilde = build_compensated_weight(w, result)

    assert w_tilde.shape == w.shape
    non_rep = [m for m in group.members if m != group.representative][0]
    assert np.allclose(w_tilde[:, non_rep], 0.0)
    gamma = group.gammas[non_rep]
    expected_rep_col = w_orig[:, group.representative] + gamma * w_orig[:, non_rep]
    assert np.allclose(w_tilde[:, group.representative], expected_rep_col)


def test_trim_to_budget_hits_the_requested_group_count():
    h, w_norms, result = _grouped()
    start = len(result.groups)
    assert start < 40, "fixture must actually merge, or the trim is untested"
    released = trim_to_budget(result, 40)
    assert len(result.groups) == 40
    assert released == 40 - start


def test_trim_is_a_noop_when_already_within_budget():
    h, w_norms, result = _grouped()
    before = len(result.groups)
    assert trim_to_budget(result, before) == 0
    assert trim_to_budget(result, before - 5) == 0
    assert len(result.groups) == before


def test_trim_releases_the_worst_justified_merges_first():
    """eps_j is what judged the merge cheap, so the costliest go back first."""
    h, w_norms, result = _grouped()
    released_eps = [e for g in result.groups
                    for m, e in g.eps_to_representative.items()]
    kept_before = sorted(released_eps, reverse=True)[:5]
    trim_to_budget(result, len(result.groups) + 5)
    still_merged = [e for g in result.groups
                    for m, e in g.eps_to_representative.items()]
    for e in kept_before:
        assert e not in still_merged


def test_trimmed_members_are_left_untouched_by_compensation():
    """A released channel keeps its own column and folds into no one."""
    rng = np.random.default_rng(7)
    h, w_norms, result = _grouped()
    target = len(result.groups) + 6
    trim_to_budget(result, target)

    w = rng.standard_normal((5, h.shape[0]))
    w_tilde = build_compensated_weight(w, result)
    singletons = [g.representative for g in result.groups if g.size == 1]
    for s in singletons:
        assert np.allclose(w_tilde[:, s], w[:, s])


def test_trim_keeps_every_channel_accounted_for():
    """No channel may be dropped or duplicated by releasing merges."""
    h, w_norms, result = _grouped()
    trim_to_budget(result, len(result.groups) + 7)
    seen = [m for g in result.groups for m in g.members]
    assert sorted(seen) == list(range(h.shape[0]))

