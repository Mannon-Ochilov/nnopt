"""Functional grouping of hidden nodes via calibration-response similarity
-- README.md Sec 2.3 / Sec 3.2-A.

This module operates purely on already-extracted functional response
vectors h_j (one vector per hidden node, built by concatenating calibration
outputs at active positions -- see README Sec 2.3 point 2 and Sec 3.1.10).
Extracting h_j from a real ONNX model + calibration set is the job of a
separate `nnopt.calibrator` module (not yet built); this module is
model-agnostic and fully testable with synthetic vectors.

Two residual metrics are provided:

  * `sin_theta_residual` -- the metric as literally derived from the
    dissertation text (README Sec 2.3 point 3-4). It is proven in
    README Sec 3.2-A to be a deterministic function of the cosine
    similarity alone (sin(theta) = sqrt(1 - cos^2) under the least-squares-
    optimal gamma), so thresholding on it carries *no information beyond*
    thresholding on cosine. Kept here ONLY for ablation studies that need
    to reproduce the original two-criteria formulation faithfully.

  * `output_weighted_residual` -- the corrected primary metric (README
    Sec 3.2-A / Sec 10 rule 7): weights the same geometric residual by
    the downstream weight column's norm and normalizes by the downstream
    layer output norm, so the criterion now depends on more than direction
    alone (a node with tiny downstream weight or tiny activation energy is
    "cheap" to merge even at a middling cosine; a node with a large
    downstream weight is not). This is what `greedy_group()` uses by
    default.

Special-case handling (README Sec 3.1.10, "last" draft): near-zero-activity
nodes (||h_j|| ~ 0) cannot serve as an anchor, and are folded into any group
with gamma ~ 0 (negligible contribution) instead of blocking the grouping
pass or being left ungrouped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS_NUM = 1e-9  # numerical-stability guard, README's xi/eps_num


# --------------------------------------------------------------------------
# Pairwise geometry
# --------------------------------------------------------------------------

def cosine_similarity(h_j: np.ndarray, h_p: np.ndarray, eps_num: float = EPS_NUM) -> float:
    num = float(np.dot(h_j, h_p))
    denom = float(np.linalg.norm(h_j) * np.linalg.norm(h_p)) + eps_num
    return num / denom


def compensation_gamma(h_j: np.ndarray, h_p: np.ndarray, eps_num: float = EPS_NUM) -> float:
    """Least-squares-optimal scalar compensation: gamma minimizing
    ||h_j - gamma*h_p||^2  =>  gamma = <h_j,h_p> / ||h_p||^2 (README 2.3)."""
    denom = float(np.dot(h_p, h_p)) + eps_num
    return float(np.dot(h_j, h_p)) / denom


def sin_theta_residual(cos_theta: float) -> float:
    """README Sec 3.2-A: the original 'qoldiq xatolik' reduces exactly to
    sin(theta) = sqrt(1 - cos^2) under the LS-optimal gamma -- i.e. it is
    NOT independent information from the cosine threshold. Ablation-only.
    """
    return math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))


def output_weighted_residual(
    h_j: np.ndarray,
    h_p: np.ndarray,
    w_col_j_norm: float,
    y_norm: float,
    eps_num: float = EPS_NUM,
) -> tuple[float, float, float]:
    """README Sec 3.2-A corrected primary metric:

        eps_j = ||W[:,j]|| * ||h_j - gamma*h_p|| / (||Y|| + xi)
              = ||W[:,j]|| * ||h_j|| * sin(theta) / (||Y|| + xi)

    Returns (eps_j, cos_theta, gamma) so callers get the intermediate
    diagnostics for free.
    """
    cos_theta = cosine_similarity(h_j, h_p, eps_num)
    gamma = compensation_gamma(h_j, h_p, eps_num)
    sin_theta = sin_theta_residual(cos_theta)
    h_j_norm = float(np.linalg.norm(h_j))
    eps_j = w_col_j_norm * h_j_norm * sin_theta / (y_norm + eps_num)
    return eps_j, cos_theta, gamma


# --------------------------------------------------------------------------
# Greedy grouping
# --------------------------------------------------------------------------

@dataclass
class FunctionalGroup:
    representative: int
    members: list[int]  # includes representative
    gammas: dict[int, float]  # member -> compensation coeff vs representative (repr excluded)
    cos_to_representative: dict[int, float]
    eps_to_representative: dict[int, float]

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class GroupingResult:
    groups: list[FunctionalGroup]
    tau: float
    eps_threshold: float
    zero_activity_nodes: list[int]

    def representative_indices(self) -> list[int]:
        return [g.representative for g in self.groups]

    def node_to_group(self) -> dict[int, int]:
        """node index -> index into self.groups"""
        mapping = {}
        for gi, g in enumerate(self.groups):
            for m in g.members:
                mapping[m] = gi
        return mapping


def greedy_group(
    h_vectors: np.ndarray,  # shape (num_nodes, response_dim)
    w_col_norms: np.ndarray,  # shape (num_nodes,) -- ||W[:, j]|| in the downstream layer
    y_norm: float,  # ||Y|| of the downstream layer's calibration output
    tau: float,
    eps_threshold: float,
    zero_activity_eps: float = 1e-6,
    metric: str = "output_weighted",  # "output_weighted" | "sin_theta" (ablation)
    abs_cosine: bool = False,
) -> GroupingResult:
    """README Sec 2.3 point 4 (sequential greedy) + Sec 3.1.10 (near-zero
    activity special case) + Sec 3.1.9 (this function evaluates ONE
    (tau, eps_threshold) candidate pair; sweeping the sorted candidate
    lists top-down is the caller's/cascade's responsibility).

    A node j joins the group anchored at p iff BOTH:
        cos(j, p) >= tau                    (direction criterion)
        eps_metric(j, p) <= eps_threshold    (magnitude/impact criterion)

    `abs_cosine` compares |cos| instead. A pair at cos = -0.95 is as
    mergeable as one at +0.95 -- gamma is fitted signed and
    build_compensated_weight already applies it signed -- so the signed gate
    refuses a merge the rest of the machinery supports. Measured, the
    refusal costs nothing on Whisper and something real on open_llama_3b: at
    the tau = 0.99 operating point the two agree to the digit on all four
    Whisper layers sampled, while on Llama the largest |cos| is 0.8488
    against a largest signed cos of 0.7681, and at tau = 0.70 the absolute
    form reaches 6.5x as many channels.

    It stays OFF by default even so. Every operating point in the paper was
    measured with the signed gate, and the agreement on Whisper is exact only
    at tau = 0.99; at tau = 0.95 the two differ slightly, so flipping the
    default would silently change artifacts that have already been evaluated.
    """
    if metric not in ("output_weighted", "sin_theta"):
        raise ValueError(f"unknown metric {metric!r}")

    num_nodes = h_vectors.shape[0]
    norms = np.linalg.norm(h_vectors, axis=1)
    zero_activity = [i for i in range(num_nodes) if norms[i] < zero_activity_eps]
    zero_set = set(zero_activity)

    ungrouped = set(range(num_nodes)) - zero_set
    groups: list[FunctionalGroup] = []

    # Vectorized candidate scan. The per-anchor inner loop is O(num_nodes)
    # Python iterations, so a 4096-channel FFN costs ~17M of them and the
    # pass takes hours. Every quantity in that loop is a dot product against
    # the anchor, so one matrix-vector product per anchor computes them all
    # at once. Semantics are unchanged -- same criteria, same greedy order.
    order = np.argsort(-norms)  # anchors are taken most-active first
    alive = np.zeros(num_nodes, dtype=bool)
    alive[list(ungrouped)] = True

    for anchor in order:
        if not alive[anchor]:
            continue
        alive[anchor] = False
        members = [int(anchor)]
        gammas: dict[int, float] = {}
        cos_map: dict[int, float] = {}
        eps_map: dict[int, float] = {}

        h_anchor = h_vectors[anchor]
        cand = np.flatnonzero(alive)
        if cand.size:
            # Dot every row against the anchor and then select, rather than
            # gathering the live rows and multiplying those. The gather is a
            # COPY of (live x dim) float64 -- about 49 MB on a 4096-channel
            # layer -- repeated once per anchor, and it dominates: measured at
            # 4096 x 3072 with half the channels live, gather-then-multiply
            # takes 81.2 s against 7.6 s for one gemv over the contiguous
            # matrix followed by a 1-D select, a 10.7x difference. The cost
            # falls hardest on layers whose channels barely merge, since
            # almost every channel then becomes an anchor. Same dot products,
            # same results -- only the memory traffic differs.
            dots = (h_vectors @ h_anchor)[cand]
            anchor_norm = float(norms[anchor])
            cos_all = dots / (norms[cand] * anchor_norm + EPS_NUM)
            gamma_all = dots / (float(np.dot(h_anchor, h_anchor)) + EPS_NUM)
            sin_all = np.sqrt(np.maximum(0.0, 1.0 - cos_all * cos_all))
            if metric == "sin_theta":
                eps_all = sin_all
            else:
                eps_all = w_col_norms[cand] * norms[cand] * sin_all / (y_norm + EPS_NUM)

            direction = np.abs(cos_all) if abs_cosine else cos_all
            take = (direction >= tau) & (eps_all <= eps_threshold)
            for idx in np.flatnonzero(take):
                j = int(cand[idx])
                members.append(j)
                gammas[j] = float(gamma_all[idx])
                cos_map[j] = float(cos_all[idx])
                eps_map[j] = float(eps_all[idx])
                alive[j] = False

        groups.append(
            _finalize_group(anchor, members, gammas, cos_map, eps_map, h_vectors, w_col_norms, y_norm, metric)
        )

    # Fold zero-activity nodes into the largest group with gamma ~= 0
    # (README Sec 3.1.10: they cannot anchor and contribute negligibly).
    if groups and zero_activity:
        target = max(groups, key=lambda g: g.size)
        for z in zero_activity:
            target.members.append(z)
            target.gammas[z] = 0.0
            target.cos_to_representative[z] = 0.0
            target.eps_to_representative[z] = 0.0
    elif zero_activity and not groups:
        # Degenerate case: every node has ~zero activity. Group them all
        # under an arbitrary representative with gamma=0 rather than crash.
        rep = zero_activity[0]
        groups.append(
            FunctionalGroup(
                representative=rep,
                members=list(zero_activity),
                gammas={z: 0.0 for z in zero_activity[1:]},
                cos_to_representative={z: 0.0 for z in zero_activity[1:]},
                eps_to_representative={z: 0.0 for z in zero_activity[1:]},
            )
        )

    return GroupingResult(groups=groups, tau=tau, eps_threshold=eps_threshold, zero_activity_nodes=zero_activity)


def _finalize_group(
    anchor: int,
    members: list[int],
    gammas: dict[int, float],
    cos_map: dict[int, float],
    eps_map: dict[int, float],
    h_vectors: np.ndarray,
    w_col_norms: np.ndarray,
    y_norm: float,
    metric: str,
) -> FunctionalGroup:
    """Representative selection, per the dissertation's Fig 2.7 box 3:

        j*_t = argmax_{j in C_t} sim(h_j, mu_t)

    i.e. the group member whose response is most similar *in direction* to
    the group mean mu_t -- measured with the SAME cosine similarity the
    grouping criterion itself uses, NOT Euclidean distance to the mean.

    The distinction is load-bearing, not cosmetic: amplitude differences
    between members are already handled separately by the compensation
    coefficient gamma (README Sec 2.3 point 5), so penalizing a member for
    having a larger magnitude -- which is exactly what argmin ||h_j - mu||
    does -- would double-count amplitude and can elect a worse-aligned
    representative. Example: with mu = [1,0], h_a = [5,0] (cos = 1.0,
    Euclid = 4.0) is the correct representative, yet Euclidean distance
    would pick h_b = [1.1,0.3] (cos = 0.965, Euclid = 0.32) instead.

    If the elected representative differs from the greedy anchor, gammas
    are recomputed relative to it ("re-anchoring").
    """
    if len(members) == 1:
        return FunctionalGroup(anchor, members, {}, {}, {})

    group_h = h_vectors[members]
    mean_h = group_h.mean(axis=0)
    sims = np.array([cosine_similarity(group_h[i], mean_h) for i in range(len(members))])
    representative = members[int(np.argmax(sims))]

    if representative == anchor:
        return FunctionalGroup(anchor, members, gammas, cos_map, eps_map)

    # Re-anchor: recompute gamma/cos/eps of every other member relative to
    # the newly chosen representative.
    h_rep = h_vectors[representative]
    new_gammas, new_cos, new_eps = {}, {}, {}
    for m in members:
        if m == representative:
            continue
        h_m = h_vectors[m]
        cos_theta = cosine_similarity(h_m, h_rep)
        gamma = compensation_gamma(h_m, h_rep)
        if metric == "sin_theta":
            eps_val = sin_theta_residual(cos_theta)
        else:
            eps_val, _, _ = output_weighted_residual(h_m, h_rep, float(w_col_norms[m]), y_norm)
        new_gammas[m] = gamma
        new_cos[m] = cos_theta
        new_eps[m] = eps_val
    return FunctionalGroup(representative, members, new_gammas, new_cos, new_eps)


# --------------------------------------------------------------------------
# Weight-matrix compensation (README (1.24)-style masking)
# --------------------------------------------------------------------------

def trim_to_budget(result: GroupingResult, target_groups: int) -> int:
    """Release merges until `target_groups` groups remain, worst merge first.

    Grouping is driven by tau, so a caller working to a channel BUDGET can
    only approach it from a discrete ladder of tau values, and the step that
    first meets the budget generally overshoots it -- on a Whisper encoder
    layer, asking for 45% removal and landing on 57.5%. Charging the budget
    for that extra 12.5 points would misattribute the damage.

    Undoing a merge is exact rather than approximate: the member becomes a
    singleton group again, and `build_compensated_weight` then leaves its
    column untouched and folds nothing into a representative on its behalf.
    Which merges to undo is the criterion's own call -- eps_j is precisely the
    quantity the grouping used to judge the merge cheap, so the merges it
    judged most expensive are released first.

    Returns the number of merges released. A `target_groups` at or below the
    current group count is a no-op.
    """
    released = 0
    candidates = []
    for gi, grp in enumerate(result.groups):
        for m in grp.members:
            if m != grp.representative:
                candidates.append((grp.eps_to_representative.get(m, 0.0), gi, m))
    candidates.sort(reverse=True)

    for _, gi, m in candidates:
        if len(result.groups) >= target_groups:
            break
        grp = result.groups[gi]
        if m not in grp.members:
            continue
        grp.members.remove(m)
        grp.gammas.pop(m, None)
        grp.cos_to_representative.pop(m, None)
        grp.eps_to_representative.pop(m, None)
        result.groups.append(FunctionalGroup(m, [m], {}, {}, {}))
        released += 1
    return released


def build_compensated_weight(w: np.ndarray, result: GroupingResult) -> np.ndarray:
    """README Sec 2.3 point 5: fold non-representative columns' contribution
    into their group's representative column, zero-mask the rest. `w` has
    shape (out_features, in_features) where in_features == num hidden nodes
    (columns index hidden nodes, matching the downstream Linear/Gemm layer's
    weight layout: y = w @ h).

    Physical shape is UNCHANGED (README: "matritsaning fizik hajmi
    o'zgarmaydi") -- this is an intermediate matrix for CUR column
    candidate selection, not a final compression.
    """
    w_tilde = w.copy()
    for group in result.groups:
        rep = group.representative
        for m in group.members:
            if m == rep:
                continue
            gamma = group.gammas.get(m, 0.0)
            w_tilde[:, rep] += gamma * w[:, m]
            w_tilde[:, m] = 0.0
    return w_tilde
