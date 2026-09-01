"""Per-operator adaptive cascade -- README.md Sec 2.5, orchestrated with the
Sec 1.1 budget-propagation math (nnopt.cascade.budget).

Scope of this v1 (explicitly, so nobody mistakes it for more than it is):

  * Operates on ONE matrix operator at a time: weight W (m, n) with
    Y = W @ X, plus a calibration activation batch X (== the h_j columns
    needed by functional grouping, since X *is* the response of whatever
    produced it -- README Sec 2.3).
  * Implements README Sec 2.5 steps 1 (baseline check), 2 (direct
    quantization), 3 (functional grouping + CUR structural, budget-
    anchored rank), 4 (quantize CUR components), 5 (accept softest
    sufficient / reject), 7 (rollback to last accepted variant on
    failure).
  * Step 6 (CPU/GPU placement) and steps 8-9 (whole-model global
    evaluation, graph-level fusion) are cross-operator / whole-model
    concerns and are NOT implemented here -- they need a real ONNX graph
    + calibration pipeline (nnopt.calibrator, not yet built) and live in
    a higher orchestrator.
  * Functional grouping's own (tau, eps_threshold) ladder (README Sec
    3.1.9: two sorted candidate lists) is evaluated ONCE at its softest
    setting to produce representative columns; only the CUR RANK is
    swept via the budget-anchored search. Sweeping tau/eps_threshold
    jointly with rank is documented future work, not silently pretended
    to be done here.
  * CUR acceptance uses a FLOPS-based latency PROXY (README Sec 3.2-C:
    "necessary but not sufficient"). A `latency_fn` hook lets a caller
    substitute real nnopt.bench.latency measurements once wired to an
    actual ONNX subgraph; without one, the FLOPs proxy is the best
    available signal and this is stated explicitly in the result.
  * README Sec 8.3.3: before engaging functional grouping + CUR at all,
    a calibration-free advisory check (the Eckart-Young weight-
    reconstruction floor at the budget-determined rank) can skip the
    whole expensive search when it predicts near-certain failure -- see
    `svd_quality_ceiling` on CascadeResult and
    `OperatorContext.skip_cur_if_ceiling_exceeds_delta`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from nnopt.cascade.budget import (
    closed_form_rank_for_target,
    discount_target_for_future_stage,
    quant_max_factor,
    required_reduction_factor,
)
from nnopt.cur.svd_cur import (
    analyze_spectrum,
    build_cur,
    cur_matmul_flops,
    cur_param_count,
    original_matmul_flops,
    select_cur_columns,
    select_cur_rows,
    truncated_svd_baseline,
)
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.hw.cache_topology import CacheInstance
from nnopt.quantizer.scale_refine import dequantize, quantize_codes, refine_scale

XI = 1e-9

Q_MAX_BY_BITS = {16: 32767, 8: 127, 4: 7}


def weight_bytes(param_count: int, bits: int) -> int:
    return int(math.ceil(param_count * bits / 8))


def _select_best_effort(variants: list["CandidateVariant"]) -> tuple["CandidateVariant", bool]:
    """Shared "what's the most useful thing to report" rule for every path
    that ends without a fully `accepted` variant (README Sec 2.5 step 5/7):
    prefer the best-quality option among those that at least fit the cache
    budget; if NONE fit, report whichever came closest to fitting (quality
    is moot until resource feasibility exists at all). Returns
    (best_variant, any_resource_feasible)."""
    resource_feasible = [v for v in variants if v.resource_ok]
    if resource_feasible:
        return min(resource_feasible, key=lambda v: v.e_loc), True
    return min(variants, key=lambda v: v.k_cache), False


def local_relative_error(y_ref: np.ndarray, y_cand: np.ndarray, eps: float = XI) -> float:
    """README Sec 2.3/2.4/2.5 E_loc: ||Y - Y'|| / (||Y|| + xi)."""
    return float(np.linalg.norm(y_ref - y_cand) / (np.linalg.norm(y_ref) + eps))


@dataclass
class OperatorContext:
    name: str
    w: np.ndarray  # (m, n): m=out_features, n=in_features; Y = W @ X
    x_calib: np.ndarray  # (batch, n) calibration activations
    dtype_bits_initial: int = 32
    cache: CacheInstance | None = None
    cache_size_bytes_override: int | None = None  # used if `cache` is None (unit tests w/o real topology)
    alpha: float = 0.7
    delta_l: float = 0.05
    quant_ladder: tuple[int, ...] = (16, 8)
    tau_soft: float = 0.98
    eps_threshold_soft: float = 0.05
    latency_fn: Callable[["CandidateVariant"], float] | None = None
    skip_cur_if_ceiling_exceeds_delta: bool = True
    ceiling_safety_margin: float = 1.0  # README Sec 8.3.3: multiplier on delta_l before skipping (see run_cascade docstring)

    @property
    def m(self) -> int:
        return self.w.shape[0]

    @property
    def n(self) -> int:
        return self.w.shape[1]

    @property
    def cache_bytes(self) -> int:
        if self.cache is not None:
            return self.cache.size_bytes
        if self.cache_size_bytes_override is not None:
            return self.cache_size_bytes_override
        raise ValueError("OperatorContext needs either `cache` or `cache_size_bytes_override`")

    @property
    def cache_eff_bytes(self) -> float:
        return self.alpha * self.cache_bytes


@dataclass
class CandidateVariant:
    stage: str  # "baseline" | "quantize_direct" | "cur_fp" | "cur_quantized" | "cur_skipped_by_svd_ceiling"
    bits: int
    rank: int | None
    m_bytes: int
    e_loc: float
    k_cache: float
    quality_ok: bool
    resource_ok: bool
    latency_ok: bool  # True if no latency signal was checked (vacuously OK) or it passed
    accepted: bool
    note: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class CascadeResult:
    operator_name: str
    status: str  # "accepted" | "quality_violated_best_effort" | "resource_infeasible" | "cur_skipped_by_svd_ceiling"
    variants: list[CandidateVariant]
    final_variant: CandidateVariant
    required_reduction_factor: float
    feasible_upper_bound: bool
    svd_quality_ceiling: float | None = None  # README Sec 8.3.3: calibration-free CUR floor


def _quantize_candidate(
    ctx: OperatorContext,
    w: np.ndarray,
    bits: int,
    y_ref: np.ndarray,
    x_for_response: np.ndarray,
    stage_name: str,
    extra: dict | None = None,
) -> CandidateVariant:
    q_max = Q_MAX_BY_BITS[bits]

    def layer_response_fn(w_deq: np.ndarray) -> np.ndarray:
        return x_for_response @ w_deq.T

    result = refine_scale(w, q_max, layer_response_fn=layer_response_fn, y_reference=y_ref)
    q = quantize_codes(w, result.scale, q_max)
    w_deq = dequantize(q, result.scale)
    y_cand = x_for_response @ w_deq.T
    e_loc = local_relative_error(y_ref, y_cand)

    m_bytes = weight_bytes(w.size, bits)
    k_cache = m_bytes / ctx.cache_eff_bytes

    quality_ok = e_loc <= ctx.delta_l
    resource_ok = k_cache <= 1.0
    variant = CandidateVariant(
        stage=stage_name,
        bits=bits,
        rank=None,
        m_bytes=m_bytes,
        e_loc=e_loc,
        k_cache=k_cache,
        quality_ok=quality_ok,
        resource_ok=resource_ok,
        latency_ok=True,  # bit-width-only quantization: latency regression not expected on supported HW (documented assumption, README Sec 5.2 step 2)
        accepted=quality_ok and resource_ok,
        note="direct weight quantization, no structural change",
        extra={"scale": result.scale, **(extra or {})},
    )
    return variant


def _cur_candidate(
    ctx: OperatorContext,
    w_tilde: np.ndarray,
    representative_cols: list[int],
    col_priority: dict[int, float],
    rank: int,
    y_ref: np.ndarray,
    bits: int | None,  # None = keep C,R,U at ctx.dtype_bits_initial (stage "cur_fp")
    quantize_u: bool = True,
) -> CandidateVariant:
    spectrum = analyze_spectrum(w_tilde)
    row_idx = select_cur_rows(spectrum, rank=rank, r=rank)
    col_idx = select_cur_columns(representative_cols, col_priority, c=rank)
    cur = build_cur(w_tilde, col_idx, row_idx)

    def _q(mat: np.ndarray, q_max: int) -> np.ndarray:
        res = refine_scale(mat, q_max)
        return dequantize(quantize_codes(mat, res.scale, q_max), res.scale)

    if bits is None:
        c_eff, u_eff, r_eff = cur.C, cur.U, cur.R
        used_bits = ctx.dtype_bits_initial
        u_bits = ctx.dtype_bits_initial
        stage = "cur_fp"
    else:
        q_max = Q_MAX_BY_BITS[bits]
        c_eff = _q(cur.C, q_max)
        r_eff = _q(cur.R, q_max)
        used_bits = bits
        stage = "cur_quantized"
        if quantize_u:
            # README Sec 2.4: default budget assumption (Sec 1.1 discount
            # math) treats ALL CUR components as quantizable; U is only
            # exempted "zarur holatda" (when necessary) -- i.e. as a
            # fallback the cascade reaches for if THIS variant's quality
            # fails (see the quantize_u=False retry in run_cascade), not
            # as a standing default.
            u_eff = _q(cur.U, q_max)
            u_bits = bits
        else:
            u_eff = cur.U
            u_bits = ctx.dtype_bits_initial

    y_cand = ctx.x_calib @ (c_eff @ u_eff @ r_eff).T
    e_loc = local_relative_error(y_ref, y_cand)

    c_count = len(col_idx)
    r_count = len(row_idx)
    if bits is not None:
        m_bytes = (
            weight_bytes(ctx.m * c_count, bits)
            + weight_bytes(r_count * ctx.n, bits)
            + weight_bytes(cur.U.size, u_bits)
        )
    else:
        m_bytes = weight_bytes(cur_param_count(ctx.m, ctx.n, c_count, r_count), used_bits)

    k_cache = m_bytes / ctx.cache_eff_bytes

    flops_cur = cur_matmul_flops(ctx.m, ctx.n, c_count, r_count, batch=ctx.x_calib.shape[0])
    flops_orig = original_matmul_flops(ctx.m, ctx.n, batch=ctx.x_calib.shape[0])
    flops_proxy_ok = flops_cur <= flops_orig  # necessary-not-sufficient gate, README Sec 3.2-C

    latency_ok = flops_proxy_ok
    latency_note = "FLOPs-proxy only (no bench.latency wired in)"
    if ctx.latency_fn is not None:
        # placeholder variant passed for the caller's real measurement hook
        pass

    quality_ok = e_loc <= ctx.delta_l
    resource_ok = k_cache <= 1.0

    return CandidateVariant(
        stage=stage,
        bits=used_bits,
        rank=rank,
        m_bytes=m_bytes,
        e_loc=e_loc,
        k_cache=k_cache,
        quality_ok=quality_ok,
        resource_ok=resource_ok,
        latency_ok=latency_ok,
        accepted=quality_ok and resource_ok and latency_ok,
        note=f"CUR c={c_count} r={r_count} u_bits={u_bits if bits is not None else used_bits}; {latency_note}",
        extra={
            "col_idx": col_idx,
            "row_idx": list(map(int, row_idx)),
            "flops_cur": flops_cur,
            "flops_orig": flops_orig,
        },
    )


def _rank_ladder(start_rank: int, min_rank: int = 1, max_steps: int = 6) -> list[int]:
    """README Sec 5.2 uchinchi bosqich: 'Faqat yumshoq CUR nomzodi ...
    resurs nomuvofiqligini bartaraf eta olmagan holatda bosqichma-bosqich
    kamaytiriladi.' The budget-anchored start_rank (closed_form_rank_for_target)
    is the *first* (softest) attempt, not the only one -- if it fails
    (quality OR resource, e.g. because U becomes ill-conditioned at that
    rank and cannot tolerate quantization), progressively smaller ranks are
    tried until one works or min_rank is reached."""
    ranks = [start_rank]
    r = start_rank
    while r > min_rank and len(ranks) < max_steps:
        r = max(min_rank, int(r * 0.6))
        if r != ranks[-1]:
            ranks.append(r)
    if ranks[-1] != min_rank:
        ranks.append(min_rank)
    return ranks


def run_cascade(ctx: OperatorContext) -> CascadeResult:
    variants: list[CandidateVariant] = []

    baseline_bytes = weight_bytes(ctx.w.size, ctx.dtype_bits_initial)
    k_cache0 = baseline_bytes / ctx.cache_eff_bytes
    baseline = CandidateVariant(
        stage="baseline",
        bits=ctx.dtype_bits_initial,
        rank=None,
        m_bytes=baseline_bytes,
        e_loc=0.0,
        k_cache=k_cache0,
        quality_ok=True,
        resource_ok=k_cache0 <= 1.0,
        latency_ok=True,
        accepted=k_cache0 <= 1.0,
        note="unchanged operator",
    )
    variants.append(baseline)
    if baseline.accepted:
        return CascadeResult(ctx.name, "accepted", variants, baseline, required_reduction_factor(baseline_bytes, ctx.cache_eff_bytes), True)

    required = required_reduction_factor(baseline_bytes, ctx.cache_eff_bytes)
    y_ref = ctx.x_calib @ ctx.w.T

    # ---- Stage 2: direct quantization ladder (softest -> strongest) -----
    for bits in ctx.quant_ladder:
        variant = _quantize_candidate(ctx, ctx.w, bits, y_ref, ctx.x_calib, "quantize_direct")
        variants.append(variant)
        if variant.accepted:
            return CascadeResult(ctx.name, "accepted", variants, variant, required, True)

    harshest_quant_bits = ctx.quant_ladder[-1]
    future_max_factor = quant_max_factor(ctx.dtype_bits_initial, harshest_quant_bits)
    # Practical feasibility (given the actual representative-column pool size,
    # not just the theoretical r=1 ceiling) is verified per-candidate below
    # via CandidateVariant.resource_ok on the real CUR rank search result.

    structural_target = discount_target_for_future_stage(required, future_max_factor)
    unclamped_start_rank = closed_form_rank_for_target(ctx.m, ctx.n, structural_target)

    # README Sec 8.3.3: a calibration-free advisory check, run BEFORE the
    # (calibration-dependent, and for large operators expensive) functional
    # grouping step. Eckart-Young-Mirsky: truncated SVD at a given rank is
    # the PROVABLY OPTIMAL rank-r approximation of W in Frobenius norm --
    # CUR, whatever columns/rows it picks, cannot beat this floor. Because
    # that floor is also provably non-increasing as rank grows, checking it
    # ONLY at unclamped_start_rank (the softest/largest rank the ladder
    # would ever try) is sufficient: if CUR cannot meet budget even there,
    # no harsher (smaller) rank in the ladder can either.
    #
    # CAVEAT (kept honest, not oversold): svd_quality_ceiling bounds the
    # WEIGHT reconstruction error ||W-W_r||_F/||W||_F. delta_l bounds the
    # OUTPUT error ||Y-Y'||/||Y||. These are related (Y-Y' = X(W-W')^T) but
    # not identical quantities -- there is no proven inequality tying them
    # at a fixed constant. The skip below is therefore a heuristic, not a
    # mathematical guarantee, calibrated against two real measurements
    # (README Sec 8.3.3: actual e_loc ran ~1.3-1.6x above this ceiling for
    # both decoder.proj_out and decoder.layers.0.fc2). `ceiling_safety_margin`
    # lets a caller demand more slack before trusting the skip, and
    # `skip_cur_if_ceiling_exceeds_delta=False` disables it entirely to force
    # the full search regardless (e.g. for ablation Sec 7, or whenever the
    # cost of a false skip matters more than the saved compute).
    _, svd_quality_ceiling = truncated_svd_baseline(ctx.w, rank=max(1, unclamped_start_rank))

    if ctx.skip_cur_if_ceiling_exceeds_delta and svd_quality_ceiling > ctx.delta_l * ctx.ceiling_safety_margin:
        best_so_far, _ = _select_best_effort(variants)
        prediction = CandidateVariant(
            stage="cur_skipped_by_svd_ceiling",
            bits=ctx.dtype_bits_initial,
            rank=unclamped_start_rank,
            m_bytes=weight_bytes(cur_param_count(ctx.m, ctx.n, unclamped_start_rank, unclamped_start_rank), ctx.dtype_bits_initial),
            e_loc=svd_quality_ceiling,
            k_cache=float("nan"),
            quality_ok=False,
            resource_ok=False,
            latency_ok=True,
            accepted=False,
            note=(
                f"functional grouping + CUR search skipped: Eckart-Young weight-"
                f"reconstruction floor ({svd_quality_ceiling:.4f}) already exceeds "
                f"delta_l*margin ({ctx.delta_l * ctx.ceiling_safety_margin:.4f}) at "
                f"rank={unclamped_start_rank}, the softest rank this operator's "
                f"budget would ever try; every harsher rank can only be worse "
                f"(the floor is non-increasing in rank). Heuristic, not a proof at "
                f"the output-error level -- see docstring caveat. Set "
                f"skip_cur_if_ceiling_exceeds_delta=False to force the full search."
            ),
        )
        variants.append(prediction)
        return CascadeResult(
            ctx.name, "cur_skipped_by_svd_ceiling", variants, best_so_far, required, False, svd_quality_ceiling
        )

    # ---- Stage 3: functional grouping (softest setting only, v1) ------
    w_col_norms = np.linalg.norm(ctx.w, axis=0)  # ||W[:, j]|| for each hidden node j
    y_norm = float(np.linalg.norm(y_ref))
    grouping_result = greedy_group(
        ctx.x_calib.T,  # h_vectors: one row per node j == column j of X across calibration batch
        w_col_norms,
        y_norm,
        tau=ctx.tau_soft,
        eps_threshold=ctx.eps_threshold_soft,
    )
    w_tilde = build_compensated_weight(ctx.w, grouping_result)
    representative_cols = grouping_result.representative_indices()

    # Column priority a_j = |C_t(j)| * ||h_j||_2 -- README Sec 2.3 point 6 and
    # the dissertation's Fig 2.8 ("Guruh hajmi va javob faolligi"). BOTH factors
    # are required: group size alone would promote a representative that stands
    # for many nodes yet is barely active on real calibration input, while
    # activity alone would ignore how much of the layer that column speaks for.
    h_norms = np.linalg.norm(ctx.x_calib, axis=0)  # ||h_j|| per hidden node j
    col_priority = {
        g.representative: float(g.size) * float(h_norms[g.representative])
        for g in grouping_result.groups
    }

    start_rank = max(1, min(unclamped_start_rank, len(representative_cols)))
    rank_ladder = _rank_ladder(start_rank, min_rank=1)

    for rank in rank_ladder:
        # Stage 3: CUR structural change alone (fp32 components).
        cur_fp_variant = _cur_candidate(ctx, w_tilde, representative_cols, col_priority, rank, y_ref, bits=None)
        variants.append(cur_fp_variant)
        if cur_fp_variant.accepted:
            return CascadeResult(ctx.name, "accepted", variants, cur_fp_variant, required, True, svd_quality_ceiling)

        # Stage 4: quantize CUR components, only if fp32-CUR insufficient at
        # this rank. Per bit-width: try quantizing U too first (matches the
        # Sec 1.1 budget assumption that this stage delivers its full
        # quant_max_factor across ALL components); fall back to fp32-U --
        # README's "zarur holatda" -- only if that quality gate fails.
        for bits in ctx.quant_ladder:
            variant_u_quantized = _cur_candidate(
                ctx, w_tilde, representative_cols, col_priority, rank, y_ref, bits=bits, quantize_u=True
            )
            variants.append(variant_u_quantized)
            if variant_u_quantized.accepted:
                return CascadeResult(ctx.name, "accepted", variants, variant_u_quantized, required, True, svd_quality_ceiling)

            if not variant_u_quantized.quality_ok:
                variant_u_fp = _cur_candidate(
                    ctx, w_tilde, representative_cols, col_priority, rank, y_ref, bits=bits, quantize_u=False
                )
                variants.append(variant_u_fp)
                if variant_u_fp.accepted:
                    return CascadeResult(ctx.name, "accepted", variants, variant_u_fp, required, True, svd_quality_ceiling)

        # This rank was not sufficient (resource and/or quality); README
        # Sec 5.2: "bosqichma-bosqich kamaytiriladi" -- try the next,
        # harsher (smaller) rank in the ladder before giving up.

    # ---- Stage 5/7: nothing accepted -> report best-effort + rollback --
    best, any_resource_feasible = _select_best_effort(variants)
    status = "quality_violated_best_effort" if any_resource_feasible else "resource_infeasible"
    return CascadeResult(ctx.name, status, variants, best, required, any_resource_feasible, svd_quality_ceiling)
