"""Integration tests for nnopt.cascade.operator_cascade (README Sec 2.5 + Sec 1.1)."""

from __future__ import annotations

import numpy as np
import pytest

from nnopt.cascade.budget import closed_form_rank_for_target, discount_target_for_future_stage, quant_max_factor, required_reduction_factor
from nnopt.cascade.operator_cascade import CandidateVariant, OperatorContext, _select_best_effort, run_cascade


def _low_rank_weight(m, n, true_rank, noise, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, true_rank))
    b = rng.standard_normal((true_rank, n))
    w = a @ b
    if noise > 0:
        w = w + noise * rng.standard_normal((m, n))
    return w


def test_svd_quality_ceiling_predicts_concentrated_vs_flat_spectrum():
    """README Sec 8.3.3: a calibration-free, weight-only diagnostic (the
    Eckart-Young floor at the budget-determined rank) should distinguish a
    matrix CUR can plausibly handle (concentrated spectrum, like the real
    proj_out finding) from one it cannot (flat spectrum, like the real fc2
    finding) -- WITHOUT running functional grouping or CUR at all.
    """
    m, n = 256, 256
    rng = np.random.default_rng(9)
    x = rng.standard_normal((64, n)) * 0.1

    concentrated = _low_rank_weight(m, n, true_rank=4, noise=0.01, seed=10)
    flat = rng.standard_normal((m, n))  # full rank, energy spread evenly

    cache_size = (m * n * 4 / 6.0) / 0.7  # force CUR engagement (>4x needed) for both

    result_concentrated = run_cascade(OperatorContext(
        name="concentrated", w=concentrated, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=cache_size, alpha=0.7, delta_l=0.5,
    ))
    result_flat = run_cascade(OperatorContext(
        name="flat", w=flat, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=cache_size, alpha=0.7, delta_l=0.5,
    ))

    assert result_concentrated.svd_quality_ceiling is not None
    assert result_flat.svd_quality_ceiling is not None
    assert result_concentrated.svd_quality_ceiling < result_flat.svd_quality_ceiling, (
        f"concentrated-spectrum matrix should have a much lower (better) SVD "
        f"floor than a flat-spectrum one at the same target rank: "
        f"concentrated={result_concentrated.svd_quality_ceiling:.4f} "
        f"flat={result_flat.svd_quality_ceiling:.4f}"
    )
    # With the ceiling this bad, the flat case is expected to trip the skip
    # (the feature this test now also covers).
    assert result_flat.status == "cur_skipped_by_svd_ceiling"

    # Force the full search on both (skip disabled) to confirm the ceiling's
    # prediction actually holds at the OUTPUT-error level too, not just in
    # the weight-reconstruction floor compared above.
    result_concentrated_forced = run_cascade(OperatorContext(
        name="concentrated", w=concentrated, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=cache_size, alpha=0.7, delta_l=0.5,
        skip_cur_if_ceiling_exceeds_delta=False,
    ))
    result_flat_forced = run_cascade(OperatorContext(
        name="flat", w=flat, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=cache_size, alpha=0.7, delta_l=0.5,
        skip_cur_if_ceiling_exceeds_delta=False,
    ))
    assert result_concentrated_forced.final_variant.e_loc < result_flat_forced.final_variant.e_loc


def test_cur_column_priority_uses_both_group_size_and_activity():
    """Dissertation Fig 2.8 / README Sec 2.3 point 6: the CUR column ranking
    score is a_j = |C_t(j)| * ||h_j||_2 -- BOTH group size and calibration
    response activity.

    The two factors are put in DIRECT CONFLICT here, which is the only way
    to tell the correct score from a group-size-only one:
      * nodes 0..3 are near-parallel and weakly active -> one group of 4,
        its representative scores 4 on size but only ~1 on activity;
      * node 4 is alone but massively active -> scores 1 on size, ~200 on
        activity.
    Size-only ranking picks the 4-member group's representative first;
    the correct a_j (= size * ||h_j||) picks node 4 first.
    """
    rng = np.random.default_rng(7)
    n_pos = 60
    n_nodes = 5

    shared_dir = rng.standard_normal(n_pos)
    shared_dir /= np.linalg.norm(shared_dir)
    lone_dir = rng.standard_normal(n_pos)
    lone_dir -= np.dot(lone_dir, shared_dir) * shared_dir  # make it orthogonal
    lone_dir /= np.linalg.norm(lone_dir)

    x_calib = np.zeros((n_pos, n_nodes))
    for j in range(4):  # weakly-active, near-parallel cluster
        x_calib[:, j] = (1.0 + 0.05 * j) * shared_dir + 1e-4 * rng.standard_normal(n_pos)
    x_calib[:, 4] = 200.0 * lone_dir  # lone but dominant on real calibration input

    m = 32
    w = rng.standard_normal((m, n_nodes))

    fp32_bytes = m * n_nodes * 4
    ctx = OperatorContext(
        name="activity_priority_op",
        w=w,
        x_calib=x_calib,
        dtype_bits_initial=32,
        cache_size_bytes_override=(fp32_bytes / 6.0) / 0.7,  # force CUR engagement
        alpha=0.7,
        delta_l=0.5,
        tau_soft=0.98,
        eps_threshold_soft=0.5,
        skip_cur_if_ceiling_exceeds_delta=False,  # this test is about column ordering WITHIN CUR
    )
    result = run_cascade(ctx)

    cur_variants = [v for v in result.variants if v.rank is not None and "col_idx" in v.extra]
    assert cur_variants, "expected at least one CUR variant to be attempted"
    chosen_columns = cur_variants[0].extra["col_idx"]
    assert chosen_columns, "CUR must select at least one column"
    assert chosen_columns[0] == 4, (
        f"node 4 (singleton group, but ~200x the calibration activity of the "
        f"4-member cluster) must outrank the cluster's representative; got "
        f"ordering {chosen_columns}. A group-size-only priority would rank the "
        f"cluster first -- the ||h_j|| factor of a_j is missing or ignored."
    )


def test_svd_ceiling_skip_can_be_disabled_and_reaches_the_same_ceiling_either_way():
    """The skip is an efficiency/reporting shortcut, not a different
    computation: svd_quality_ceiling itself must be identical whether or
    not the skip is allowed to act on it (it's computed before the branch)."""
    rng = np.random.default_rng(11)
    m, n = 128, 128
    w = rng.standard_normal((m, n))  # flat spectrum -> high ceiling
    x = rng.standard_normal((32, n)) * 0.1
    cache_size = (m * n * 4 / 8.0) / 0.7

    skipped = run_cascade(OperatorContext(
        name="op", w=w, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=cache_size, alpha=0.7, delta_l=0.05,
    ))
    forced = run_cascade(OperatorContext(
        name="op", w=w, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=cache_size, alpha=0.7, delta_l=0.05,
        skip_cur_if_ceiling_exceeds_delta=False,
    ))

    assert skipped.status == "cur_skipped_by_svd_ceiling"
    assert forced.status != "cur_skipped_by_svd_ceiling"
    assert skipped.svd_quality_ceiling == pytest.approx(forced.svd_quality_ceiling)
    # the skip must actually have saved work: far fewer variants tried
    assert len(skipped.variants) < len(forced.variants)


def _dummy_variant(stage, e_loc, k_cache, resource_ok):
    return CandidateVariant(
        stage=stage, bits=8, rank=None, m_bytes=0, e_loc=e_loc, k_cache=k_cache,
        quality_ok=False, resource_ok=resource_ok, latency_ok=True, accepted=False,
    )


def test_select_best_effort_prefers_resource_feasible_over_lower_eloc():
    """README Sec 2.5 step 5/7 fallback rule, tested directly at the unit
    level (an end-to-end run_cascade scenario cannot isolate this: whenever
    the SVD-ceiling skip fires, the current budget/discount math happens to
    always leave the rank unconstrained-if-quantize-alone-was-resource_ok,
    so a resource_ok-but-quality-failing quantize variant essentially never
    coexists with a fired skip in practice -- see the discount formula in
    run_cascade. The rule matters regardless, e.g. for the ordinary
    end-of-ladder fallback and for future extensions (more quant_ladder
    entries, per-channel quantization) that could decouple that coincidence.)

    A variant with e_loc=0 (like "baseline" always trivially has, being its
    own reference) but resource_ok=False must NOT be picked over a
    resource-feasible variant with worse e_loc.
    """
    baseline = _dummy_variant("baseline", e_loc=0.0, k_cache=18.0, resource_ok=False)
    fp16 = _dummy_variant("quantize_direct", e_loc=0.0001, k_cache=9.0, resource_ok=False)
    int8_ok_but_lossy = _dummy_variant("quantize_direct", e_loc=0.02, k_cache=0.9, resource_ok=True)

    best, any_feasible = _select_best_effort([baseline, fp16, int8_ok_but_lossy])
    assert any_feasible
    assert best is int8_ok_but_lossy, (
        f"expected the only resource-feasible variant to win regardless of its "
        f"higher e_loc; got stage={best.stage} e_loc={best.e_loc} -- naive "
        f"min(e_loc) over ALL variants would wrongly pick baseline (e_loc=0, "
        f"but never actually fits the cache budget)"
    )


def test_select_best_effort_falls_back_to_closest_k_cache_when_nothing_fits():
    baseline = _dummy_variant("baseline", e_loc=0.0, k_cache=18.0, resource_ok=False)
    fp16 = _dummy_variant("quantize_direct", e_loc=0.0001, k_cache=9.0, resource_ok=False)
    int8 = _dummy_variant("quantize_direct", e_loc=0.02, k_cache=4.5, resource_ok=False)

    best, any_feasible = _select_best_effort([baseline, fp16, int8])
    assert not any_feasible
    assert best is int8  # smallest k_cache among the non-feasible ones


def test_ceiling_safety_margin_widens_or_narrows_the_skip_window():
    rng = np.random.default_rng(12)
    m, n = 96, 96
    w = rng.standard_normal((m, n))
    x = rng.standard_normal((32, n)) * 0.1
    cache_size = (m * n * 4 / 8.0) / 0.7

    lenient = run_cascade(OperatorContext(  # large margin -> harder to trigger skip
        name="op", w=w, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=cache_size, alpha=0.7, delta_l=0.05,
        ceiling_safety_margin=100.0,
    ))
    assert lenient.status != "cur_skipped_by_svd_ceiling"


def test_baseline_accepted_when_already_fits():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 8))
    x = rng.standard_normal((16, 8))
    ctx = OperatorContext(
        name="tiny_op", w=w, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=10 * 1024 * 1024, alpha=0.7,
    )
    result = run_cascade(ctx)
    assert result.status == "accepted"
    assert result.final_variant.stage == "baseline"
    assert len(result.variants) == 1  # nothing else should even be attempted


def test_quantize_direct_alone_is_sufficient_and_cur_is_skipped():
    rng = np.random.default_rng(1)
    m, n = 64, 64
    w = rng.standard_normal((m, n))
    x = rng.standard_normal((256, n))

    fp32_bytes = m * n * 4
    # cache_eff chosen so required reduction is 3x: FP16 (2x) insufficient, INT8 (4x) sufficient
    cache_size = (fp32_bytes / 3.0) / 0.7
    ctx = OperatorContext(
        name="mid_op", w=w, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=cache_size, alpha=0.7,
        delta_l=0.2,  # generous quality budget; this test is about stage *selection*, not quantization precision
    )
    result = run_cascade(ctx)
    assert result.status == "accepted"
    assert result.final_variant.stage == "quantize_direct"
    assert result.final_variant.bits == 8
    stages_tried = [v.stage for v in result.variants]
    assert "cur_fp" not in stages_tried and "cur_quantized" not in stages_tried


def test_cur_engaged_when_quantization_alone_cannot_reach_target():
    """The README Sec 1.1 worked-example scenario end-to-end: baseline needs
    more than INT8's 4x max, forcing the cascade into the CUR stage, with
    the rank anchored by the budget-discounted residual (not the full
    requirement)."""
    m, n, true_rank = 256, 256, 8
    w = _low_rank_weight(m, n, true_rank=true_rank, noise=0.01, seed=2)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((128, n)) * 0.1  # small activations -> favors quality

    fp32_bytes = m * n * 4
    required = 6.0  # > INT8's 4x ceiling, forces CUR engagement
    cache_size = (fp32_bytes / required) / 0.7
    ctx = OperatorContext(
        name="big_low_rank_op", w=w, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=cache_size, alpha=0.7,
        delta_l=0.1,
    )
    result = run_cascade(ctx)

    stages_tried = [v.stage for v in result.variants]
    assert "quantize_direct" in stages_tried
    assert any(s in ("cur_fp", "cur_quantized") for s in stages_tried), (
        f"expected CUR to be engaged, stages tried: {stages_tried}"
    )
    assert result.status == "accepted", [
        (v.stage, v.bits, v.rank, round(v.e_loc, 4), round(v.k_cache, 3), v.accepted) for v in result.variants
    ]

    # cross-check the rank actually used against the budget math directly
    future_max = quant_max_factor(32, 8)
    discounted = discount_target_for_future_stage(required, future_max)
    expected_rank_ceiling = closed_form_rank_for_target(m, n, discounted)
    cur_variants = [v for v in result.variants if v.rank is not None]
    assert all(v.rank <= expected_rank_ceiling + 1 for v in cur_variants)  # +1 slack for len(representative_cols) clamp


def test_resource_infeasible_reports_cleanly_instead_of_crashing():
    rng = np.random.default_rng(4)
    m, n = 32, 32
    w = rng.standard_normal((m, n))
    x = rng.standard_normal((64, n))
    ctx = OperatorContext(
        name="impossible_op", w=w, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=1,  # absurdly tiny, unreachable even at INT8+CUR r=1
        alpha=0.7, delta_l=0.5,
    )
    result = run_cascade(ctx)
    assert result.status in ("resource_infeasible", "quality_violated_best_effort", "cur_skipped_by_svd_ceiling")
    assert result.final_variant is not None  # must not crash; must report *something*


def test_quality_violation_reports_best_effort_not_a_crash():
    """Full-rank random noise has no structure for CUR to exploit at the
    tiny rank a very demanding target forces -- quality must fail, and the
    cascade should report the best *resource-feasible* variant by quality
    rather than silently accepting a bad reconstruction."""
    rng = np.random.default_rng(5)
    m, n = 128, 128
    w = rng.standard_normal((m, n))  # full rank, no low-rank structure at all
    x = rng.standard_normal((64, n))

    fp32_bytes = m * n * 4
    cache_size = (fp32_bytes / 50.0) / 0.7  # extremely demanding, unreachable at good quality
    ctx = OperatorContext(
        name="noisy_full_rank_op", w=w, x_calib=x, dtype_bits_initial=32,
        cache_size_bytes_override=cache_size, alpha=0.7, delta_l=0.01,  # strict quality budget
    )
    result = run_cascade(ctx)
    assert result.status in ("quality_violated_best_effort", "resource_infeasible", "cur_skipped_by_svd_ceiling")
    assert result.final_variant is not None
    if result.status == "quality_violated_best_effort":
        feasible = [v for v in result.variants if v.resource_ok]
        assert result.final_variant.e_loc == min(v.e_loc for v in feasible)
