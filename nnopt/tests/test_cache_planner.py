"""Tests for the model-level planner.

The properties worth pinning down are the ones the design argument rests on:
that the miss objective is continuous and rewards fitting without requiring
it, that the ladder is monotone so lazy evaluation may stop at the first
failure, and that an unreachable target is reported as unreachable instead of
being approximated by an ever-harsher configuration.
"""

from dataclasses import replace

import pytest

from nnopt.cascade.cache_planner import (
    MIB,
    CachePlan,
    PartSpec,
    Treatment,
    feasibility,
    miss_bytes,
    part_bytes,
    plan,
    reachable_keep,
    required_factor,
)

# The reference machine's Whisper, in bytes: encoder layers are half FFN,
# decoder layers a quarter attention-heavy (self + cross), which is exactly
# what makes the decoder harder to fit.
ENC = PartSpec("enkoder", int(48 * MIB), 24, int(32 * MIB), reuse=1500)
DEC = PartSpec("dekoder", int(64 * MIB), 24, int(32 * MIB), reuse=1)


def test_part_bytes_scales_with_bits_and_keep():
    assert part_bytes(ENC, Treatment(32, 1.0)) == pytest.approx(48 * MIB)
    assert part_bytes(ENC, Treatment(8, 1.0)) == pytest.approx(12 * MIB)
    # Only the prunable half responds to keep.
    assert part_bytes(ENC, Treatment(32, 0.5)) == pytest.approx(32 * MIB)


def test_spec_rejects_impossible_geometry():
    with pytest.raises(ValueError):
        PartSpec("x", 10, 1, 20)
    with pytest.raises(ValueError):
        PartSpec("x", 10, 1, 5, reuse=0)


def test_treatment_validates_its_domain():
    with pytest.raises(ValueError):
        Treatment(keep=0.0)
    with pytest.raises(ValueError):
        Treatment(bits=7)


def test_miss_is_continuous_across_the_budget_boundary():
    """No jump at the fit/no-fit line -- the property a gate does not have.

    Continuity here is Lipschitz rather than flat: crossing the budget changes
    the SLOPE by the reuse factor, because every overflowing byte is re-read
    once per reuse. So the bound to check is reuse * delta, not a constant.
    """
    budget = 16.8 * MIB
    delta, reuse = 1024, 1500
    below = PartSpec("p", int(budget) - delta, 1, int(budget) // 2, reuse=reuse)
    above = PartSpec("p", int(budget) + delta, 1, int(budget) // 2, reuse=reuse)
    t = Treatment(32, 1.0)
    gap = abs(miss_bytes(above, t, budget) - miss_bytes(below, t, budget))
    assert gap <= reuse * 2 * delta


def test_miss_rewards_fitting_without_requiring_it():
    budget = 16.8 * MIB
    fits = miss_bytes(ENC, Treatment(8, 0.5), budget)      # 8 MiB/layer
    over = miss_bytes(ENC, Treatment(8, 1.0), budget)      # 12 MiB/layer
    way_over = miss_bytes(ENC, Treatment(32, 1.0), budget)  # 48 MiB/layer
    assert fits < over < way_over


def test_reuse_decides_how_much_overflow_hurts():
    """The encoder/decoder asymmetry the whole cascade rests on."""
    budget = 8.4 * MIB
    t = Treatment(8, 1.0)  # 16 MiB/layer at fp32 -> 4 MiB, which would FIT;
    # the asymmetry only shows once the part actually overflows, so size it up.
    streamed = PartSpec("dek", int(64 * MIB), 1, int(32 * MIB), reuse=1)
    reused = PartSpec("enk", int(64 * MIB), 1, int(32 * MIB), reuse=1500)
    assert part_bytes(reused, t) > budget
    assert miss_bytes(reused, t, budget) > 100 * miss_bytes(streamed, t, budget)


def test_ladder_is_monotone_in_both_size_and_misses():
    p = plan([ENC, DEC], 24 * MIB)
    sizes = [r.total_bytes for r in p.rungs]
    misses = [r.miss for r in p.rungs]
    assert sizes == sorted(sizes, reverse=True)
    assert misses == sorted(misses, reverse=True)


def test_each_rung_changes_exactly_one_part():
    p = plan([ENC, DEC], 24 * MIB)
    for prev, cur in zip(p.rungs, p.rungs[1:]):
        changed = [n for n in cur.treatments
                   if cur.treatments[n] != prev.treatments[n]]
        assert len(changed) == 1
        assert changed[0] in cur.step


def test_quantization_is_spent_before_structural_reduction():
    p = plan([ENC, DEC], 24 * MIB)
    for r in p.rungs:
        for name, t in r.treatments.items():
            if t.keep < 1.0:
                assert t.bits == 8, f"{name} pruned while still fp32"


def test_plan_reaches_fit_on_the_reference_machine():
    """L3 = 24 MiB: INT8 alone puts both parts inside 16.8 MiB."""
    p = plan([ENC, DEC], 24 * MIB)
    int8_only = next(r for r in p.rungs
                     if all(t.bits == 8 and t.keep == 1.0
                            for t in r.treatments.values()))
    assert int8_only.all_fit


def test_smaller_cache_needs_structural_help():
    p = plan([ENC, DEC], 12 * MIB)
    int8_only = next(r for r in p.rungs
                     if all(t.bits == 8 and t.keep == 1.0
                            for t in r.treatments.values()))
    assert not int8_only.all_fit
    assert any(r.all_fit or r.index == len(p.rungs) - 1 for r in p.rungs)


def test_feasibility_flags_the_unreachable_decoder():
    """A 4 MiB L3 cannot hold the decoder's attention alone, at any keep."""
    v = {x.part: x for x in feasibility([ENC, DEC], 4 * MIB)}
    assert not v["dekoder"].feasible
    assert "byudjetdan katta" in v["dekoder"].note


def test_feasibility_reports_int8_sufficiency_on_the_real_machine():
    v = {x.part: x for x in feasibility([ENC, DEC], 24 * MIB)}
    assert v["enkoder"].feasible and v["dekoder"].feasible
    assert v["dekoder"].after_quant == pytest.approx(64 / 16.8 / 4, rel=1e-3)
    assert "INT8 yetarli" in v["dekoder"].note


def test_feasibility_quantifies_the_12_mib_demand():
    """The number the L3=12 experiment was built around: 45% of the FFN."""
    v = {x.part: x for x in feasibility([ENC, DEC], 12 * MIB)}
    assert v["enkoder"].keep_needed == pytest.approx(0.55, abs=0.01)
    assert v["dekoder"].keep_needed == pytest.approx(0.05, abs=0.01)


def test_reachable_keep_negative_when_fixed_part_alone_overflows():
    assert reachable_keep(DEC, 8, 3 * MIB) < 0


def test_required_factor_matches_hand_arithmetic():
    assert required_factor(DEC, 0.7 * 24 * MIB) == pytest.approx(3.81, abs=0.01)
    assert required_factor(ENC, 0.7 * 24 * MIB) == pytest.approx(2.86, abs=0.01)


def test_plan_rejects_bad_inputs():
    with pytest.raises(ValueError):
        plan([], 24 * MIB)
    with pytest.raises(ValueError):
        plan([ENC], 24 * MIB, alpha=0.0)


TAU_RUNGS = (("tau=0.99", 0.830), ("tau=0.97", 0.799), ("tau=0.95", 0.764))


def test_structural_ladder_replaces_the_uniform_fractions():
    p = plan([ENC, DEC], 24 * MIB, structural_ladder=TAU_RUNGS)
    tags = {t.tag for r in p.rungs for t in r.treatments.values() if t.tag}
    assert tags <= {"tau=0.99", "tau=0.97", "tau=0.95"}
    assert tags, "strukturaviy pog'onalar umuman ishlatilmadi"


def test_structural_rungs_appear_in_the_step_labels():
    p = plan([ENC], 24 * MIB, structural_ladder=TAU_RUNGS)
    steps = " ".join(r.step for r in p.rungs)
    assert "tau=0.99" in steps


def test_ladder_stays_monotone_with_named_rungs():
    p = plan([ENC, DEC], 12 * MIB, structural_ladder=TAU_RUNGS)
    sizes = [r.total_bytes for r in p.rungs]
    assert sizes == sorted(sizes, reverse=True)


def test_unsorted_structural_ladder_is_refused():
    """A ladder that is not mildest-first would break the lazy stop rule."""
    with pytest.raises(ValueError, match="tartiblangan"):
        plan([ENC], 24 * MIB,
             structural_ladder=(("a", 0.7), ("b", 0.9)))


def test_full_keep_rung_is_inserted_when_absent():
    p = plan([ENC], 24 * MIB, structural_ladder=TAU_RUNGS)
    assert p.rungs[0].treatments["enkoder"].keep == 1.0
    assert p.rungs[0].treatments["enkoder"].tag == ""


def test_unsupported_part_gets_no_structural_rungs():
    """Rungs are cumulative, so one unbuildable step poisons every rung above
    it -- the planner must leave them out rather than emit them and rely on
    the walk to cope."""
    dec = replace(DEC, structural_supported=False)
    p = plan([ENC, dec], 24 * MIB, structural_ladder=TAU_RUNGS)
    for r in p.rungs:
        assert r.treatments["dekoder"].keep == 1.0
    assert any(r.treatments["enkoder"].keep < 1.0 for r in p.rungs)


def test_excluding_a_part_lets_the_other_go_further():
    """The concrete cost of the old behaviour: encoder rungs beyond the first
    were unreachable because a decoder step sat in front of them."""
    dec = replace(DEC, structural_supported=False)
    with_dec = plan([ENC, DEC], 24 * MIB, structural_ladder=TAU_RUNGS)
    without = plan([ENC, dec], 24 * MIB, structural_ladder=TAU_RUNGS)

    def first_dec_prune(pl):
        return next((r.index for r in pl.rungs
                     if r.treatments["dekoder"].keep < 1.0), None)

    def enc_rungs_before(pl, limit):
        return sum(1 for r in pl.rungs[:limit]
                   if r.treatments["enkoder"].keep < 1.0)

    blocked = first_dec_prune(with_dec)
    assert blocked is not None
    assert enc_rungs_before(with_dec, blocked) == 1
    assert enc_rungs_before(without, len(without.rungs)) == len(TAU_RUNGS)


def test_prunable_bytes_still_reported_when_unsupported():
    """Feasibility must still say what fitting would demand, even where the
    toolchain cannot deliver it."""
    dec = replace(DEC, structural_supported=False)
    v = {x.part: x for x in feasibility([dec], 12 * MIB)}
    assert v["dekoder"].keep_needed == pytest.approx(0.05, abs=0.01)


def test_treatment_carries_its_realisation_tag():
    t = Treatment(bits=8, keep=0.83, tag="tau=0.99")
    assert t.tag == "tau=0.99"
    assert Treatment(bits=8, keep=0.83) != t  # the tag distinguishes them


def test_plan_is_a_cache_plan_with_readable_summary():
    p = plan([ENC, DEC], 24 * MIB)
    assert isinstance(p, CachePlan)
    text = p.summary()
    assert "enkoder" in text and "dekoder" in text and "byudjet" in text
