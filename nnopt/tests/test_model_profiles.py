"""Tests for the model-profile layer.

The framework was written against Whisper and assumed it in three places: two
parts, those builders, and a metric where lower is better. A profile supplies
all three. The third is the one worth guarding hardest -- mBERT is scored by
accuracy, so a budget rule written for word error rate would accept exactly
the rungs it exists to reject, and nothing about that failure would look wrong
in the output.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from model_profiles import PROFILES, PartSource, get_profile  # noqa: E402
from optimize import resolve_budget  # noqa: E402


def test_all_three_models_are_registered():
    assert set(PROFILES) == {"whisper", "mbert", "llama"}


def test_llama_derives_its_spec_without_a_graph():
    """It has no ONNX export, so the sizes come from the architecture's own
    config. The shares are what the planner acts on, so both are checked
    against the shapes rather than against a remembered byte count."""
    p = get_profile("llama")
    src = p.parts()[0]
    if not os.path.exists(os.path.join(src.path, "config.json")):
        pytest.skip("open_llama_3b yuklanmagan")
    s = p.spec(src)
    h, m, n = 3200, 8640, 26
    assert s.n_layers == n
    assert s.prunable_bytes == 3 * h * m * 4          # gate, up, down
    assert s.per_layer_bytes == 4 * h * h * 4 + s.prunable_bytes
    assert s.reuse == 1                                # batch-1 autoregressive


def test_llama_ladder_offers_criterion_endorsed_rungs_before_forced_ones():
    """The first ladder here listed only 10/20/30%, which sits entirely ABOVE
    what the criterion endorses on this model (0.6-7.3%) -- so every rung
    offered was a forced one. The mildest rung must be one the criterion
    itself asks for."""
    lad = get_profile("llama").structural_ladder()
    tau_rungs = [(t, k) for t, k in lad if t.startswith("tau=")]
    assert tau_rungs, "tau pog'onalari yo'q"
    assert lad[0] == tau_rungs[0], "zinapoya majburiy pog'ona bilan boshlanadi"
    # Every tau rung is milder than every forced one.
    forced = [k for t, k in lad if "majburiy" in t]
    assert min(k for _, k in tau_rungs) > max(forced)


def test_llama_prunes_before_it_quantizes(monkeypatch):
    """Order is load-bearing, not conventional. Removing a channel folds its
    contribution into the survivors and widens the per-row range 188x
    (Sec 4.4); the measured result is that the quantizer's compensation
    absorbs that residual. Quantizing first perturbs weights the quantizer
    already fitted, which is a different and worse pipeline -- and this
    function shipped that way until it was caught."""
    import model_profiles as mp

    calls = []
    fake = type("M", (), {"eval": lambda s: None,
                          "state_dict": lambda s: {}})()
    monkeypatch.setattr(mp, "quantize_ffn_int8",
                        lambda *a, **k: calls.append("quant"))
    import llama_structural_refusal as lsr
    monkeypatch.setattr(lsr, "prune_model",
                        lambda *a, **k: calls.append("prune") or [])
    import torch
    monkeypatch.setattr(torch, "save", lambda *a, **k: None)
    monkeypatch.setattr(type(get_profile("llama")), "_calib_batches",
                        staticmethod(lambda n: torch.zeros(1, 4,
                                                           dtype=torch.long)))
    # The real artifact may exist on disk (the walk builds it), and a cached
    # early return would make this test observe nothing. Order is what is
    # under test, not caching.
    import os.path
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    import transformers
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(lambda *a, **k: fake))

    get_profile("llama").build("dekoder-stek", 8, 0.8, "20% kanal (majburiy)",
                               None)
    assert calls == ["prune", "quant"], calls


def test_llama_calibration_size_reaches_the_artifact_name():
    """Same failure the tau filenames and the mBERT slice both had: the
    channels kept and the scales fitted depend on how much calibration text
    was used, so two amounts must not share a file. Calibration size is not
    cosmetic on this model -- quadrupling it moved GPTQ from worst to best
    at INT4."""
    from calib_utils import CalibSet
    p = get_profile("llama")
    assert p._calib_segments(CalibSet("validation", 0, 2)) == 2
    assert p._calib_segments(CalibSet("validation", 0, 8)) == 8
    # A missing set falls back rather than producing 0 rows; a zero-sized one
    # cannot reach here at all, because CalibSet refuses it on construction.
    assert p._calib_segments(None) >= 1
    with pytest.raises(ValueError):
        CalibSet("validation", 0, 0)


def test_llama_refuses_more_calibration_than_the_corpus_holds(monkeypatch):
    """Asking for more than exists would silently return fewer rows, and the
    artifact would then carry a name claiming an amount it never saw."""
    import model_profiles as mp
    p = get_profile("llama")
    monkeypatch.setattr(type(p), "_available_calib_segments",
                        staticmethod(lambda: 4))
    from calib_utils import CalibSet
    with pytest.raises(ValueError, match="kalibrlash segmenti"):
        p.check_data_split(CalibSet("validation", 0, 99), "test", 24)
    p.check_data_split(CalibSet("validation", 0, 4), "test", 24)


def test_llama_margin_lives_in_nll_space_not_perplexity_space():
    """The walk resamples per-segment NLL but the ceiling is a perplexity.
    Read naively, a 5% relative budget at base 7.5466 is 0.3773 in
    perplexity units — about 8x looser than the ln(1.05) = 0.0488 it means
    in NLL units — and the walk measurably accepted an over-budget rung
    (8.0687 against a 7.9240 ceiling) before this conversion existed."""
    import math
    p = get_profile("llama")
    base, ceiling = 7.5466, 7.5466 * 1.05
    m = p.paired_margin(base, ceiling)
    assert m == pytest.approx(math.log(1.05))
    # The measured over-budget rung must now fail the rule.
    nll_delta_of_forced_10pct = math.log(8.0687 / base)
    assert nll_delta_of_forced_10pct > m
    # And an accepted rung must still pass.
    assert math.log(7.6001 / base) < m


def test_llama_refuses_a_bit_width_it_has_no_weights_for():
    """INT8 and FP32 are cached artifacts; anything else would have to be
    invented, and inventing it silently is how a wrong number gets a table."""
    p = get_profile("llama")
    with pytest.raises(ValueError, match="INT4"):
        p.build("dekoder-stek", 4, 1.0, "", None)


def test_unknown_model_is_refused():
    with pytest.raises(ValueError, match="noma'lum model"):
        get_profile("gpt-9")


def test_metric_direction_differs_between_the_two():
    assert get_profile("whisper").higher_is_better is False
    assert get_profile("mbert").higher_is_better is True


def test_absolute_budget_follows_the_metric_direction():
    """0.02 worse is a HIGHER error rate and a LOWER accuracy."""
    wer_cap, _ = resolve_budget(0.1761, wer_delta=0.02)
    acc_cap, _ = resolve_budget(0.2656, wer_delta=0.02, higher_is_better=True)
    assert wer_cap == pytest.approx(0.1961)
    assert acc_cap == pytest.approx(0.2456)


def test_relative_budget_follows_the_metric_direction():
    wer_cap, _ = resolve_budget(0.2000, wer_eps=0.10)
    acc_cap, _ = resolve_budget(0.2000, wer_eps=0.10, higher_is_better=True)
    assert wer_cap == pytest.approx(0.22)
    assert acc_cap == pytest.approx(0.18)


def test_budget_text_names_the_metric_and_relation():
    _, text = resolve_budget(0.2656, wer_delta=0.02, metric="aniqlik",
                             higher_is_better=True)
    assert "aniqlik" in text and ">=" in text
    _, text = resolve_budget(0.1761, wer_delta=0.02, metric="WER")
    assert "WER" in text and "<=" in text


def test_whisper_parts_mark_the_decoder_unbuildable():
    parts = {p.name: p for p in get_profile("whisper").parts()}
    assert parts["enkoder"].structural_supported is True
    assert parts["dekoder"].structural_supported is False
    assert parts["enkoder"].reuse > parts["dekoder"].reuse


def test_mbert_is_a_single_part_model():
    parts = get_profile("mbert").parts()
    assert [p.name for p in parts] == ["enkoder"]
    assert parts[0].structural_supported is True


def test_ladders_are_mildest_first_for_every_profile():
    for name in PROFILES:
        keeps = [k for _, k in get_profile(name).structural_ladder()]
        assert keeps == sorted(keeps, reverse=True), name
        assert all(0.0 < k <= 1.0 for k in keeps), name


def test_every_profile_owns_a_data_split_check():
    """Skipping the check for a model whose data is split elsewhere would
    remove a safeguard that has caught a real failure, so each profile must
    provide one rather than opt out."""
    for name in PROFILES:
        assert callable(getattr(get_profile(name), "check_data_split", None)), name


def test_whisper_split_check_rejects_an_overlap():
    from calib_utils import CalibSet
    p = get_profile("whisper")
    with pytest.raises(ValueError, match="kesishadi"):
        p.check_data_split(CalibSet("validation", 0, 6), "validation", 100)
    p.check_data_split(CalibSet("validation", 100, 6), "validation", 100)


def test_mbert_calibration_slice_reaches_the_artifact_name():
    """Channels kept depend on the calibration text, so two slices must not
    produce the same filename -- the failure the tau filenames already had."""
    from calib_utils import CalibSet
    p = get_profile("mbert")
    a = p._calib_slice(CalibSet("validation", 0, 400))
    b = p._calib_slice(CalibSet("validation", 400, 400))
    assert a[2] != b[2]
    assert a[0] == 0 and b[0] == 400


def test_mbert_evaluation_never_overlaps_its_calibration():
    """However the slice moves, evaluation starts after calibration ends."""
    from mbert_task_metric import load_texts
    for skip, n in ((0, 400), (200, 100), (1000, 50)):
        calib, ev = load_texts(skip, n, n_eval=200)
        assert not (set(calib) & set(ev)), (skip, n)


def test_part_source_carries_what_the_planner_needs():
    s = PartSource("x", "m.onnx", {"batch": 1}, 128)
    assert s.structural_supported is True    # default: assume buildable
    assert s.reuse == 128
