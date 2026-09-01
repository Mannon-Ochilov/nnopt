"""Tests for how an accuracy budget is stated.

"Let WER be 0.05 worse" has two readings, and against this work's 0.1761
baseline they differ by 0.04 -- far more than the gap between neighbouring
rungs of the ladder, so picking the wrong one selects a different model. These
tests pin each reading and pin the refusal to guess between them.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from optimize import resolve_budget  # noqa: E402

BASE = 0.1761


def test_absolute_delta_adds_to_the_baseline():
    cap, text = resolve_budget(BASE, wer_delta=0.05)
    assert cap == pytest.approx(0.2261)
    assert "mutlaq" in text


def test_relative_eps_scales_the_baseline():
    cap, text = resolve_budget(BASE, wer_eps=0.05)
    assert cap == pytest.approx(0.1849, abs=1e-4)
    assert "nisbiy" in text


def test_the_two_readings_really_do_differ():
    """If they ever coincided these tests would be proving nothing."""
    a, _ = resolve_budget(BASE, wer_delta=0.05)
    b, _ = resolve_budget(BASE, wer_eps=0.05)
    assert abs(a - b) > 0.03


def test_hard_cap_is_taken_as_given():
    cap, text = resolve_budget(BASE, wer_max=0.20)
    assert cap == 0.20
    assert "+0.0239" in text  # the headroom it leaves, stated explicitly


def test_no_budget_returns_none_rather_than_a_default():
    """A silent default here would let a run proceed with no stopping rule."""
    assert resolve_budget(BASE) == (None, None)


@pytest.mark.parametrize("kwargs", [
    {"wer_delta": 0.05, "wer_eps": 0.05},
    {"wer_max": 0.2, "wer_delta": 0.05},
    {"wer_max": 0.2, "wer_eps": 0.05, "wer_delta": 0.05},
])
def test_more_than_one_form_is_refused(kwargs):
    with pytest.raises(ValueError, match="bitta shaklda"):
        resolve_budget(BASE, **kwargs)


def test_zero_is_a_budget_not_an_absence():
    """0.0 is falsy; a `if not delta` check here would drop a real budget."""
    cap, _ = resolve_budget(BASE, wer_delta=0.0)
    assert cap == pytest.approx(BASE)
