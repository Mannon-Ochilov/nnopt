"""Tests for the named calibration set.

Two properties matter here and neither is cosmetic. A calibration set must
have an identity that reaches the filenames of what it produces, because
artifacts built from different utterances are different artifacts. And it must
be able to say whether it overlaps an evaluation slice, because scoring a
build on the data that shaped it is the one methodological failure this
pipeline has actually produced.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from calib_utils import CalibSet  # noqa: E402


def test_tag_distinguishes_every_field():
    tags = {CalibSet("validation", 0, 6).tag,
            CalibSet("validation", 6, 6).tag,
            CalibSet("validation", 0, 12).tag,
            CalibSet("test", 0, 6).tag}
    assert len(tags) == 4


def test_tag_is_filename_safe():
    tag = CalibSet("validation", 3, 8).tag
    assert tag == "validation-s3n8"
    assert not set(tag) & set(r'\/:*?"<>| ')


def test_path_follows_the_split():
    assert CalibSet("validation").path.endswith("cv_uz_validation.npz")
    assert CalibSet("test").path.endswith("cv_uz_test.npz")


def test_rejects_unknown_split_and_bad_sizes():
    with pytest.raises(ValueError):
        CalibSet("train")
    with pytest.raises(ValueError):
        CalibSet("validation", n=0)
    with pytest.raises(ValueError):
        CalibSet("validation", skip=-1)


def test_overlap_is_false_across_different_splits():
    """Calibrating on validation and scoring on test never collides."""
    assert not CalibSet("validation", 0, 100).overlaps("test", 0, 100)


def test_overlap_detects_a_shared_prefix():
    calib = CalibSet("validation", 0, 6)
    assert calib.overlaps("validation", 0, 100)
    assert calib.overlaps("validation", 5, 10)


def test_overlap_allows_adjacent_ranges():
    """Touching but disjoint slices are fine -- the check must not be greedy."""
    calib = CalibSet("validation", 0, 6)
    assert not calib.overlaps("validation", 6, 100)
    assert not CalibSet("validation", 300, 6).overlaps("validation", 0, 300)


def test_indices_match_skip_and_n():
    assert list(CalibSet("validation", 4, 3).indices()) == [4, 5, 6]


def test_check_disjoint_raises_only_on_real_overlap():
    from cascade_runner import check_disjoint

    check_disjoint(CalibSet("validation", 0, 6), "test", 300)
    check_disjoint(CalibSet("validation", 300, 6), "validation", 300)
    with pytest.raises(ValueError, match="kesishadi"):
        check_disjoint(CalibSet("validation", 0, 6), "validation", 100)


def test_map_path_separates_budgets_and_calibration_sets():
    from l3_12_cascade import map_path

    seen = {map_path(0, 0.45, "d", CalibSet("validation", 0, 6)),
            map_path(0, 0.30, "d", CalibSet("validation", 0, 6)),
            map_path(0, 0.45, "d", CalibSet("validation", 6, 6)),
            map_path(1, 0.45, "d", CalibSet("validation", 0, 6))}
    assert len(seen) == 4
