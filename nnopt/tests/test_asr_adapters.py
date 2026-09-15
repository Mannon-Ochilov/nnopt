"""Tests for the ASR adapter seam.

The framework's generality claim is precise, and these tests are what keep it
that way: planning is architecture-independent, scoring goes through a narrow
interface, and a model with no adapter is refused rather than guessed at. Only
Whisper is implemented, so the tests check the CONTRACT and the refusal, not a
second architecture that does not exist here.

WhisperAdapter is not constructed: it loads a feature extractor and a tokenizer
from disk, which makes it an integration dependency rather than a unit one. Its
conformance is checked on the class instead.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from asr_adapters import ADAPTERS, AsrAdapter, WhisperAdapter, get_adapter  # noqa: E402


def test_only_the_measured_architecture_is_registered():
    """A half-written adapter would imply coverage the experiments lack."""
    assert set(ADAPTERS) == {"whisper"}


def test_unknown_adapter_is_refused_with_the_way_forward():
    with pytest.raises(ValueError) as exc:
        get_adapter("wav2vec2")
    msg = str(exc.value)
    assert "noma'lum adapter" in msg
    assert "AsrAdapter" in msg  # says what implementing one requires


def test_whisper_adapter_declares_the_protocol_surface():
    for attr in ("has_decoder", "encoder_feed", "transcribe", "normalize"):
        assert hasattr(WhisperAdapter, attr)
    assert WhisperAdapter.has_decoder is True


def test_protocol_accepts_any_conforming_object():
    """The interface is the generality claim, so it must be satisfiable
    without inheriting from anything in this repository."""

    class Minimal:
        has_decoder = False

        def encoder_feed(self, wave):
            return {"x": np.asarray(wave)[None, :]}

        def transcribe(self, dec_session, encoder_output):
            return "salom"

        def normalize(self, text):
            return text.lower()

    assert isinstance(Minimal(), AsrAdapter)


def test_protocol_rejects_an_incomplete_adapter():
    class MissingTranscribe:
        has_decoder = False

        def encoder_feed(self, wave):
            return {}

        def normalize(self, text):
            return text

    assert not isinstance(MissingTranscribe(), AsrAdapter)


def test_encoder_only_adapters_are_representable():
    """`has_decoder = False` is the whole of encoder-only support; the planner
    already treats such a model as having one part."""

    class EncoderOnly:
        has_decoder = False

        def encoder_feed(self, wave):
            return {"input_values": np.asarray(wave)[None, :]}

        def transcribe(self, dec_session, encoder_output):
            assert dec_session is None
            return ""

        def normalize(self, text):
            return text

    a = EncoderOnly()
    assert isinstance(a, AsrAdapter)
    assert a.transcribe(None, np.zeros((1, 4, 3))) == ""
