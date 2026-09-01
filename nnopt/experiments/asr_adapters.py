"""The seam between the framework and a particular ASR model.

Most of the framework does not care what model it is given. The cache target,
the miss objective, the candidate ladder and the structural search all work in
bytes and graph topology, and `nnopt.profiler.blocks` finds the reducible
block by structure rather than by operator name -- so a transformer ASR model
this code has never seen is PLANNED for correctly.

Scoring is the one place where that generality has to stop, because turning
audio into a hypothesis needs a feature front end, an output vocabulary and a
decoding loop, and those are properties of a model rather than of a graph. The
design answer is a narrow interface: three methods, listed below. Anything the
framework needs from a model in order to put a number on it goes through them,
and nothing else in the pipeline touches Whisper-specific code.

Only Whisper is implemented, because Whisper is the model this work measures.
That is a deliberate scope, not an oversight: shipping a half-written adapter
for an architecture with no data behind it would imply a coverage claim the
experiments do not support. A model without an adapter can still be planned
for -- the useful half, and it works today -- and the runner says plainly that
it cannot be scored rather than producing a number that means nothing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AsrAdapter(Protocol):
    """What the runner needs from a model in order to score it.

    Implementing these three methods (plus `has_decoder`) is the entire cost
    of bringing a new ASR architecture into the framework; nothing else in the
    planner or the runner has to change. `has_decoder = False` covers
    encoder-only models, which the planner already treats as a one-part model.
    """

    #: True when the model has a separate autoregressive decoder graph.
    has_decoder: bool

    def encoder_feed(self, wave: np.ndarray) -> dict:
        """Input dict for one utterance, ready for the encoder session."""

    def transcribe(self, dec_session, encoder_output: np.ndarray) -> str:
        """Text from the encoder output; `dec_session` is None when
        `has_decoder` is False."""

    def normalize(self, text: str) -> str:
        """Scoring-time normalisation, applied to both reference and
        hypothesis."""


class WhisperAdapter:
    """Encoder-decoder Whisper with greedy decoding, as measured in this work."""

    has_decoder = True

    def __init__(self, model_dir=None, language="uz", task="transcribe"):
        from calib_utils import MODEL_DIR, TARGET_SR
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        self.sr = TARGET_SR
        self.fe = WhisperFeatureExtractor.from_pretrained(model_dir or MODEL_DIR)
        self.tok = WhisperTokenizer.from_pretrained(model_dir or MODEL_DIR)
        self.prompt_ids = [t for _, t in self.tok.get_decoder_prompt_ids(
            language=language, task=task)]

    def encoder_feed(self, wave):
        feats = self.fe(wave, sampling_rate=self.sr,
                        return_tensors="np").input_features
        return {"input_features": feats.astype(np.float32)}

    def transcribe(self, dec_session, encoder_output):
        from wer_cer_whole_network import greedy_decode

        ids = greedy_decode(dec_session, encoder_output, self.prompt_ids)
        return self.tok.decode(ids, skip_special_tokens=True)

    def normalize(self, text):
        from wer_cer_whole_network import normalize

        return normalize(text)


ADAPTERS = {"whisper": WhisperAdapter}


def get_adapter(name, **kwargs):
    if name not in ADAPTERS:
        raise ValueError(
            f"noma'lum adapter {name!r}; mavjud: {sorted(ADAPTERS)}. "
            f"Yangi arxitektura uchun AsrAdapter interfeysini bajaring "
            f"(encoder_feed, transcribe, normalize).")
    return ADAPTERS[name](**kwargs)
