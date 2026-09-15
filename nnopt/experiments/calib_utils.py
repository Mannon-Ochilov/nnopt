"""Shared calibration plumbing for the experiment scripts.

Reads audio from the LOCAL cache (models/_calib_cache/, built once by
cache_audio_locally.py) rather than streaming from HuggingFace -- the
streaming path repeatedly died mid-run on DNS/read timeouts, which is not
an acceptable dependency for benchmarks that have to be reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from nnopt.calibrator.activation_capture import (
    ActivationCapture,
    active_mask_from_lengths,
    build_response_vectors,
)

MODEL_DIR = "models/hh"
ENCODER_PATH = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
AUDIO_CACHE = "models/_calib_cache/cv_uz_validation.npz"
SPLIT_CACHE = {"validation": AUDIO_CACHE,
               "test": "models/_calib_cache/cv_uz_test.npz"}
TARGET_SR = 16000
START_OF_TRANSCRIPT = 50258


@dataclass(frozen=True)
class CalibSet:
    """Which utterances calibrate a build, named rather than assumed.

    Every compression decision in this work is a function of the calibration
    data -- which channels look collinear, which scale minimises error, which
    rank is worth keeping. Leaving that data implicit made two problems easy
    to miss: artifacts built from different calibration sets were cached under
    the same name, and a set could silently overlap the utterances the result
    was later scored on. Both are addressed here by making the set an explicit
    value with an identity (`tag`) that callers put in their filenames.
    """

    split: str = "validation"
    skip: int = 0
    n: int = 6
    # Whisper pads every clip to 30 s, so an encoder tensor is 1500 positions
    # wide no matter how long the audio actually was. Common Voice uz clips
    # average 5.1 s, which makes ~83% of those positions padding. Folding them
    # into h_j is what build_response_vectors' own docstring forbids, and it
    # was being done: capture_activations passed active_mask=None. Masking is
    # the correct behaviour and therefore the default; the flag exists so the
    # unmasked runs stay reproducible for the comparison that justifies it,
    # and it is part of `tag` so the two never share an artifact filename.
    masked: bool = True

    def __post_init__(self):
        if self.split not in SPLIT_CACHE:
            raise ValueError(f"noma'lum split {self.split!r}; "
                             f"mavjud: {sorted(SPLIT_CACHE)}")
        if self.n <= 0 or self.skip < 0:
            raise ValueError("n musbat, skip manfiy bo'lmasligi kerak")

    @property
    def path(self) -> str:
        return SPLIT_CACHE[self.split]

    @property
    def tag(self) -> str:
        return f"{self.split}-s{self.skip}n{self.n}" + ("m" if self.masked else "")

    def indices(self) -> range:
        return range(self.skip, self.skip + self.n)

    def overlaps(self, split: str, skip: int, n: int) -> bool:
        """True when this set shares utterances with the given selection."""
        if split != self.split:
            return False
        return not (self.skip + self.n <= skip or skip + n <= self.skip)


def load_audio(skip: int = 0, take: int | None = None, source: str | None = None):
    """Returns (waveforms, texts) from the local cache."""
    z = np.load(source or AUDIO_CACHE, allow_pickle=True)
    flat, lengths, texts = z["audio"], z["lengths"], z["texts"]
    waves, off = [], 0
    for ln in lengths:
        waves.append(flat[off:off + int(ln)])
        off += int(ln)
    pairs = list(zip(waves, list(texts)))[skip: None if take is None else skip + take]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def decoder_feeds(skip: int = 0, take: int | None = None):
    """Real decoder feeds: encoder states from the real encoder, input_ids
    from the reference transcript (teacher forcing)."""
    waves, texts = load_audio(skip, take)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    enc = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    feeds = []
    for wav, text in zip(waves, texts):
        f = fe(wav, sampling_rate=TARGET_SR, return_tensors="np").input_features.astype(np.float32)
        (eh,) = enc.run(None, {"input_features": f})
        ids = np.array(
            [[START_OF_TRANSCRIPT, *prompt_ids, *tok(text, add_special_tokens=False).input_ids]],
            dtype=np.int64,
        )
        feeds.append({"input_ids": ids, "encoder_hidden_states": eh.astype(np.float32)})
    return feeds


def feeds_for(calib: CalibSet):
    """Encoder feeds for a named calibration set."""
    return encoder_feeds(calib.skip, calib.n, source=calib.path)


# 16 kHz audio -> mel hop of 160 samples -> 3000 frames for 30 s; the encoder's
# second conv has stride 2, so 3000 frames become 1500 positions. Hence one
# encoder position per 320 input samples, and everything past the real audio
# is padding.
SAMPLES_PER_POSITION = 320
ENCODER_POSITIONS = 1500


def encoder_positions_for(calib: CalibSet):
    """Real (non-padding) encoder positions per utterance in a calibration set.

    Whisper pads to a fixed 30 s window, so this is the only thing that
    distinguishes a real frame from silence the feature extractor invented.
    """
    z = np.load(calib.path, allow_pickle=True)
    lengths = z["lengths"][calib.indices()]
    return [min(int(np.ceil(int(ln) / SAMPLES_PER_POSITION)), ENCODER_POSITIONS)
            for ln in lengths]


def encoder_feeds(skip: int = 0, take: int | None = None, source: str | None = None):
    waves, _ = load_audio(skip, take, source)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    return [
        {"input_features": fe(w, sampling_rate=TARGET_SR, return_tensors="np").input_features.astype(np.float32)}
        for w in waves
    ]


def capture_activations(model_path, tensor_names, feeds, max_rows=512, seed=0,
                        active_positions=None):
    """Run `feeds` through `model_path`, capturing the named internal
    tensors and turning each into an (rows, num_nodes) calibration matrix.
    Rows are randomly subsampled to `max_rows` to keep downstream SVD /
    grouping tractable.

    `active_positions` gives the real length, in sequence positions, of each
    feed. Supply it for any padded input -- which for a Whisper encoder means
    always, since the 30 s window is fixed. Without it every padding position
    enters h_j as if it were speech, and on Common Voice uz that is roughly
    five of every six rows. Passing None keeps the old behaviour and is only
    correct when the feeds are genuinely unpadded.
    """
    rng = np.random.default_rng(seed)
    cap = ActivationCapture(model_path, tensor_names=list(tensor_names))
    collected = {nm: [] for nm in tensor_names}
    for i, feed in enumerate(feeds, 1):
        npos = None if active_positions is None else active_positions[i - 1]
        for nm, arr in cap.run_batch(feed).items():
            # Only a padded sequence axis needs a mask; a tensor without one
            # (or already at its true length) is passed through untouched.
            mask = None
            if npos is not None and arr.ndim == 3 and arr.shape[1] > npos:
                mask = active_mask_from_lengths([npos] * arr.shape[0],
                                                arr.shape[1])
            collected[nm].append(build_response_vectors(arr, active_mask=mask))
        print(f"  calib {i}/{len(feeds)}", flush=True)
    out = {}
    for nm, chunks in collected.items():
        x = np.concatenate(chunks, axis=1).T
        if x.shape[0] > max_rows:
            x = x[rng.choice(x.shape[0], max_rows, replace=False)]
        out[nm] = x.astype(np.float64)
    return out


def weighted_matmul_profiles(model_path, free_dims):
    from nnopt.profiler.graph_profiler import profile_onnx_model

    profs = profile_onnx_model(model_path, free_dims=free_dims)
    return [p for p in profs if p.weight_initializer is not None]


def weight_for_operator(model, initializers, profile, x_calib):
    """Return W oriented as (m, n) with Y = X @ W.T, matching x_calib's
    column count."""
    w = onnx.numpy_helper.to_array(initializers[profile.weight_initializer]).astype(np.float64)
    if w.shape[1] != x_calib.shape[1]:
        w = w.T
    return w
