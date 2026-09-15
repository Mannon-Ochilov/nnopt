"""One-off: pull the first N Common Voice Uzbek validation samples down to a
local .npz so every later experiment is network-independent. The HF
streaming path kept dropping mid-run (DNS/read timeouts), which is not an
acceptable dependency for a benchmark that has to be reproducible.

Stores the resampled 16 kHz mono waveform + reference text per sample.
"""

import io
import os

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

OUT = "models/_calib_cache/cv_uz_validation.npz"
# 24 was enough while calibration was the only consumer. WER/CER evaluation
# needs far more: with 8 held-out utterances the metric could not separate
# 0.0729 from 0.0729 (README Sec 8.3.11-D), so every "no difference" verdict
# was unresolvable rather than measured.
N_SAMPLES = 120
TARGET_SR = 16000


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    waves, texts = [], []
    for i, row in enumerate(ds):
        if i >= N_SAMPLES:
            break
        data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            data = resample_poly(data, TARGET_SR, sr)
        waves.append(data.astype(np.float32))
        texts.append(row["text"])
        print(f"  cached {i+1}/{N_SAMPLES}  ({len(data)/TARGET_SR:.1f}s)", flush=True)

    np.savez_compressed(
        OUT,
        texts=np.array(texts, dtype=object),
        lengths=np.array([len(w) for w in waves], dtype=np.int64),
        audio=np.concatenate(waves),
    )
    print(f"\nwrote {OUT}: {N_SAMPLES} samples, {os.path.getsize(OUT)/1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
