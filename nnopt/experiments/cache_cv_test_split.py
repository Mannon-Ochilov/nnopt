"""Cache the Common Voice Uzbek TEST split for a statistically usable WER.

Two problems with the current evaluation set are fixed at once.

Size. Every headline WER comes from 80 utterances, where the bootstrap
interval on the central claim spans [-0.0020, +0.0240]. Differences below
about 0.03 WER are therefore unresolvable, which is wider than most of the
effects being compared -- the "statistically indistinguishable from FP32"
verdict is true but weakly supported. Interval width shrinks as 1/sqrt(n), so
going from 80 to ~600 utterances tightens it by roughly 2.7x.

Split. Calibration and evaluation both came from the VALIDATION split
(utterances 0-11 and 12-91 respectively). Disjoint, but drawn from the same
pool, and validation is not the split the literature reports. Moving
evaluation to TEST keeps calibration where it is and makes the numbers
comparable, at no cost.

The waveform is stored resampled to 16 kHz mono next to its reference text,
so later runs never touch the network -- the streaming path dropped
mid-run often enough that it is not an acceptable benchmark dependency.
"""

import io
import os

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

OUT = "models/_calib_cache/cv_uz_test.npz"
DATASET = "yakhyo/mozilla-common-voice-uzbek"
N_SAMPLES = int(os.environ.get("N_SAMPLES", "600"))
TARGET_SR = 16000
MAX_SECONDS = 30.0          # Whisper's receptive field; longer clips are cropped anyway


def open_split():
    """Prefer the test split; fall back to validation if it is absent.

    Which split was actually used is printed and stored, because a silent
    fallback would quietly undo the comparability this script exists for.
    """
    for split in ("test", "validation"):
        try:
            ds = load_dataset(DATASET, split=split, streaming=True)
            print(f"split: {split}")
            return ds, split
        except (ValueError, KeyError) as exc:
            print(f"  '{split}' ochilmadi ({type(exc).__name__}), keyingisi sinaladi")
    raise RuntimeError("na 'test', na 'validation' split ochildi")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds, split = open_split()
    ds = ds.cast_column("audio", Audio(decode=False))

    waves, texts, skipped = [], [], 0
    for row in ds:
        if len(waves) >= N_SAMPLES:
            break
        data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            data = resample_poly(data, TARGET_SR, sr)
        if len(data) > MAX_SECONDS * TARGET_SR or not str(row["text"]).strip():
            skipped += 1
            continue
        waves.append(data.astype(np.float32))
        texts.append(row["text"])
        if len(waves) % 50 == 0:
            print(f"  {len(waves)}/{N_SAMPLES}", flush=True)

    np.savez_compressed(
        OUT,
        texts=np.array(texts, dtype=object),
        lengths=np.array([len(w) for w in waves], dtype=np.int64),
        audio=np.concatenate(waves),
        split=np.array([split], dtype=object),
    )
    dur = sum(len(w) for w in waves) / TARGET_SR
    print(f"\nsaqlandi: {OUT}")
    print(f"  {len(waves)} namuna ({skipped} tashlandi), jami {dur/60:.1f} daqiqa audio")
    print(f"  hajm: {os.path.getsize(OUT)/1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
