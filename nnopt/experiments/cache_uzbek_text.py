"""Cache a larger Uzbek text corpus for LM calibration.

Two problems with reusing the 120 cached ASR transcripts for Llama:

  * Volume. The cache-anchored rank for Llama's FFN is 1487, and the
    rows/rank >= 10 rule established on Whisper (README Sec 8.3.10-A) then
    demands ~15000 calibration rows. 60 transcripts give roughly 5000
    tokens -- a ratio near 3, deep inside the regime where the
    factorization memorizes the calibration set.
  * Padding. The transcripts are single sentences; padding them to a fixed
    length means most captured positions are PAD, which is exactly the
    contamination Fig 2.5 forbids and that was already fixed once for
    Whisper.

Text is tiny compared to audio, so a few thousand sentences cost almost
nothing to fetch. Only the transcript column is pulled -- no audio.
"""

import os

import numpy as np
from datasets import Audio, load_dataset

OUT = "models/_calib_cache/uz_text.npz"
N_SENTENCES = 8000
SPLITS = ["train", "validation", "test"]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    texts = []
    for split in SPLITS:
        if len(texts) >= N_SENTENCES:
            break
        try:
            ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split=split,
                              streaming=True)
            # decode=False keeps the audio bytes opaque: any attempt to decode
            # pulls in torchcodec, whose shared library will not load in this
            # environment. This is the same workaround used by
            # cache_audio_locally.py; only the transcript is read here.
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as exc:
            print(f"  {split}: ochilmadi ({type(exc).__name__}: {str(exc)[:80]})")
            continue
        got = 0
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) > 5:
                texts.append(t)
                got += 1
            if len(texts) >= N_SENTENCES:
                break
        print(f"  {split}: {got} ta jumla (jami {len(texts)})", flush=True)

    if not texts:
        raise SystemExit("matn olinmadi")
    np.savez_compressed(OUT, texts=np.array(texts, dtype=object))
    chars = sum(len(t) for t in texts)
    print(f"\nsaqlandi: {OUT}")
    print(f"  {len(texts)} jumla, {chars:,} belgi, "
          f"{os.path.getsize(OUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
