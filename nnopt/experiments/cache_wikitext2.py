"""Cache the WikiText-2 test split for standard perplexity evaluation.

Every result so far used Uzbek Common Voice transcripts, which keeps the
language domain consistent with the ASR work but makes the language-model
numbers incomparable to the literature: a perplexity of 230 cannot be read
against published Llama results without a shared benchmark.

WikiText-2 is the benchmark GPTQ, AWQ, SVD-LLM and SliceGPT all report, so
adding it makes our numbers directly checkable against those papers.

The standard protocol (as used in those works) joins the test split with
newlines, tokenizes once, and evaluates non-overlapping segments of a fixed
length; segment length matters for comparability, so 2048 is kept.
"""

import os

import numpy as np
from datasets import load_dataset

OUT = "models/_calib_cache/wikitext2_test.npz"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("wikitext-2-raw-v1 yuklanmoqda...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    # Calibration must not come from the evaluation split. The published
    # protocol draws calibration segments from train (or C4), so train is
    # cached too; only the first slice is kept since GPTQ needs a few
    # hundred thousand characters at most.
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib = "\n\n".join(train["text"][:20000])
    np.savez_compressed(OUT, text=np.array([text], dtype=object),
                        calib=np.array([calib], dtype=object))
    print(f"saqlandi: {OUT}")
    print(f"  test  : {len(test)} qator, {len(text):,} belgi")
    print(f"  calib : {len(calib):,} belgi (train dan, test bilan kesishmaydi)")
    print(f"  hajm  : {os.path.getsize(OUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
