"""Fetch mBERT and export it to ONNX for the same analysis pipeline.

The question this opens: does the method transfer beyond Whisper? mBERT is
a useful second object because it differs on every axis that matters here --
encoder-only, 12 layers instead of 24, hidden 768 vs 1024, FFN width 3072
vs 4096, and a text (not speech) input distribution.

Calibration and evaluation text reuse the Uzbek Common Voice transcripts
already cached for the ASR work, so the language domain stays constant and
no new corpus download is needed.
"""

import os

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "bert-base-multilingual-cased"
OUT_DIR = "models/mbert_onnx"
SEQ_LEN = 128


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    onnx_path = f"{OUT_DIR}/model.onnx"
    if os.path.exists(onnx_path):
        print(f"allaqachon mavjud: {onnx_path}")
        return

    print(f"yuklanmoqda: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    model.eval()
    tok.save_pretrained(OUT_DIR)

    dummy = tok("Bu sinov matni", return_tensors="pt", padding="max_length",
                max_length=SEQ_LEN, truncation=True)
    print("ONNX ga eksport qilinmoqda...")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
        # torch 2.13 defaults to the dynamo exporter, which needs onnxscript;
        # the TorchScript path is enough for a static BERT graph and keeps
        # the dependency set unchanged.
        dynamo=False,
    )
    size = os.path.getsize(onnx_path) / 1024**2
    print(f"saqlandi: {onnx_path}  ({size:.0f} MiB)")

    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    mm = [n for n in m.graph.node if n.op_type == "MatMul"]
    print(f"graf: {len(m.graph.node)} tugun, {len(mm)} MatMul")


if __name__ == "__main__":
    main()
