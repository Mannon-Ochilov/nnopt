"""Finish the cascade track: convert the FP32-on-INT8-grid cascade model to
external-data form, drop the huge single-file copy, then run ORT
quantize_dynamic to get REAL MatMulInteger kernels.

Split out from whole_network_table.py because the first attempt died on a
full disk: a 1.5 GiB single-file proto plus ORT's temporary "-inferred"
copy did not fit. External-data form keeps the proto tiny, and each step
here frees the previous artifact before the next one is written.
"""

import os
import shutil

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

SRC = "models/_whole_net/dec_cascade_grid.onnx"
EXT_DIR = "models/_whole_net_ext"
EXT = f"{EXT_DIR}/dec_cascade_grid.onnx"
DST = f"{EXT_DIR}/dec_cascade_int8.onnx"


def free_gb():
    return shutil.disk_usage("C:\\").free / 1024**3


def main():
    print(f"free: {free_gb():.2f} GB")
    os.makedirs(EXT_DIR, exist_ok=True)

    if not os.path.exists(EXT):
        print("step 1: re-save with external data...")
        m = onnx.load(SRC)
        onnx.save(m, EXT, save_as_external_data=True, all_tensors_to_one_file=True,
                  location="weights.bin", size_threshold=1024)
        del m
        print(f"  proto {os.path.getsize(EXT)/1024**2:.1f} MiB, "
              f"weights {os.path.getsize(f'{EXT_DIR}/weights.bin')/1024**2:.0f} MiB, free: {free_gb():.2f} GB")

    if os.path.exists(SRC):
        print("step 2: drop the single-file copy...")
        os.remove(SRC)
        print(f"  free: {free_gb():.2f} GB")

    print("step 3: quantize_dynamic (lossless on our grid -> real INT8 kernels)...")
    quantize_dynamic(EXT, DST, weight_type=QuantType.QInt8, use_external_data_format=True)
    print(f"  done, free: {free_gb():.2f} GB")
    for f in sorted(os.listdir(EXT_DIR)):
        print(f"    {f:32s} {os.path.getsize(os.path.join(EXT_DIR, f))/1024**2:8.1f} MiB")


if __name__ == "__main__":
    main()
