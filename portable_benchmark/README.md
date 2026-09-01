# Portable benchmark — second platform

The framework derives its compression target from the target machine's cache,
so the claim that matters is not how one configuration behaves on the machine
it was developed on, but whether the ordering holds elsewhere. This kit
measures that on any CPU, in a few minutes, without a network connection or an
audio corpus.

Run it on three machines with **different L3 sizes** and the cross-platform
table can be filled.

## Requirements

- Python 3.9+ (tested to 3.12)
- `pip install onnxruntime numpy`
- ~2.5 GB of disk for the bundled models

No internet access is needed at run time.

## Running

```bash
cd portable_benchmark
python benchmark.py --quick     # 1-2 min: check everything works
python benchmark.py             # full measurement, 7 rounds
```

Results are written to `results/benchmark_<machine>_<date>.json` and printed
as a table. **Return that JSON file.**

Optional: `python benchmark.py --threads 4` also measures a multi-threaded
configuration. The default is a single thread, matching the article's
protocol.

## Conditions that matter

| Condition | Why |
|---|---|
| Close other heavy applications (browser, updates) | machine drift distorts the result |
| On a laptop: **plugged in**, power saving off | on battery the CPU clocks down |
| Let the machine cool after sustained load | thermal throttling costs 20–30% |
| Leave the machine idle during the run | |

Configurations are measured **interleaved** (A-B-C-A-B-C), not one at a time
to completion. Blocking a benchmark by configuration lets thermal and load
drift be read as a size effect; that produced a spurious result once in this
project, and the interleaved design is the correction.

## Which machines are worth measuring

The development machine has 24 MiB of L3. Cache sizes as different from that
as possible are the most informative:

| Class | Example | What it adds |
|---|---|---|
| Small cache, ARM | Raspberry Pi 5 (2 MiB) | the cascade's derived plan changes — the most valuable point |
| Small cache, x86 | Intel N100 / mini-PC (6 MiB) | the edge-device class |
| Mid | laptop i5/i7 (8–12 MiB) | the most common class |
| Large | Ryzen / server (32+ MiB) | the upper end |

Three machines are enough, and any of them helps. An old laptop counts — what
matters is that its L3 differs from 24 MiB.

## What is and is not measured

**Measured:** latency (ms), run-to-run spread, speedup ratio, and a machine
passport — CPU, cores, L3, RAM, **memory speed and module count**, OS,
onnxruntime version.

Memory speed is collected because capacity alone cannot be used to read a
memory-bound result: single- and dual-channel DDR4-3200 differ by a factor of
two in peak bandwidth. `memory_peak_GBs` is derived as
`modules × MT/s × 8 bytes` and assumes one channel per populated module — the
usual arrangement, but a ceiling rather than a measurement. On Linux the
memory fields need DMI access (`sudo`); without it they are left empty, like
any other value the platform will not report.

**Not measured:** WER. It is a property of the artifact, not of the machine —
the same ONNX file produces the same transcription anywhere. Each
configuration's WER is already in the script (the article's 300-utterance TEST
figure) and is shown in the table for reference. This is why no audio corpus
ships with the kit and why a run takes minutes rather than hours.

If the L3 size cannot be read on a platform, the field is left **empty** rather
than estimated.

## Expected outcome, written down in advance

- **tau=0.99** should be at least as fast as blind INT8: it is 11% smaller and
  performs less arithmetic.
- The speedup **ratio will differ on another machine**, and that is not a
  contradiction — it depends on the FLOP/memory balance of the platform. It
  will be reported as measured.
- On a small-cache machine (Pi 5, N100) the cascade's derived case changes;
  whether the prediction matches is checked separately.

Whichever way the result falls, it is recorded as measured: the text follows
the result, not the other way round.

## Troubleshooting

| Problem | Fix |
|---|---|
| `onnxruntime not found` | `pip install onnxruntime numpy` |
| `At least two models are required` | check that `models/` copied completely (5 `.onnx` files, ~2.3 GB) |
| Install fails on ARM / Raspberry Pi | `pip install onnxruntime` usually works; if not, run `python3 -m pip install --upgrade pip` and retry |
| FP32 model does not fit in memory | it is skipped automatically — the remaining four are sufficient |
