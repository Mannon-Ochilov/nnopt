# Measurement logs

Raw records, one file per quantity. Each begins with a header naming its
source result file, the date it was recorded, the machine, and the protocol
under which it was taken.

Regenerate all of them from the stored result files:

```bash
python nnopt/experiments/make_logs.py
```

## Which log backs which table

Table numbers follow the English manuscript.

| Article table | Log | Source result file |
|---|---|---|
| 3, 4, 5, 6, 13 — derived targets, cases, sensitivity | `redundancy_reported.log` (Llama block) | `results_llama.json` |
| 7 — compensation and quantization granularity | `wer_all_configurations.log` | `results_prune_perchannel.json` |
| 8 — quantizer × structural reduction (2×2) | `wer_all_configurations.log` | `results_gptq_pruning.json` |
| 9 — whole-model stopping rule | `wer_all_configurations.log` | `results_whole_model_cascade.json` |
| 10 — structural methods at equal budget | `wer_all_configurations.log` | `results_final_wer_testsplit.json` |
| **11 — reference configurations** | **`latency_library_blocked.log`** (as published) · `latency_library_interleaved.log` (corrected) | `results_latency_library.json` · `..._interleaved.json` |
| 12 — cache and memory traffic | `hardware_counters.log` | `results_vtune_whole_model.json` |
| **14, 15 — redundancy across architectures** | **`redundancy_reported.log`** | `results_ffn_prune.json`, `results_mbert.json`, `results_llama.json` |
| criterion comparison | `criterion_comparison.log` | `results_mbert_criterion.json` |

## Measured twice

Two quantities have a second run. Both are published, each stating its own
protocol; neither is presented as superseding the other by fiat.

**Latency.** `latency_library_blocked.log` measured each configuration to
completion before moving to the next — this is the run in Table 11.
`latency_library_interleaved.log` cycles the configurations A-B-C-A-B-C and is
the corrected record. Eleven of twelve rows reproduce within run-to-run spread
(−0.4% to +6.7%); the blind INT8 baseline does not, at 8658.2 ms against
6981.2 ms. INT8 was measured first in the blocked sequence, and its blocked
value lies 17% above the maximum of seven interleaved repeats (6724–7413 ms),
so it is outside that distribution rather than an extreme sample of it.
Because every speedup in Table 11 is computed against that baseline, the whole
ratio column moves with it: 1.06×–1.14× interleaved against 1.29×–1.40×
blocked. The interleaved value also agrees to within 0.5% with the independent
measurement of the same artifact in Table 8 (6940 ms). Blocking a benchmark by
configuration lets machine drift be read as a size effect.

**Redundancy.** `redundancy_reported.log` holds the runs behind Tables 11 and
12. Their calibration activations were captured without an attention mask, and
both models pad heavily — Whisper to a 30 s window, mBERT to 128 tokens — so
padded positions entered the response vectors. `redundancy_masked.log` repeats
the measurement with the mask applied and over every layer rather than a
sample. `criterion_comparison_masked.log` is the same correction applied to
the criterion comparison.

## Not part of the article

`hardware_counters.log` also carries last-level-cache miss counts
(`results_llc_miss_count.json`) and `redundancy_masked.log` carries the
calibration-size sweep. These were measured after submission and are included
because they bear on the same claims, not because the article reports them.
