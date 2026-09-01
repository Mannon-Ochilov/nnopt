# Measurement logs

Raw records, one file per quantity. Each begins with a header naming its
source result file, the date it was recorded, the machine, and the protocol
under which it was taken.

Regenerate all of them from the stored result files:

```bash
python nnopt/experiments/make_logs.py
```

## Which log backs which table

| Article table | Log | Source result file |
|---|---|---|
| 1, 2, 3, 15 — derived targets, cases, sensitivity | `redundancy_reported.log` (Llama block) | `results_llama.json` |
| 4 — compensation and quantization granularity | `wer_all_configurations.log` | `results_prune_perchannel.json` |
| 5 — quantizer × structural reduction (2×2) | `wer_all_configurations.log` | `results_gptq_pruning.json` |
| 6 — whole-model stopping rule | `wer_all_configurations.log` | `results_whole_model_cascade.json` |
| 7 — structural methods at equal budget | `wer_all_configurations.log` | `results_final_wer_testsplit.json` |
| **8 — reference configurations** | **`latency_library_blocked.log`** | `results_latency_library.json` |
| 10 — cache and memory traffic | `hardware_counters.log` | `results_vtune_whole_model.json` |
| **11, 12 — redundancy across architectures** | **`redundancy_reported.log`** | `results_ffn_prune.json`, `results_mbert.json`, `results_llama.json` |
| criterion comparison | `criterion_comparison.log` | `results_mbert_criterion.json` |

## Measured twice

Two quantities have a second run. Both are published, each stating its own
protocol; neither is presented as superseding the other by fiat.

**Latency.** `latency_library_blocked.log` measured each configuration to
completion before moving to the next, with sessions cached between repeats —
this is the run in Table 8. `latency_library_interleaved.log` cycles the
configurations A-B-C-A-B-C. Eleven of twelve rows reproduce within run-to-run
spread; the blind INT8 baseline does not (8658.2 ms against 6981.2 ms), and
every ratio computed against that baseline moves with it. Blocking a benchmark
by configuration lets machine drift be read as a size effect.

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
