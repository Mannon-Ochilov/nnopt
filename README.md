# A Cache-Aware Compression Framework for Transformers on CPU-Class Hardware

Reference implementation and measurement code for *"A Cache-Aware Compression
Framework: Deriving the Compression Target and Stopping Rule for Transformers
on CPU-Class Hardware."*

Post-training compression usually begins with a rate someone chose in
advance — a bit width, a rank, a pruning percentage — and the target device
enters only afterwards, as the benchmark. This framework inverts that order.
The compression requirement is **derived from the target machine's cache
budget**, and a separate, task-level accuracy gate decides how much further it
is safe to go. The two questions are kept apart:

1. How much must this model shrink for *this* hardware?
2. Once that minimum is met, how much more can be removed without losing
   accuracy?

The goal is not to make the model as small as possible, but to fit the
operator's working set into the processor's effective L3 budget and then stop.

## The derived target

```
budget  =  α · M_cache                 α = 0.7  (30% left to activations,
                                                 KV state, other processes)
ρ_i     =  max(1, M_eff,i / budget)    minimum reduction operator i must reach
```

On the Intel Tiger Lake H platform used in the paper (L3 = 24 MiB, budget
16.8 MiB), Whisper-medium requires **3.81×** for a decoder layer and **2.86×**
for an encoder layer. INT8 supplies roughly 4×, so in both cases the hardware
minimum is met by quantization alone — but that is the hardware minimum, not
the stopping point.

The cascade distinguishes three cases: FP32 already fits; final INT8 makes it
fit; INT8 is insufficient and structural reduction must run first. In every
case **INT8 is the last stage, never the first.** Structural work precedes it
because least-squares compensation redistributes weight mass, and the
quantization scales must be fitted to the resulting distribution.

## Structural reduction is a merge, not a mask

A channel is not zeroed. Channels with near-identical functional responses on
calibration data are identified, the removed channel's output contribution is
transferred to its representative by least squares, and only then is it
physically deleted — so the FFN's intermediate width and both projection
matrices actually shrink.

This creates a downstream requirement. Because the removed channel's weight
mass is folded into the representative, the row-wise dynamic range becomes
sharply uneven: measured spread rises from **9.6× to 188.4×**. Per-tensor INT8
is then no longer adequate and per-channel quantization becomes necessary. A
2×2 experiment shows the quality cost of structural reduction also depends on
which quantizer follows it.

## Results

**Whisper-medium encoder, Uzbek ASR.** Calibration responses marked 17.1% of
FFN channels as functionally redundant. Removing them with compensation cut
encoder memory a further 11% beyond GPTQ-only (300 → 267 MiB) and reduced
execution time by about 3%, with no statistically distinguishable WER change:

```
ΔWER = -0.0014,  95% CI [-0.0111, +0.0096]   (300 test utterances)
```

**Whole model.**

| Configuration | Size | Compression | WER |
|---|---|---|---|
| FP32 | 2915 MiB | 1.0× | 0.1761 |
| **cascade** | **705 MiB** | **4.14×** | **0.1833** |
| aggressive | 546 MiB | 5.34× | 0.6101 |

The accuracy gate is what separates the middle row from the last: pushing past
the derived target collapses quality.

**Hardware counters.** Memory stalls fall **2.41×** and L3 pressure from
**2.4% to 1.0%**, against a **1.91×** reduction in total execution time — the
memory component shrinks faster than wall time, so the model becomes less
memory-bound rather than merely smaller.

## L3 is a resource criterion, not a latency cliff

Tested directly rather than assumed. Crossing the 16.8 MiB effective budget
changes execution time by only **1.08×** on an optimized blocked INT8 GEMM,
because blocking keeps *tiles* resident rather than whole matrices. On a naive
block-streaming kernel the same crossing costs **1.56× on average and up to
2.3× at 64 MiB**.

The residency effect is therefore a property of the runtime kernel's blocking
and reuse strategy, not a universal property of the model. Accordingly the
cache budget is used to size the required compression, not as a latency
threshold in the objective.

## Redundancy takes different forms in different architectures

| Model | FFN structure | What the diagnostic finds | Branch chosen |
|---|---|---|---|
| Whisper encoder | GELU | high pairwise functional similarity | compensated structured pruning |
| mBERT | GELU | very little; forced pruning degrades quality | final INT8, stop |
| open_llama_3b | SwiGLU (gated) | low pairwise collinearity, distributed redundancy remains | activation-aware low-rank |

The generalizable part of the framework is therefore not one pruning
algorithm, but the diagnostic and decision mechanism that selects the
appropriate reduction for a given model.

## Reproducing

```bash
python -m venv .venv
.venv/bin/pip install -e nnopt

python nnopt/experiments/wer_master_table.py     # all measured WER results
python nnopt/experiments/global_tau_range.py     # per-layer redundancy profile
python nnopt/experiments/l3_12_feasibility.py    # derived targets per cache size
```

Python ≥ 3.10 (measured on 3.12.8; onnxruntime 1.28, numpy 2.4, torch
2.13+cpu, transformers 4.57). Platform: Intel i7-11850H, L3 = 24 MiB,
single-threaded unless stated; counter collection via Intel VTune.

Model checkpoints and audio are not included. Scripts that need them print the
expected path and exit if it is absent.

Confidence intervals are percentile bootstrap over utterances (2000
resamples). Comparisons between two configurations use the **paired**
interval, since both are scored on the same utterances and pairing removes the
between-utterance variance they share; overlapping marginal intervals are not
evidence of equivalence.

## Measurement logs

`nnopt/logs/` holds the raw record — every configuration measured, every field
recorded, with the protocol and machine stated in each file's header.
Regenerate with `python nnopt/experiments/make_logs.py`.

Where a quantity was measured more than once, the run the article reports and
any later re-measurement sit next to each other, each stating its protocol:

```
latency_library_blocked.log        run 1, blocked design — the run in Table 8
latency_library_interleaved.log    run 2, interleaved re-measurement
redundancy_reported.log            the runs behind Tables 11 and 12
redundancy_masked.log              same quantity, padding masked out
criterion_comparison.log           removal criteria at an equal budget
criterion_comparison_masked.log    same comparison, padding masked out
wer_all_configurations.log         every scored variant, grouped by eval set
hardware_counters.log              VTune uarch-exploration and LLC miss counts
```

Two quantities were measured twice. The latency library differs on one row
because the first run was blocked by configuration and the second interleaved.
The redundancy diagnostics differ because calibration activations were
originally captured without an attention mask — Whisper pads every clip to a
30 s window and mBERT to 128 tokens, so padded positions entered the response
vectors. Both runs are published in each case: the disagreement and the design
that produced it are part of the record, not something to resolve silently in
favour of the later number.

## Layout

```
nnopt/nnopt/          profiler, calibrator, grouping, quantizer, cascade
nnopt/experiments/    measurement scripts and their results_*.json
nnopt/logs/           raw measurement logs
nnopt/tests/          unit and integration tests
portable_benchmark/   run the configurations on another machine
```

`portable_benchmark/` is self-contained: two dependencies, no network, no
audio corpus, a few minutes per machine. Because the compression target is
derived from the cache, the question that matters is whether the ordering
holds on hardware with a different L3 — see its README.

## Citation

```bibtex
@article{ochilov_cache_aware,
  title  = {A Cache-Aware Compression Framework: Deriving the Compression
            Target and Stopping Rule for Transformers on CPU-Class Hardware},
  author = {Ochilov, Mannon and Khujayarov, Ilyos and Kholdorov, Shohruh
            and Narzullayev, Oybek and Musaev, Muhammadjon},
  year   = {2026}
}
```
