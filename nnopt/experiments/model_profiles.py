"""What the framework needs to know about a model, gathered in one place.

The planner and the walk were written against Whisper and quietly assumed it
in three places: that a model has an encoder and a decoder, that those
particular builders can produce its variants, and that the quality metric is
one where LOWER is better. None of the three is a property of the method.

A profile supplies exactly what varies:

  parts()               the pieces the cascade decides over, with the graph
                        path, the free dimensions needed to resolve shapes,
                        how often a weight is reused per pass, and whether a
                        structural variant of that piece can be built at all
  structural_ladder()   the operating points the criterion offers, mildest
                        first
  build()               one artifact for one part under one treatment
  evaluate()            a score for a set of artifacts, plus the per-sample
                        outcomes the paired stopping rule needs

`higher_is_better` is separate and load-bearing: mBERT is scored by masked
token accuracy, where a rung that raises the number is an improvement, and a
budget rule written for word error rate would accept exactly the rungs it
should reject.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

MIB = 1024.0 ** 2


@dataclass(frozen=True)
class PartSource:
    """One piece of a model, as the planner needs to see it."""

    name: str
    path: str
    dims: dict
    reuse: int
    structural_supported: bool = True


class WhisperProfile:
    """Encoder-decoder ASR, scored by word error rate."""

    name = "whisper"
    metric = "WER"
    higher_is_better = False
    baseline_hint = 0.1761

    def parts(self):
        from calib_utils import DECODER_PATH, ENCODER_PATH
        return [
            PartSource("enkoder", ENCODER_PATH,
                       {"batch_size": 1, "encoder_sequence_length": 1500},
                       1500, True),
            # The decoder's feed-forward is reduced by low-rank factorization
            # in this codebase, not by channel removal, so no structural rung
            # for it can be built -- and an unbuildable rung would block every
            # rung above it, the encoder's included.
            PartSource("dekoder", DECODER_PATH,
                       {"batch_size": 1, "decoder_sequence_length": 16,
                        "encoder_sequence_length": 1500},
                       1, False),
        ]

    def structural_ladder(self):
        from cascade_runner import structural_ladder
        return structural_ladder()

    def build(self, part, bits, keep, tag, calib):
        from cascade_runner import build_decoder, build_encoder
        if part == "enkoder":
            return build_encoder(bits, keep, 1.0, calib, tag)
        return build_decoder(bits, keep)

    def evaluate(self, paths, n_eval, split, calib=None):
        from cascade_runner import evaluate
        r = evaluate(paths["enkoder"], paths.get("dekoder"), n_eval, split)
        r["score"] = r["wer"]
        r["per_sample"] = r["per_sample_wer"]
        return r

    def check_data_split(self, calib, split, n_eval):
        """Calibration and evaluation utterances are indexed into the same
        cached split here, so they can overlap and the ranges are checked."""
        from cascade_runner import check_disjoint
        check_disjoint(calib, split, n_eval)


class MBertProfile:
    """Encoder-only masked LM, scored by masked-token accuracy.

    Two things differ from Whisper beyond the obvious. The score improves
    upwards, so the budget compares in the other direction. And the criterion
    finds almost no collinear channels in this model at any threshold
    (Sec 4.11), which makes a tau-indexed ladder meaningless here -- the rungs
    would all sit at the same size. The ladder is therefore given in removal
    ratios, which is the honest way to ask "what would forcing it cost", and
    the measured answer is that it costs a great deal (Sec 4.11, table 39).
    """

    name = "mbert"
    metric = "masked-LM aniqligi"
    higher_is_better = True
    baseline_hint = 0.2656

    def parts(self):
        from mbert_analysis import MBERT_ONNX, SEQ_LEN
        return [PartSource("enkoder", MBERT_ONNX,
                           {"batch": 8, "seq": SEQ_LEN}, SEQ_LEN, True)]

    #: Which channels to drop. Measured per operator on this model, the
    #: fluctuation score wins in 12 layers out of 12 against forcing the
    #: cosine criterion down to the budget (Sec 4.11), which is what one
    #: should expect: cosine looks for collinearity and mBERT has almost
    #: none, so forcing it selects among channels it does not itself judge
    #: redundant. End to end the advantage is NOT established -- accuracy
    #: improves but pseudo-perplexity worsens -- so this is the better of two
    #: options rather than a settled one.
    criterion = "fluctuation"

    def structural_ladder(self):
        # The ladder has to be able to express what the cache target asks
        # for. At L3 = 6 MiB this model needs 57% of the FFN removed, and a
        # ladder that stops at 30% would report "no rung fits" for a reason
        # that is a property of the ladder rather than of the model.
        return (("10% kanal", 0.90), ("20% kanal", 0.80),
                ("30% kanal", 0.70), ("40% kanal", 0.60),
                ("50% kanal", 0.50), ("60% kanal", 0.40))

    @staticmethod
    def _calib_slice(calib):
        """How this model reads a CalibSet.

        The split field addresses cached AUDIO and has no meaning for a text
        corpus, so only skip and n are used -- but they ARE used, and the tag
        they produce reaches the artifact name. Ignoring the calibration set
        here, which is what this profile did at first, makes two models built
        from different sentences share a filename and a cached score.
        """
        skip = getattr(calib, "skip", 0) if calib else 0
        n = getattr(calib, "n", None) if calib else None
        n = 400 if not n or n < 50 else n
        return skip, n, f"s{skip}n{n}"

    def build(self, part, bits, keep, tag, calib):
        from mbert_analysis import MBERT_ONNX, MBERT_DIR
        from mbert_task_metric import OUT_DIR, build_int8, build_pruned, load_texts
        from cascade_runner import NotWired
        from transformers import AutoTokenizer

        if bits == 32 and keep >= 1.0:
            return MBERT_ONNX
        if bits != 8:
            raise NotWired(f"mBERT uchun INT{bits} ulanmagan")
        if keep >= 1.0:
            # Dynamic quantization derives activation scales at run time, so
            # this arm uses no calibration text at all and needs no tag.
            return build_int8(MBERT_ONNX, f"{OUT_DIR}/mbert_int8.onnx")
        skip, n, ctag = self._calib_slice(calib)
        tok = AutoTokenizer.from_pretrained(MBERT_DIR)
        calib_texts, _ = load_texts(skip, n)
        return build_pruned(calib_texts, tok, removal=round(1.0 - keep, 2),
                            calib_tag=ctag, criterion=self.criterion)

    def evaluate(self, paths, n_eval, split, calib=None):
        from mbert_task_metric import load_texts, masked_batches, score
        from mbert_analysis import MBERT_DIR
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MBERT_DIR)
        skip, n, _ = self._calib_slice(calib)
        _, eval_texts = load_texts(skip, n)
        batches = masked_batches(tok, eval_texts[:n_eval])
        r = score(paths["enkoder"], batches)
        r["score"] = r["acc"]
        r["per_sample"] = r["hits"]
        r["wer"] = r["acc"]          # the runner's cache field, kept uniform
        r["wer_lo"] = r["wer_hi"] = r["acc"]
        return r

    def check_data_split(self, calib, split, n_eval):
        """This model's texts are split before either side sees them, so the
        guarantee is structural rather than range-based -- but it is asserted
        here rather than assumed. Skipping the check outright would remove a
        safeguard that has caught a real failure in this pipeline (Sec 4.9),
        so the disjointness is verified on the actual returned slices.
        """
        from mbert_task_metric import load_texts
        skip, n, tag = self._calib_slice(calib)
        calib_texts, eval_texts = load_texts(skip, n)
        if set(calib_texts) & set(eval_texts[:n_eval]):
            raise ValueError(f"mBERT: kalibrlash ({tag}) va baholash "
                             f"matnlari kesishadi")
        if n_eval > len(eval_texts):
            raise ValueError(f"mBERT: baholash uchun {n_eval} matn "
                             f"so'raldi, kalibrlashdan ({tag}) keyin "
                             f"mavjudi {len(eval_texts)}")


class LlamaProfile:
    """Decoder-only causal LM, scored by WikiText-2 perplexity.

    The third architecture, and the one where the cascade's low-rank branch is
    actually exercised: Whisper never reaches case 3, so this model carries
    the only end-to-end evidence for that stage (Sec 4.11, table 44).

    Two things about it do not fit the encoder shape. Its feed-forward has TWO
    expanding projections into a gated activation, so one channel decision
    touches three matrices rather than two; and it carries no bias term, which
    the mean-correction the structural stage relies on needs somewhere to go.
    Both are handled in llama_structural_refusal.py, which is where a build
    for this profile comes from.

    The planner's part is the whole decoder stack. Reuse is 1: at batch 1 a
    weight is read once per token and the layer has been evicted before the
    next token returns to it -- the same argument as Whisper's decoder.
    """

    name = "llama"
    metric = "perplexity"
    higher_is_better = False
    baseline_hint = 7.547

    def parts(self):
        from wikitext2_int4 import MODEL_DIR
        # There is no ONNX export of this model; the Llama work runs through
        # PyTorch. The path is the model DIRECTORY, and `spec()` below reads
        # the sizes from its config rather than from a graph.
        return [PartSource("dekoder-stek", MODEL_DIR,
                           {"batch": 1, "seq": 2048}, 1, True)]

    def spec(self, src):
        """Part sizes without a graph, from the architecture's own definition.

        The framework normally reads per-layer bytes off an ONNX graph. That
        route is unavailable here, and the alternative is not a guess: for a
        Llama block the shapes ARE the config. Attention is four hidden x
        hidden projections, the feed-forward is two hidden -> intermediate
        expansions plus one contraction back, and the feed-forward is exactly
        the block the structural stage is allowed to touch. Embeddings and the
        output head are excluded for the same reason they are on the other two
        models: they are not part of the repeated stack the cache target is
        derived over.
        """
        import json
        import os

        from nnopt.cascade.cache_planner import PartSpec

        with open(os.path.join(src.path, "config.json"), encoding="utf-8") as f:
            c = json.load(f)
        h, m = c["hidden_size"], c["intermediate_size"]
        bytes_per = 4                                    # planning in FP32
        attn = 4 * h * h * bytes_per
        ffn = 3 * h * m * bytes_per                      # gate, up, down
        return PartSpec(name=src.name,
                        per_layer_bytes=attn + ffn,
                        n_layers=c["num_hidden_layers"],
                        prunable_bytes=ffn,
                        reuse=src.reuse,
                        structural_supported=src.structural_supported)

    #: Which criterion realises a FORCED rung, i.e. one past where tau stops
    #: endorsing removal. Measured: at a 20% budget the fluctuation score
    #: reaches perplexity 8.906 against the cosine grouping's 9.490
    #: (Sec 4.14). Below the forced rungs the ladder is tau-indexed and this
    #: does not apply -- tau IS the cosine criterion, run without a budget.
    criterion = "fluctuation"

    #: Average share of feed-forward channels the criterion itself endorses
    #: removing at each tau, measured per layer on this model (Sec 4.1,
    #: results_llama.json). The spread is the reason these are here at all:
    #: at tau = 0.90 the first block yields 25.0% and the twenty-first 0.3%,
    #: so the removal is NOT uniform and a ratio cannot express it.
    TAU_POINTS = ((0.99, 0.0063), (0.95, 0.0379), (0.90, 0.0725))

    def structural_ladder(self):
        """Criterion-endorsed rungs first, forced rungs only after.

        The first version of this ladder listed 10/20/30% and nothing else,
        which was wrong in a way worth recording: the criterion's whole range
        on this model sits BELOW 10%, so every rung the framework offered was
        one the criterion did not endorse. That is the forcing the cascade's
        "mildest sufficient change" principle exists to avoid, and on a model
        with little redundancy it is exactly where the principle bites.

        The forced rungs are kept because the cache target here is
        unreachable by a wide margin (28x on a 24 MiB L3), so the walk must
        still be able to say what pushing past the criterion would cost --
        and the measured answer is that it costs a great deal (Sec 4.14).
        """
        rungs = [(f"tau={t:.2f}", 1.0 - share) for t, share in self.TAU_POINTS]
        rungs += [("10% kanal (majburiy)", 0.90),
                  ("20% kanal (majburiy)", 0.80),
                  ("30% kanal (majburiy)", 0.70)]
        return tuple(rungs)

    @staticmethod
    def _calib_segments(calib):
        """How this model reads a CalibSet.

        `split` addresses cached Uzbek AUDIO and is meaningless for WikiText,
        and `skip` has no use here because the train field is read from the
        front. What DOES carry over is `n`: the number of 2048-token
        calibration segments. It is used rather than ignored, and it reaches
        the artifact name, because two models calibrated on different amounts
        of text are different models -- measurably so on this one.
        """
        from wikitext2_int4 import N_CALIB_SEGMENTS
        n = getattr(calib, "n", None) if calib else None
        return N_CALIB_SEGMENTS if not n or n < 1 else int(n)

    @staticmethod
    def _calib_batches(n_seg):
        """The calibration segments both stages see, resolved once.

        Loading them separately in each stage is how the argument stops being
        honoured: the amount would then be whatever each caller's default
        happened to be, and the two stages could disagree without any error.

        Returned as a [n, seq] tensor, which is what capture_group indexes;
        the pruning stage iterates one segment at a time and gets its own view
        at the call site rather than a second copy of the data.
        """
        from transformers import AutoTokenizer
        from wikitext2_int4 import MODEL_DIR, load_segments
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        _, calib = load_segments(tok)
        return calib[:min(n_seg, len(calib))]

    @staticmethod
    def _available_calib_segments():
        import numpy as np
        from transformers import AutoTokenizer
        from wikitext2_int4 import MODEL_DIR, SEQ_LEN, WIKI_CACHE
        z = np.load(WIKI_CACHE, allow_pickle=True)
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        ids = tok(str(z["calib"][0]), return_tensors="pt").input_ids[0]
        return len(ids) // SEQ_LEN

    def build(self, part, bits, keep, tag, calib):
        """One artifact for one treatment, as a saved state dict.

        Structural removal runs FIRST and quantization second, matching
        build_encoder for Whisper. The order is not a convention: removing a
        channel folds its contribution into the survivors, which widens the
        per-row weight range by two orders of magnitude (188x, Sec 4.4), and
        the measured finding there is that the quantizer's error compensation
        absorbs exactly that residual. Quantizing first would hand the
        quantizer the ORIGINAL weights and then perturb them afterwards, so
        the interaction the cascade depends on never happens -- and the same
        2x2 measurement shows what that costs: with the two stages not
        composing, forcing the structural step turns a -0.0014 change into
        +0.0084.

        This function had the two the wrong way round at first. Nothing about
        the output looked wrong; it was simply a different, worse pipeline.
        """
        import os

        import torch

        from llama_structural_refusal import prune_model
        from wikitext2_int4 import MODEL_DIR, WEIGHT_CACHE, apply_weights

        if bits == 32 and keep >= 1.0:
            return MODEL_DIR                       # untouched: load from source
        if bits not in (8, 32):
            raise ValueError(f"Llama uchun INT{bits} kvantlash tayyorlanmagan")

        out_dir = "models/_llama_cascade"
        os.makedirs(out_dir, exist_ok=True)
        # A tau rung and a ratio rung can land on nearly the same total size
        # while removing entirely different channels, so the tag -- not the
        # size -- decides which was built and names the file. Keying on keep
        # alone is how a tau artifact would be silently served for a forced
        # rung.
        tau = float(tag.split("=")[1]) if tag.startswith("tau=") else None
        removal = round(1.0 - keep, 2)
        stem = (f"tau{int(tau*100)}" if tau is not None
                else f"r{int(removal*100)}_{self.criterion}")
        # The calibration size belongs in the name for the same reason it does
        # on the other two profiles: which channels are kept and which scales
        # are fitted both depend on it, so two amounts must not share a file.
        nseg = self._calib_segments(calib)
        path = f"{out_dir}/llama_int{bits}_{stem}_c{nseg}.pt"
        if os.path.exists(path):
            return path

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.float32, low_cpu_mem_usage=True)
        model.eval()

        calib_x = self._calib_batches(nseg)
        # capture_down_input runs one segment per forward pass and needs the
        # batch axis kept; capture_group indexes the tensor itself.
        segs = [calib_x[i:i + 1] for i in range(len(calib_x))]
        pruned = False
        if tau is not None:
            prune_model(model, None, "cosine", calib_t=segs, tau=tau)
            pruned = True
        elif removal > 0.0:
            prune_model(model, removal, self.criterion, calib_t=segs)
            pruned = True

        if bits == 8:
            if pruned:
                # The cached INT8 weights are full width, and after removal
                # the matrices are narrower -- so they cannot be reused, and
                # reusing them is not what is wanted anyway: the scales must
                # be fitted to the COMPENSATED weights for the absorption in
                # Sec 4.4 to happen at all.
                quantize_ffn_int8(model, calib_x)
            else:
                if not os.path.exists(f"{WEIGHT_CACHE}/int8_ours_L0.npz"):
                    from wikitext2_int4 import build_all_weights
                    build_all_weights("int8", 127)
                apply_weights(model, "int8", "ours")

        torch.save(model.state_dict(), path)
        del model
        return path

    def evaluate(self, paths, n_eval, split, calib=None):
        """WikiText-2 perplexity, plus the per-segment losses the paired
        stopping rule needs. Perplexity is exp of a token-weighted mean, so
        the per-sample quantity resampled has to be the per-segment mean NLL,
        not the segment's own perplexity."""
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from wikitext2_int4 import MODEL_DIR, load_segments

        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        test, _ = load_segments(tok)
        test = test[:n_eval] if n_eval and n_eval < len(test) else test

        path = paths["dekoder-stek"]
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.float32, low_cpu_mem_usage=True)
        if path != MODEL_DIR:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            # A pruned stack has narrower feed-forward tensors than the config
            # declares, so the shapes are adopted from the artifact rather
            # than forced onto the freshly constructed module.
            _adopt_shapes(model, sd)
            model.load_state_dict(sd, strict=False)
        model.eval()

        per_seg = []
        with torch.no_grad():
            for i in range(len(test)):
                ids = test[i:i + 1]
                logits = model(input_ids=ids).logits[:, :-1]
                tgt = ids[:, 1:]
                nll = float(torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1), reduction="mean"))
                per_seg.append(nll)
        del model
        ppl = float(np.exp(np.mean(per_seg)))
        return {"score": ppl, "ppl": ppl, "per_sample": per_seg,
                "wer": ppl, "wer_lo": ppl, "wer_hi": ppl}

    @staticmethod
    def paired_margin(base_score, ceiling):
        """The accuracy budget, converted to the units the stopping rule
        actually compares in.

        The score is exp(mean NLL) but the paired bootstrap resamples
        per-segment NLL, so a perplexity ceiling C over a baseline B is an
        NLL margin of ln(C) - ln(B) = ln(C/B). Without this conversion a 5%
        relative budget read as 0.05*B in perplexity units is ~8x looser
        than intended (0.3773 against ln(1.05) = 0.0488 at B = 7.55), and
        the walk accepted a rung that was visibly over its own ceiling.
        """
        import math
        return math.log(ceiling / base_score)

    def check_data_split(self, calib, split, n_eval):
        """Calibration comes from WikiText's TRAIN field and evaluation from
        its TEST field, so the two cannot overlap however the slice moves.
        What can still go wrong is asking for more calibration than the train
        field holds, which would silently return fewer rows than requested --
        and calibration size is not a detail on this model: quadrupling it
        moved GPTQ from worst to best at INT4 (Sec 4.9a). So the amount is
        checked even though the disjointness is structural."""
        from wikitext2_int4 import SEQ_LEN
        want = self._calib_segments(calib)
        have = self._available_calib_segments()
        if want > have:
            raise ValueError(
                f"Llama: {want} ta kalibrlash segmenti ({want*SEQ_LEN} qator) "
                f"so'raldi, WikiText-2 train keshida {have} tasi bor")


def quantize_ffn_int8(model, calib, verbose=True):
    """Per-channel INT8 over the feed-forward matrices of a model in memory.

    The equivalent for Whisper is build_gptq_model, which likewise runs on the
    already-pruned graph. Calibration activations are captured from THIS
    model, so the scales see the compensated weights and the widened row
    ranges they produce -- which is the whole reason the two stages are
    ordered this way.
    """
    import numpy as np
    import torch

    from nnopt.quantizer.per_channel import (quantize_codes_pc,
                                             refine_scales_per_channel)
    from wikitext2_int4 import FFN, LAYERS_PER_GROUP, capture_group

    n_layers = len(model.model.layers)
    for start in range(0, n_layers, LAYERS_PER_GROUP):
        group = list(range(start, min(start + LAYERS_PER_GROUP, n_layers)))
        x_by = capture_group(model, calib, group)
        for li in group:
            mlp = model.model.layers[li].mlp
            for nm in FFN:
                lin = getattr(mlp, nm)
                w = lin.weight.detach().numpy().astype(np.float64)
                res = refine_scales_per_channel(w, 127, x_calib=x_by[(li, nm)])
                wq = quantize_codes_pc(w, res.scales, 127) * res.scales
                with torch.no_grad():
                    lin.weight.copy_(torch.from_numpy(wq).float())
        del x_by
        if verbose:
            print(f"    INT8 {group[-1]+1}/{n_layers} qatlam", flush=True)


def _adopt_shapes(model, sd):
    """Resize the feed-forward modules to whatever the artifact holds.

    A structurally reduced stack no longer matches its own config, and
    load_state_dict would reject it. Rebuilding the module from a corrected
    config is the cleaner-looking option but a worse one: the removal is
    per-layer, so a single intermediate_size cannot describe the result.
    """
    import torch

    for li, layer in enumerate(model.model.layers):
        for nm in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{li}.mlp.{nm}.weight"
            if key not in sd:
                continue
            want = sd[key].shape
            lin = getattr(layer.mlp, nm)
            if tuple(lin.weight.shape) == tuple(want):
                continue
            lin.out_features, lin.in_features = int(want[0]), int(want[1])
            lin.weight = torch.nn.Parameter(torch.empty(want))
        bkey = f"model.layers.{li}.mlp.down_proj.bias"
        if bkey in sd and layer.mlp.down_proj.bias is None:
            # The fluctuation criterion folds the removed channels' mean into
            # a bias this architecture does not natively carry (Sec 4.14).
            layer.mlp.down_proj.bias = torch.nn.Parameter(
                torch.empty(sd[bkey].shape))


PROFILES = {p.name: p for p in (WhisperProfile(), MBertProfile(),
                                LlamaProfile())}


def get_profile(name):
    if name not in PROFILES:
        raise ValueError(f"noma'lum model {name!r}; mavjud: {sorted(PROFILES)}")
    return PROFILES[name]
