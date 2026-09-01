"""Model-level planning: given a cache size and an accuracy budget, what to try.

The operator cascade (nnopt.cascade.operator_cascade) decides one matrix at a
time against a fixed target. What it cannot do is answer the question a user
actually has -- "here is my model, my L3, and how much WER I can afford; give
me the best configuration" -- because that question spans operators, involves
an accuracy budget only measurable end to end, and has to tolerate targets no
configuration can reach.

Two decisions shape this module.

**Cache residency is an objective, not a gate.** Requiring the working set to
fit alpha*L3 makes the goal binary, and a binary goal is brittle exactly where
this one is: on the reference machine the decoder's decision sat 0.033 away
from flipping in alpha, and 1.1 MiB away from flipping in L3. Worse, when the
target is unreachable -- a 12 MiB L3 asks 45% of the channels of every encoder
layer, where redundancy exists only in the early ones -- a gate forces the
cascade to keep cutting past the point its own criterion endorses. So the
objective here is MISS BYTES (`miss_bytes`), which falls smoothly as the
footprint shrinks, rewards fitting without demanding it, and degrades
gracefully when fitting is impossible.

**The ladder is ordered, and evaluation is lazy.** Each candidate costs about
an hour to build and half an hour to score, so an exhaustive sweep is not an
option. `plan()` emits a totally ordered ladder, mildest first, by repeatedly
tightening whichever part currently wastes the most memory traffic. A caller
walks it from the bottom and stops at the first rung that breaks the accuracy
budget; everything above that rung is strictly more aggressive and cannot
recover it.

The miss model is an analytic proxy, not a measurement: a part whose per-layer
weights fit the budget is read once per pass, and the overflow is re-read on
each reuse of the layer. For a Whisper encoder (1500 positions per pass) that
makes overflow expensive; for a batch-1 decoder (one use per token) reuse is
1 and the proxy correctly collapses to plain byte count. Validating it against
hardware counters is a separate job -- see experiments/vtune_whole_model.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

MIB = 1024.0 ** 2


@dataclass(frozen=True)
class PartSpec:
    """One structurally homogeneous piece of the model, e.g. the encoder.

    `prunable_bytes` is the share of a layer the structural stage is allowed
    to touch -- the FFN, on the architectures measured here. Attention is
    excluded, and that exclusion is what makes some cache targets unreachable
    rather than merely expensive, so it is carried explicitly rather than
    assumed to be the whole layer.
    """

    name: str
    per_layer_bytes: int
    n_layers: int
    prunable_bytes: int
    reuse: int = 1
    #: Whether the toolchain can actually BUILD a structurally reduced version
    #: of this part. It is separate from `prunable_bytes`, which says how much
    #: of the layer the cache target could in principle draw on: the
    #: feasibility report needs the second even where the first is false.
    #: Enumerating configurations that cannot be built is worse than useless
    #: -- the rungs are cumulative, so one unbuildable step poisons every rung
    #: above it, and a walk that stops there silently forgoes the reductions
    #: that ARE available on other parts.
    structural_supported: bool = True

    def __post_init__(self):
        if self.prunable_bytes > self.per_layer_bytes:
            raise ValueError(f"{self.name}: prunable_bytes exceeds per_layer_bytes")
        if self.per_layer_bytes <= 0 or self.n_layers <= 0:
            raise ValueError(f"{self.name}: sizes must be positive")
        if self.reuse < 1:
            raise ValueError(f"{self.name}: reuse must be at least 1")


@dataclass(frozen=True)
class Treatment:
    """What has been applied to a part: a bit width and a structural setting.

    `keep` is the fraction of prunable bytes retained IN TOTAL, and `tag`
    names how that total is realised. The distinction is not cosmetic. The
    first version of this planner enumerated keep ratios and left the builder
    to remove that fraction from every layer uniformly, which turned out to be
    a dominated family: measured against the same baseline on the same
    utterances, our criterion at tau = 0.99 was both SMALLER (267 MiB against
    281) and significantly more accurate (dWER +0.0077 against +0.0186,
    difference -0.0108 [-0.0239, -0.0003]). A uniform cut spends the budget
    where redundancy is not, so the ladder now carries the criterion's own
    operating points and `tag` says which one.
    """

    bits: int = 32
    keep: float = 1.0
    tag: str = ""

    def __post_init__(self):
        if not 0.0 < self.keep <= 1.0:
            raise ValueError("keep must lie in (0, 1]")
        if self.bits not in (32, 16, 8, 4):
            raise ValueError(f"unsupported bit width {self.bits}")


def part_bytes(spec: PartSpec, t: Treatment) -> float:
    """Per-layer bytes after quantization and structural reduction."""
    fixed = spec.per_layer_bytes - spec.prunable_bytes
    return (fixed + spec.prunable_bytes * t.keep) * (t.bits / 32.0)


def miss_bytes(spec: PartSpec, t: Treatment, budget_bytes: float) -> float:
    """Proxy for bytes moved from memory per pass over the whole part.

    Weights that fit the budget are fetched once and then reused in place;
    whatever overflows is re-fetched on every reuse. The result is continuous
    in the footprint and has no discontinuity at the budget, which is the
    property that makes it usable as an objective rather than a test.
    """
    per_layer = part_bytes(spec, t)
    overflow = max(0.0, per_layer - budget_bytes)
    return spec.n_layers * (per_layer + overflow * (spec.reuse - 1))


def required_factor(spec: PartSpec, budget_bytes: float) -> float:
    """How far the fp32 per-layer footprint is from fitting."""
    return spec.per_layer_bytes / budget_bytes if budget_bytes > 0 else float("inf")


def reachable_keep(spec: PartSpec, t_bits: int, budget_bytes: float) -> float:
    """Keep ratio that would put the part exactly inside budget, if any.

    Returns 0.0 or less when even deleting every prunable byte leaves the
    part over budget -- the infeasible case, which the caller should report
    rather than approximate.
    """
    fixed = spec.per_layer_bytes - spec.prunable_bytes
    scale = t_bits / 32.0
    if spec.prunable_bytes == 0:
        return 1.0 if fixed * scale <= budget_bytes else -1.0
    return (budget_bytes / scale - fixed) / spec.prunable_bytes


@dataclass
class Rung:
    """One candidate configuration, with everything needed to judge it."""

    index: int
    treatments: dict[str, Treatment]
    total_bytes: float
    miss: float
    fits: dict[str, bool]
    step: str  # what changed relative to the previous rung

    @property
    def all_fit(self) -> bool:
        return all(self.fits.values())


@dataclass
class CachePlan:
    budget_bytes: float
    l3_bytes: float
    alpha: float
    specs: list[PartSpec]
    rungs: list[Rung] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"L3 = {self.l3_bytes/MIB:.0f} MiB, alpha = {self.alpha}, "
                 f"byudjet = {self.budget_bytes/MIB:.1f} MiB"]
        for s in self.specs:
            need = required_factor(s, self.budget_bytes)
            lines.append(f"  {s.name:9s} qatlam {s.per_layer_bytes/MIB:5.1f} MiB "
                         f"(qisqartirilishi mumkin {s.prunable_bytes/MIB:5.1f}), "
                         f"talab {need:5.2f}x, qayta ishlatish {s.reuse}")
        return "\n".join(lines)


def plan(
    specs: list[PartSpec],
    l3_bytes: float,
    alpha: float = 0.7,
    bit_ladder: tuple[int, ...] = (32, 8),
    keep_ladder: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
    structural_ladder: tuple[tuple[str, float], ...] | None = None,
    max_rungs: int = 24,
) -> CachePlan:
    """Order the candidates from mildest to most aggressive.

    Two rules, in this order.

    Quantization everywhere first, structural reduction only afterwards. This
    is the cascade's own backward-scheduling principle (Sec 1.1): a stage is
    asked for the residual left by the cheaper stage, never the reverse. It is
    enforced here as a phase boundary rather than a weight, because a weight
    cannot express it. Ranking every step by traffic-removed-per-risk with a
    tunable risk price was the first attempt and it fails structurally: the
    encoder's reuse factor of 1500 makes its overflow dominate every other
    term, so at L3 = 12 MiB the planner pruned the encoder to half its
    channels before quantizing the decoder at all -- five structural cuts
    taken ahead of a nearly free fourfold reduction. No finite risk price
    fixes that, since the imbalance grows with reuse.

    Within a phase, take the step that removes the most traffic. That is where
    a greedy is appropriate: the steps are comparable in kind, so their sizes
    are the only thing left to order them by.

    `structural_ladder` supplies the structural operating points as
    (tag, total keep) pairs, mildest first, so a caller can hand the planner
    the settings its criterion actually produces instead of round fractions.
    Without it the ladder falls back to `keep_ladder`, i.e. uniform removal --
    kept only so the planner remains usable for a model whose criterion has
    not been characterised, and documented on `Treatment` as the weaker
    option.

    The ladder stops when no part can be tightened further, or at `max_rungs`.
    It does NOT stop when everything fits: a caller minimising misses may
    still want the rungs beyond that point, and a caller with a strict
    accuracy budget simply never reaches them.
    """
    if not specs:
        raise ValueError("no parts to plan for")
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must lie in (0, 1]")
    budget = alpha * l3_bytes

    ladder = (tuple(structural_ladder) if structural_ladder
              else tuple(("", k) for k in keep_ladder))
    if ladder[0][1] != 1.0:
        ladder = (("", 1.0),) + ladder
    keeps = [k for _, k in ladder]
    if keeps != sorted(keeps, reverse=True):
        raise ValueError("structural_ladder yumshoqdan qattiqqa tartiblangan "
                         "bo'lishi kerak")

    current = {s.name: Treatment(bits=bit_ladder[0], keep=ladder[0][1],
                                 tag=ladder[0][0])
               for s in specs}
    by_name = {s.name: s for s in specs}

    def snapshot(index, step):
        return Rung(
            index=index,
            treatments=dict(current),
            total_bytes=sum(part_bytes(by_name[n], t) * by_name[n].n_layers
                            for n, t in current.items()),
            miss=sum(miss_bytes(by_name[n], t, budget) for n, t in current.items()),
            fits={n: part_bytes(by_name[n], t) <= budget for n, t in current.items()},
            step=step,
        )

    out = CachePlan(budget_bytes=budget, l3_bytes=l3_bytes, alpha=alpha,
                    specs=list(specs))
    out.rungs.append(snapshot(0, "boshlang'ich (o'zgartirishsiz)"))

    for i in range(1, max_rungs):
        quant_steps, prune_steps = [], []
        for s in specs:
            t = current[s.name]
            bi = bit_ladder.index(t.bits) if t.bits in bit_ladder else 0
            if bi + 1 < len(bit_ladder):
                nxt = replace(t, bits=bit_ladder[bi + 1])
                gain = miss_bytes(s, t, budget) - miss_bytes(s, nxt, budget)
                quant_steps.append((gain, s.name, nxt,
                                    f"{s.name}: INT{bit_ladder[bi+1]}"))
                continue
            ki = next((i for i, (tag, k) in enumerate(ladder)
                       if tag == t.tag and k == t.keep), 0)
            if (ki + 1 >= len(ladder) or s.prunable_bytes == 0
                    or not s.structural_supported):
                continue
            nxt_tag, nxt_keep = ladder[ki + 1]
            nxt = replace(t, keep=nxt_keep, tag=nxt_tag)
            gain = miss_bytes(s, t, budget) - miss_bytes(s, nxt, budget)
            how = nxt_tag if nxt_tag else f"{(1-nxt_keep)*100:.0f}%"
            prune_steps.append((gain, s.name, nxt,
                                f"{s.name}: qisqartirish {how}"))

        options = quant_steps or prune_steps
        if not options:
            break
        options.sort(key=lambda o: -o[0])
        _, name, nxt, label = options[0]
        current[name] = nxt
        out.rungs.append(snapshot(i, label))

    return out


@dataclass
class Verdict:
    """Whether the target is reachable at all, and what it would demand."""

    part: str
    required: float
    after_quant: float
    keep_needed: float
    feasible: bool
    note: str


def feasibility(specs: list[PartSpec], l3_bytes: float, alpha: float = 0.7,
                bits: int = 8) -> list[Verdict]:
    """Report, per part, what fitting would cost -- before anything is built.

    This is the check that turns "the cascade produced a bad model" into "the
    target was never reachable", and it is cheap enough to run first every
    time.
    """
    budget = alpha * l3_bytes
    out = []
    for s in specs:
        need = required_factor(s, budget)
        after = need / (32.0 / bits)
        keep = reachable_keep(s, bits, budget)
        if need <= 1.0:
            out.append(Verdict(s.name, need, after, 1.0, True,
                               "fp32 holicha sig'adi"))
        elif after <= 1.0:
            out.append(Verdict(s.name, need, after, 1.0, True,
                               f"INT{bits} yetarli"))
        elif keep <= 0.0:
            out.append(Verdict(s.name, need, after, keep, False,
                               "qisqartirilmaydigan qism byudjetdan katta"))
        else:
            out.append(Verdict(s.name, need, after, keep, True,
                               f"qisqartiriladigan qismning "
                               f"{(1-keep)*100:.0f}% i olib tashlanishi kerak"))
    return out
