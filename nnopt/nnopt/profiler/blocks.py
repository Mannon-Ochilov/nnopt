"""Finding the reducible block of a transformer, without knowing its names.

The cascade's structural stage needs one thing from a model: an axis that is
one operator's OUTPUT width and the next operator's INPUT width, with nothing
between them but elementwise work. Removing a coordinate of such an axis
shrinks both matrices at once and leaves every other tensor shape untouched --
which is why the method removes channels exactly rather than approximating a
rank.

Until now that block was located by matching operator names, `/fc1/` and
`/fc2/`, which are Whisper's. Every other transformer ASR model spells it
differently -- wav2vec2 exports `intermediate_dense` and `output_dense`,
Conformers have two half-step feed-forwards per block, and a name-matched
search finds nothing in either. Silently finding nothing is the dangerous
failure: the planner would report zero prunable bytes and conclude, wrongly,
that no cache target is reachable.

So the block is identified by the property the method actually requires. A
pair (A, B) qualifies when

  * B consumes A's output, possibly through elementwise nodes (activation,
    bias add, dropout-as-identity) but through no other matrix operator;
  * they share the intermediate width exactly, A.n == B.k;
  * that width EXPANDS on A, n > k.

The expansion condition is what keeps attention out. A value projection feeds
an output projection through a reshape and shares its width, but does not
expand it; the redundancy this method exploits lives in the widened axis, and
including the attention path would let the planner promise reductions the
structural stage cannot deliver there.

What this module does NOT do is decide reuse. How often a weight is revisited
depends on how the model is driven -- an encoder that sees a whole utterance
per pass reuses weights once per position, an autoregressive decoder at batch
1 does not reuse them at all -- and that is a property of the caller's serving
loop rather than of the graph. It stays an argument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import onnx

# Nodes that may sit between the two halves of a reducible pair. They act
# coordinate-wise on the intermediate axis, so removing a coordinate commutes
# with them; anything not on this list breaks that and ends the search.
ELEMENTWISE = frozenset({
    "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Erf", "Tanh", "Sigmoid",
    "Relu", "Gelu", "LeakyRelu", "Elu", "Exp", "Log", "Clip", "Cast",
    "Identity", "Dropout", "Neg", "Abs", "HardSigmoid", "HardSwish", "Sin",
    "Cos", "Softplus", "Mish", "Swish", "QuickGelu", "BiasGelu",
})
MATMUL_OPS = frozenset({"MatMul", "Gemm", "MatMulInteger", "QLinearMatMul",
                        "FusedMatMul"})

# Layer numbering conventions seen across exported transformers.
LAYER_PATTERNS = (
    r"/layers\.(\d+)/", r"/layer\.(\d+)/", r"/blocks?\.(\d+)/",
    r"/h\.(\d+)/", r"\blayers?_(\d+)\b", r"\bblock_(\d+)\b",
)


def layer_index(name: str) -> int:
    """Layer number from an operator name, or -1 when it carries none."""
    for pat in LAYER_PATTERNS:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return -1


@dataclass(frozen=True)
class ReduciblePair:
    """Two operators sharing an intermediate axis the cascade may shrink."""

    expand: str          # operator name that produces the wide axis
    contract: str        # operator name that consumes it
    expand_weight: str
    contract_weight: str
    width: int           # size of the shared axis
    layer: int

    @property
    def params(self) -> int:
        return 0  # filled by the caller, which holds the shapes


def _consumers(graph):
    out = {}
    for nd in graph.node:
        for i in nd.input:
            out.setdefault(i, []).append(nd)
    return out


def _elementwise_closure(graph, start_tensor, max_hops=12):
    """Tensors reachable from `start_tensor` through elementwise nodes only.

    The walk stops at any matrix operator, so a pair can never be matched
    across an intervening projection -- which would make the "shared axis"
    claim false.
    """
    consumers = _consumers(graph)
    seen, frontier = {start_tensor}, [start_tensor]
    for _ in range(max_hops):
        nxt = []
        for t in frontier:
            for nd in consumers.get(t, []):
                if nd.op_type not in ELEMENTWISE:
                    continue
                for o in nd.output:
                    if o not in seen:
                        seen.add(o)
                        nxt.append(o)
        if not nxt:
            break
        frontier = nxt
    return seen


def find_reducible_pairs(model, profiles, require_expansion=True):
    """Locate every (expand, contract) pair the structural stage can act on."""
    graph = model.graph
    by_output = {p.output_name: p for p in profiles if p.weight_initializer}
    weighted = [p for p in profiles if p.weight_initializer]

    pairs = []
    for a in weighted:
        if require_expansion and a.n <= a.k:
            continue
        reach = _elementwise_closure(graph, a.output_name)
        for b in weighted:
            if b is a or b.activation_input not in reach:
                continue
            if b.k != a.n:
                continue
            if b.output_name in by_output and by_output[b.output_name] is a:
                continue
            pairs.append(ReduciblePair(
                expand=a.name, contract=b.name,
                expand_weight=a.weight_initializer,
                contract_weight=b.weight_initializer,
                width=a.n, layer=layer_index(a.name)))
    return pairs


@dataclass
class PartBreakdown:
    """Per-layer byte accounting for one part of a model."""

    n_layers: int
    per_layer_bytes: int
    reducible_bytes: int
    pairs: list[ReduciblePair]
    largest_layer: int

    @property
    def fixed_bytes(self) -> int:
        return self.per_layer_bytes - self.reducible_bytes


def breakdown(model_path, profiles, bytes_per_param=4):
    """Layer sizes and reducible share, derived from the graph alone.

    Returns the numbers a `PartSpec` needs. The layer chosen is the largest,
    because the cache target binds on the worst layer rather than the average
    one -- a model with one oversized layer does not fit merely because its
    others do.
    """
    model = onnx.load(model_path, load_external_data=False)
    shapes = {i.name: tuple(i.dims) for i in model.graph.initializer}
    pairs = find_reducible_pairs(model, profiles)

    per_layer, reducible = {}, {}
    for p in profiles:
        if not p.weight_initializer or p.weight_initializer not in shapes:
            continue
        li = layer_index(p.name)
        if li < 0:
            continue
        per_layer[li] = per_layer.get(li, 0) + int(np.prod(shapes[p.weight_initializer]))

    counted = set()
    for pr in pairs:
        if pr.layer < 0:
            continue
        for w in (pr.expand_weight, pr.contract_weight):
            if w in counted or w not in shapes:
                continue
            counted.add(w)
            reducible[pr.layer] = reducible.get(pr.layer, 0) + int(np.prod(shapes[w]))

    if not per_layer:
        raise ValueError(f"{model_path}: qatlamga bo'lingan operatorlar topilmadi")
    li = max(per_layer, key=lambda k: per_layer[k])
    return PartBreakdown(
        n_layers=len(per_layer),
        per_layer_bytes=per_layer[li] * bytes_per_param,
        reducible_bytes=reducible.get(li, 0) * bytes_per_param,
        pairs=[pr for pr in pairs if pr.layer == li],
        largest_layer=li,
    )
