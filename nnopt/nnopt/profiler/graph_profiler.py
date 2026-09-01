"""Operator-level resource profiling for ONNX graphs -- README.md Sec 2.2.

For every Gemm / MatMul (and, optionally, Conv-as-GEMM) operator in the
graph, this module computes the static resource quantities defined in
Sec 2.2 of the project README:

    M_W, M_X, M_Y, M_tmp, M_total   (bytes)
    FLOPs, Bytes                     -> AI (arithmetic intensity)
    K_cache = M_eff / (alpha * M_cache)   -- once a target cache is chosen
    D, d                             -- absolute / relative residual demand

Two *honest* limitations, called out explicitly rather than hidden:

1.  M_eff (the "active working set at any instant during execution") is a
    function of the BLAS/oneDNN kernel's internal tiling strategy and is
    NOT recoverable from a static ONNX graph. We therefore report the
    upper bound M_eff = M_total (variant "a" from README Sec 2.2.2) by
    default, and a simplified blocked-GEMM estimate (variant "b") as an
    opt-in alternative once representative tile sizes are supplied.

2.  M_tmp (temporary/workspace buffers) is kernel- and runtime-dependent
    and defaults to 0 here. Do not treat M_total as a precise number --
    treat it as a lower/upper bracket to be refined empirically (README
    Sec 8.3, point 2).

3.  Which cache level (L2 vs L3) is "the" target for an operator depends
    on *which logical processors execute it*, itself a runtime scheduling
    decision (thread affinity / intra-op parallelism). This module does
    not guess that. `evaluate_against_cache()` takes an explicit
    CacheInstance (see nnopt.hw.cache_topology) so the caller states the
    assumption; `profile_onnx_model` separately reports both a
    single-core (L2) and an all-core (L3) K_cache so downstream cascade
    logic (README Sec 2.5 step 6, CPU/GPU + thread placement) can pick
    the one that matches the actual execution plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import shape_inference

try:
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    _HAS_SYMBOLIC_SHAPE_INFER = True
except ImportError:  # pragma: no cover - onnxruntime always installed per pyproject, defensive only
    _HAS_SYMBOLIC_SHAPE_INFER = False

from nnopt.hw.cache_topology import CacheInstance

# ONNX TensorProto.DataType -> bits per element.
_ELEM_TYPE_BITS: dict[int, int] = {
    1: 32,   # FLOAT
    2: 8,    # UINT8
    3: 8,    # INT8
    4: 16,   # UINT16
    5: 16,   # INT16
    6: 32,   # INT32
    7: 64,   # INT64
    9: 1,    # BOOL (approximate; ONNX packs bool as 1 byte at runtime, not 1 bit)
    10: 16,  # FLOAT16
    11: 64,  # DOUBLE
    12: 32,  # UINT32
    13: 64,  # UINT64
    16: 16,  # BFLOAT16
}
_ELEM_TYPE_NAMES: dict[int, str] = {
    1: "FP32", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16", 6: "INT32",
    7: "INT64", 9: "BOOL", 10: "FP16", 11: "FP64", 12: "UINT32", 13: "UINT64",
    16: "BF16",
}

_MATMUL_LIKE_OPS = {"Gemm", "MatMul"}


class ShapeResolutionError(RuntimeError):
    """Raised when a tensor dimension is symbolic and not covered by free_dims."""


@dataclass
class OperatorResourceProfile:
    """Static resource profile of a single matrix operator (README Sec 2.2)."""

    name: str
    op_type: str
    weight_initializer: str | None
    activation_input: str  # graph tensor name feeding this operator (== the h_j source for calibration)
    output_name: str
    dtype_name: str
    dtype_bits: int

    # Effective GEMM shape after resolving Gemm transA/transB or MatMul
    # batch/broadcast semantics: output = (batch * m, n), reduction = k.
    batch: int
    m: int
    k: int
    n: int

    M_W: int
    M_X: int
    M_Y: int
    M_tmp: int

    @property
    def M_total(self) -> int:
        return self.M_W + self.M_X + self.M_Y + self.M_tmp

    @property
    def flops(self) -> int:
        # Multiply-accumulate = 2 FLOPs, batched GEMM: 2 * batch * m * k * n.
        return 2 * self.batch * self.m * self.k * self.n

    @property
    def bytes_io(self) -> int:
        # Bytes moved between the operator and memory: weights read once,
        # activations read once, output written once. (M_tmp intentionally
        # excluded -- it is not necessarily off-chip traffic.)
        return self.M_W + self.M_X + self.M_Y

    @property
    def arithmetic_intensity(self) -> float:
        b = self.bytes_io
        return self.flops / b if b > 0 else math.inf

    # M_eff variant (a): upper bound = whole operator footprint.
    @property
    def m_eff_upper(self) -> int:
        return self.M_total

    def m_eff_blocked(self, k_block: int, n_block: int) -> int:
        """M_eff variant (b): simplified blocked-GEMM estimate (README Sec 2.2.2).

        Assumes a standard tiled GEMM kernel that streams k_block columns of
        the reduction dimension and n_block columns of the output at a time,
        keeping one (m x k_block) weight tile, one (k_block x n_block) input
        tile and one (m x n_block) output tile resident simultaneously:

            M_eff ~= q/8 * (m*k_block + k_block*n_block + m*n_block)

        This is a coarse model (real kernels double-buffer, vectorize, and
        block M too) -- it exists to give a *lower* bound to bracket against
        the m_eff_upper *upper* bound, not as a precise prediction.
        """
        q_bytes = self.dtype_bits / 8
        k_b = min(k_block, self.k)
        n_b = min(n_block, self.n)
        elems = self.m * k_b + k_b * n_b + self.m * n_b
        return int(q_bytes * elems)


@dataclass
class CacheFitResult:
    """Sec 2.2 keshga moslik baholash natijasi for one operator + one cache."""

    operator: str
    cache_level: int
    cache_size_bytes: int
    alpha: float
    m_eff: int
    m_cache_eff: float
    k_cache: float  # M_eff / M_cache_eff
    d_abs: int  # max(0, M_eff - M_cache_eff)
    d_rel: float  # d_abs / (M_eff + xi)
    is_critical: bool  # K_cache > 1


XI = 1e-9  # small positive constant guarding division by zero (README Sec 2.2/2.5)


def evaluate_against_cache(
    profile: OperatorResourceProfile,
    cache: CacheInstance,
    alpha: float = 0.7,
    m_eff: int | None = None,
) -> CacheFitResult:
    """README (2.6)-(2.9): K_cache, D, d for one operator against one cache."""
    eff = profile.m_eff_upper if m_eff is None else m_eff
    m_cache_eff = alpha * cache.size_bytes
    k_cache = eff / m_cache_eff if m_cache_eff > 0 else math.inf
    d_abs = max(0, eff - int(m_cache_eff))
    d_rel = d_abs / (eff + XI)
    return CacheFitResult(
        operator=profile.name,
        cache_level=cache.level,
        cache_size_bytes=cache.size_bytes,
        alpha=alpha,
        m_eff=eff,
        m_cache_eff=m_cache_eff,
        k_cache=k_cache,
        d_abs=d_abs,
        d_rel=d_rel,
        is_critical=k_cache > 1.0,
    )


# --------------------------------------------------------------------------
# ONNX graph walking
# --------------------------------------------------------------------------

def _dim_to_int(dim, free_dims: dict[str, int]) -> int:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param") and dim.dim_param in free_dims:
        return int(free_dims[dim.dim_param])
    label = dim.dim_param or "<unnamed>"
    raise ShapeResolutionError(
        f"Unresolved symbolic dimension '{label}'. Pass it via free_dims=, "
        f"e.g. free_dims={{'{label}': 1}}."
    )


def _build_shape_and_dtype_index(
    model: onnx.ModelProto,
    use_symbolic_shape_infer: bool = True,
) -> tuple[dict[str, list], dict[str, int]]:
    """value_info name -> (dims,) and name -> elem_type, covering inputs,
    outputs, and inferred intermediate value_info entries.

    Plain onnx.shape_inference.infer_shapes does a single local pass and
    frequently gives up (emitting placeholder 'unk__N' dims) on transformer
    exports whose attention blocks route through Reshape/Transpose chains
    with runtime-computed target shapes -- observed in practice to leave
    >95% of a Whisper encoder's MatMul operators unresolved. ONNX Runtime's
    symbolic shape inference tool reasons across those chains and resolves
    the overwhelming majority of them, so it is used by default when
    available (onnxruntime is a hard dependency of this package); the
    plain inferencer remains a fallback for environments without it.
    """
    if use_symbolic_shape_infer and _HAS_SYMBOLIC_SHAPE_INFER:
        try:
            inferred = SymbolicShapeInference.infer_shapes(
                model, auto_merge=True, guess_output_rank=False, verbose=0
            )
        except Exception as exc:  # pragma: no cover - defensive fallback for pathological graphs
            print(f"[graph_profiler] symbolic shape inference failed ({exc}); falling back to onnx.shape_inference")
            inferred = shape_inference.infer_shapes(model, strict_mode=False)
    else:
        inferred = shape_inference.infer_shapes(model, strict_mode=False)

    shapes: dict[str, list] = {}
    dtypes: dict[str, int] = {}
    graph = inferred.graph
    for collection in (graph.input, graph.output, graph.value_info):
        for vi in collection:
            if vi.type.HasField("tensor_type"):
                tt = vi.type.tensor_type
                shapes[vi.name] = list(tt.shape.dim) if tt.HasField("shape") else None
                dtypes[vi.name] = tt.elem_type
    return shapes, dtypes


def _initializer_index(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {init.name: init for init in model.graph.initializer}


def _resolve_shape(
    name: str,
    shapes: dict[str, list],
    initializers: dict[str, onnx.TensorProto],
    free_dims: dict[str, int],
) -> list[int]:
    if name in initializers:
        return list(initializers[name].dims)
    dims = shapes.get(name)
    if dims is None:
        raise ShapeResolutionError(
            f"No shape information available for tensor '{name}' "
            f"(missing from shape_inference output; is the graph malformed?)."
        )
    return [_dim_to_int(d, free_dims) for d in dims]


def _resolve_dtype_bits(
    name: str,
    dtypes: dict[str, int],
    initializers: dict[str, onnx.TensorProto],
) -> tuple[int, str]:
    if name in initializers:
        elem_type = initializers[name].data_type
    else:
        elem_type = dtypes.get(name, 1)  # default FP32 if unknown
    return _ELEM_TYPE_BITS.get(elem_type, 32), _ELEM_TYPE_NAMES.get(elem_type, f"type{elem_type}")


def _attr(node: onnx.NodeProto, key: str, default):
    for a in node.attribute:
        if a.name == key:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
    return default


def _profile_gemm(
    node: onnx.NodeProto,
    shapes: dict[str, list],
    dtypes: dict[str, int],
    initializers: dict[str, onnx.TensorProto],
    free_dims: dict[str, int],
) -> OperatorResourceProfile:
    a_name, b_name = node.input[0], node.input[1]
    c_name = node.input[2] if len(node.input) > 2 else None
    trans_a = bool(_attr(node, "transA", 0))
    trans_b = bool(_attr(node, "transB", 0))

    a_shape = _resolve_shape(a_name, shapes, initializers, free_dims)
    b_shape = _resolve_shape(b_name, shapes, initializers, free_dims)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ShapeResolutionError(f"Gemm '{node.name}' expects rank-2 inputs, got {a_shape}, {b_shape}")

    m, k_a = (a_shape[1], a_shape[0]) if trans_a else (a_shape[0], a_shape[1])
    k_b, n = (b_shape[1], b_shape[0]) if trans_b else (b_shape[0], b_shape[1])
    if k_a != k_b:
        raise ShapeResolutionError(f"Gemm '{node.name}' inner dims mismatch: {k_a} vs {k_b}")

    weight_name = b_name if b_name in initializers else (a_name if a_name in initializers else None)
    act_name = a_name if weight_name == b_name else b_name

    w_bits, w_type = _resolve_dtype_bits(weight_name or b_name, dtypes, initializers)
    x_bits, _ = _resolve_dtype_bits(act_name, dtypes, initializers)
    out_name = node.output[0]
    y_bits, _ = _resolve_dtype_bits(out_name, dtypes, initializers)

    m_w = (k_a * n) * w_bits // 8 if weight_name else 0
    m_x = (m * k_a) * x_bits // 8
    m_y = (m * n) * y_bits // 8
    if c_name:
        bias_shape = _resolve_shape(c_name, shapes, initializers, free_dims)
        bias_bits, _ = _resolve_dtype_bits(c_name, dtypes, initializers)
        m_w += int(np.prod(bias_shape)) * bias_bits // 8

    return OperatorResourceProfile(
        name=node.name or out_name,
        op_type="Gemm",
        weight_initializer=weight_name,
        activation_input=act_name,
        output_name=out_name,
        dtype_name=w_type,
        dtype_bits=w_bits,
        batch=1,
        m=m,
        k=k_a,
        n=n,
        M_W=m_w,
        M_X=m_x,
        M_Y=m_y,
        M_tmp=0,
    )


def _profile_matmul(
    node: onnx.NodeProto,
    shapes: dict[str, list],
    dtypes: dict[str, int],
    initializers: dict[str, onnx.TensorProto],
    free_dims: dict[str, int],
) -> OperatorResourceProfile:
    a_name, b_name = node.input[0], node.input[1]
    a_shape = _resolve_shape(a_name, shapes, initializers, free_dims)
    b_shape = _resolve_shape(b_name, shapes, initializers, free_dims)
    if len(a_shape) < 2 or len(b_shape) < 2:
        raise ShapeResolutionError(
            f"MatMul '{node.name}' with rank<2 operand not supported by this profiler "
            f"(shapes {a_shape}, {b_shape})."
        )

    m, k_a = a_shape[-2], a_shape[-1]
    k_b, n = b_shape[-2], b_shape[-1]
    if k_a != k_b:
        raise ShapeResolutionError(f"MatMul '{node.name}' inner dims mismatch: {k_a} vs {k_b}")

    batch_a = a_shape[:-2]
    batch_b = b_shape[:-2]
    batch_dims = batch_a if len(batch_a) >= len(batch_b) else batch_b
    batch = int(np.prod(batch_dims)) if batch_dims else 1

    weight_name = b_name if b_name in initializers else (a_name if a_name in initializers else None)
    act_name = a_name if weight_name == b_name else b_name

    w_bits, w_type = _resolve_dtype_bits(weight_name or b_name, dtypes, initializers)
    x_bits, _ = _resolve_dtype_bits(act_name, dtypes, initializers)
    out_name = node.output[0]
    y_bits, _ = _resolve_dtype_bits(out_name, dtypes, initializers)

    # Weight bytes counted once (not multiplied by batch) when it is a
    # genuine shared-weight projection (2-D initializer); if the "weight"
    # side itself carries a batch dimension (e.g. batched matmul with two
    # runtime tensors, no initializer at all), M_W is 0 and both operands
    # are counted as activations via M_X/M_Y bookkeeping below.
    if weight_name is not None:
        w_shape = _resolve_shape(weight_name, shapes, initializers, free_dims)
        m_w = int(np.prod(w_shape)) * w_bits // 8
    else:
        m_w = 0

    m_x = batch * m * k_a * x_bits // 8
    if weight_name is None:
        # both operands are runtime activations: count the second one too.
        m_x += batch * k_b * n * x_bits // 8
    m_y = batch * m * n * y_bits // 8

    return OperatorResourceProfile(
        name=node.name or out_name,
        op_type="MatMul",
        weight_initializer=weight_name,
        activation_input=act_name,
        output_name=out_name,
        dtype_name=w_type,
        dtype_bits=w_bits,
        batch=batch,
        m=m,
        k=k_a,
        n=n,
        M_W=m_w,
        M_X=m_x,
        M_Y=m_y,
        M_tmp=0,
    )


def profile_onnx_model(
    model_path: str,
    free_dims: dict[str, int] | None = None,
    on_error: str = "warn",
    use_symbolic_shape_infer: bool = True,
) -> list[OperatorResourceProfile]:
    """Profile every Gemm/MatMul operator in an ONNX model (README Sec 2.2).

    Parameters
    ----------
    model_path: path to a .onnx file.
    free_dims: mapping of symbolic dimension name -> concrete value, e.g.
        {"batch_size": 1, "sequence_length": 128}. Required for any model
        with dynamic axes (virtually all transformer exports).
    on_error: "warn" (skip operator, print a note) or "raise".
    use_symbolic_shape_infer: use onnxruntime's symbolic shape inference
        (recommended, resolves far more shapes on real transformer exports
        than plain onnx.shape_inference -- see _build_shape_and_dtype_index).
    """
    free_dims = free_dims or {}
    model = onnx.load(model_path)
    shapes, dtypes = _build_shape_and_dtype_index(model, use_symbolic_shape_infer=use_symbolic_shape_infer)
    initializers = _initializer_index(model)

    profiles: list[OperatorResourceProfile] = []
    for node in model.graph.node:
        if node.op_type not in _MATMUL_LIKE_OPS:
            continue
        try:
            if node.op_type == "Gemm":
                profiles.append(_profile_gemm(node, shapes, dtypes, initializers, free_dims))
            else:
                profiles.append(_profile_matmul(node, shapes, dtypes, initializers, free_dims))
        except ShapeResolutionError as exc:
            if on_error == "raise":
                raise
            print(f"[graph_profiler] skipping '{node.name or node.output[0]}' ({node.op_type}): {exc}")
    return profiles


def summarize(profiles: list[OperatorResourceProfile]) -> str:
    lines = [f"{'name':40s} {'op':6s} {'m':>6s} {'k':>6s} {'n':>6s} {'dtype':5s} {'M_total(KiB)':>13s} {'AI':>8s}"]
    for p in profiles:
        lines.append(
            f"{p.name[:40]:40s} {p.op_type:6s} {p.m:6d} {p.k:6d} {p.n:6d} {p.dtype_name:5s} "
            f"{p.M_total / 1024:13.1f} {p.arithmetic_intensity:8.2f}"
        )
    return "\n".join(lines)
