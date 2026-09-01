"""Capturing intermediate ONNX activations for calibration -- README.md
Sec 2.3 point 2 ("h_j" functional response vectors) and Sec 2.4 point on
calibration inputs/outputs for quantization scale refinement.

This module is audio/model-agnostic: it augments an ONNX graph so ONNX
Runtime returns any requested intermediate tensor alongside the normal
outputs, then runs a batch of feeds through it. What the feeds actually
*are* (real speech mel-spectrograms, or a placeholder input for pipeline
testing) is entirely the caller's concern -- see
nnopt.calibrator.synthetic_inputs for a clearly-labeled placeholder
generator used until a real Uzbek speech calibration set is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort

try:
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    _HAS_SYMBOLIC_SHAPE_INFER = True
except ImportError:  # pragma: no cover - onnxruntime always installed per pyproject, defensive only
    _HAS_SYMBOLIC_SHAPE_INFER = False


@dataclass
class CapturedActivations:
    tensor_name: str
    batches: list[np.ndarray]  # one array per calibration batch, as returned by ORT

    def concatenate(self, axis: int = 0) -> np.ndarray:
        return np.concatenate(self.batches, axis=axis)


def _augment_model_with_outputs(model: onnx.ModelProto, tensor_names: list[str]) -> onnx.ModelProto:
    """Return a copy of `model` with each name in `tensor_names` added as an
    extra graph output, so ONNX Runtime will hand it back from session.run().
    Shape inference is run first so internal (non-input/output) tensors get
    a concrete dtype in value_info -- ONNX Runtime rejects a graph output
    declared with an UNDEFINED element type, so a typeless fallback is not
    an option for genuinely internal tensors.
    """
    if _HAS_SYMBOLIC_SHAPE_INFER:
        try:
            inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True, guess_output_rank=False, verbose=0)
        except Exception:  # pragma: no cover - defensive fallback for pathological graphs
            inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    else:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)

    augmented = onnx.ModelProto()
    augmented.CopyFrom(inferred)

    existing_outputs = {o.name for o in augmented.graph.output}
    value_info_by_name = {vi.name: vi for vi in augmented.graph.value_info}
    value_info_by_name.update({vi.name: vi for vi in augmented.graph.input})
    value_info_by_name.update({vi.name: vi for vi in augmented.graph.output})

    unresolved = [n for n in tensor_names if n not in existing_outputs and n not in value_info_by_name]
    if unresolved:
        raise ValueError(
            f"Could not determine dtype for tensor(s) {unresolved} even after shape inference -- "
            f"ONNX Runtime requires a concrete element type to expose a tensor as a graph output. "
            f"Check the tensor name(s) exist in this graph."
        )

    for name in tensor_names:
        if name in existing_outputs:
            continue
        new_output = onnx.ValueInfoProto()
        new_output.CopyFrom(value_info_by_name[name])
        augmented.graph.output.append(new_output)
        existing_outputs.add(name)

    return augmented


class ActivationCapture:
    """Wraps one ONNX model + a fixed set of intermediate tensors to
    capture. Building the augmented graph and the ORT session happens once
    in __init__; run_batch()/run_calibration_set() are cheap thereafter.
    """

    def __init__(
        self,
        model_path: str,
        tensor_names: list[str],
        providers: list[str] | None = None,
        intra_op_threads: int | None = None,
    ):
        model = onnx.load(model_path)
        augmented = _augment_model_with_outputs(model, tensor_names)
        so = ort.SessionOptions()
        if intra_op_threads is not None:
            so.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            augmented.SerializeToString(), sess_options=so, providers=providers or ["CPUExecutionProvider"]
        )
        self.tensor_names = list(tensor_names)
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._capture_indices = [self._output_names.index(n) for n in self.tensor_names]

    def run_batch(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = self.session.run(self._output_names, feeds)
        return {name: outputs[idx] for name, idx in zip(self.tensor_names, self._capture_indices)}

    def run_calibration_set(
        self, feeds_iterable
    ) -> dict[str, CapturedActivations]:
        """feeds_iterable: an iterable of feed dicts, one per calibration
        sample/batch (README Sec 2.3: 'kichik namunalar majmuasi')."""
        collected: dict[str, list[np.ndarray]] = {name: [] for name in self.tensor_names}
        for feeds in feeds_iterable:
            batch_out = self.run_batch(feeds)
            for name, arr in batch_out.items():
                collected[name].append(arr)
        return {name: CapturedActivations(name, arrs) for name, arrs in collected.items()}


def active_mask_from_lengths(lengths: list[int] | np.ndarray, max_len: int) -> np.ndarray:
    """Build the (batch, seq) boolean mask required by build_response_vectors
    from per-sample real lengths -- Fig 2.5's "To'ldiruvchi pozitsiyalar
    hisobga olinmaydi" (padding positions are not counted).

    Use this whenever calibration samples are batched together, since
    batching necessarily pads shorter samples up to max_len and those pad
    positions must NOT enter h_j. (Running each sample separately at its
    own exact length is the other valid option -- then there is no padding
    and no mask is needed.)

    NOTE for Whisper specifically: the ENCODER input is always padded to
    30 s (3000 mel frames -> 1500 encoder positions) regardless of the
    real audio duration, so any calibration of an *encoder* operator MUST
    pass a mask built from the real audio lengths. Decoder-side operators
    fed one un-padded sample at a time do not need one.
    """
    lengths = np.asarray(lengths, dtype=int)
    if np.any(lengths > max_len):
        raise ValueError(f"lengths contain values greater than max_len={max_len}: {lengths[lengths > max_len]}")
    positions = np.arange(max_len)[None, :]
    return positions < lengths[:, None]


def build_response_vectors(
    activations: np.ndarray,
    active_mask: np.ndarray | None = None,
    node_axis: int = -1,
) -> np.ndarray:
    """README Sec 2.3 point 2 / Sec 3.1.10: build one functional response
    vector h_j per hidden node j, by flattening (batch, ..., active
    positions) for that node's channel, discarding padding positions.

    `activations` shape convention: (batch, seq, hidden) is the common
    transformer case (node_axis=-1, i.e. hidden is the last axis -- this
    is exactly graph_profiler's `activation_input` tensor for a
    Linear/Gemm/MatMul operator). `active_mask` shape (batch, seq), True
    where the position is a real (non-padding) frame/token.

    `active_mask=None` means "every position in this tensor is real" and
    is ONLY correct when no padding exists -- e.g. a single sample fed at
    its own exact length. Passing None on padded/batched activations
    silently folds padding into h_j, which Fig 2.5 explicitly forbids;
    use `active_mask_from_lengths()` to build the mask in that case.

    Returns h_vectors of shape (num_nodes, num_active_positions_total),
    matching what nnopt.grouping.functional_grouping.greedy_group expects.
    """
    arr = np.moveaxis(activations, node_axis, -1)  # (..., num_nodes)
    num_nodes = arr.shape[-1]
    flat = arr.reshape(-1, num_nodes)  # (batch*seq*..., num_nodes)

    if active_mask is not None:
        mask_flat = active_mask.reshape(-1)
        if mask_flat.shape[0] != flat.shape[0]:
            raise ValueError(
                f"active_mask flattens to {mask_flat.shape[0]} positions, "
                f"but activations flatten to {flat.shape[0]} positions -- "
                f"shapes are inconsistent with node_axis={node_axis}"
            )
        flat = flat[mask_flat.astype(bool)]

    return flat.T  # (num_nodes, num_active_positions)
