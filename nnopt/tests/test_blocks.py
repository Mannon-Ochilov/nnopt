"""Tests for structural detection of the reducible block.

The point of finding the block by structure rather than by name is that the
framework should work on a transformer ASR model it has never seen. So these
tests build small graphs in several naming conventions -- Whisper's fc1/fc2,
wav2vec2's intermediate_dense/output_dense, a Conformer-style ffn -- and
require the same pair to be found in each, with no name ever consulted.

The failure that matters most is the quiet one: finding nothing. A model whose
block goes undetected reports zero reducible bytes, and the planner then
concludes no cache target is reachable when in fact the whole feed-forward
stack was available. Several tests below exist only to make that case loud.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nnopt.profiler.blocks import (
    breakdown,
    find_reducible_pairs,
    layer_index,
)
from nnopt.profiler.graph_profiler import profile_onnx_model

D_MODEL, D_FF, N_LAYERS = 16, 64, 2
FREE = {"batch": 1, "seq": 8}


def _linear(name, in_t, out_t, w_name, k, n, inits):
    inits.append(helper.make_tensor(
        w_name, TensorProto.FLOAT, [k, n],
        np.random.RandomState(0).randn(k, n).astype(np.float32).flatten()))
    return helper.make_node("MatMul", [in_t, w_name], [out_t], name=name)


def build_model(path, fmt="/layers.{i}/{op}/MatMul", ffn=("fc1", "fc2"),
                attention=True, n_layers=N_LAYERS):
    """A stack of blocks: optional attention pair, then an expanding FFN pair."""
    nodes, inits = [], []
    cur = "x"
    for i in range(n_layers):
        if attention:
            # v then o: same width in and out -- must NOT be taken as reducible.
            v = fmt.format(i=i, op="v_proj")
            o = fmt.format(i=i, op="out_proj")
            nodes.append(_linear(v, cur, f"v{i}", f"wv{i}", D_MODEL, D_MODEL, inits))
            nodes.append(helper.make_node("Add", [f"v{i}", f"v{i}"], [f"va{i}"],
                                          name=f"attn_add{i}"))
            nodes.append(_linear(o, f"va{i}", f"o{i}", f"wo{i}", D_MODEL, D_MODEL, inits))
            cur = f"o{i}"
        up = fmt.format(i=i, op=ffn[0])
        down = fmt.format(i=i, op=ffn[1])
        nodes.append(_linear(up, cur, f"h{i}", f"w1_{i}", D_MODEL, D_FF, inits))
        # An activation between the halves: the walk must pass through it.
        nodes.append(helper.make_node("Erf", [f"h{i}"], [f"g{i}"], name=f"act{i}"))
        nodes.append(_linear(down, f"g{i}", f"y{i}", f"w2_{i}", D_FF, D_MODEL, inits))
        cur = f"y{i}"

    graph = helper.make_graph(
        nodes, "synthetic_transformer",
        inputs=[helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, ["batch", "seq", D_MODEL])],
        outputs=[helper.make_tensor_value_info(
            cur, TensorProto.FLOAT, ["batch", "seq", D_MODEL])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def _pairs(path):
    profs = profile_onnx_model(path, free_dims=FREE)
    return find_reducible_pairs(onnx.load(path), profs)


def test_layer_index_reads_every_convention():
    assert layer_index("/layers.7/fc1/MatMul") == 7
    assert layer_index("/encoder/layer.3/output/dense") == 3
    assert layer_index("/blocks.11/mlp/fc") == 11
    assert layer_index("/h.5/attn/c_proj") == 5
    assert layer_index("something_without_a_number") == -1


@pytest.mark.parametrize("ffn,fmt", [
    (("fc1", "fc2"), "/layers.{i}/{op}/MatMul"),                 # Whisper
    (("intermediate_dense", "output_dense"), "/layer.{i}/{op}"),  # wav2vec2
    (("ffn1_up", "ffn1_down"), "/blocks.{i}/{op}/MatMul"),        # Conformer-ish
])
def test_block_is_found_regardless_of_naming(tmp_path, ffn, fmt):
    path = build_model(str(tmp_path / "m.onnx"), fmt=fmt, ffn=ffn)
    pairs = _pairs(path)
    assert len(pairs) == N_LAYERS
    for p in pairs:
        assert p.width == D_FF


def test_attention_pair_is_not_mistaken_for_the_block(tmp_path):
    """v -> o shares a width but does not expand it; taking it would promise
    reductions the structural stage cannot deliver there."""
    path = build_model(str(tmp_path / "m.onnx"))
    for p in _pairs(path):
        assert "proj" not in p.expand and "proj" not in p.contract


def test_pair_is_found_through_the_activation(tmp_path):
    path = build_model(str(tmp_path / "m.onnx"))
    pairs = _pairs(path)
    assert pairs and all(p.layer in range(N_LAYERS) for p in pairs)


def test_search_does_not_cross_a_matrix_operator(tmp_path):
    """Two FFNs in series must give two pairs, not four: the down-projection
    of block 0 ends the walk, so its up-projection cannot match block 1."""
    path = build_model(str(tmp_path / "m.onnx"), attention=False, n_layers=2)
    pairs = _pairs(path)
    assert len(pairs) == 2
    assert {p.layer for p in pairs} == {0, 1}


def test_breakdown_accounts_for_every_weight_once(tmp_path):
    path = build_model(str(tmp_path / "m.onnx"))
    profs = profile_onnx_model(path, free_dims=FREE)
    b = breakdown(path, profs)
    attn = 2 * D_MODEL * D_MODEL
    ffn = 2 * D_MODEL * D_FF
    assert b.n_layers == N_LAYERS
    assert b.per_layer_bytes == (attn + ffn) * 4
    assert b.reducible_bytes == ffn * 4
    assert b.fixed_bytes == attn * 4


def test_breakdown_picks_the_largest_layer(tmp_path):
    """The cache target binds on the worst layer, not the average one."""
    path = str(tmp_path / "m.onnx")
    build_model(path, n_layers=3)
    profs = profile_onnx_model(path, free_dims=FREE)
    b = breakdown(path, profs)
    assert b.largest_layer in range(3)
    assert b.reducible_bytes > 0


def test_model_without_a_reducible_block_reports_zero_not_a_guess(tmp_path):
    """An attention-only stack has nothing this method can remove, and the
    honest answer is zero rather than a nearest match."""
    path = str(tmp_path / "m.onnx")
    nodes, inits = [], []
    nodes.append(_linear("/layers.0/v_proj", "x", "v", "wv", D_MODEL, D_MODEL, inits))
    nodes.append(_linear("/layers.0/out_proj", "v", "o", "wo", D_MODEL, D_MODEL, inits))
    graph = helper.make_graph(
        nodes, "attn_only",
        inputs=[helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, ["batch", "seq", D_MODEL])],
        outputs=[helper.make_tensor_value_info(
            "o", TensorProto.FLOAT, ["batch", "seq", D_MODEL])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)

    profs = profile_onnx_model(path, free_dims=FREE)
    b = breakdown(path, profs)
    assert b.reducible_bytes == 0
    assert b.per_layer_bytes > 0


def test_breakdown_raises_when_no_layer_numbering_exists(tmp_path):
    path = str(tmp_path / "m.onnx")
    build_model(path, fmt="/{op}_plain")
    profs = profile_onnx_model(path, free_dims=FREE)
    with pytest.raises(ValueError, match="qatlamga"):
        breakdown(path, profs)
