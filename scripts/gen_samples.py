#!/usr/bin/env python3
"""Generate small ONNX models that exercise Netron Plus.

Every model is written as <name>.onnx together with <name>.shapes.json, the
result of ONNX's reference shape inference on the same model, which the Rust
tests compare against the app's own inference. `index.json` lists the models
for the web demo.
"""
import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

rng = np.random.default_rng(0)


def init(name, shape, dtype=np.float32, scale=0.1):
    return numpy_helper.from_array((rng.standard_normal(shape) * scale).astype(dtype), name)


def const_i64(name, values):
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name)


def ref_shapes(model):
    stripped = onnx.ModelProto()
    stripped.CopyFrom(model)
    del stripped.graph.value_info[:]
    try:
        inferred = shape_inference.infer_shapes(stripped, data_prop=True)
    except Exception as e:  # noqa: BLE001
        print(f"  shape inference failed: {e}")
        return None
    out = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
        t = vi.type.tensor_type
        if not vi.type.HasField("tensor_type"):
            continue
        if not t.HasField("shape"):
            out[vi.name] = {"dtype": t.elem_type, "shape": None}
            continue
        dims = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append(None)
        out[vi.name] = {"dtype": t.elem_type, "shape": dims}
    return out


def save(model, out_dir, name, desc, index, external=False, check=True):
    path = os.path.join(out_dir, name + ".onnx")
    if check:
        try:
            onnx.checker.check_model(model)
        except Exception as e:  # noqa: BLE001
            print(f"  checker warning for {name}: {e}")
    shapes = ref_shapes(model)
    if external:
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=name + ".onnx.data",
            size_threshold=0,
        )
    else:
        onnx.save_model(model, path)
    if shapes is not None:
        with open(os.path.join(out_dir, name + ".shapes.json"), "w") as f:
            json.dump(shapes, f, indent=1)
    index.append({"name": name, "file": name + ".onnx", "description": desc})
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


class Builder:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def node(self, op, inputs, name=None, outputs=1, **attrs):
        self.n += 1
        base = name or f"/{op}_{self.n}"
        outs = [f"{base}_output_{i}" for i in range(outputs)]
        self.nodes.append(helper.make_node(op, inputs, outs, name=base, **attrs))
        return outs[0] if outputs == 1 else outs

    def weight(self, name, shape, dtype=np.float32, scale=0.1):
        self.inits.append(init(name, shape, dtype, scale))
        return name

    def i64(self, name, values):
        self.inits.append(const_i64(name, values))
        return name


def tiny_cnn(out, index):
    b = Builder()

    def conv(x, cin, cout, k, s, name):
        w = b.weight(f"{name}.weight", (cout, cin, k, k))
        bias = b.weight(f"{name}.bias", (cout,))
        return b.node("Conv", [x, w, bias], name=f"/{name}/Conv", kernel_shape=[k, k], strides=[s, s], pads=[k // 2] * 4)

    def bn(x, c, name):
        s = b.weight(f"{name}.weight", (c,))
        bb = b.weight(f"{name}.bias", (c,))
        m = b.weight(f"{name}.running_mean", (c,))
        b.inits.append(numpy_helper.from_array(np.abs(rng.standard_normal(c)).astype(np.float32) + 0.5, f"{name}.running_var"))
        return b.node("BatchNormalization", [x, s, bb, m, f"{name}.running_var"], name=f"/{name}/BatchNormalization", epsilon=1e-5)

    def relu(x, name):
        return b.node("Relu", [x], name=f"/{name}/Relu")

    x = relu(bn(conv("input", 3, 16, 3, 1, "stem.conv"), 16, "stem.bn"), "stem")
    x = b.node("MaxPool", [x], name="/stem/MaxPool", kernel_shape=[2, 2], strides=[2, 2])
    y = relu(bn(conv(x, 16, 16, 3, 1, "layer1.0.conv1"), 16, "layer1.0.bn1"), "layer1.0.relu1")
    y = bn(conv(y, 16, 16, 3, 1, "layer1.0.conv2"), 16, "layer1.0.bn2")
    x = relu(b.node("Add", [x, y], name="/layer1.0/Add"), "layer1.0")
    y = relu(bn(conv(x, 16, 32, 3, 2, "layer2.0.conv1"), 32, "layer2.0.bn1"), "layer2.0.relu1")
    y = bn(conv(y, 32, 32, 3, 1, "layer2.0.conv2"), 32, "layer2.0.bn2")
    sc = bn(conv(x, 16, 32, 1, 2, "layer2.0.downsample.0"), 32, "layer2.0.downsample.1")
    x = relu(b.node("Add", [sc, y], name="/layer2.0/Add"), "layer2.0")
    x = b.node("GlobalAveragePool", [x], name="/GlobalAveragePool")
    x = b.node("Flatten", [x], name="/Flatten", axis=1)
    w = b.weight("fc.weight", (10, 32))
    bias = b.weight("fc.bias", (10,))
    x = b.node("Gemm", [x, w, bias], name="/fc/Gemm", transB=1)
    b.nodes.append(helper.make_node("Softmax", [x], ["probs"], name="/Softmax", axis=1))
    g = helper.make_graph(
        b.nodes,
        "tiny_cnn",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, 32, 32])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["N", 10])],
        b.inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="netron-plus-samples")
    m.doc_string = "Small ResNet-style CNN with a dynamic batch dimension."
    helper.set_model_props(m, {"author": "netron-plus", "task": "classification"})
    save(m, out, "tiny_cnn", "ResNet-style CNN, dynamic batch", index)


def fp16_detector(out, index):
    b = Builder()
    f16 = np.float16

    def conv(x, cin, cout, k, s, name):
        w = b.weight(f"{name}.weight", (cout, cin, k, k), f16)
        bias = b.weight(f"{name}.bias", (cout,), f16)
        return b.node("Conv", [x, w, bias], name=f"/{name}/Conv", kernel_shape=[k, k], strides=[s, s], pads=[k // 2] * 4)

    def silu(x, name):
        s = b.node("Sigmoid", [x], name=f"/{name}/Sigmoid")
        return b.node("Mul", [x, s], name=f"/{name}/Mul")

    x0 = silu(conv("images", 3, 16, 3, 2, "model.0.conv"), "model.0.act")
    x1 = silu(conv(x0, 16, 32, 3, 2, "model.1.conv"), "model.1.act")
    x2 = silu(conv(x1, 32, 32, 3, 2, "model.2.conv"), "model.2.act")
    b.inits.append(numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), "up.scales"))
    up = b.node("Resize", [x2, "", "up.scales"], name="/model.3/Resize", mode="nearest")
    cat = b.node("Concat", [up, x1], name="/model.4/Concat", axis=1)
    h = silu(conv(cat, 64, 32, 3, 1, "model.5.conv"), "model.5.act")
    det = conv(h, 32, 18, 1, 1, "model.6.m.0")
    shp = b.i64("det.shape", [1, 3, 6, -1])
    r = b.node("Reshape", [det, shp], name="/model.6/Reshape")
    t = b.node("Transpose", [r], name="/model.6/Transpose", perm=[0, 1, 3, 2])
    split = b.i64("det.split", [2, 2, 2])
    xy, wh, conf = b.node("Split", [t, split], name="/model.6/Split", outputs=3, axis=3)
    xy = b.node("Sigmoid", [xy], name="/model.6/xy/Sigmoid")
    conf = b.node("Sigmoid", [conf], name="/model.6/conf/Sigmoid")
    wh = b.node("Exp", [wh], name="/model.6/wh/Exp")
    b.nodes.append(helper.make_node("Concat", [xy, wh, conf], ["output0"], name="/model.6/Concat_out", axis=3))
    g = helper.make_graph(
        b.nodes,
        "fp16_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT16, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT16, [1, 3, 64, 6])],
        b.inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], producer_name="netron-plus-samples")
    save(m, out, "fp16_detector", "YOLO-like detection head with float16 weights", index)


def transformer_block(out, index):
    b = Builder()
    d, heads, hd, ff = 64, 4, 16, 256
    emb = b.weight("embed.weight", (1000, d))
    x = b.node("Gather", [emb, "input_ids"], name="/embed/Gather", axis=0)

    def layernorm(x, name):
        s = b.weight(f"{name}.weight", (d,))
        bb = b.weight(f"{name}.bias", (d,))
        return b.node("LayerNormalization", [x, s, bb], name=f"/{name}/LayerNormalization", axis=-1, epsilon=1e-5)

    def linear(x, din, dout, name):
        w = b.weight(f"{name}.weight", (din, dout))
        bias = b.weight(f"{name}.bias", (dout,))
        y = b.node("MatMul", [x, w], name=f"/{name}/MatMul")
        return b.node("Add", [y, bias], name=f"/{name}/Add")

    h = layernorm(x, "ln1")
    shape = b.node("Shape", [h], name="/attn/Shape")
    zero = b.i64("c0", [0])
    one = b.i64("c1", [1])
    bt = b.node("Gather", [shape, zero], name="/attn/Gather_b", axis=0)
    st = b.node("Gather", [shape, one], name="/attn/Gather_s", axis=0)
    heads_c = b.i64("heads", [heads])
    hd_c = b.i64("head_dim", [hd])
    d_c = b.i64("model_dim", [d])
    split_shape = b.node("Concat", [bt, st, heads_c, hd_c], name="/attn/Concat", axis=0)
    merge_shape = b.node("Concat", [bt, st, d_c], name="/attn/Concat_merge", axis=0)

    def heads_of(t, name, perm):
        r = b.node("Reshape", [t, split_shape], name=f"/attn/{name}/Reshape")
        return b.node("Transpose", [r], name=f"/attn/{name}/Transpose", perm=perm)

    q = heads_of(linear(h, d, d, "attn.q"), "q", [0, 2, 1, 3])
    k = heads_of(linear(h, d, d, "attn.k"), "k", [0, 2, 3, 1])
    v = heads_of(linear(h, d, d, "attn.v"), "v", [0, 2, 1, 3])
    scores = b.node("MatMul", [q, k], name="/attn/MatMul_qk")
    b.inits.append(numpy_helper.from_array(np.array(4.0, dtype=np.float32), "attn.scale"))
    scores = b.node("Div", [scores, "attn.scale"], name="/attn/Div")
    probs = b.node("Softmax", [scores], name="/attn/Softmax", axis=-1)
    ctx = b.node("MatMul", [probs, v], name="/attn/MatMul_v")
    ctx = b.node("Transpose", [ctx], name="/attn/Transpose_out", perm=[0, 2, 1, 3])
    ctx = b.node("Reshape", [ctx, merge_shape], name="/attn/Reshape_out")
    x = b.node("Add", [x, linear(ctx, d, d, "attn.o")], name="/residual1/Add")
    h = layernorm(x, "ln2")
    h = b.node("Gelu", [linear(h, d, ff, "mlp.fc1")], name="/mlp/Gelu")
    y = linear(h, ff, d, "mlp.fc2")
    b.nodes.append(helper.make_node("Add", [x, y], ["hidden_states"], name="/residual2/Add"))
    g = helper.make_graph(
        b.nodes,
        "transformer_block",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])],
        [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, ["batch", "seq", d])],
        b.inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)], producer_name="netron-plus-samples")
    save(m, out, "transformer_block", "Transformer block with dynamic batch/sequence (Shape→Gather→Concat→Reshape)", index)


def control_flow(out, index):
    then_g = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["then_out"], name="then_add")],
        "then_branch",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2, 3])],
    )
    else_g = helper.make_graph(
        [helper.make_node("Mul", ["x", "x"], ["else_out"], name="else_mul")],
        "else_branch",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2, 3])],
    )
    body = helper.make_graph(
        [
            helper.make_node("Add", ["v_in", "x"], ["v_out"], name="body_add"),
            helper.make_node("Identity", ["cond_in"], ["cond_out"], name="body_cond"),
            helper.make_node("Identity", ["v_out"], ["scan_out"], name="body_scan"),
        ],
        "loop_body",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_in", TensorProto.FLOAT, [2, 3]),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("v_out", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [2, 3]),
        ],
    )
    nodes = [
        helper.make_node("If", ["cond"], ["y"], name="branch", then_branch=then_g, else_branch=else_g),
        helper.make_node("Loop", ["trip", "true", "y"], ["v_final", "scans"], name="accumulate", body=body),
    ]
    g = helper.make_graph(
        nodes,
        "control_flow",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [
            helper.make_tensor_value_info("v_final", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("scans", TensorProto.FLOAT, [None, 2, 3]),
        ],
        [
            numpy_helper.from_array(np.array(3, dtype=np.int64), "trip"),
            numpy_helper.from_array(np.array(True), "true"),
        ],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 16)], producer_name="netron-plus-samples")
    save(m, out, "control_flow", "If and Loop subgraphs with outer-scope references", index)


def quantized(out, index):
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zp"], ["qx"], name="quant"),
        helper.make_node(
            "QLinearConv",
            ["qx", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp", "bias"],
            ["qy"],
            name="qconv",
            kernel_shape=[3, 3],
        ),
        helper.make_node("DequantizeLinear", ["qy", "y_scale", "y_zp"], ["y"], name="dequant"),
        helper.make_node("Flatten", ["qx"], ["qx_flat"], name="flatten", axis=1),
        helper.make_node("MatMulInteger", ["qx_flat", "mw", "x_zp", "mw_zp"], ["mm"], name="matmul_int"),
    ]
    inits = [
        numpy_helper.from_array(np.array(0.02, dtype=np.float32), "x_scale"),
        numpy_helper.from_array(np.array(128, dtype=np.uint8), "x_zp"),
        numpy_helper.from_array(rng.integers(-127, 127, (8, 4, 3, 3)).astype(np.int8), "w"),
        numpy_helper.from_array(np.full(8, 0.01, dtype=np.float32), "w_scale"),
        numpy_helper.from_array(np.zeros(8, dtype=np.int8), "w_zp"),
        numpy_helper.from_array(np.array(0.05, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.array(128, dtype=np.uint8), "y_zp"),
        numpy_helper.from_array(rng.integers(-1000, 1000, 8).astype(np.int32), "bias"),
        numpy_helper.from_array(rng.integers(0, 255, (256, 16)).astype(np.uint8), "mw"),
        numpy_helper.from_array(np.array(128, dtype=np.uint8), "mw_zp"),
    ]
    g = helper.make_graph(
        nodes,
        "quantized",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 6, 6]),
            helper.make_tensor_value_info("mm", TensorProto.INT32, [1, 16]),
        ],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], producer_name="netron-plus-samples")
    save(m, out, "quantized", "QDQ / QLinearConv / MatMulInteger with int8 and uint8 weights", index)


def external_data(out, index):
    b = Builder()
    w1 = b.weight("fc1.weight", (64, 128))
    w2 = b.weight("fc2.weight", (128, 10))
    h = b.node("MatMul", ["x", w1], name="/fc1/MatMul")
    h = b.node("Relu", [h], name="/relu/Relu")
    b.nodes.append(helper.make_node("MatMul", [h, w2], ["logits"], name="/fc2/MatMul"))
    g = helper.make_graph(
        b.nodes,
        "external_mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 64])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 10])],
        b.inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="netron-plus-samples")
    save(m, out, "external_data", "MLP whose weights live in an external .data file", index, external=True)


def all_dtypes(out, index):
    specs = [
        ("t_float", TensorProto.FLOAT, [1.5, -2.25, 0.0, 3.0]),
        ("t_double", TensorProto.DOUBLE, [1.5, -2.25, 1e-300, 3.0]),
        ("t_float16", TensorProto.FLOAT16, [1.5, -2.25, 65504.0, 0.0]),
        ("t_bfloat16", TensorProto.BFLOAT16, [1.5, -2.25, 1e30, 0.0]),
        ("t_int8", TensorProto.INT8, [-128, -1, 0, 127]),
        ("t_uint8", TensorProto.UINT8, [0, 1, 128, 255]),
        ("t_int16", TensorProto.INT16, [-32768, -1, 0, 32767]),
        ("t_uint16", TensorProto.UINT16, [0, 1, 1000, 65535]),
        ("t_int32", TensorProto.INT32, [-(2**31), -1, 0, 2**31 - 1]),
        ("t_uint32", TensorProto.UINT32, [0, 1, 1000, 2**32 - 1]),
        ("t_int64", TensorProto.INT64, [-(2**40), -1, 0, 2**40]),
        ("t_uint64", TensorProto.UINT64, [0, 1, 1000, 2**63]),
        ("t_bool", TensorProto.BOOL, [True, False, True, True]),
        ("t_string", TensorProto.STRING, [b"alpha", b"beta", b"", b"\xe4\xb8\xad\xe6\x96\x87"]),
        ("t_float8e4m3fn", getattr(TensorProto, "FLOAT8E4M3FN", None), [1.0, -2.0, 448.0, 0.0]),
        ("t_float8e5m2", getattr(TensorProto, "FLOAT8E5M2", None), [1.0, -2.0, 0.5, 0.0]),
        ("t_int4", getattr(TensorProto, "INT4", None), [-8, -1, 0, 7]),
        ("t_uint4", getattr(TensorProto, "UINT4", None), [0, 1, 8, 15]),
    ]
    nodes, inits, outputs = [], [], []
    for name, dt, vals in specs:
        if dt is None:
            continue
        try:
            t = helper.make_tensor(name, dt, [4], vals)
        except Exception as e:  # noqa: BLE001
            print(f"  skipping {name}: {e}")
            continue
        inits.append(t)
        nodes.append(helper.make_node("Identity", [name], [name + "_out"], name="id_" + name))
        outputs.append(helper.make_tensor_value_info(name + "_out", dt, [4]))
    raw = numpy_helper.from_array(rng.standard_normal((3, 4, 5)).astype(np.float32), "t_raw3d")
    inits.append(raw)
    nodes.append(helper.make_node("Identity", ["t_raw3d"], ["t_raw3d_out"], name="id_raw3d"))
    outputs.append(helper.make_tensor_value_info("t_raw3d_out", TensorProto.FLOAT, [3, 4, 5]))
    g = helper.make_graph(nodes, "all_dtypes", [], outputs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21)], producer_name="netron-plus-samples")
    save(m, out, "all_dtypes", "Initializers of every ONNX element type", index)


def big_graph(out, index, blocks=300):
    b = Builder()
    x = "input"
    for i in range(blocks):
        w = b.weight(f"blocks.{i}.conv.weight", (8, 8, 1, 1))
        y = b.node("Conv", [x, w], name=f"/blocks.{i}/conv/Conv", kernel_shape=[1, 1])
        y = b.node("Relu", [y], name=f"/blocks.{i}/act/Relu")
        if i % 3 == 2:
            s = b.node("Sigmoid", [y], name=f"/blocks.{i}/gate/Sigmoid")
            y = b.node("Mul", [y, s], name=f"/blocks.{i}/gate/Mul")
        x = b.node("Add", [x, y], name=f"/blocks.{i}/Add")
    b.nodes.append(helper.make_node("Identity", [x], ["output"], name="/output/Identity"))
    g = helper.make_graph(
        b.nodes,
        "big_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 16, 16])],
        b.inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="netron-plus-samples")
    save(m, out, "big_graph", f"{blocks} residual blocks (~{len(b.nodes)} nodes) for layout stress tests", index)


def local_functions(out, index):
    swish = helper.make_function(
        "local",
        "Swish",
        ["X"],
        ["Y"],
        [
            helper.make_node("Sigmoid", ["X"], ["S"]),
            helper.make_node("Mul", ["X", "S"], ["Y"]),
        ],
        [helper.make_opsetid("", 18)],
    )
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["c"], name="conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Swish", ["c"], ["y"], name="swish", domain="local"),
    ]
    g = helper.make_graph(
        nodes,
        "functions",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 16, 16])],
        [init("w", (4, 3, 3, 3))],
    )
    m = helper.make_model(
        g,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("local", 1)],
        functions=[swish],
        producer_name="netron-plus-samples",
    )
    save(m, out, "functions", "Model-local function (Swish) called from the graph", index)


def custom_attrs(out, index):
    sub = helper.make_graph(
        [helper.make_node("Neg", ["a"], ["b"])],
        "attr_graph",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])],
    )
    node = helper.make_node(
        "FancyOp",
        ["x", "", "w"],
        ["y", "unused"],
        name="fancy",
        domain="custom.test",
        alpha=0.5,
        count=3,
        mode="fast",
        table=helper.make_tensor("table", TensorProto.FLOAT, [2, 2], [1, 2, 3, 4]),
        graph=sub,
        weights=[0.1, 0.2, 0.3],
        sizes=[1, 2, 3, 4],
        names=["a", "b"],
        tensors=[helper.make_tensor("t1", TensorProto.INT64, [2], [7, 8])],
    )
    dense = np.zeros((4, 5), dtype=np.float32)
    dense[0, 1] = 1.0
    dense[3, 4] = 2.0
    values = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), "sparse_w")
    indices = numpy_helper.from_array(np.array([1, 19], dtype=np.int64), "sparse_w_indices")
    sparse = helper.make_sparse_tensor(values, indices, [4, 5])
    g = helper.make_graph(
        [node, helper.make_node("Relu", ["y"], ["z"], name="relu")],
        "custom_attrs",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 5])],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, None)],
        [init("w", (5,))],
        sparse_initializer=[sparse],
    )
    m = helper.make_model(
        g,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("custom.test", 1)],
        producer_name="netron-plus-samples",
    )
    save(m, out, "custom_attrs", "Custom-domain op with every attribute type and a sparse initializer", index, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.json")
    index = []
    for fn in (tiny_cnn, fp16_detector, transformer_block, control_flow, quantized, external_data,
               all_dtypes, big_graph, local_functions, custom_attrs):
        try:
            fn(args.out, index)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {fn.__name__}: {e}")
            raise
    existing = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f)
    names = {e["name"] for e in index}
    merged = index + [e for e in existing if e["name"] not in names]
    with open(index_path, "w") as f:
        json.dump(merged, f, indent=1)


if __name__ == "__main__":
    main()
