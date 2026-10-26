//! End-to-end tests of the JSON API on a model built in code.

use netron_core::proto::tensor_shape_proto::{dimension, Dimension};
use netron_core::proto::type_proto;
use netron_core::proto::*;
use netron_core::{NoSource, Workspace};
use prost::Message;
use serde_json::{json, Value};

fn dim_param(p: &str) -> Dimension {
    Dimension {
        denotation: String::new(),
        value: Some(dimension::Value::DimParam(p.into())),
    }
}

fn dim(v: i64) -> Dimension {
    Dimension {
        denotation: String::new(),
        value: Some(dimension::Value::DimValue(v)),
    }
}

fn tensor_type(elem: i32, dims: Vec<Dimension>) -> TypeProto {
    TypeProto {
        denotation: String::new(),
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: elem,
            shape: Some(TensorShapeProto { dim: dims }),
        })),
    }
}

fn vi(name: &str, tp: TypeProto) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(tp),
        ..Default::default()
    }
}

fn float_tensor(name: &str, dims: &[i64]) -> TensorProto {
    let n: i64 = dims.iter().product();
    TensorProto {
        name: name.into(),
        dims: dims.to_vec(),
        data_type: 1,
        float_data: (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect(),
        ..Default::default()
    }
}

fn int_tensor(name: &str, vals: &[i64]) -> TensorProto {
    TensorProto {
        name: name.into(),
        dims: vec![vals.len() as i64],
        data_type: 7,
        int64_data: vals.to_vec(),
        ..Default::default()
    }
}

fn attr_ints(name: &str, v: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: 7,
        ints: v.to_vec(),
        ..Default::default()
    }
}

fn attr_int(name: &str, v: i64) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        r#type: 2,
        i: v,
        ..Default::default()
    }
}

fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str], attrs: Vec<AttributeProto>) -> NodeProto {
    NodeProto {
        op_type: op.into(),
        name: name.into(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        attribute: attrs,
        ..Default::default()
    }
}

fn build_model() -> Vec<u8> {
    let graph = GraphProto {
        name: "test".into(),
        node: vec![
            node("Conv", "/features/conv/Conv", &["x", "W", "B"], &["c"], vec![attr_ints("kernel_shape", &[3, 3]), attr_ints("pads", &[1, 1, 1, 1])]),
            node("Relu", "/features/relu/Relu", &["c"], &["r"], vec![]),
            node("MaxPool", "/features/pool/MaxPool", &["r"], &["p"], vec![attr_ints("kernel_shape", &[2, 2]), attr_ints("strides", &[2, 2])]),
            node("Shape", "/head/Shape", &["p"], &["s"], vec![]),
            node("Gather", "/head/Gather", &["s", "zero"], &["n"], vec![attr_int("axis", 0)]),
            node("Unsqueeze", "/head/Unsqueeze", &["n", "axes0"], &["n1"], vec![]),
            node("Concat", "/head/Concat", &["n1", "minus1"], &["shape"], vec![attr_int("axis", 0)]),
            node("Reshape", "/head/Reshape", &["p", "shape"], &["flat"], vec![]),
            node("Gemm", "/head/fc/Gemm", &["flat", "FW", "FB"], &["logits"], vec![attr_int("transB", 1)]),
            node("Softmax", "/head/Softmax", &["logits"], &["y"], vec![attr_int("axis", 1)]),
            node("Identity", "", &["unused_w"], &["dead"], vec![]),
        ],
        initializer: vec![
            float_tensor("W", &[8, 3, 3, 3]),
            float_tensor("B", &[8]),
            float_tensor("FW", &[10, 8 * 16 * 16]),
            float_tensor("FB", &[10]),
            float_tensor("unused_w", &[4]),
            TensorProto {
                name: "zero".into(),
                data_type: 7,
                int64_data: vec![0],
                ..Default::default()
            },
            int_tensor("axes0", &[0]),
            int_tensor("minus1", &[-1]),
        ],
        input: vec![vi("x", tensor_type(1, vec![dim_param("N"), dim(3), dim(32), dim(32)]))],
        output: vec![vi("y", tensor_type(1, vec![dim_param("N"), dim(10)]))],
        ..Default::default()
    };
    let model = ModelProto {
        ir_version: 8,
        producer_name: "netron-plus-test".into(),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(graph),
        ..Default::default()
    };
    model.encode_to_vec()
}

fn call(ws: &Workspace, method: &str, params: Value) -> Value {
    let out = ws
        .call(method, &params)
        .unwrap_or_else(|e| panic!("{} failed: {:#}", method, e));
    serde_json::from_str(&out).unwrap()
}

fn find_value(table: &Value, name: &str) -> Value {
    table
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == name)
        .cloned()
        .unwrap_or_else(|| panic!("value {} not found", name))
}

#[test]
fn full_api_roundtrip() {
    let ws = Workspace::new();
    let summary: Value = serde_json::from_str(
        &ws.open_bytes("test.onnx".into(), None, build_model(), Box::new(NoSource))
            .expect("open"),
    )
    .unwrap();
    let id = summary["id"].as_u64().unwrap();
    assert_eq!(summary["stats"]["nodes"], 11);
    assert_eq!(summary["irVersion"], 8);
    assert_eq!(summary["inputs"][0]["name"], "x");
    let params = summary["totals"]["params"].as_u64().unwrap();
    assert_eq!(params, 8 * 27 + 8 + 10 * 2048 + 10 + 4);
    assert!(summary["symbols"].as_array().unwrap().iter().any(|s| s["name"] == "N"));
    // Costs are estimated with the dynamic batch taken as 1.
    assert_eq!(summary["totals"]["assumed"], json!(["N"]));
    assert!(summary["totals"]["flops"].as_u64().unwrap() > 0);

    // Shapes through the Shape→Gather→Unsqueeze→Concat→Reshape pattern.
    let values = call(&ws, "valueTable", json!({"model": id, "graph": 0}));
    assert_eq!(find_value(&values, "c")["shape"], "N×8×32×32");
    assert_eq!(find_value(&values, "p")["shape"], "N×8×16×16");
    assert_eq!(find_value(&values, "flat")["shape"], "N×2048");
    assert_eq!(find_value(&values, "logits")["shape"], "N×10");

    // Graph view.
    let view = call(&ws, "graph", json!({"model": id, "graph": 0, "options": {"showNames": true}}));
    let vnodes = view["nodes"].as_array().unwrap();
    assert!(vnodes.iter().any(|n| n["kind"] == "input"));
    assert!(vnodes.iter().any(|n| n["kind"] == "output"));
    let conv = vnodes.iter().find(|n| n["title"] == "Conv").unwrap();
    assert_eq!(conv["lines"].as_array().unwrap().len(), 2); // W + B
    assert_eq!(conv["name"], "/features/conv/Conv");
    for e in view["edges"].as_array().unwrap() {
        assert!(e["points"].as_array().unwrap().len() >= 4, "edge without route: {}", e);
    }
    // Shape → Gather → Unsqueeze → Concat is folded into the Reshape node.
    assert_eq!(view["foldedNodes"], 4);
    assert!(!vnodes.iter().any(|n| n["title"] == "Shape"));
    let reshape = vnodes.iter().find(|n| n["title"] == "Reshape").unwrap();
    assert!(
        reshape["lines"].as_array().unwrap().iter().any(|l| l["v"] == "= [N, -1]"),
        "{}",
        reshape
    );
    let unfolded = call(&ws, "graph", json!({"model": id, "graph": 0, "options": {"foldShapes": false}}));
    assert_eq!(unfolded["foldedNodes"], 0);
    assert!(unfolded["nodes"].as_array().unwrap().iter().any(|n| n["title"] == "Shape"));

    // Collapsing a module turns its nodes into one node; nested paths are
    // absorbed by the outermost collapsed module.
    let col = call(
        &ws,
        "graph",
        json!({"model": id, "graph": 0, "options": {"collapsed": ["features", "features/conv"], "foldShapes": false}}),
    );
    let cnodes = col["nodes"].as_array().unwrap();
    let mods: Vec<&Value> = cnodes.iter().filter(|n| n["kind"] == "module").collect();
    assert_eq!(mods.len(), 1, "{:?}", mods);
    assert_eq!(mods[0]["module"], "features");
    assert_eq!(mods[0]["members"], 3);
    assert_eq!(mods[0]["ids"], json!([0, 1, 2]));
    assert_eq!(mods[0]["value"], -1);
    assert_eq!(col["collapsedNodes"], 3);
    assert!(!cnodes.iter().any(|n| n["title"] == "Conv" || n["title"] == "MaxPool"));
    let module_index = cnodes.iter().position(|n| n["kind"] == "module").unwrap();
    let input_index = cnodes.iter().position(|n| n["kind"] == "input").unwrap();
    let cedges = col["edges"].as_array().unwrap();
    assert!(cedges.iter().all(|e| e["from"] != e["to"]));
    assert!(cedges
        .iter()
        .any(|e| e["from"] == input_index && e["to"] == module_index));
    assert!(cedges.iter().any(|e| e["from"] == module_index));
    for e in cedges {
        assert!(e["points"].as_array().unwrap().len() >= 4, "edge without route: {}", e);
    }
    // Collapsing everything still yields a valid graph.
    let all = call(
        &ws,
        "graph",
        json!({"model": id, "graph": 0, "options": {"collapsed": ["features", "head"]}}),
    );
    let all_mods = all["nodes"].as_array().unwrap().iter().filter(|n| n["kind"] == "module").count();
    assert_eq!(all_mods, 2);

    // Overrides make every shape concrete and enable FLOPs.
    let s2 = call(&ws, "setOverrides", json!({"model": id, "params": {"N": 2}}));
    assert_eq!(s2["outputs"][0]["shape"], json!([2, 10]));
    let flops = s2["totals"]["flops"].as_u64().unwrap();
    let conv_macs = 2 * 8 * 32 * 32 * 27;
    assert!(flops > 2 * conv_macs, "flops {}", flops);
    assert!(s2["memory"]["peakBytes"].as_u64().unwrap() > 0);

    // Node and value details.
    let nodes = call(&ws, "nodeTable", json!({"model": id, "graph": 0}));
    let conv_row = nodes.as_array().unwrap().iter().find(|r| r["op"] == "Conv").unwrap().clone();
    assert_eq!(conv_row["macs"].as_u64().unwrap(), conv_macs);
    let detail = call(&ws, "node", json!({"model": id, "graph": 0, "node": 0}));
    assert_eq!(detail["inputs"][1]["label"], "W");
    assert!(detail["docUrl"].as_str().unwrap().contains("onnx__Conv"));
    let w = find_value(&values, "W");
    let wd = call(&ws, "value", json!({"model": id, "graph": 0, "value": w["id"]}));
    let ti = wd["tensor"]["index"].as_u64().unwrap();
    let stats = call(&ws, "tensorStats", json!({"model": id, "graph": 0, "tensor": ti}));
    assert_eq!(stats["count"], 216);
    assert!(stats["histogram"]["counts"].as_array().unwrap().len() > 1);
    let page = call(&ws, "tensorValues", json!({"model": id, "graph": 0, "tensor": ti, "offset": 10, "count": 5}));
    assert_eq!(page["values"].as_array().unwrap().len(), 5);
    let npy = ws.export(&json!({"model": id, "graph": 0, "tensor": ti, "format": "npy"})).unwrap();
    assert_eq!(&npy[..6], b"\x93NUMPY");

    // Health checks find the dead Identity node and the unused initializer path.
    let issues = call(&ws, "checks", json!({"model": id, "deep": true}));
    let codes: Vec<String> = issues
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["code"].as_str().unwrap().to_string())
        .collect();
    assert!(codes.contains(&"dead-node".to_string()), "{:?}", codes);
    assert!(codes.contains(&"dynamic-input".to_string()), "{:?}", codes);

    let tree = call(&ws, "hierarchy", json!({"model": id, "graph": 0}));
    assert!(tree["children"].as_array().unwrap().iter().any(|c| c["name"] == "features"));
    let hits = call(&ws, "search", json!({"model": id, "query": "op:Conv"}));
    assert_eq!(hits.as_array().unwrap().len(), 1);
    let weights = call(&ws, "weightTable", json!({"model": id}));
    assert!(weights.as_array().unwrap().len() >= 5);
}

fn open(ws: &Workspace, bytes: Vec<u8>) -> (u64, Value) {
    let summary: Value = serde_json::from_str(
        &ws.open_bytes("test.onnx".into(), None, bytes, Box::new(NoSource))
            .expect("open"),
    )
    .unwrap();
    (summary["id"].as_u64().unwrap(), summary)
}

fn symbol(f: &Value, tex: &str) -> Value {
    f["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["tex"] == tex)
        .cloned()
        .unwrap_or_else(|| panic!("symbol {} not found in {}", tex, f))
}

#[test]
fn op_functions() {
    let ws = Workspace::new();
    let (id, _) = open(&ws, build_model());

    // Conv: attributes from the model, defaults, and values derived from shapes.
    let conv = call(&ws, "opFunction", json!({"model": id, "graph": 0, "node": 0}));
    assert_eq!(conv["op"], "Conv");
    assert_eq!(conv["node"], 0);
    assert!(conv["since"].as_i64().is_some());
    let formulas = conv["formulas"].as_array().unwrap();
    assert!(formulas.iter().any(|f| f["label"] == ""));
    assert!(formulas.iter().any(|f| f["label"] == "shape"));
    let k = symbol(&conv, r"\mathbf{K}");
    assert_eq!(k["value"], r"(3,\, 3)");
    assert_eq!(k["source"], "set");
    assert_eq!(k["kind"], "attr");
    assert_eq!(symbol(&conv, r"\mathbf{s}")["source"], "default");
    let c = symbol(&conv, "C");
    assert_eq!(c["value"], "3");
    assert_eq!(c["source"], "derived");
    assert_eq!(c["binding"], "X.shape[1]");
    assert_eq!(symbol(&conv, "M")["value"], "8");
    assert_eq!(
        symbol(&conv, r"\operatorname{shape}(X)")["value"],
        r"(\textit{N},\, 3,\, 32,\, 32)"
    );

    // Constant inputs and shape arithmetic provide input values.
    let gather = call(&ws, "opFunction", json!({"model": id, "graph": 0, "node": 4}));
    assert_eq!(symbol(&gather, r"\text{indices}")["value"], "0");
    let reshape = call(&ws, "opFunction", json!({"model": id, "graph": 0, "node": 7}));
    assert_eq!(symbol(&reshape, r"\text{shape}")["value"], r"(\textit{N},\, -1)");

    // The entry matching the opset is used (Softmax changed in opset 13).
    let softmax = call(&ws, "opFunction", json!({"model": id, "graph": 0, "node": 9}));
    assert_eq!(softmax["since"], 13);
    assert_eq!(symbol(&softmax, r"\text{axis}")["value"], "1");
    assert_eq!(symbol(&softmax, r"\text{axis}")["source"], "set");
    let gemm = call(&ws, "opFunction", json!({"model": id, "graph": 0, "node": 8}));
    assert_eq!(symbol(&gemm, r"\text{transB}")["value"], "1");
    assert_eq!(symbol(&gemm, r"\alpha")["source"], "default");
    assert_eq!(symbol(&gemm, r"\alpha")["value"], "1");

    // Generic definitions by operator type; unknown operators have none.
    let relu = call(&ws, "opFunction", json!({"model": id, "op": "Relu"}));
    assert!(relu["node"].is_null());
    assert!(relu["docUrl"].as_str().unwrap().contains("onnx__Relu"));
    assert!(!relu["summary"]["zh"].as_str().unwrap().is_empty());
    let custom = call(&ws, "opFunction", json!({"model": id, "op": "Foo", "domain": "my.domain"}));
    assert!(custom["since"].is_null());
    assert!(custom["formulas"].as_array().unwrap().is_empty());
    assert!(ws.call("opFunction", &json!({"model": id})).is_err());
}

#[test]
fn local_functions_link_to_call_sites() {
    let swish = FunctionProto {
        name: "Swish".into(),
        domain: "local".into(),
        input: vec!["X".into()],
        output: vec!["Y".into()],
        node: vec![
            node("Sigmoid", "", &["X"], &["S"], vec![]),
            node("Mul", "", &["X", "S"], &["Y"], vec![]),
        ],
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 18,
        }],
        ..Default::default()
    };
    // An overload of the same function whose body references an attribute.
    let leaky = FunctionProto {
        name: "Swish".into(),
        domain: "local".into(),
        overload: "leaky".into(),
        input: vec!["X".into()],
        output: vec!["Y".into()],
        attribute: vec!["a".into()],
        node: vec![node(
            "LeakyRelu",
            "",
            &["X"],
            &["Y"],
            vec![AttributeProto {
                name: "alpha".into(),
                r#type: 1,
                ref_attr_name: "a".into(),
                ..Default::default()
            }],
        )],
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 18,
        }],
        ..Default::default()
    };
    let mut caller = node("Swish", "swish", &["x"], &["y"], vec![]);
    caller.domain = "local".into();
    let mut caller2 = node(
        "Swish",
        "swish_leaky",
        &["y"],
        &["y2"],
        vec![AttributeProto {
            name: "a".into(),
            r#type: 1,
            f: 0.2,
            ..Default::default()
        }],
    );
    caller2.domain = "local".into();
    caller2.overload = "leaky".into();
    let graph = GraphProto {
        name: "g".into(),
        node: vec![caller, caller2, node("Relu", "relu", &["y2"], &["z"], vec![])],
        input: vec![vi("x", tensor_type(1, vec![dim(1), dim(4)]))],
        output: vec![vi("z", tensor_type(1, vec![dim(1), dim(4)]))],
        ..Default::default()
    };
    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: 18,
            },
            OperatorSetIdProto {
                domain: "local".into(),
                version: 1,
            },
        ],
        graph: Some(graph),
        functions: vec![swish, leaky],
        ..Default::default()
    };
    let ws = Workspace::new();
    let (id, summary) = open(&ws, model.encode_to_vec());
    let graph_id = |label: &str| {
        summary["graphs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["label"] == label)
            .unwrap_or_else(|| panic!("graph {} not found", label))["id"]
            .as_u64()
            .unwrap()
    };
    let fg = graph_id("function local::Swish");
    let fg2 = graph_id("function local::Swish:leaky");
    assert_ne!(fg, fg2);

    // Calls resolve by (domain, name, overload).
    let detail = call(&ws, "node", json!({"model": id, "graph": 0, "node": 0}));
    assert_eq!(detail["function"]["id"], fg);
    assert_eq!(detail["function"]["nodes"], 2);
    let detail2 = call(&ws, "node", json!({"model": id, "graph": 0, "node": 1}));
    assert_eq!(detail2["function"]["id"], fg2);
    assert_eq!(detail2["function"]["nodes"], 1);
    let relu = call(&ws, "node", json!({"model": id, "graph": 0, "node": 2}));
    assert!(relu["function"].is_null());

    let view = call(&ws, "graph", json!({"model": id, "graph": 0}));
    let vnodes = view["nodes"].as_array().unwrap();
    let swish_box = vnodes.iter().find(|n| n["node"] == 0).unwrap();
    assert_eq!(swish_box["func"], fg);
    let leaky_box = vnodes.iter().find(|n| n["node"] == 1).unwrap();
    assert_eq!(leaky_box["func"], fg2);
    let relu_box = vnodes.iter().find(|n| n["title"] == "Relu").unwrap();
    assert!(relu_box.get("func").is_none());

    let f = call(&ws, "opFunction", json!({"model": id, "graph": 0, "node": 0}));
    assert_eq!(f["function"]["id"], fg);
    assert!(f["since"].is_null());
    let by_type = call(&ws, "opFunction", json!({"model": id, "op": "Swish", "domain": "local"}));
    assert_eq!(by_type["function"]["id"], fg);

    // An attribute referencing a function parameter has no value of its own.
    let inner = call(&ws, "opFunction", json!({"model": id, "graph": fg2, "node": 0}));
    assert_eq!(inner["op"], "LeakyRelu");
    let alpha = symbol(&inner, r"\alpha");
    assert_eq!(alpha["source"], "unknown");
    assert!(alpha["value"].is_null());
    assert_eq!(alpha["binding"], "alpha = @a");
}

#[test]
fn reduction_axes_follow_noop_flag() {
    let reduce = |op: &str, name: &str, inputs: &[&str], noop: Option<i64>| {
        let attrs = noop.map(|v| vec![attr_int("noop_with_empty_axes", v)]).unwrap_or_default();
        node(op, name, inputs, &[name], attrs)
    };
    let graph = GraphProto {
        name: "reduce".into(),
        node: vec![
            reduce("ReduceSum", "noop", &["x"], Some(1)),
            reduce("ReduceSum", "empty", &["x", "none"], None),
            reduce("ReduceSum", "given", &["x", "ax"], Some(1)),
            reduce("ReduceMax", "omitted", &["x"], None),
        ],
        initializer: vec![
            TensorProto {
                name: "none".into(),
                dims: vec![0],
                data_type: 7,
                ..Default::default()
            },
            int_tensor("ax", &[1]),
        ],
        input: vec![vi("x", tensor_type(1, vec![dim(2), dim(3)]))],
        output: ["noop", "empty", "given", "omitted"]
            .iter()
            .map(|n| ValueInfoProto {
                name: n.to_string(),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 18,
        }],
        graph: Some(graph),
        ..Default::default()
    };
    let ws = Workspace::new();
    let (id, _) = open(&ws, model.encode_to_vec());
    let axes = |n: usize| {
        let f = call(&ws, "opFunction", json!({"model": id, "graph": 0, "node": n}));
        symbol(&f, r"\mathcal{A}")["value"].as_str().unwrap().to_string()
    };
    // Missing axes with noop_with_empty_axes = 1: nothing is reduced.
    assert_eq!(axes(0), r"\varnothing\ (\text{no reduction})");
    // An empty axes tensor with the default flag reduces every axis.
    assert_eq!(axes(1), r"(0,\, 1)");
    // Given axes win over the flag.
    assert_eq!(axes(2), "(1)");
    assert_eq!(axes(3), r"(0,\, 1)");
}

#[test]
fn rejects_garbage() {
    let ws = Workspace::new();
    assert!(ws
        .open_bytes("x.onnx".into(), None, b"hello world, not a model".to_vec(), Box::new(NoSource))
        .is_err());
    assert!(ws
        .open_bytes("x.onnx".into(), None, Vec::new(), Box::new(NoSource))
        .is_err());
}
