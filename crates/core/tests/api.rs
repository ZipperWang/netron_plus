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
