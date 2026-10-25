//! Display helpers for attribute values.

use crate::dtype;
use crate::model::{AttrValue, Graph, GraphId, Model};
use crate::proto::TensorProto;
use crate::shape;
use crate::tensor;

pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{}…", t)
    }
}

fn list(items: impl Iterator<Item = String>, n: usize, max: usize) -> String {
    let v: Vec<String> = items.take(max).collect();
    let mut s = v.join(", ");
    if n > max {
        s.push_str(&format!(", … (+{})", n - max));
    }
    s
}

pub fn tensor_text(t: &TensorProto) -> String {
    let dims = shape::dims_label(Some(&shape::from_i64s(&t.dims)));
    match tensor::inline_values(t, 8) {
        Some(v) => format!("{} {}", dtype::name(t.data_type), v),
        None => format!("{}⟨{}⟩", dtype::name(t.data_type), dims),
    }
}

fn graph_text(model: &Model, gid: GraphId) -> String {
    match model.graphs.get(gid) {
        Some(sg) => {
            if sg.name.is_empty() {
                format!("graph ({} nodes)", sg.nodes.len())
            } else {
                format!("graph '{}' ({} nodes)", sg.name, sg.nodes.len())
            }
        }
        None => "graph".to_string(),
    }
}

fn attr_text(model: &Model, g: &Graph, v: &AttrValue, max_items: usize) -> String {
    match v {
        AttrValue::Undefined => String::new(),
        AttrValue::Float(f) => tensor::fmt_float(*f as f64),
        AttrValue::Int(i) => i.to_string(),
        AttrValue::String(s) => String::from_utf8_lossy(s).into_owned(),
        AttrValue::Tensor(ti) => tensor_text(&g.tensors[*ti]),
        AttrValue::Graph(gid) => graph_text(model, *gid),
        AttrValue::SparseTensor(sp) => format!(
            "sparse {}⟨{}⟩ nnz={}",
            dtype::name(sp.dtype),
            shape::dims_label(Some(&shape::from_i64s(&sp.dims))),
            sp.nnz
        ),
        AttrValue::Type(t) => t.display.clone(),
        AttrValue::Floats(v) => list(
            v.iter().map(|f| tensor::fmt_float(*f as f64)),
            v.len(),
            max_items,
        ),
        AttrValue::Ints(v) => list(v.iter().map(|i| i.to_string()), v.len(), max_items),
        AttrValue::Strings(v) => list(
            v.iter().map(|s| format!("{:?}", String::from_utf8_lossy(s))),
            v.len(),
            max_items,
        ),
        AttrValue::Tensors(v) => list(
            v.iter().map(|ti| tensor_text(&g.tensors[*ti])),
            v.len(),
            max_items,
        ),
        AttrValue::Graphs(v) => list(v.iter().map(|gid| graph_text(model, *gid)), v.len(), max_items),
        AttrValue::SparseTensors(v) => format!("{} sparse tensors", v.len()),
        AttrValue::Types(v) => list(v.iter().map(|t| t.display.clone()), v.len(), max_items),
    }
}

/// One-line rendering for node boxes.
pub fn attr_short(model: &Model, g: &Graph, v: &AttrValue, max_chars: usize) -> String {
    truncate(&attr_text(model, g, v, 16), max_chars)
}

/// Full rendering for the inspector.
pub fn attr_full(model: &Model, g: &Graph, v: &AttrValue) -> String {
    attr_text(model, g, v, 4096)
}
