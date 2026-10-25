//! Model-wide analysis: shape inference, costs, memory and totals.

use crate::cost::{node_cost, NodeCost};
use crate::dtype;
use crate::infer::{self, GraphShapes, Overrides};
use crate::model::{Model, ValueKind};
use crate::shape::{numel, Dim};
use crate::tensor;
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStats {
    pub weights_bytes: u64,
    /// Sum of all intermediate tensors (no buffer reuse).
    pub activation_bytes: u64,
    /// Peak of live tensors when executing nodes in topological order.
    pub peak_bytes: u64,
    pub peak_node: Option<usize>,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub known_values: usize,
    pub unknown_values: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Totals {
    pub params: u64,
    pub param_bytes: u64,
    pub macs: u64,
    pub flops: u64,
    pub costed_nodes: usize,
    pub uncosted_nodes: usize,
    pub uncosted_ops: Vec<String>,
    /// Dynamic input dimensions that were taken as 1 for costs and memory.
    pub assumed: Vec<String>,
}

pub struct Analysis {
    pub overrides: Overrides,
    pub shapes: Vec<GraphShapes>,
    pub costs: Vec<Vec<NodeCost>>,
    pub memory: MemoryStats,
    pub totals: Totals,
}

pub fn analyze(model: &Model, overrides: Overrides) -> Analysis {
    let shapes = infer::infer(model, &overrides);
    // Shapes shown to the user keep symbolic dimensions; costs and memory
    // are computed with every still-dynamic input dimension taken as 1.
    let (assumed, cost_overrides) = assume_dynamic(model, &overrides);
    let cost_owned: Vec<GraphShapes>;
    let cost_shapes: &[GraphShapes] = if assumed.is_empty() {
        &shapes
    } else {
        cost_owned = infer::infer(model, &cost_overrides);
        &cost_owned
    };
    let costs: Vec<Vec<NodeCost>> = model
        .graphs
        .iter()
        .zip(cost_shapes)
        .map(|(g, s)| g.nodes.iter().map(|n| node_cost(g, n, &s.states)).collect())
        .collect();
    let mut totals = compute_totals(model, &costs);
    totals.assumed = assumed;
    let memory = compute_memory(model, &cost_shapes[0]);
    Analysis {
        overrides,
        shapes,
        costs,
        memory,
        totals,
    }
}

/// Overrides that additionally set every unresolved dynamic dimension of the
/// main graph inputs to 1, plus the names of those dimensions.
fn assume_dynamic(model: &Model, ov: &Overrides) -> (Vec<String>, Overrides) {
    let main = model.main();
    let mut out = ov.clone();
    let mut assumed = Vec::new();
    for &vid in &main.inputs {
        let v = &main.values[vid];
        if v.kind != ValueKind::Input {
            continue;
        }
        let Some(shape) = v.declared.as_ref().and_then(|d| d.shape.as_ref()) else {
            continue;
        };
        let mut per: Vec<Option<i64>> = ov.inputs.get(&v.name).cloned().unwrap_or_default();
        if per.len() < shape.len() {
            per.resize(shape.len(), None);
        }
        let mut changed = false;
        for (i, d) in shape.iter().enumerate() {
            if per[i].is_some() {
                continue;
            }
            match d {
                Dim::Param(p) => {
                    if !out.params.contains_key(p) {
                        out.params.insert(p.clone(), 1);
                        assumed.push(p.clone());
                    }
                }
                Dim::Unknown => {
                    per[i] = Some(1);
                    changed = true;
                    assumed.push(format!("{}[{}]", v.name, i));
                }
                Dim::Known(_) => {}
            }
        }
        if changed {
            out.inputs.insert(v.name.clone(), per);
        }
    }
    (assumed, out)
}

fn compute_totals(model: &Model, costs: &[Vec<NodeCost>]) -> Totals {
    let mut t = Totals::default();
    for g in &model.graphs {
        for v in &g.values {
            if !matches!(v.kind, ValueKind::Initializer | ValueKind::Constant) {
                continue;
            }
            if let Some(ti) = v.tensor {
                let tp = &g.tensors[ti];
                if dtype::is_weight_type(tp.data_type) {
                    t.params += tensor::element_count(tp);
                    t.param_bytes += tensor::byte_size(tp).unwrap_or(0);
                }
            } else if let Some(sp) = &v.sparse {
                if dtype::is_weight_type(sp.dtype) {
                    t.params += sp.nnz;
                }
            }
        }
    }
    let main = model.main();
    let mut uncosted: BTreeSet<String> = BTreeSet::new();
    for (ni, c) in costs[0].iter().enumerate() {
        match c.flops {
            Some(f) => {
                t.flops = t.flops.saturating_add(f);
                t.macs = t.macs.saturating_add(c.macs.unwrap_or(0));
                t.costed_nodes += 1;
            }
            None => {
                t.uncosted_nodes += 1;
                if uncosted.len() < 24 {
                    uncosted.insert(main.nodes[ni].op_type.clone());
                }
            }
        }
    }
    t.uncosted_ops = uncosted.into_iter().collect();
    t
}

fn compute_memory(model: &Model, gs: &GraphShapes) -> MemoryStats {
    let g = model.main();
    let st = &gs.states;
    let bytes = |vid: usize| -> Option<u64> {
        let s = &st[vid];
        dtype::bytes_for(s.dtype, numel(s.shape.as_deref()?)?)
    };
    let mut m = MemoryStats::default();
    for v in &g.values {
        if matches!(v.kind, ValueKind::Initializer | ValueKind::Constant) {
            if let Some(ti) = v.tensor {
                m.weights_bytes += tensor::byte_size(&g.tensors[ti]).unwrap_or(0);
            }
        }
    }
    let mut pos = vec![0usize; g.nodes.len()];
    for (p, &ni) in g.topo.iter().enumerate() {
        pos[ni] = p;
    }
    let mut last: Vec<Option<usize>> = vec![None; g.values.len()];
    for (vid, v) in g.values.iter().enumerate() {
        for &(ni, _) in &v.consumers {
            let p = pos[ni];
            last[vid] = Some(last[vid].map_or(p, |l| l.max(p)));
        }
    }
    let mut live: u64 = 0;
    for &vid in &g.inputs {
        if g.values[vid].kind == ValueKind::Input {
            if let Some(b) = bytes(vid) {
                m.input_bytes += b;
                live += b;
            }
        }
    }
    for &vid in &g.outputs {
        if let Some(b) = bytes(vid) {
            m.output_bytes += b;
        }
    }
    m.peak_bytes = live;
    for (p, &ni) in g.topo.iter().enumerate() {
        let node = &g.nodes[ni];
        for &vid in node.outputs.iter().flatten() {
            if g.values[vid].kind == ValueKind::Constant {
                continue;
            }
            match bytes(vid) {
                Some(b) => {
                    live += b;
                    m.activation_bytes += b;
                    m.known_values += 1;
                }
                None => m.unknown_values += 1,
            }
        }
        if live > m.peak_bytes {
            m.peak_bytes = live;
            m.peak_node = Some(ni);
        }
        let mut freed: Vec<usize> = Vec::new();
        for &vid in node.inputs.iter().flatten().chain(node.implicit_inputs.iter()) {
            let v = &g.values[vid];
            if matches!(v.kind, ValueKind::Initializer | ValueKind::Constant)
                || v.is_output
                || freed.contains(&vid)
            {
                continue;
            }
            if last[vid] == Some(p) {
                if let Some(b) = bytes(vid) {
                    live = live.saturating_sub(b);
                }
                freed.push(vid);
            }
        }
        for &vid in node.outputs.iter().flatten() {
            let v = &g.values[vid];
            if v.consumers.is_empty() && !v.is_output && v.kind != ValueKind::Constant {
                if let Some(b) = bytes(vid) {
                    live = live.saturating_sub(b);
                }
            }
        }
    }
    m
}
