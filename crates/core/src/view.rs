//! Builds the renderable graph (boxes, lines and routed edges) for one graph.

use crate::analysis::Analysis;
use crate::dtype;
use crate::format;
use crate::layout::{self, LEdge, LNode, LayoutParams};
use crate::model::{module_path, Graph, GraphId, Model, ValueKind};
use crate::ops;
use crate::shape::{self, Dim};
use crate::tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FontMetrics {
    /// Advance widths for ASCII 32..=126 in the title font.
    pub title: Vec<f64>,
    /// Advance widths for ASCII 32..=126 in the body font.
    pub body: Vec<f64>,
    pub title_wide: f64,
    pub body_wide: f64,
    pub title_height: f64,
    pub line_height: f64,
    pub pad_x: f64,
    pub pad_y: f64,
}

impl Default for FontMetrics {
    fn default() -> Self {
        FontMetrics {
            title: Vec::new(),
            body: Vec::new(),
            title_wide: 12.0,
            body_wide: 11.0,
            title_height: 22.0,
            line_height: 15.0,
            pad_x: 8.0,
            pad_y: 4.0,
        }
    }
}

impl FontMetrics {
    fn width(table: &[f64], wide: f64, fallback: f64, s: &str) -> f64 {
        s.chars()
            .map(|c| {
                let code = c as u32;
                if (32..=126).contains(&code) {
                    table.get((code - 32) as usize).copied().unwrap_or(fallback)
                } else {
                    wide
                }
            })
            .sum()
    }

    fn title_w(&self, s: &str) -> f64 {
        Self::width(&self.title, self.title_wide, 7.2, s)
    }

    fn body_w(&self, s: &str) -> f64 {
        Self::width(&self.body, self.body_wide, 6.6, s)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ViewOptions {
    pub show_initializers: bool,
    pub show_attributes: bool,
    pub show_names: bool,
    pub show_constants: bool,
    pub edge_labels: bool,
    pub detach_long_edges: bool,
    /// Hide nodes that only compute known shape values (Shape → Gather →
    /// Concat ...) and show the computed values inside their consumers.
    pub fold_shapes: bool,
    /// Module paths (see `model::module_path`) shown as single nodes.
    pub collapsed: Vec<String>,
    pub max_attributes: usize,
    pub metrics: FontMetrics,
}

impl Default for ViewOptions {
    fn default() -> Self {
        ViewOptions {
            show_initializers: true,
            show_attributes: false,
            show_names: false,
            show_constants: false,
            edge_labels: true,
            detach_long_edges: true,
            fold_shapes: true,
            collapsed: Vec::new(),
            max_attributes: 8,
            metrics: FontMetrics::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewLine {
    pub k: String,
    pub v: String,
    /// Value id for weight lines, -1 for attributes.
    pub value: i64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewNode {
    /// "op" | "module" | "input" | "output" | "outer" | "missing"
    pub kind: &'static str,
    pub node: i64,
    pub value: i64,
    pub title: String,
    pub name: String,
    pub cat: &'static str,
    pub lines: Vec<ViewLine>,
    pub module: String,
    pub sub: bool,
    /// Number of nodes inside a collapsed module node.
    #[serde(skip_serializing_if = "is_zero")]
    pub members: usize,
    /// IR node ids represented by a collapsed module node.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ids: Vec<usize>,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewEdge {
    pub from: usize,
    pub to: usize,
    pub value: usize,
    /// Tensor name.
    pub name: String,
    pub label: String,
    /// Flattened polyline `[x0, y0, x1, y1, ...]`.
    pub points: Vec<f64>,
    pub implicit: bool,
    pub detached: bool,
    pub back: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphView {
    pub graph: GraphId,
    pub nodes: Vec<ViewNode>,
    pub edges: Vec<ViewEdge>,
    pub width: f64,
    pub height: f64,
    pub hidden_constants: usize,
    pub folded_nodes: usize,
    pub collapsed_nodes: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// Is `module` equal to `path` or nested inside it?
fn in_module(module: &str, path: &str) -> bool {
    module.len() >= path.len()
        && module.starts_with(path)
        && (module.len() == path.len() || module.as_bytes()[path.len()] == b'/')
}

fn weight_line(g: &Graph, vid: usize, label: String) -> ViewLine {
    let v = &g.values[vid];
    let text = match v.tensor {
        Some(ti) => {
            let t = &g.tensors[ti];
            let inline = if tensor::element_count(t) <= 4 && t.data_type != dtype::STRING {
                tensor::inline_values(t, 4)
            } else {
                None
            };
            match inline {
                Some(s) => format!("= {}", s),
                None => format!("⟨{}⟩", shape::dims_label(Some(&shape::from_i64s(&t.dims)))),
            }
        }
        None => match &v.sparse {
            Some(sp) => format!("⟨{}⟩ sparse", shape::dims_label(Some(&shape::from_i64s(&sp.dims)))),
            None => "⟨?⟩".to_string(),
        },
    };
    ViewLine {
        k: label,
        v: text,
        value: vid as i64,
    }
}

fn contents_label(d: &[Dim], scalar: bool) -> String {
    if scalar && d.len() == 1 {
        return format!("= {}", d[0].label());
    }
    let mut parts: Vec<String> = d.iter().take(8).map(Dim::label).collect();
    if d.len() > 8 {
        parts.push(format!("… +{}", d.len() - 8));
    }
    format!("= [{}]", parts.join(", "))
}

pub fn build_view(model: &Model, an: &Analysis, gid: GraphId, opt: &ViewOptions) -> anyhow::Result<GraphView> {
    let g = model.graph(gid)?;
    let st = &an.shapes[gid].states;
    let m = &opt.metrics;
    let mut vnodes: Vec<ViewNode> = Vec::new();
    let mut node_view: Vec<Option<usize>> = vec![None; g.nodes.len()];
    let mut value_view: HashMap<usize, usize> = HashMap::new();
    let mut hidden_constants = 0usize;
    let mut folded_nodes = 0usize;
    let mut collapsed_nodes = 0usize;

    // Collapsed modules: every node maps to the outermost collapsed module
    // that contains it.
    let modules: Vec<String> = g.nodes.iter().map(|n| module_path(&n.name)).collect();
    let collapsed: Vec<&str> = opt
        .collapsed
        .iter()
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .collect();
    let mut group_of: Vec<Option<usize>> = vec![None; g.nodes.len()];
    if !collapsed.is_empty() {
        for (ni, m) in modules.iter().enumerate() {
            group_of[ni] = collapsed
                .iter()
                .enumerate()
                .filter(|(_, c)| in_module(m, c))
                .min_by_key(|(_, c)| c.len())
                .map(|(i, _)| i);
        }
    }
    let mut group_view: Vec<Option<usize>> = vec![None; collapsed.len()];

    // Values whose contents shape inference fully determined (shape
    // arithmetic); nodes producing only such values can be folded away.
    let known_contents = |vid: usize| -> bool {
        let v = &g.values[vid];
        v.kind == ValueKind::Intermediate
            && !v.is_output
            && matches!(&st[vid].ints, Some(c) if c.len() <= 32 && c.iter().all(|d| !d.is_unknown()))
    };
    let mut folded = vec![false; g.nodes.len()];
    if opt.fold_shapes {
        for (ni, node) in g.nodes.iter().enumerate() {
            let mut outs = node.outputs.iter().flatten().peekable();
            if node.subgraphs.is_empty() && outs.peek().is_some() && outs.all(|&o| known_contents(o)) {
                folded[ni] = true;
            }
        }
    }
    let folded_value = |vid: usize| -> bool {
        g.values[vid].producer.map_or(false, |(p, _)| folded[p])
    };
    let embedded = |vid: usize| -> bool {
        let v = &g.values[vid];
        match v.kind {
            ValueKind::Initializer => true,
            ValueKind::Constant => !opt.show_constants,
            _ => folded_value(vid),
        }
    };

    let measure = |title: &str, name: &str, lines: &[ViewLine]| -> (f64, f64) {
        let mut w = m.title_w(title);
        if !name.is_empty() {
            w = w.max(m.body_w(name));
        }
        for l in lines {
            w = w.max(m.body_w(&l.k) + 6.0 + m.body_w(&l.v));
        }
        let mut h = m.title_height;
        let body_lines = lines.len() + usize::from(!name.is_empty());
        if body_lines > 0 {
            h += body_lines as f64 * m.line_height + m.pad_y * 2.0;
        }
        ((w + 2.0 * m.pad_x).max(44.0).ceil(), h.ceil())
    };

    // Graph inputs (excluding initializers).
    for &vid in &g.inputs {
        let v = &g.values[vid];
        if v.kind != ValueKind::Input {
            continue;
        }
        let ty = shape::type_label(dtype::name(st[vid].dtype), st[vid].shape.as_deref());
        let lines = vec![ViewLine {
            k: String::new(),
            v: ty,
            value: -1,
        }];
        let (w, h) = measure(&v.name, "", &lines);
        value_view.insert(vid, vnodes.len());
        vnodes.push(ViewNode {
            kind: "input",
            node: -1,
            value: vid as i64,
            title: v.name.clone(),
            name: String::new(),
            cat: "input",
            lines,
            module: String::new(),
            sub: false,
            members: 0,
            ids: Vec::new(),
            x: 0.0,
            y: 0.0,
            w,
            h,
        });
    }

    // Operator nodes in topological order.
    for &ni in &g.topo {
        let node = &g.nodes[ni];
        if node.op_type == "Constant" && node.is_standard() && !opt.show_constants {
            if let Some(o) = node.output(0) {
                if g.values[o].kind == ValueKind::Constant {
                    hidden_constants += 1;
                    continue;
                }
            }
        }
        if folded[ni] {
            folded_nodes += 1;
            continue;
        }
        if let Some(gi) = group_of[ni] {
            collapsed_nodes += 1;
            if let Some(vn) = group_view[gi] {
                node_view[ni] = Some(vn);
                continue;
            }
            let path = collapsed[gi];
            let mut counts: Vec<(String, usize)> = Vec::new();
            let mut members = 0usize;
            for (nj, n) in g.nodes.iter().enumerate() {
                if group_of[nj] != Some(gi) || folded[nj] {
                    continue;
                }
                if n.op_type == "Constant" && n.is_standard() && !opt.show_constants {
                    continue;
                }
                members += 1;
                match counts.iter_mut().find(|c| c.0 == n.op_type) {
                    Some(c) => c.1 += 1,
                    None => counts.push((n.op_type.clone(), 1)),
                }
            }
            counts.sort_by(|a, b| b.1.cmp(&a.1));
            let mut lines: Vec<ViewLine> = counts
                .iter()
                .take(4)
                .map(|(op, c)| ViewLine {
                    k: op.clone(),
                    v: format!("×{}", c),
                    value: -1,
                })
                .collect();
            if counts.len() > 4 {
                let rest: usize = counts[4..].iter().map(|c| c.1).sum();
                lines.push(ViewLine {
                    k: "…".into(),
                    v: format!("×{}", rest),
                    value: -1,
                });
            }
            let title = path.rsplit('/').next().unwrap_or(path).to_string();
            let name = if opt.show_names || title != path { path.to_string() } else { String::new() };
            let (w, h) = measure(&title, &name, &lines);
            let vn = vnodes.len();
            group_view[gi] = Some(vn);
            node_view[ni] = Some(vn);
            vnodes.push(ViewNode {
                kind: "module",
                node: -1,
                value: -1,
                title,
                name,
                cat: "module",
                lines,
                module: path.to_string(),
                sub: false,
                members,
                ids: (0..g.nodes.len()).filter(|&nj| group_of[nj] == Some(gi)).collect(),
                x: 0.0,
                y: 0.0,
                w,
                h,
            });
            continue;
        }
        let mut lines = Vec::new();
        if opt.show_initializers {
            for (slot, inp) in node.inputs.iter().enumerate() {
                let Some(vid) = *inp else { continue };
                let label = ops::input_name(&node.domain, &node.op_type, slot);
                if folded_value(vid) {
                    let scalar = st[vid].shape.as_ref().map_or(false, |s| s.is_empty());
                    let text = contents_label(st[vid].ints.as_deref().unwrap_or(&[]), scalar);
                    lines.push(ViewLine {
                        k: label,
                        v: text,
                        value: vid as i64,
                    });
                } else if embedded(vid) {
                    lines.push(weight_line(g, vid, label));
                }
            }
        }
        if opt.show_attributes {
            for a in node.attributes.iter().take(opt.max_attributes) {
                lines.push(ViewLine {
                    k: a.name.clone(),
                    v: format::attr_short(model, g, &a.value, 36),
                    value: -1,
                });
            }
            if node.attributes.len() > opt.max_attributes {
                lines.push(ViewLine {
                    k: "…".into(),
                    v: format!("+{}", node.attributes.len() - opt.max_attributes),
                    value: -1,
                });
            }
        }
        let name = if opt.show_names { node.name.clone() } else { String::new() };
        let title = if ops::is_standard_domain(&node.domain) || node.domain == "com.microsoft" {
            node.op_type.clone()
        } else {
            format!("{}::{}", node.domain, node.op_type)
        };
        let (w, h) = measure(&title, &name, &lines);
        node_view[ni] = Some(vnodes.len());
        vnodes.push(ViewNode {
            kind: "op",
            node: ni as i64,
            value: -1,
            title,
            name,
            cat: ops::category(&node.domain, &node.op_type).as_str(),
            lines,
            module: modules[ni].clone(),
            sub: !node.subgraphs.is_empty(),
            members: 0,
            ids: Vec::new(),
            x: 0.0,
            y: 0.0,
            w,
            h,
        });
    }

    // Edges.
    struct PendingEdge {
        from: usize,
        to: usize,
        value: usize,
        implicit: bool,
    }
    let mut pending: Vec<PendingEdge> = Vec::new();
    let mut seen: std::collections::HashSet<(usize, usize, usize)> = std::collections::HashSet::new();

    let source_of = |vid: usize, vnodes: &mut Vec<ViewNode>, value_view: &mut HashMap<usize, usize>| -> Option<usize> {
        let v = &g.values[vid];
        if let Some((p, _)) = v.producer {
            if let Some(vn) = node_view[p] {
                return Some(vn);
            }
        }
        if let Some(&vn) = value_view.get(&vid) {
            return Some(vn);
        }
        let kind = match v.kind {
            ValueKind::OuterScope => "outer",
            ValueKind::Unresolved => "missing",
            ValueKind::Constant => "outer",
            ValueKind::Input => "input",
            _ => return None,
        };
        let ty = shape::type_label(dtype::name(st[vid].dtype), st[vid].shape.as_deref());
        let lines = vec![ViewLine {
            k: String::new(),
            v: ty,
            value: -1,
        }];
        let (w, h) = measure(&v.name, "", &lines);
        value_view.insert(vid, vnodes.len());
        vnodes.push(ViewNode {
            kind,
            node: -1,
            value: vid as i64,
            title: v.name.clone(),
            name: String::new(),
            cat: kind,
            lines,
            module: String::new(),
            sub: false,
            members: 0,
            ids: Vec::new(),
            x: 0.0,
            y: 0.0,
            w,
            h,
        });
        Some(vnodes.len() - 1)
    };

    for &ni in &g.topo {
        let Some(to) = node_view[ni] else { continue };
        let node = &g.nodes[ni];
        for vid in node.inputs.iter().flatten().copied() {
            if embedded(vid) {
                continue;
            }
            if let Some(from) = source_of(vid, &mut vnodes, &mut value_view) {
                if from != to && seen.insert((from, to, vid)) {
                    pending.push(PendingEdge {
                        from,
                        to,
                        value: vid,
                        implicit: false,
                    });
                }
            }
        }
        for &vid in &node.implicit_inputs {
            if embedded(vid) {
                continue;
            }
            if let Some(from) = source_of(vid, &mut vnodes, &mut value_view) {
                if from != to && seen.insert((from, to, vid)) {
                    pending.push(PendingEdge {
                        from,
                        to,
                        value: vid,
                        implicit: true,
                    });
                }
            }
        }
    }

    // Graph outputs.
    for &vid in &g.outputs {
        let v = &g.values[vid];
        let ty = shape::type_label(dtype::name(st[vid].dtype), st[vid].shape.as_deref());
        let lines = vec![ViewLine {
            k: String::new(),
            v: ty,
            value: -1,
        }];
        let (w, h) = measure(&v.name, "", &lines);
        let out_idx = vnodes.len();
        vnodes.push(ViewNode {
            kind: "output",
            node: -1,
            value: vid as i64,
            title: v.name.clone(),
            name: String::new(),
            cat: "output",
            lines,
            module: String::new(),
            sub: false,
            members: 0,
            ids: Vec::new(),
            x: 0.0,
            y: 0.0,
            w,
            h,
        });
        let src = if embedded(vid) {
            None
        } else {
            source_of(vid, &mut vnodes, &mut value_view)
        };
        if let Some(from) = src {
            if seen.insert((from, out_idx, vid)) {
                pending.push(PendingEdge {
                    from,
                    to: out_idx,
                    value: vid,
                    implicit: false,
                });
            }
        }
    }

    // Count distinct consumers per value to decide which long edges may be
    // detached (typical for masks / position ids shared by every layer).
    let mut fanout: HashMap<usize, usize> = HashMap::new();
    for e in &pending {
        *fanout.entry(e.value).or_insert(0) += 1;
    }

    let lnodes: Vec<LNode> = vnodes.iter().map(|n| LNode { w: n.w, h: n.h }).collect();
    let ledges: Vec<LEdge> = pending
        .iter()
        .map(|e| LEdge {
            src: e.from,
            dst: e.to,
            weight: if e.implicit { 0.3 } else { 1.0 },
            detachable: opt.detach_long_edges && fanout.get(&e.value).copied().unwrap_or(0) >= 3,
        })
        .collect();
    let params = LayoutParams {
        rank_sep: if opt.edge_labels { 40.0 } else { 30.0 },
        node_sep: 24.0,
        edge_sep: 10.0,
        detach_span: 16,
    };
    let lay = layout::layout(&lnodes, &ledges, &params);

    for (i, n) in vnodes.iter_mut().enumerate() {
        n.x = lay.x[i] - n.w / 2.0;
        n.y = lay.y[i] - n.h / 2.0;
    }
    // A tensor feeding several nodes is labeled once (on its first routed edge).
    let mut labeled: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let edges: Vec<ViewEdge> = pending
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let routed = !lay.detached[i] && !lay.points[i].is_empty();
            let label = match st[e.value].shape.as_deref() {
                Some(s) if opt.edge_labels && !s.is_empty() && routed && labeled.insert((e.from, e.value)) => {
                    shape::dims_label(Some(s))
                }
                _ => String::new(),
            };
            ViewEdge {
                from: e.from,
                to: e.to,
                value: e.value,
                name: g.values[e.value].name.clone(),
                label,
                points: lay.points[i].iter().flat_map(|&(x, y)| [x, y]).collect(),
                implicit: e.implicit,
                detached: lay.detached[i],
                back: lay.reversed[i],
            }
        })
        .collect();

    Ok(GraphView {
        graph: gid,
        nodes: vnodes,
        edges,
        width: lay.width,
        height: lay.height,
        hidden_constants,
        folded_nodes,
        collapsed_nodes,
    })
}
