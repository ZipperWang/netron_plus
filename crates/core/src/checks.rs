//! Model health checks ("lint") useful before deployment.

use crate::analysis::Analysis;
use crate::dtype;
use crate::model::{Model, ValueKind};
use crate::ops;
use crate::tensor::{self, DataSource};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Issue {
    /// "error" | "warning" | "info"
    pub severity: &'static str,
    pub code: &'static str,
    /// English fallback message; the UI localizes by `code` + `params`.
    pub message: String,
    pub graph: usize,
    pub node: Option<usize>,
    pub value: Option<usize>,
    pub count: usize,
    pub items: Vec<String>,
    pub params: BTreeMap<String, String>,
}

const MAX_ITEMS: usize = 50;

fn issue(severity: &'static str, code: &'static str, graph: usize, message: String) -> Issue {
    Issue {
        severity,
        code,
        message,
        graph,
        node: None,
        value: None,
        count: 1,
        items: Vec::new(),
        params: BTreeMap::new(),
    }
}

fn with_param(mut i: Issue, k: &str, v: impl ToString) -> Issue {
    i.params.insert(k.to_string(), v.to_string());
    i
}

fn aggregate(
    severity: &'static str,
    code: &'static str,
    graph: usize,
    names: Vec<String>,
    what: &str,
) -> Option<Issue> {
    if names.is_empty() {
        return None;
    }
    let count = names.len();
    let mut i = issue(severity, code, graph, format!("{} {}", count, what));
    i.count = count;
    i.items = names.into_iter().take(MAX_ITEMS).collect();
    Some(with_param(i, "count", count))
}

pub fn run(model: &Model, an: &Analysis, src: &dyn DataSource, deep: bool) -> Vec<Issue> {
    let mut out: Vec<Issue> = Vec::new();
    let imported: HashSet<String> = model
        .opsets
        .iter()
        .map(|(d, _)| if ops::is_standard_domain(d) { String::new() } else { d.clone() })
        .collect();
    let mut custom_domains: BTreeMap<String, usize> = BTreeMap::new();
    let mut deprecated: BTreeMap<String, usize> = BTreeMap::new();
    let mut missing_domains: BTreeMap<String, usize> = BTreeMap::new();

    for (gid, g) in model.graphs.iter().enumerate() {
        let is_function = matches!(g.owner, crate::model::GraphOwner::Function { .. });
        for b in &g.issues {
            let (sev, msg) = match b.code {
                "duplicate-producer" => ("error", format!("Value '{}' is produced more than once", b.name)),
                "duplicate-initializer" => ("warning", format!("Initializer '{}' is defined more than once", b.name)),
                _ => ("warning", b.name.clone()),
            };
            let mut i = with_param(issue(sev, b.code, gid, msg), "name", &b.name);
            i.node = b.node;
            i.value = b.value;
            out.push(i);
        }
        if g.has_cycle {
            out.push(issue("error", "cycle", gid, "The graph contains a cycle".into()));
        } else if !g.sorted {
            out.push(issue(
                "warning",
                "not-topo-sorted",
                gid,
                "Nodes are not stored in topological order (required by the ONNX spec)".into(),
            ));
        }

        let mut unused_init = Vec::new();
        let mut init_in_inputs = Vec::new();
        for (vid, v) in g.values.iter().enumerate() {
            match v.kind {
                ValueKind::Unresolved => {
                    let mut i = with_param(
                        issue("error", "unresolved-input", gid, format!("'{}' is used but never defined", v.name)),
                        "name",
                        &v.name,
                    );
                    i.value = Some(vid);
                    i.node = v.consumers.first().map(|c| c.0);
                    out.push(i);
                }
                ValueKind::Initializer => {
                    if v.consumers.is_empty() && !v.is_output {
                        unused_init.push(v.name.clone());
                    }
                    if v.is_input {
                        init_in_inputs.push(v.name.clone());
                    }
                }
                ValueKind::Input => {
                    if v.consumers.is_empty() && !v.is_output && !is_function {
                        let mut i = with_param(
                            issue("warning", "unused-input", gid, format!("Graph input '{}' is never used", v.name)),
                            "name",
                            &v.name,
                        );
                        i.value = Some(vid);
                        out.push(i);
                    }
                }
                _ => {}
            }
        }
        out.extend(aggregate("warning", "unused-initializer", gid, unused_init, "unused initializers"));
        out.extend(aggregate(
            "info",
            "initializer-in-inputs",
            gid,
            init_in_inputs,
            "initializers are also listed as graph inputs (pre IR v4 style)",
        ));

        let mut dead = Vec::new();
        let mut unused_out = Vec::new();
        let mut names: HashMap<&str, usize> = HashMap::new();
        let mut empty_names = 0usize;
        let mut identity = 0usize;
        let mut foldable: BTreeMap<String, usize> = BTreeMap::new();
        for node in &g.nodes {
            let outs: Vec<usize> = node.outputs.iter().flatten().copied().collect();
            let used: Vec<bool> = outs
                .iter()
                .map(|&o| !g.values[o].consumers.is_empty() || g.values[o].is_output)
                .collect();
            let label = if node.name.is_empty() { node.op_type.clone() } else { node.name.clone() };
            if !outs.is_empty() && used.iter().all(|u| !u) {
                dead.push(label.clone());
            } else {
                for (k, &o) in outs.iter().enumerate() {
                    if !used[k] {
                        unused_out.push(format!("{} ({})", g.values[o].name, label));
                    }
                }
            }
            if node.name.is_empty() {
                empty_names += 1;
            } else {
                *names.entry(node.name.as_str()).or_insert(0) += 1;
            }
            if node.op_type == "Identity" && node.is_standard() {
                identity += 1;
            }
            let ins: Vec<usize> = node.inputs.iter().flatten().copied().collect();
            if !ins.is_empty()
                && node.is_standard()
                && node.op_type != "Constant"
                && node.subgraphs.is_empty()
                && ins.iter().all(|&v| {
                    matches!(g.values[v].kind, ValueKind::Initializer | ValueKind::Constant)
                })
            {
                *foldable.entry(node.op_type.clone()).or_insert(0) += 1;
            }
            let dom = if ops::is_standard_domain(&node.domain) { String::new() } else { node.domain.clone() };
            if !dom.is_empty() && dom != "ai.onnx.ml" {
                *custom_domains.entry(dom.clone()).or_insert(0) += 1;
            }
            if !is_function && !imported.contains(&dom) && !(dom.is_empty() && imported.is_empty()) {
                *missing_domains.entry(if dom.is_empty() { "ai.onnx".to_string() } else { dom.clone() }).or_insert(0) += 1;
            }
            if dom.is_empty() && ops::deprecated_since(&node.op_type).is_some() {
                *deprecated.entry(node.op_type.clone()).or_insert(0) += 1;
            }
        }
        out.extend(aggregate("warning", "dead-node", gid, dead, "nodes whose outputs are never used"));
        out.extend(aggregate("info", "unused-output", gid, unused_out, "node outputs are never used"));
        let dups: Vec<String> = names
            .into_iter()
            .filter(|(_, c)| *c > 1)
            .map(|(n, c)| format!("{} (×{})", n, c))
            .collect();
        out.extend(aggregate("warning", "duplicate-node-name", gid, dups, "node names are not unique"));
        if empty_names > 0 {
            let mut i = issue("info", "empty-node-name", gid, format!("{} nodes have no name", empty_names));
            i.count = empty_names;
            out.push(with_param(i, "count", empty_names));
        }
        if identity > 0 {
            let mut i = issue("info", "identity-nodes", gid, format!("{} Identity nodes can usually be removed", identity));
            i.count = identity;
            out.push(with_param(i, "count", identity));
        }
        if !foldable.is_empty() {
            let count: usize = foldable.values().sum();
            let mut i = issue(
                "info",
                "foldable-nodes",
                gid,
                format!("{} nodes only consume constants and can be constant-folded", count),
            );
            i.count = count;
            i.items = foldable.iter().map(|(k, v)| format!("{} ×{}", k, v)).collect();
            out.push(with_param(i, "count", count));
        }
    }

    for (dom, count) in &custom_domains {
        let mut i = issue(
            "info",
            "custom-domain",
            0,
            format!("{} nodes use the custom operator domain '{}'", count, dom),
        );
        i.count = *count;
        out.push(with_param(with_param(i, "domain", dom), "count", count));
    }
    for (dom, count) in &missing_domains {
        let mut i = issue(
            "error",
            "missing-opset",
            0,
            format!("{} nodes use domain '{}' which has no opset import", count, dom),
        );
        i.count = *count;
        out.push(with_param(with_param(i, "domain", dom), "count", count));
    }
    let std_opset = model.opset("").unwrap_or(0);
    for (op, count) in &deprecated {
        let since = ops::deprecated_since(op).unwrap_or(0);
        let sev = if std_opset >= since && since > 1 { "warning" } else { "info" };
        let mut i = issue(sev, "deprecated-op", 0, format!("{} {} nodes use a deprecated operator", count, op));
        i.count = *count;
        out.push(with_param(with_param(i, "op", op), "count", count));
    }
    if std_opset > 0 && std_opset < 11 {
        out.push(with_param(
            issue("info", "old-opset", 0, format!("The model uses opset {}; many runtimes prefer opset 11 or newer", std_opset)),
            "opset",
            std_opset,
        ));
    }

    // Dynamic inputs.
    let main = model.main();
    let st0 = &an.shapes[0].states;
    for &vid in &main.inputs {
        let v = &main.values[vid];
        if v.kind != ValueKind::Input {
            continue;
        }
        let declared = v.declared.as_ref().and_then(|d| d.shape.clone());
        let dynamic = match &declared {
            Some(s) => s.iter().any(|d| !d.is_known()),
            None => true,
        };
        if dynamic {
            let shape = crate::shape::dims_label(declared.as_deref());
            let mut i = issue(
                "info",
                "dynamic-input",
                0,
                format!("Input '{}' has a dynamic shape [{}]", v.name, shape),
            );
            i.value = Some(vid);
            out.push(with_param(with_param(i, "name", &v.name), "shape", shape));
        }
    }

    // Shape inference coverage.
    let unknown: Vec<String> = main
        .values
        .iter()
        .enumerate()
        .filter(|(vid, v)| v.kind == ValueKind::Intermediate && st0[*vid].shape.is_none())
        .map(|(_, v)| v.name.clone())
        .collect();
    out.extend(aggregate(
        "info",
        "unknown-shapes",
        0,
        unknown,
        "intermediate tensors have unknown shapes",
    ));
    if an.totals.uncosted_nodes > 0 {
        let mut i = issue(
            "info",
            "uncosted-nodes",
            0,
            format!("{} nodes have no FLOPs estimate", an.totals.uncosted_nodes),
        );
        i.count = an.totals.uncosted_nodes;
        i.items = an.totals.uncosted_ops.clone();
        out.push(with_param(i, "count", an.totals.uncosted_nodes));
    }

    // External data files.
    let mut ext_locations: BTreeMap<String, usize> = BTreeMap::new();
    for g in &model.graphs {
        for t in &g.tensors {
            if let Some(e) = tensor::external_info(t) {
                *ext_locations.entry(e.location).or_insert(0) += 1;
            }
        }
    }
    for (loc, count) in &ext_locations {
        match src.external_exists(loc) {
            Some(false) => {
                let mut i = issue(
                    "error",
                    "external-missing",
                    0,
                    format!("External data file '{}' ({} tensors) was not found", loc, count),
                );
                i.count = *count;
                out.push(with_param(with_param(i, "file", loc), "count", count));
            }
            Some(true) => {
                let mut i = issue(
                    "info",
                    "external-data",
                    0,
                    format!("{} tensors are stored in external file '{}'", count, loc),
                );
                i.count = *count;
                out.push(with_param(with_param(i, "file", loc), "count", count));
            }
            None => {}
        }
    }

    if deep {
        let mut nan_inf = Vec::new();
        let mut zeros = Vec::new();
        let mut fp16 = Vec::new();
        let mut errors = Vec::new();
        for g in &model.graphs {
            for v in &g.values {
                if !matches!(v.kind, ValueKind::Initializer | ValueKind::Constant) {
                    continue;
                }
                let Some(ti) = v.tensor else { continue };
                let t = &g.tensors[ti];
                if !dtype::is_numeric(t.data_type) {
                    continue;
                }
                match tensor::compute_stats(t, src, 8) {
                    Ok(s) => {
                        if s.nan + s.pos_inf + s.neg_inf > 0 {
                            nan_inf.push(format!("{} (NaN {}, Inf {})", v.name, s.nan, s.pos_inf + s.neg_inf));
                        }
                        if s.count > 1 && s.zeros == s.count && dtype::is_float(t.data_type) {
                            zeros.push(v.name.clone());
                        }
                        if s.fp16_overflow > 0 {
                            fp16.push(format!("{} ({} values)", v.name, s.fp16_overflow));
                        }
                    }
                    Err(e) => errors.push(format!("{}: {}", v.name, e)),
                }
            }
        }
        out.extend(aggregate("error", "nan-inf-weights", 0, nan_inf, "weight tensors contain NaN or Inf"));
        out.extend(aggregate("warning", "zero-weights", 0, zeros, "float weight tensors are all zeros"));
        out.extend(aggregate(
            "warning",
            "fp16-overflow",
            0,
            fp16,
            "float32 weight tensors have values outside the float16 range (±65504)",
        ));
        out.extend(aggregate("warning", "weight-read-error", 0, errors, "weight tensors could not be read"));
    }

    let rank = |s: &str| match s {
        "error" => 0,
        "warning" => 1,
        _ => 2,
    };
    out.sort_by_key(|i| rank(i.severity));
    out
}
