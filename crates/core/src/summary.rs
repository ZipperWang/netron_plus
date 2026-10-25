//! Serializable summaries, tables and the module hierarchy.

use crate::analysis::{Analysis, MemoryStats, Totals};
use crate::checks::Issue;
use crate::dtype;
use crate::infer::ShapeSource;
use crate::model::{module_path, GraphId, GraphOwner, Model, ValueKind};
use crate::ops;
use crate::shape::{self, Dim};
use crate::tensor::{self, DataSource};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}

pub fn kvs(v: &[(String, String)]) -> Vec<KeyValue> {
    v.iter()
        .map(|(k, v)| KeyValue {
            key: k.clone(),
            value: v.clone(),
        })
        .collect()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Opset {
    pub domain: String,
    pub version: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphInfo {
    pub id: usize,
    pub name: String,
    pub label: String,
    pub kind: &'static str,
    pub parent: Option<usize>,
    pub parent_node: Option<usize>,
    pub attribute: Option<String>,
    pub nodes: usize,
    pub depth: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IoInfo {
    pub value: usize,
    pub name: String,
    pub dtype: String,
    pub shape: Option<Vec<Dim>>,
    pub declared: Option<Vec<Dim>>,
    pub display: String,
    pub bytes: Option<u64>,
    pub doc: String,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Stats {
    pub nodes: usize,
    pub nodes_total: usize,
    pub op_types: usize,
    pub initializers: usize,
    pub values: usize,
    pub graphs: usize,
    pub subgraphs: usize,
    pub functions: usize,
    pub unknown_shapes: usize,
    pub intermediate_values: usize,
    pub custom_ops: usize,
    pub hidden_constants: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DtypeRow {
    pub dtype: String,
    pub tensors: usize,
    pub elements: u64,
    pub bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OpRow {
    pub op: String,
    pub domain: String,
    pub category: &'static str,
    pub count: usize,
    pub params: u64,
    pub flops: u64,
    pub macs: u64,
    pub costed: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CategoryRow {
    pub category: &'static str,
    pub count: usize,
    pub flops: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TensorRow {
    pub graph: usize,
    pub value: usize,
    pub name: String,
    pub dtype: String,
    pub shape: String,
    pub elements: u64,
    pub bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolInfo {
    pub name: String,
    pub value: Option<i64>,
    pub inputs: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OverridesOut {
    pub params: BTreeMap<String, i64>,
    pub inputs: BTreeMap<String, Vec<Option<i64>>>,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct IssueCounts {
    pub error: usize,
    pub warning: usize,
    pub info: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSummary {
    pub id: u32,
    pub file_name: String,
    pub file_path: Option<String>,
    pub file_size: u64,
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc: String,
    pub opsets: Vec<Opset>,
    pub metadata: Vec<KeyValue>,
    pub graphs: Vec<GraphInfo>,
    pub inputs: Vec<IoInfo>,
    pub outputs: Vec<IoInfo>,
    pub stats: Stats,
    pub dtypes: Vec<DtypeRow>,
    pub ops: Vec<OpRow>,
    pub categories: Vec<CategoryRow>,
    pub largest: Vec<TensorRow>,
    pub symbols: Vec<SymbolInfo>,
    pub overrides: OverridesOut,
    pub memory: MemoryStats,
    pub totals: Totals,
    pub issues: IssueCounts,
    pub external_data: bool,
}

fn graph_kind(o: &GraphOwner) -> &'static str {
    match o {
        GraphOwner::Main => "main",
        GraphOwner::Subgraph { .. } => "subgraph",
        GraphOwner::Function { .. } => "function",
        GraphOwner::Training { .. } => "training",
    }
}

fn io_info(model: &Model, an: &Analysis, gid: GraphId, vid: usize) -> IoInfo {
    let g = &model.graphs[gid];
    let v = &g.values[vid];
    let s = &an.shapes[gid].states[vid];
    let bytes = s
        .shape
        .as_deref()
        .and_then(shape::numel)
        .and_then(|n| dtype::bytes_for(s.dtype, n));
    IoInfo {
        value: vid,
        name: v.name.clone(),
        dtype: dtype::name(s.dtype).to_string(),
        shape: s.shape.clone(),
        declared: v.declared.as_ref().and_then(|d| d.shape.clone()),
        display: match &v.declared {
            Some(d) if !d.display.is_empty() && d.kind != crate::model::TypeKind::Tensor => d.display.clone(),
            _ => shape::type_label(dtype::name(s.dtype), s.shape.as_deref()),
        },
        bytes,
        doc: v.doc.clone(),
    }
}

pub fn build(id: u32, model: &Model, an: &Analysis, issues: &[Issue]) -> ModelSummary {
    let main = model.main();
    let graphs: Vec<GraphInfo> = model
        .graphs
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let (parent, parent_node, attribute) = match &g.owner {
                GraphOwner::Subgraph {
                    parent,
                    node,
                    attribute,
                } => (Some(*parent), Some(*node), Some(attribute.clone())),
                _ => (None, None, None),
            };
            GraphInfo {
                id: i,
                name: g.name.clone(),
                label: g.label(&model.graphs),
                kind: graph_kind(&g.owner),
                parent,
                parent_node,
                attribute,
                nodes: g.nodes.len(),
                depth: g.depth,
            }
        })
        .collect();

    let inputs: Vec<IoInfo> = main
        .inputs
        .iter()
        .filter(|&&v| main.values[v].kind == ValueKind::Input)
        .map(|&v| io_info(model, an, 0, v))
        .collect();
    let outputs: Vec<IoInfo> = main.outputs.iter().map(|&v| io_info(model, an, 0, v)).collect();

    // Operator statistics over the main graph and its control-flow subgraphs.
    let mut ops_map: BTreeMap<(String, String), OpRow> = BTreeMap::new();
    let mut cats: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();
    let mut stats = Stats {
        nodes: main.nodes.len(),
        graphs: model.graphs.len(),
        ..Default::default()
    };
    for (gid, g) in model.graphs.iter().enumerate() {
        stats.nodes_total += g.nodes.len();
        match g.owner {
            GraphOwner::Subgraph { .. } => stats.subgraphs += 1,
            GraphOwner::Function { .. } => stats.functions += 1,
            _ => {}
        }
        if !matches!(g.owner, GraphOwner::Main | GraphOwner::Subgraph { .. }) {
            continue;
        }
        for (ni, n) in g.nodes.iter().enumerate() {
            let c = &an.costs[gid][ni];
            let cat = ops::category(&n.domain, &n.op_type);
            let dom = if ops::is_standard_domain(&n.domain) { String::new() } else { n.domain.clone() };
            if !dom.is_empty() && dom != "com.microsoft" && dom != "ai.onnx.ml" {
                stats.custom_ops += 1;
            }
            let row = ops_map.entry((n.op_type.clone(), dom.clone())).or_insert_with(|| OpRow {
                op: n.op_type.clone(),
                domain: dom,
                category: cat.as_str(),
                count: 0,
                params: 0,
                flops: 0,
                macs: 0,
                costed: 0,
            });
            row.count += 1;
            row.params += c.params;
            if let Some(f) = c.flops {
                row.flops = row.flops.saturating_add(f);
                row.costed += 1;
            }
            row.macs = row.macs.saturating_add(c.macs.unwrap_or(0));
            let e = cats.entry(cat.as_str()).or_insert((0, 0));
            e.0 += 1;
            e.1 = e.1.saturating_add(c.flops.unwrap_or(0));
            if n.op_type == "Constant" && n.is_standard() {
                stats.hidden_constants += 1;
            }
        }
    }
    let mut ops_rows: Vec<OpRow> = ops_map.into_values().collect();
    ops_rows.sort_by(|a, b| b.count.cmp(&a.count).then(a.op.cmp(&b.op)));
    stats.op_types = ops_rows.len();
    let mut categories: Vec<CategoryRow> = cats
        .into_iter()
        .map(|(category, (count, flops))| CategoryRow {
            category,
            count,
            flops,
        })
        .collect();
    categories.sort_by(|a, b| b.count.cmp(&a.count));

    // Tensors by data type and the largest weights.
    let mut dt_map: BTreeMap<i32, (usize, u64, u64)> = BTreeMap::new();
    let mut largest: Vec<TensorRow> = Vec::new();
    let mut external = false;
    for (gid, g) in model.graphs.iter().enumerate() {
        for (vid, v) in g.values.iter().enumerate() {
            if !matches!(v.kind, ValueKind::Initializer | ValueKind::Constant) {
                continue;
            }
            if gid == 0 && v.kind == ValueKind::Initializer {
                stats.initializers += 1;
            }
            let Some(ti) = v.tensor else { continue };
            let t = &g.tensors[ti];
            if tensor::external_info(t).is_some() {
                external = true;
            }
            let n = tensor::element_count(t);
            let b = tensor::byte_size(t).unwrap_or(0);
            let e = dt_map.entry(t.data_type).or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += n;
            e.2 += b;
            largest.push(TensorRow {
                graph: gid,
                value: vid,
                name: v.name.clone(),
                dtype: dtype::name(t.data_type).to_string(),
                shape: shape::dims_label(Some(&shape::from_i64s(&t.dims))),
                elements: n,
                bytes: b,
            });
        }
    }
    largest.sort_by(|a, b| b.bytes.cmp(&a.bytes));
    largest.truncate(15);
    let mut dtypes: Vec<DtypeRow> = dt_map
        .into_iter()
        .map(|(d, (tensors, elements, bytes))| DtypeRow {
            dtype: dtype::name(d).to_string(),
            tensors,
            elements,
            bytes,
        })
        .collect();
    dtypes.sort_by(|a, b| b.bytes.cmp(&a.bytes));

    let st0 = &an.shapes[0].states;
    stats.values = main.values.len();
    for (vid, v) in main.values.iter().enumerate() {
        if v.kind == ValueKind::Intermediate {
            stats.intermediate_values += 1;
            if st0[vid].shape.is_none() {
                stats.unknown_shapes += 1;
            }
        }
    }

    // Symbolic dimensions of the main inputs.
    let mut sym_inputs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for &vid in &main.inputs {
        let v = &main.values[vid];
        if v.kind != ValueKind::Input {
            continue;
        }
        if let Some(d) = &v.declared {
            for dim in d.shape.iter().flatten() {
                if let Dim::Param(p) = dim {
                    sym_inputs.entry(p.clone()).or_default().insert(v.name.clone());
                }
            }
        }
    }
    for p in an.overrides.params.keys() {
        sym_inputs.entry(p.clone()).or_default();
    }
    let symbols: Vec<SymbolInfo> = sym_inputs
        .into_iter()
        .map(|(name, inputs)| SymbolInfo {
            value: an.overrides.params.get(&name).copied(),
            name,
            inputs: inputs.into_iter().collect(),
        })
        .collect();

    let mut counts = IssueCounts::default();
    for i in issues {
        match i.severity {
            "error" => counts.error += 1,
            "warning" => counts.warning += 1,
            _ => counts.info += 1,
        }
    }

    ModelSummary {
        id,
        file_name: model.file_name.clone(),
        file_path: model.file_path.clone(),
        file_size: model.file_size,
        ir_version: model.ir_version,
        producer_name: model.producer_name.clone(),
        producer_version: model.producer_version.clone(),
        domain: model.domain.clone(),
        model_version: model.model_version,
        doc: model.doc.clone(),
        opsets: model
            .opsets
            .iter()
            .map(|(d, v)| Opset {
                domain: if d.is_empty() { "ai.onnx".into() } else { d.clone() },
                version: *v,
            })
            .collect(),
        metadata: kvs(&model.metadata),
        graphs,
        inputs,
        outputs,
        stats,
        dtypes,
        ops: ops_rows,
        categories,
        largest,
        symbols,
        overrides: OverridesOut {
            params: an.overrides.params.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            inputs: an.overrides.inputs.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        },
        memory: an.memory.clone(),
        totals: an.totals.clone(),
        issues: counts,
        external_data: external,
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeRow {
    pub id: usize,
    pub order: usize,
    pub name: String,
    pub op: String,
    pub domain: String,
    pub cat: &'static str,
    pub inputs: String,
    pub outputs: String,
    pub params: u64,
    pub macs: Option<u64>,
    pub flops: Option<u64>,
    pub out_bytes: Option<u64>,
    pub read_bytes: Option<u64>,
    pub module: String,
}

pub fn node_table(model: &Model, an: &Analysis, gid: GraphId) -> anyhow::Result<Vec<NodeRow>> {
    let g = model.graph(gid)?;
    let st = &an.shapes[gid].states;
    let costs = &an.costs[gid];
    let shape_of = |vid: usize| shape::dims_label(st[vid].shape.as_deref());
    Ok(g.topo
        .iter()
        .enumerate()
        .map(|(order, &ni)| {
            let n = &g.nodes[ni];
            let c = &costs[ni];
            let inputs: Vec<String> = n
                .inputs
                .iter()
                .flatten()
                .filter(|&&v| !matches!(g.values[v].kind, ValueKind::Initializer | ValueKind::Constant))
                .map(|&v| shape_of(v))
                .collect();
            let outputs: Vec<String> = n.outputs.iter().flatten().map(|&v| shape_of(v)).collect();
            NodeRow {
                id: ni,
                order,
                name: n.name.clone(),
                op: n.op_type.clone(),
                domain: n.domain.clone(),
                cat: ops::category(&n.domain, &n.op_type).as_str(),
                inputs: inputs.join("; "),
                outputs: outputs.join("; "),
                params: c.params,
                macs: c.macs,
                flops: c.flops,
                out_bytes: c.write_bytes,
                read_bytes: c.read_bytes,
                module: module_path(&n.name),
            }
        })
        .collect())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValueRow {
    pub id: usize,
    pub name: String,
    pub kind: &'static str,
    pub dtype: String,
    pub shape: String,
    pub elements: Option<u64>,
    pub bytes: Option<u64>,
    pub producer: Option<usize>,
    pub producer_op: String,
    pub consumers: usize,
    pub is_input: bool,
    pub is_output: bool,
    pub tensor: Option<usize>,
    pub source: ShapeSource,
    pub external: bool,
}

pub fn value_table(model: &Model, an: &Analysis, gid: GraphId) -> anyhow::Result<Vec<ValueRow>> {
    let g = model.graph(gid)?;
    let gs = &an.shapes[gid];
    Ok(g.values
        .iter()
        .enumerate()
        .map(|(vid, v)| {
            let s = &gs.states[vid];
            let elements = s.shape.as_deref().and_then(shape::numel);
            let bytes = match v.tensor {
                Some(ti) => tensor::byte_size(&g.tensors[ti]),
                None => elements.and_then(|n| dtype::bytes_for(s.dtype, n)),
            };
            ValueRow {
                id: vid,
                name: v.name.clone(),
                kind: v.kind.as_str(),
                dtype: dtype::name(s.dtype).to_string(),
                shape: shape::dims_label(s.shape.as_deref()),
                elements,
                bytes,
                producer: v.producer.map(|p| p.0),
                producer_op: v
                    .producer
                    .map(|p| g.nodes[p.0].op_type.clone())
                    .unwrap_or_default(),
                consumers: v.consumers.len(),
                is_input: v.is_input,
                is_output: v.is_output,
                tensor: v.tensor,
                source: gs.sources[vid],
                external: v
                    .tensor
                    .map(|ti| tensor::external_info(&g.tensors[ti]).is_some())
                    .unwrap_or(false),
            }
        })
        .collect())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WeightRow {
    pub graph: usize,
    pub value: usize,
    pub tensor: usize,
    pub name: String,
    pub dtype: String,
    pub shape: String,
    pub elements: u64,
    pub bytes: u64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub abs_mean: Option<f64>,
    pub zeros: u64,
    pub nan: u64,
    pub inf: u64,
    pub fp16_overflow: u64,
    pub consumers: String,
    pub error: Option<String>,
}

pub fn weight_table(model: &Model, src: &dyn DataSource) -> Vec<WeightRow> {
    let mut rows = Vec::new();
    for (gid, g) in model.graphs.iter().enumerate() {
        for (vid, v) in g.values.iter().enumerate() {
            if !matches!(v.kind, ValueKind::Initializer | ValueKind::Constant) {
                continue;
            }
            let Some(ti) = v.tensor else { continue };
            let t = &g.tensors[ti];
            if !dtype::is_numeric(t.data_type) {
                continue;
            }
            let mut consumers: Vec<String> = v
                .consumers
                .iter()
                .filter(|c| c.0 < g.nodes.len())
                .map(|c| g.nodes[c.0].op_type.clone())
                .collect();
            consumers.sort();
            consumers.dedup();
            let mut row = WeightRow {
                graph: gid,
                value: vid,
                tensor: ti,
                name: v.name.clone(),
                dtype: dtype::name(t.data_type).to_string(),
                shape: shape::dims_label(Some(&shape::from_i64s(&t.dims))),
                elements: tensor::element_count(t),
                bytes: tensor::byte_size(t).unwrap_or(0),
                min: None,
                max: None,
                mean: None,
                std: None,
                abs_mean: None,
                zeros: 0,
                nan: 0,
                inf: 0,
                fp16_overflow: 0,
                consumers: consumers.join(", "),
                error: None,
            };
            match tensor::compute_stats(t, src, 8) {
                Ok(s) => {
                    row.min = s.min;
                    row.max = s.max;
                    row.mean = s.mean;
                    row.std = s.std;
                    row.abs_mean = s.abs_mean;
                    row.zeros = s.zeros;
                    row.nan = s.nan;
                    row.inf = s.pos_inf + s.neg_inf;
                    row.fp16_overflow = s.fp16_overflow;
                }
                Err(e) => row.error = Some(e.to_string()),
            }
            rows.push(row);
        }
    }
    rows
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModuleNode {
    pub name: String,
    pub path: String,
    pub nodes: usize,
    pub params: u64,
    pub flops: u64,
    pub children: Vec<ModuleNode>,
}

struct TmpModule {
    name: String,
    path: String,
    nodes: usize,
    params: u64,
    flops: u64,
    children: Vec<usize>,
    index: HashMap<String, usize>,
}

pub fn hierarchy(model: &Model, an: &Analysis, gid: GraphId) -> anyhow::Result<ModuleNode> {
    let g = model.graph(gid)?;
    let mut arena: Vec<TmpModule> = vec![TmpModule {
        name: String::new(),
        path: String::new(),
        nodes: 0,
        params: 0,
        flops: 0,
        children: Vec::new(),
        index: HashMap::new(),
    }];
    for (ni, n) in g.nodes.iter().enumerate() {
        let c = &an.costs[gid][ni];
        let flops = c.flops.unwrap_or(0);
        arena[0].nodes += 1;
        arena[0].params += c.params;
        arena[0].flops = arena[0].flops.saturating_add(flops);
        let path = module_path(&n.name);
        if path.is_empty() {
            continue;
        }
        let mut cur = 0usize;
        let mut acc = String::new();
        for seg in path.split('/') {
            if !acc.is_empty() {
                acc.push('/');
            }
            acc.push_str(seg);
            let found = arena[cur].index.get(seg).copied();
            let next = match found {
                Some(i) => i,
                None => {
                    let i = arena.len();
                    arena.push(TmpModule {
                        name: seg.to_string(),
                        path: acc.clone(),
                        nodes: 0,
                        params: 0,
                        flops: 0,
                        children: Vec::new(),
                        index: HashMap::new(),
                    });
                    arena[cur].children.push(i);
                    arena[cur].index.insert(seg.to_string(), i);
                    i
                }
            };
            cur = next;
            arena[cur].nodes += 1;
            arena[cur].params += c.params;
            arena[cur].flops = arena[cur].flops.saturating_add(flops);
        }
    }
    fn convert(arena: &[TmpModule], i: usize) -> ModuleNode {
        let t = &arena[i];
        ModuleNode {
            name: t.name.clone(),
            path: t.path.clone(),
            nodes: t.nodes,
            params: t.params,
            flops: t.flops,
            children: t.children.iter().map(|&c| convert(arena, c)).collect(),
        }
    }
    Ok(convert(&arena, 0))
}
