//! JSON API shared by the desktop (Tauri) and web (WebAssembly) front ends.
//!
//! Every request is `method` + JSON `params`; every response is a JSON string,
//! so both hosts are thin wrappers around [`Workspace`].

use crate::analysis::{self, Analysis};
use crate::checks;
use crate::cost::NodeCost;
use crate::dtype;
use crate::format;
use crate::infer::{Overrides, ShapeSource};
use crate::model::{self, AttrValue, GraphId, Model, TypeInfo, IMPLICIT_SLOT};
use crate::ops;
use crate::shape::{self, Dim};
use crate::summary::{self, KeyValue};
use crate::tensor::{self, DataSource, ExternalInfo, FileSource};
use crate::view::{self, ViewOptions};
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub struct Document {
    pub id: u32,
    pub model: Model,
    pub source: Box<dyn DataSource>,
    analysis: RwLock<Arc<Analysis>>,
}

impl Document {
    pub fn analysis(&self) -> Arc<Analysis> {
        self.analysis
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

#[derive(Default)]
pub struct Workspace {
    docs: Mutex<HashMap<u32, Arc<Document>>>,
    next: AtomicU32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelReq {
    model: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GraphReq {
    model: u32,
    #[serde(default)]
    graph: usize,
    #[serde(default)]
    options: Option<ViewOptions>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct NodeReq {
    model: u32,
    graph: usize,
    node: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ValueReq {
    model: u32,
    graph: usize,
    value: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TensorReq {
    model: u32,
    graph: usize,
    tensor: usize,
    #[serde(default)]
    bins: Option<usize>,
    #[serde(default)]
    offset: Option<u64>,
    #[serde(default)]
    count: Option<u64>,
    #[serde(default)]
    format: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ChecksReq {
    model: u32,
    #[serde(default)]
    deep: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OverridesReq {
    model: u32,
    #[serde(default)]
    params: HashMap<String, Option<i64>>,
    #[serde(default)]
    inputs: HashMap<String, Vec<Option<i64>>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchReq {
    model: u32,
    query: String,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeRef {
    pub graph: usize,
    pub id: usize,
    pub name: String,
    pub op: String,
    pub slot: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValueBrief {
    pub id: usize,
    pub name: String,
    pub kind: &'static str,
    pub dtype: String,
    pub shape: Option<Vec<Dim>>,
    pub source: ShapeSource,
    pub producer: Option<NodeRef>,
    pub consumers: usize,
    pub is_input: bool,
    pub is_output: bool,
    pub tensor: Option<usize>,
    pub preview: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SlotInfo {
    pub slot: usize,
    pub label: String,
    pub value: Option<ValueBrief>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttrInfo {
    pub name: String,
    pub type_name: &'static str,
    pub value: String,
    pub doc: String,
    pub ref_attr_name: String,
    pub graphs: Vec<usize>,
    pub tensor: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SubgraphRef {
    pub id: usize,
    pub label: String,
    pub nodes: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeDetail {
    pub graph: usize,
    pub id: usize,
    pub order: usize,
    pub name: String,
    pub op: String,
    pub domain: String,
    pub category: &'static str,
    pub opset: Option<i64>,
    pub doc: String,
    pub doc_url: Option<String>,
    pub inputs: Vec<SlotInfo>,
    pub outputs: Vec<SlotInfo>,
    pub implicit_inputs: Vec<ValueBrief>,
    pub attributes: Vec<AttrInfo>,
    pub subgraphs: Vec<SubgraphRef>,
    pub cost: NodeCost,
    pub metadata: Vec<KeyValue>,
    pub module: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TensorMeta {
    pub index: usize,
    pub dtype: String,
    pub dims: Vec<i64>,
    pub elements: u64,
    pub bytes: Option<u64>,
    pub storage: &'static str,
    pub external: Option<ExternalInfo>,
    pub doc: String,
    pub preview: Vec<String>,
    pub numeric: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValueDetail {
    pub graph: usize,
    pub id: usize,
    pub name: String,
    pub kind: &'static str,
    pub dtype: String,
    pub shape: Option<Vec<Dim>>,
    pub declared: Option<TypeInfo>,
    pub source: ShapeSource,
    pub is_input: bool,
    pub is_output: bool,
    pub doc: String,
    pub elements: Option<u64>,
    pub bytes: Option<u64>,
    pub producer: Option<NodeRef>,
    pub consumers: Vec<NodeRef>,
    pub tensor: Option<TensorMeta>,
    pub contents: Option<Vec<Dim>>,
    pub outer: Option<[usize; 2]>,
    pub quantization: Vec<KeyValue>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub kind: &'static str,
    pub graph: usize,
    pub id: usize,
    pub name: String,
    pub op: String,
    pub detail: String,
}

fn parse<T: for<'de> Deserialize<'de>>(params: &Json) -> Result<T> {
    serde_json::from_value(params.clone()).map_err(|e| anyhow!("Invalid request parameters: {}", e))
}

fn to_json<T: Serialize>(v: &T) -> Result<String> {
    Ok(serde_json::to_string(v)?)
}

fn node_ref(model: &Model, gid: GraphId, ni: usize, slot: Option<usize>) -> NodeRef {
    let n = &model.graphs[gid].nodes[ni];
    NodeRef {
        graph: gid,
        id: ni,
        name: n.name.clone(),
        op: n.op_type.clone(),
        slot,
    }
}

fn value_brief(model: &Model, an: &Analysis, gid: GraphId, vid: usize) -> ValueBrief {
    let g = &model.graphs[gid];
    let v = &g.values[vid];
    let s = &an.shapes[gid].states[vid];
    ValueBrief {
        id: vid,
        name: v.name.clone(),
        kind: v.kind.as_str(),
        dtype: dtype::name(s.dtype).to_string(),
        shape: s.shape.clone(),
        source: an.shapes[gid].sources[vid],
        producer: v.producer.map(|(p, slot)| node_ref(model, gid, p, Some(slot))),
        consumers: v.consumers.len(),
        is_input: v.is_input,
        is_output: v.is_output,
        tensor: v.tensor,
        preview: v.tensor.and_then(|ti| tensor::inline_values(&g.tensors[ti], 8)),
    }
}

impl Workspace {
    pub fn new() -> Self {
        Workspace {
            docs: Mutex::new(HashMap::new()),
            next: AtomicU32::new(1),
        }
    }

    fn docs(&self) -> std::sync::MutexGuard<'_, HashMap<u32, Arc<Document>>> {
        self.docs.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn doc(&self, id: u32) -> Result<Arc<Document>> {
        self.docs()
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow!("Model {} is not open", id))
    }

    /// Parse a model from memory and return its summary as JSON.
    pub fn open_bytes(
        &self,
        name: String,
        path: Option<String>,
        data: Vec<u8>,
        source: Box<dyn DataSource>,
    ) -> Result<String> {
        let model = model::load(data, name, path)?;
        let an = analysis::analyze(&model, Overrides::default());
        let mut id = self.next.fetch_add(1, Ordering::Relaxed);
        if id == 0 {
            id = self.next.fetch_add(1, Ordering::Relaxed);
        }
        let issues = checks::run(&model, &an, source.as_ref(), false);
        let json = to_json(&summary::build(id, &model, &an, &issues))?;
        let doc = Arc::new(Document {
            id,
            model,
            source,
            analysis: RwLock::new(Arc::new(an)),
        });
        self.docs().insert(id, doc);
        Ok(json)
    }

    /// Read a model file from disk (external data is resolved next to it).
    pub fn open_path(&self, path: &str) -> Result<String> {
        let data = std::fs::read(path).map_err(|e| anyhow!("Cannot read {}: {}", path, e))?;
        let p = std::path::Path::new(path);
        let name = p
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string());
        let dir = p.parent().map(|d| d.to_path_buf()).unwrap_or_default();
        self.open_bytes(name, Some(path.to_string()), data, Box::new(FileSource { dir }))
    }

    pub fn close(&self, id: u32) {
        self.docs().remove(&id);
    }

    /// [`Workspace::call`] with the parameters given as a JSON string.
    pub fn call_str(&self, method: &str, params: &str) -> Result<String> {
        let v: Json = if params.trim().is_empty() {
            Json::Null
        } else {
            serde_json::from_str(params)?
        };
        self.call(method, &v)
    }

    /// [`Workspace::export`] with the parameters given as a JSON string.
    pub fn export_str(&self, params: &str) -> Result<Vec<u8>> {
        let v: Json = serde_json::from_str(params)?;
        self.export(&v)
    }

    pub fn call(&self, method: &str, params: &Json) -> Result<String> {
        match method {
            "summary" => {
                let r: ModelReq = parse(params)?;
                let d = self.doc(r.model)?;
                let an = d.analysis();
                let issues = checks::run(&d.model, &an, d.source.as_ref(), false);
                to_json(&summary::build(d.id, &d.model, &an, &issues))
            }
            "close" => {
                let r: ModelReq = parse(params)?;
                self.close(r.model);
                Ok("null".into())
            }
            "graph" => {
                let r: GraphReq = parse(params)?;
                let d = self.doc(r.model)?;
                let an = d.analysis();
                let opts = r.options.unwrap_or_default();
                to_json(&view::build_view(&d.model, &an, r.graph, &opts)?)
            }
            "node" => {
                let r: NodeReq = parse(params)?;
                let d = self.doc(r.model)?;
                to_json(&self.node_detail(&d, r.graph, r.node)?)
            }
            "value" => {
                let r: ValueReq = parse(params)?;
                let d = self.doc(r.model)?;
                to_json(&self.value_detail(&d, r.graph, r.value)?)
            }
            "tensorStats" => {
                let r: TensorReq = parse(params)?;
                let d = self.doc(r.model)?;
                let t = tensor_ref(&d.model, r.graph, r.tensor)?;
                to_json(&tensor::compute_stats(t, d.source.as_ref(), r.bins.unwrap_or(64))?)
            }
            "tensorValues" => {
                let r: TensorReq = parse(params)?;
                let d = self.doc(r.model)?;
                let t = tensor_ref(&d.model, r.graph, r.tensor)?;
                let count = r.count.unwrap_or(256).min(100_000);
                to_json(&tensor::values_page(t, d.source.as_ref(), r.offset.unwrap_or(0), count)?)
            }
            "nodeTable" => {
                let r: GraphReq = parse(params)?;
                let d = self.doc(r.model)?;
                to_json(&summary::node_table(&d.model, &d.analysis(), r.graph)?)
            }
            "valueTable" => {
                let r: GraphReq = parse(params)?;
                let d = self.doc(r.model)?;
                to_json(&summary::value_table(&d.model, &d.analysis(), r.graph)?)
            }
            "weightTable" => {
                let r: ModelReq = parse(params)?;
                let d = self.doc(r.model)?;
                to_json(&summary::weight_table(&d.model, d.source.as_ref()))
            }
            "hierarchy" => {
                let r: GraphReq = parse(params)?;
                let d = self.doc(r.model)?;
                to_json(&summary::hierarchy(&d.model, &d.analysis(), r.graph)?)
            }
            "checks" => {
                let r: ChecksReq = parse(params)?;
                let d = self.doc(r.model)?;
                to_json(&checks::run(&d.model, &d.analysis(), d.source.as_ref(), r.deep))
            }
            "setOverrides" => {
                let r: OverridesReq = parse(params)?;
                let d = self.doc(r.model)?;
                let mut ov = Overrides::default();
                for (k, v) in r.params {
                    if let Some(v) = v {
                        if v < 0 {
                            bail!("Dimension '{}' must not be negative", k);
                        }
                        ov.params.insert(k, v);
                    }
                }
                for (k, v) in r.inputs {
                    if v.iter().any(|x| x.is_some()) {
                        ov.inputs.insert(k, v);
                    }
                }
                let an = Arc::new(analysis::analyze(&d.model, ov));
                *d.analysis.write().unwrap_or_else(|e| e.into_inner()) = an.clone();
                let issues = checks::run(&d.model, &an, d.source.as_ref(), false);
                to_json(&summary::build(d.id, &d.model, &an, &issues))
            }
            "search" => {
                let r: SearchReq = parse(params)?;
                let d = self.doc(r.model)?;
                to_json(&search(&d.model, &r.query, r.limit.unwrap_or(300)))
            }
            _ => bail!("Unknown method '{}'", method),
        }
    }

    /// Export a tensor; `format` is "npy" (default), "csv" or "raw".
    pub fn export(&self, params: &Json) -> Result<Vec<u8>> {
        let r: TensorReq = parse(params)?;
        let d = self.doc(r.model)?;
        let t = tensor_ref(&d.model, r.graph, r.tensor)?;
        match r.format.as_deref().unwrap_or("npy") {
            "npy" => tensor::to_npy(t, d.source.as_ref()),
            "csv" => tensor::to_csv(t, d.source.as_ref()),
            "raw" => {
                if let Some(e) = tensor::external_info(t) {
                    d.source.read_external(&e.location, e.offset, e.length)
                } else if !t.raw_data.is_empty() {
                    Ok(t.raw_data.to_vec())
                } else {
                    // Typed storage: serialize through the .npy writer and
                    // strip the header.
                    let npy = tensor::to_npy(t, d.source.as_ref())?;
                    let hl = u16::from_le_bytes([npy[8], npy[9]]) as usize;
                    Ok(npy[10 + hl..].to_vec())
                }
            }
            other => bail!("Unknown export format '{}'", other),
        }
    }

    fn node_detail(&self, d: &Document, gid: GraphId, ni: usize) -> Result<NodeDetail> {
        let model = &d.model;
        let an = d.analysis();
        let g = model.graph(gid)?;
        let n = g
            .nodes
            .get(ni)
            .ok_or_else(|| anyhow!("node {} does not exist", ni))?;
        let order = g.topo.iter().position(|&x| x == ni).unwrap_or(ni);
        let inputs = n
            .inputs
            .iter()
            .enumerate()
            .map(|(slot, v)| SlotInfo {
                slot,
                label: ops::input_name(&n.domain, &n.op_type, slot),
                value: v.map(|vid| value_brief(model, &an, gid, vid)),
            })
            .collect();
        let outputs = n
            .outputs
            .iter()
            .enumerate()
            .map(|(slot, v)| SlotInfo {
                slot,
                label: ops::output_name(&n.domain, &n.op_type, slot),
                value: v.map(|vid| value_brief(model, &an, gid, vid)),
            })
            .collect();
        let attributes = n
            .attributes
            .iter()
            .map(|a| {
                let (graphs, tensor) = match &a.value {
                    AttrValue::Graph(s) => (vec![*s], None),
                    AttrValue::Graphs(s) => (s.clone(), None),
                    AttrValue::Tensor(t) => (Vec::new(), Some(*t)),
                    _ => (Vec::new(), None),
                };
                AttrInfo {
                    name: a.name.clone(),
                    type_name: a.value.type_name(),
                    value: format::attr_full(model, g, &a.value),
                    doc: a.doc.clone(),
                    ref_attr_name: a.ref_attr_name.clone(),
                    graphs,
                    tensor,
                }
            })
            .collect();
        let subgraphs = n
            .subgraphs
            .iter()
            .map(|&s| SubgraphRef {
                id: s,
                label: model.graphs[s].label(&model.graphs),
                nodes: model.graphs[s].nodes.len(),
            })
            .collect();
        Ok(NodeDetail {
            graph: gid,
            id: ni,
            order,
            name: n.name.clone(),
            op: n.op_type.clone(),
            domain: n.domain.clone(),
            category: ops::category(&n.domain, &n.op_type).as_str(),
            opset: g.opset(&n.domain),
            doc: n.doc.clone(),
            doc_url: ops::doc_url(&n.domain, &n.op_type),
            inputs,
            outputs,
            implicit_inputs: n
                .implicit_inputs
                .iter()
                .map(|&v| value_brief(model, &an, gid, v))
                .collect(),
            attributes,
            subgraphs,
            cost: an.costs[gid][ni].clone(),
            metadata: summary::kvs(&n.metadata),
            module: model::module_path(&n.name),
        })
    }

    fn value_detail(&self, d: &Document, gid: GraphId, vid: usize) -> Result<ValueDetail> {
        let model = &d.model;
        let an = d.analysis();
        let g = model.graph(gid)?;
        let v = g
            .values
            .get(vid)
            .ok_or_else(|| anyhow!("value {} does not exist", vid))?;
        let s = &an.shapes[gid].states[vid];
        let elements = s.shape.as_deref().and_then(shape::numel);
        let tensor = v.tensor.map(|ti| {
            let t = &g.tensors[ti];
            let ext = tensor::external_info(t);
            let storage = if ext.is_some() {
                "external"
            } else if !t.raw_data.is_empty() {
                "raw"
            } else {
                "typed"
            };
            let preview = tensor::values_page(t, d.source.as_ref(), 0, 64)
                .map(|p| p.values)
                .unwrap_or_default();
            TensorMeta {
                index: ti,
                dtype: dtype::name(t.data_type).to_string(),
                dims: t.dims.clone(),
                elements: tensor::element_count(t),
                bytes: tensor::byte_size(t),
                storage,
                external: ext,
                doc: t.doc_string.clone(),
                preview,
                numeric: dtype::is_numeric(t.data_type),
            }
        });
        let bytes = match &tensor {
            Some(t) => t.bytes,
            None => elements.and_then(|n| dtype::bytes_for(s.dtype, n)),
        };
        let quantization = g
            .quant_annotations
            .iter()
            .filter(|(name, _)| name == &v.name)
            .flat_map(|(_, kv)| summary::kvs(kv))
            .collect();
        Ok(ValueDetail {
            graph: gid,
            id: vid,
            name: v.name.clone(),
            kind: v.kind.as_str(),
            dtype: dtype::name(s.dtype).to_string(),
            shape: s.shape.clone(),
            declared: v.declared.clone(),
            source: an.shapes[gid].sources[vid],
            is_input: v.is_input,
            is_output: v.is_output,
            doc: v.doc.clone(),
            elements,
            bytes,
            producer: v.producer.map(|(p, slot)| node_ref(model, gid, p, Some(slot))),
            consumers: v
                .consumers
                .iter()
                .map(|&(c, slot)| node_ref(model, gid, c, if slot == IMPLICIT_SLOT { None } else { Some(slot) }))
                .collect(),
            tensor,
            contents: s.ints.clone(),
            outer: v.outer.map(|(a, b)| [a, b]),
            quantization,
        })
    }
}

fn tensor_ref(model: &Model, gid: GraphId, ti: usize) -> Result<&crate::proto::TensorProto> {
    let g = model.graph(gid)?;
    g.tensors
        .get(ti)
        .ok_or_else(|| anyhow!("tensor {} does not exist", ti))
}

fn search(model: &Model, query: &str, limit: usize) -> Vec<SearchHit> {
    let q = query.trim().to_lowercase();
    let mut hits = Vec::new();
    if q.is_empty() {
        return hits;
    }
    // "op:Conv" restricts to operator types.
    let (op_only, q) = match q.strip_prefix("op:") {
        Some(rest) => (true, rest.trim().to_string()),
        None => (false, q),
    };
    for (gid, g) in model.graphs.iter().enumerate() {
        for (ni, n) in g.nodes.iter().enumerate() {
            if hits.len() >= limit {
                return hits;
            }
            let op = n.op_type.to_lowercase();
            let matched = if op_only {
                op == q
            } else {
                op == q || n.name.to_lowercase().contains(&q)
            };
            if matched {
                hits.push(SearchHit {
                    kind: "node",
                    graph: gid,
                    id: ni,
                    name: n.name.clone(),
                    op: n.op_type.clone(),
                    detail: model::module_path(&n.name),
                });
            }
        }
        if op_only {
            continue;
        }
        for (vid, v) in g.values.iter().enumerate() {
            if hits.len() >= limit {
                return hits;
            }
            if v.name.to_lowercase().contains(&q) {
                hits.push(SearchHit {
                    kind: "value",
                    graph: gid,
                    id: vid,
                    name: v.name.clone(),
                    op: v.kind.as_str().to_string(),
                    detail: String::new(),
                });
            }
        }
    }
    hits
}
