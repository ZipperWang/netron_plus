//! In-memory model representation built from the protobuf messages.
//!
//! Every graph (main graph, control-flow subgraphs, local functions) becomes a
//! [`Graph`] with resolved value references, so later passes never have to
//! look names up again. Tensor payloads are moved out of the protobuf tree into
//! [`Graph::tensors`] without copying.

use crate::dtype;
use crate::proto::{self, attr_type};
use crate::shape::{self, Dim};
use anyhow::{anyhow, bail, Result};
use prost::Message;
use serde::Serialize;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub type GraphId = usize;
pub type NodeId = usize;
pub type ValueId = usize;

/// Consumer slot used for implicit (subgraph) uses of a value.
pub const IMPLICIT_SLOT: usize = usize::MAX;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum TypeKind {
    #[default]
    Unknown,
    Tensor,
    SparseTensor,
    Sequence,
    Map,
    Optional,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TypeInfo {
    pub kind: TypeKind,
    pub dtype: i32,
    pub shape: Option<Vec<Dim>>,
    pub denotation: String,
    pub display: String,
}

impl TypeInfo {
    pub fn tensor(dtype: i32, shape: Option<Vec<Dim>>) -> TypeInfo {
        let display = shape::type_label(dtype::name(dtype), shape.as_deref());
        TypeInfo {
            kind: TypeKind::Tensor,
            dtype,
            shape,
            denotation: String::new(),
            display,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ValueKind {
    Input,
    Initializer,
    Constant,
    Intermediate,
    OuterScope,
    Unresolved,
}

impl ValueKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ValueKind::Input => "input",
            ValueKind::Initializer => "initializer",
            ValueKind::Constant => "constant",
            ValueKind::Intermediate => "intermediate",
            ValueKind::OuterScope => "outerScope",
            ValueKind::Unresolved => "unresolved",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Value {
    pub name: String,
    pub kind: ValueKind,
    pub is_input: bool,
    pub is_output: bool,
    pub declared: Option<TypeInfo>,
    /// Index into [`Graph::tensors`] for initializers and constants.
    pub tensor: Option<usize>,
    pub sparse: Option<SparseInfo>,
    pub producer: Option<(NodeId, usize)>,
    /// `(node, input slot)`; slot is [`IMPLICIT_SLOT`] for subgraph uses.
    pub consumers: Vec<(NodeId, usize)>,
    /// For outer-scope values: the value in the enclosing graph.
    pub outer: Option<(GraphId, ValueId)>,
    pub doc: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SparseInfo {
    pub dims: Vec<i64>,
    pub dtype: i32,
    pub nnz: u64,
}

#[derive(Clone, Debug)]
pub enum AttrValue {
    Undefined,
    Float(f32),
    Int(i64),
    String(Vec<u8>),
    Tensor(usize),
    Graph(GraphId),
    SparseTensor(SparseInfo),
    Type(TypeInfo),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<Vec<u8>>),
    Tensors(Vec<usize>),
    Graphs(Vec<GraphId>),
    SparseTensors(Vec<SparseInfo>),
    Types(Vec<TypeInfo>),
}

impl AttrValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            AttrValue::Undefined => "undefined",
            AttrValue::Float(_) => "float",
            AttrValue::Int(_) => "int",
            AttrValue::String(_) => "string",
            AttrValue::Tensor(_) => "tensor",
            AttrValue::Graph(_) => "graph",
            AttrValue::SparseTensor(_) => "sparse_tensor",
            AttrValue::Type(_) => "type_proto",
            AttrValue::Floats(_) => "float[]",
            AttrValue::Ints(_) => "int[]",
            AttrValue::Strings(_) => "string[]",
            AttrValue::Tensors(_) => "tensor[]",
            AttrValue::Graphs(_) => "graph[]",
            AttrValue::SparseTensors(_) => "sparse_tensor[]",
            AttrValue::Types(_) => "type_proto[]",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Attribute {
    pub name: String,
    pub value: AttrValue,
    pub doc: String,
    pub ref_attr_name: String,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub op_type: String,
    pub domain: String,
    /// Overload of the model-local function the node calls (IR 10+).
    pub overload: String,
    pub inputs: Vec<Option<ValueId>>,
    pub outputs: Vec<Option<ValueId>>,
    pub attributes: Vec<Attribute>,
    pub doc: String,
    pub subgraphs: Vec<GraphId>,
    /// Values of this graph referenced from inside subgraphs.
    pub implicit_inputs: Vec<ValueId>,
    pub metadata: Vec<(String, String)>,
}

impl Node {
    pub fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attributes
            .iter()
            .find(|a| a.name == name)
            .map(|a| &a.value)
    }

    pub fn attr_i(&self, name: &str) -> Option<i64> {
        match self.attr(name)? {
            AttrValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn attr_f(&self, name: &str) -> Option<f32> {
        match self.attr(name)? {
            AttrValue::Float(v) => Some(*v),
            AttrValue::Int(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn attr_ints(&self, name: &str) -> Option<&[i64]> {
        match self.attr(name)? {
            AttrValue::Ints(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn attr_floats(&self, name: &str) -> Option<&[f32]> {
        match self.attr(name)? {
            AttrValue::Floats(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn attr_s(&self, name: &str) -> Option<String> {
        match self.attr(name)? {
            AttrValue::String(v) => Some(String::from_utf8_lossy(v).into_owned()),
            _ => None,
        }
    }

    pub fn input(&self, slot: usize) -> Option<ValueId> {
        self.inputs.get(slot).copied().flatten()
    }

    pub fn output(&self, slot: usize) -> Option<ValueId> {
        self.outputs.get(slot).copied().flatten()
    }

    pub fn is_standard(&self) -> bool {
        crate::ops::is_standard_domain(&self.domain)
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum GraphOwner {
    Main,
    #[serde(rename_all = "camelCase")]
    Subgraph {
        parent: GraphId,
        node: NodeId,
        attribute: String,
    },
    #[serde(rename_all = "camelCase")]
    Function {
        index: usize,
        domain: String,
        name: String,
        overload: String,
    },
    #[serde(rename_all = "camelCase")]
    Training { index: usize, role: String },
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildIssue {
    pub code: &'static str,
    pub value: Option<ValueId>,
    pub node: Option<NodeId>,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub name: String,
    pub doc: String,
    pub owner: GraphOwner,
    pub depth: usize,
    pub nodes: Vec<Node>,
    pub values: Vec<Value>,
    pub value_map: HashMap<String, ValueId>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub initializers: Vec<ValueId>,
    pub tensors: Vec<proto::TensorProto>,
    pub topo: Vec<NodeId>,
    pub has_cycle: bool,
    pub sorted: bool,
    pub metadata: Vec<(String, String)>,
    pub quant_annotations: Vec<(String, Vec<(String, String)>)>,
    pub issues: Vec<BuildIssue>,
    /// Opset imports that apply to this graph (functions carry their own).
    pub opsets: Vec<(String, i64)>,
}

impl Graph {
    fn empty(owner: GraphOwner, depth: usize) -> Graph {
        Graph {
            name: String::new(),
            doc: String::new(),
            owner,
            depth,
            nodes: Vec::new(),
            values: Vec::new(),
            value_map: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
            tensors: Vec::new(),
            topo: Vec::new(),
            has_cycle: false,
            sorted: true,
            metadata: Vec::new(),
            quant_annotations: Vec::new(),
            issues: Vec::new(),
            opsets: Vec::new(),
        }
    }

    fn is_subgraph(&self) -> bool {
        matches!(self.owner, GraphOwner::Subgraph { .. })
    }

    fn define(&mut self, name: String, kind: ValueKind) -> ValueId {
        let id = self.values.len();
        self.value_map.insert(name.clone(), id);
        self.values.push(Value {
            name,
            kind,
            is_input: false,
            is_output: false,
            declared: None,
            tensor: None,
            sparse: None,
            producer: None,
            consumers: Vec::new(),
            outer: None,
            doc: String::new(),
        });
        id
    }

    /// Resolve a referenced name, creating an outer-scope / unresolved value
    /// when it is not defined in this graph.
    fn resolve(&mut self, name: &str) -> ValueId {
        if let Some(&id) = self.value_map.get(name) {
            return id;
        }
        let kind = if self.is_subgraph() {
            ValueKind::OuterScope
        } else {
            ValueKind::Unresolved
        };
        self.define(name.to_string(), kind)
    }

    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id]
    }

    pub fn find_value(&self, name: &str) -> Option<ValueId> {
        self.value_map.get(name).copied()
    }

    pub fn opset(&self, domain: &str) -> Option<i64> {
        let std = crate::ops::is_standard_domain(domain);
        self.opsets
            .iter()
            .find(|(d, _)| {
                if std {
                    crate::ops::is_standard_domain(d)
                } else {
                    d == domain
                }
            })
            .map(|(_, v)| *v)
    }

    /// Short label for the graph used in breadcrumbs and lists.
    pub fn label(&self, graphs: &[Graph]) -> String {
        match &self.owner {
            GraphOwner::Main => {
                if self.name.is_empty() {
                    "main".to_string()
                } else {
                    self.name.clone()
                }
            }
            GraphOwner::Subgraph {
                parent,
                node,
                attribute,
            } => {
                let pnode = &graphs[*parent].nodes[*node];
                let nname = if pnode.name.is_empty() {
                    format!("{}#{}", pnode.op_type, node)
                } else {
                    pnode.name.clone()
                };
                format!("{}.{}", nname, attribute)
            }
            GraphOwner::Function {
                domain,
                name,
                overload,
                ..
            } => {
                let name = if overload.is_empty() {
                    name.clone()
                } else {
                    format!("{}:{}", name, overload)
                };
                if domain.is_empty() {
                    format!("function {}", name)
                } else {
                    format!("function {}::{}", domain, name)
                }
            }
            GraphOwner::Training { index, role } => format!("training[{}].{}", index, role),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc: String,
    pub opsets: Vec<(String, i64)>,
    pub metadata: Vec<(String, String)>,
    pub graphs: Vec<Graph>,
    /// Model-local functions by `(domain, name, overload)`; the standard
    /// domain is "".
    pub functions: HashMap<(String, String, String), GraphId>,
    pub file_name: String,
    pub file_path: Option<String>,
    pub file_size: u64,
}

impl Model {
    pub fn main(&self) -> &Graph {
        &self.graphs[0]
    }

    pub fn graph(&self, id: GraphId) -> Result<&Graph> {
        self.graphs
            .get(id)
            .ok_or_else(|| anyhow!("graph {} does not exist", id))
    }

    pub fn opset(&self, domain: &str) -> Option<i64> {
        self.graphs.first().and_then(|g| g.opset(domain))
    }

    /// The model-local function with this domain, name and overload, if any.
    pub fn function(&self, domain: &str, name: &str, overload: &str) -> Option<GraphId> {
        if self.functions.is_empty() {
            return None;
        }
        self.functions
            .get(&function_key(domain, name, overload))
            .copied()
    }

    /// The model-local function a node calls, if any.
    pub fn function_of(&self, node: &Node) -> Option<GraphId> {
        self.function(&node.domain, &node.op_type, &node.overload)
    }
}

fn function_key(domain: &str, name: &str, overload: &str) -> (String, String, String) {
    let d = if crate::ops::is_standard_domain(domain) { "" } else { domain };
    (d.to_string(), name.to_string(), overload.to_string())
}

/// Decode an ONNX file and build the model representation.
pub fn load(bytes: Vec<u8>, file_name: String, file_path: Option<String>) -> Result<Model> {
    let file_size = bytes.len() as u64;
    if bytes.is_empty() {
        bail!("The file is empty");
    }
    check_magic(&bytes)?;
    let buf = prost::bytes::Bytes::from(bytes);
    let mp = proto::ModelProto::decode(buf)
        .map_err(|e| anyhow!("Not a valid ONNX model (protobuf decode failed: {})", e))?;
    let graph = match mp.graph {
        Some(g) => g,
        None => bail!("Not a valid ONNX model: the file has no graph"),
    };
    if mp.ir_version == 0 && graph.node.is_empty() && graph.input.is_empty() {
        bail!("Not a valid ONNX model: missing IR version and graph content");
    }

    let opsets: Vec<(String, i64)> = mp
        .opset_import
        .iter()
        .map(|o| (o.domain.clone(), o.version))
        .collect();

    let mut b = Builder { graphs: Vec::new() };
    b.build_graph(graph, GraphOwner::Main, 0, &opsets);

    for (index, f) in mp.functions.into_iter().enumerate() {
        let fopsets: Vec<(String, i64)> = if f.opset_import.is_empty() {
            opsets.clone()
        } else {
            f.opset_import
                .iter()
                .map(|o| (o.domain.clone(), o.version))
                .collect()
        };
        b.build_function(index, f, &fopsets);
    }

    for (index, t) in mp.training_info.into_iter().enumerate() {
        if let Some(g) = t.initialization {
            if !g.node.is_empty() {
                b.build_graph(
                    g,
                    GraphOwner::Training {
                        index,
                        role: "initialization".into(),
                    },
                    0,
                    &opsets,
                );
            }
        }
        if let Some(g) = t.algorithm {
            if !g.node.is_empty() {
                b.build_graph(
                    g,
                    GraphOwner::Training {
                        index,
                        role: "algorithm".into(),
                    },
                    0,
                    &opsets,
                );
            }
        }
    }

    let mut functions = HashMap::new();
    for (gid, g) in b.graphs.iter().enumerate() {
        if let GraphOwner::Function {
            domain,
            name,
            overload,
            ..
        } = &g.owner
        {
            functions
                .entry(function_key(domain, name, overload))
                .or_insert(gid);
        }
    }

    Ok(Model {
        ir_version: mp.ir_version,
        producer_name: mp.producer_name,
        producer_version: mp.producer_version,
        domain: mp.domain,
        model_version: mp.model_version,
        doc: mp.doc_string,
        opsets,
        metadata: kv(mp.metadata_props),
        graphs: b.graphs,
        functions,
        file_name,
        file_path,
        file_size,
    })
}

/// Reject some common non-ONNX formats with a helpful message.
fn check_magic(b: &[u8]) -> Result<()> {
    if b.len() >= 8 && &b[4..8] == b"TFL3" {
        bail!("This looks like a TensorFlow Lite model; only ONNX (.onnx) files are supported");
    }
    if b.len() >= 8 && &b[4..8] == b"ORTM" {
        bail!("This is an ONNX Runtime (.ort) flatbuffer model; please open the original .onnx file");
    }
    if b.len() >= 2 && b[0] == 0x80 && b[1] <= 5 {
        bail!("This looks like a Python pickle (PyTorch) file; export it to ONNX first");
    }
    if b.len() >= 4 && &b[0..4] == b"PK\x03\x04" {
        bail!("This is a ZIP archive (e.g. a PyTorch checkpoint); export the model to ONNX first");
    }
    if b.len() >= 8 && &b[0..8] == b"\x89HDF\r\n\x1a\n" {
        bail!("This is an HDF5 (Keras) file; convert it to ONNX first");
    }
    Ok(())
}

fn kv(props: Vec<proto::StringStringEntryProto>) -> Vec<(String, String)> {
    props.into_iter().map(|p| (p.key, p.value)).collect()
}

pub fn shape_from_proto(s: &proto::TensorShapeProto) -> Vec<Dim> {
    use proto::tensor_shape_proto::dimension::Value as DV;
    s.dim
        .iter()
        .map(|d| match &d.value {
            Some(DV::DimValue(v)) if *v >= 0 => Dim::Known(*v),
            Some(DV::DimParam(p)) if !p.is_empty() => Dim::Param(p.clone()),
            _ => Dim::Unknown,
        })
        .collect()
}

pub fn type_info(tp: Option<&proto::TypeProto>) -> TypeInfo {
    use proto::type_proto::Value as TV;
    let Some(tp) = tp else {
        return TypeInfo::default();
    };
    let mut info = match &tp.value {
        Some(TV::TensorType(t)) => TypeInfo::tensor(t.elem_type, t.shape.as_ref().map(shape_from_proto)),
        Some(TV::SparseTensorType(t)) => {
            let shape = t.shape.as_ref().map(shape_from_proto);
            let display = format!(
                "sparse {}",
                shape::type_label(dtype::name(t.elem_type), shape.as_deref())
            );
            TypeInfo {
                kind: TypeKind::SparseTensor,
                dtype: t.elem_type,
                shape,
                denotation: String::new(),
                display,
            }
        }
        Some(TV::SequenceType(s)) => {
            let inner = type_info(s.elem_type.as_deref());
            TypeInfo {
                kind: TypeKind::Sequence,
                dtype: inner.dtype,
                shape: None,
                denotation: String::new(),
                display: format!("sequence<{}>", inner.display),
            }
        }
        Some(TV::MapType(m)) => {
            let inner = type_info(m.value_type.as_deref());
            TypeInfo {
                kind: TypeKind::Map,
                dtype: inner.dtype,
                shape: None,
                denotation: String::new(),
                display: format!("map<{}, {}>", dtype::name(m.key_type), inner.display),
            }
        }
        Some(TV::OptionalType(o)) => {
            let inner = type_info(o.elem_type.as_deref());
            TypeInfo {
                kind: TypeKind::Optional,
                dtype: inner.dtype,
                shape: inner.shape.clone(),
                denotation: String::new(),
                display: format!("optional<{}>", inner.display),
            }
        }
        None => TypeInfo::default(),
    };
    info.denotation = tp.denotation.clone();
    info
}

fn infer_attr_type(a: &proto::AttributeProto) -> i32 {
    if !a.floats.is_empty() {
        attr_type::FLOATS
    } else if !a.ints.is_empty() {
        attr_type::INTS
    } else if !a.strings.is_empty() {
        attr_type::STRINGS
    } else if !a.tensors.is_empty() {
        attr_type::TENSORS
    } else if !a.graphs.is_empty() {
        attr_type::GRAPHS
    } else if a.t.is_some() {
        attr_type::TENSOR
    } else if a.g.is_some() {
        attr_type::GRAPH
    } else if a.sparse_tensor.is_some() {
        attr_type::SPARSE_TENSOR
    } else if a.tp.is_some() {
        attr_type::TYPE_PROTO
    } else if !a.s.is_empty() {
        attr_type::STRING
    } else if a.i != 0 {
        attr_type::INT
    } else if a.f != 0.0 {
        attr_type::FLOAT
    } else {
        attr_type::UNDEFINED
    }
}

fn sparse_info(s: &proto::SparseTensorProto) -> SparseInfo {
    let dtype = s.values.as_ref().map(|v| v.data_type).unwrap_or(0);
    let nnz = s
        .values
        .as_ref()
        .map(|v| shape::numel_i64(&v.dims))
        .unwrap_or(0);
    SparseInfo {
        dims: s.dims.clone(),
        dtype,
        nnz,
    }
}

struct Builder {
    graphs: Vec<Graph>,
}

impl Builder {
    fn reserve(&mut self, owner: GraphOwner, depth: usize) -> GraphId {
        let id = self.graphs.len();
        self.graphs.push(Graph::empty(owner, depth));
        id
    }

    fn build_graph(
        &mut self,
        g: proto::GraphProto,
        owner: GraphOwner,
        depth: usize,
        opsets: &[(String, i64)],
    ) -> GraphId {
        let id = self.reserve(owner.clone(), depth);
        let mut graph = Graph::empty(owner, depth);
        graph.name = g.name;
        graph.doc = g.doc_string;
        graph.metadata = kv(g.metadata_props);
        graph.opsets = opsets.to_vec();
        graph.quant_annotations = g
            .quantization_annotation
            .into_iter()
            .map(|a| (a.tensor_name, kv(a.quant_parameter_tensor_names)))
            .collect();

        // Initializers.
        for t in g.initializer {
            let name = t.name.clone();
            if graph.value_map.contains_key(&name) {
                graph.issues.push(BuildIssue {
                    code: "duplicate-initializer",
                    value: graph.value_map.get(&name).copied(),
                    node: None,
                    name: name.clone(),
                });
                continue;
            }
            let decl = TypeInfo::tensor(t.data_type, Some(shape::from_i64s(&t.dims)));
            let tidx = graph.tensors.len();
            graph.tensors.push(t);
            let vid = graph.define(name, ValueKind::Initializer);
            graph.values[vid].tensor = Some(tidx);
            graph.values[vid].declared = Some(decl);
            graph.initializers.push(vid);
        }
        for st in g.sparse_initializer {
            let name = st
                .values
                .as_ref()
                .map(|v| v.name.clone())
                .unwrap_or_default();
            if name.is_empty() || graph.value_map.contains_key(&name) {
                continue;
            }
            let info = sparse_info(&st);
            let decl = TypeInfo::tensor(info.dtype, Some(shape::from_i64s(&info.dims)));
            let vid = graph.define(name, ValueKind::Initializer);
            graph.values[vid].sparse = Some(info);
            graph.values[vid].declared = Some(decl);
            graph.initializers.push(vid);
        }

        // Graph inputs (older models also list initializers here).
        for vi in g.input {
            let ti = type_info(vi.r#type.as_ref());
            let vid = match graph.value_map.get(&vi.name) {
                Some(&v) => v,
                None => graph.define(vi.name.clone(), ValueKind::Input),
            };
            let v = &mut graph.values[vid];
            v.is_input = true;
            if v.kind != ValueKind::Initializer || v.declared.is_none() {
                v.declared = Some(ti);
            }
            if v.doc.is_empty() {
                v.doc = vi.doc_string;
            }
            graph.inputs.push(vid);
        }

        // Register node outputs before resolving inputs so that graphs that
        // are not topologically sorted still resolve.
        let nodes = g.node;
        for (ni, n) in nodes.iter().enumerate() {
            for (oi, o) in n.output.iter().enumerate() {
                if o.is_empty() {
                    continue;
                }
                match graph.value_map.get(o).copied() {
                    Some(vid) => {
                        graph.issues.push(BuildIssue {
                            code: "duplicate-producer",
                            value: Some(vid),
                            node: Some(ni),
                            name: o.clone(),
                        });
                        if graph.values[vid].producer.is_none()
                            && graph.values[vid].kind == ValueKind::Input
                        {
                            graph.values[vid].producer = Some((ni, oi));
                        }
                    }
                    None => {
                        let vid = graph.define(o.clone(), ValueKind::Intermediate);
                        graph.values[vid].producer = Some((ni, oi));
                    }
                }
            }
        }

        // value_info annotations.
        for vi in g.value_info {
            if let Some(&vid) = graph.value_map.get(&vi.name) {
                let v = &mut graph.values[vid];
                if v.declared.is_none() {
                    v.declared = Some(type_info(vi.r#type.as_ref()));
                }
                if v.doc.is_empty() {
                    v.doc = vi.doc_string;
                }
            }
        }

        // Graph outputs.
        for vi in g.output {
            let vid = graph.resolve(&vi.name);
            let v = &mut graph.values[vid];
            v.is_output = true;
            let ti = type_info(vi.r#type.as_ref());
            if v.kind != ValueKind::Initializer
                && (v.declared.is_none() || ti.kind != TypeKind::Unknown)
            {
                v.declared = Some(ti);
            }
            if v.doc.is_empty() {
                v.doc = vi.doc_string;
            }
            graph.outputs.push(vid);
        }

        // Nodes.
        for (ni, n) in nodes.into_iter().enumerate() {
            let mut attributes = Vec::with_capacity(n.attribute.len());
            let mut subgraphs = Vec::new();
            for a in n.attribute {
                let attr = self.convert_attr(&mut graph, a, id, ni, depth, &mut subgraphs);
                attributes.push(attr);
            }

            let mut inputs = Vec::with_capacity(n.input.len());
            for (slot, name) in n.input.iter().enumerate() {
                if name.is_empty() {
                    inputs.push(None);
                    continue;
                }
                let vid = graph.resolve(name);
                graph.values[vid].consumers.push((ni, slot));
                inputs.push(Some(vid));
            }
            let outputs: Vec<Option<ValueId>> = n
                .output
                .iter()
                .map(|o| {
                    if o.is_empty() {
                        None
                    } else {
                        graph.value_map.get(o).copied()
                    }
                })
                .collect();

            // Implicit inputs: names that the subgraphs take from this scope.
            let mut implicit = Vec::new();
            for &sid in &subgraphs {
                let free: Vec<(ValueId, String)> = self.graphs[sid]
                    .values
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.kind == ValueKind::OuterScope)
                    .map(|(i, v)| (i, v.name.clone()))
                    .collect();
                for (sub_vid, name) in free {
                    let vid = graph.resolve(&name);
                    self.graphs[sid].values[sub_vid].outer = Some((id, vid));
                    if !implicit.contains(&vid) {
                        implicit.push(vid);
                        graph.values[vid].consumers.push((ni, IMPLICIT_SLOT));
                    }
                }
            }

            let node = Node {
                name: n.name,
                op_type: n.op_type,
                domain: n.domain,
                overload: n.overload,
                inputs,
                outputs,
                attributes,
                doc: n.doc_string,
                subgraphs,
                implicit_inputs: implicit,
                metadata: kv(n.metadata_props),
            };
            graph.nodes.push(node);
        }

        mark_constants(&mut graph);
        topo_sort(&mut graph);
        self.graphs[id] = graph;
        id
    }

    fn build_function(&mut self, index: usize, f: proto::FunctionProto, opsets: &[(String, i64)]) {
        let owner = GraphOwner::Function {
            index,
            domain: f.domain.clone(),
            name: f.name.clone(),
            overload: f.overload.clone(),
        };
        let mut g = proto::GraphProto {
            name: f.name.clone(),
            doc_string: f.doc_string,
            node: f.node,
            value_info: f.value_info,
            metadata_props: f.metadata_props,
            ..Default::default()
        };
        g.input = f
            .input
            .into_iter()
            .map(|name| proto::ValueInfoProto {
                name,
                ..Default::default()
            })
            .collect();
        g.output = f
            .output
            .into_iter()
            .map(|name| proto::ValueInfoProto {
                name,
                ..Default::default()
            })
            .collect();
        let id = self.build_graph(g, owner, 0, opsets);
        // Function attributes are documented on the graph for display.
        let mut attrs: Vec<String> = f.attribute;
        attrs.extend(f.attribute_proto.into_iter().map(|a| a.name));
        if !attrs.is_empty() {
            self.graphs[id]
                .metadata
                .push(("attributes".to_string(), attrs.join(", ")));
        }
    }

    fn convert_attr(
        &mut self,
        graph: &mut Graph,
        a: proto::AttributeProto,
        gid: GraphId,
        ni: NodeId,
        depth: usize,
        subgraphs: &mut Vec<GraphId>,
    ) -> Attribute {
        let ty = if a.r#type != 0 {
            a.r#type
        } else {
            infer_attr_type(&a)
        };
        let opsets = graph.opsets.clone();
        let value = match ty {
            attr_type::FLOAT => AttrValue::Float(a.f),
            attr_type::INT => AttrValue::Int(a.i),
            attr_type::STRING => AttrValue::String(a.s),
            attr_type::TENSOR => match a.t {
                Some(t) => {
                    let idx = graph.tensors.len();
                    graph.tensors.push(t);
                    AttrValue::Tensor(idx)
                }
                None => AttrValue::Undefined,
            },
            attr_type::GRAPH => match a.g {
                Some(g) => {
                    let owner = GraphOwner::Subgraph {
                        parent: gid,
                        node: ni,
                        attribute: a.name.clone(),
                    };
                    let sid = self.build_graph(g, owner, depth + 1, &opsets);
                    subgraphs.push(sid);
                    AttrValue::Graph(sid)
                }
                None => AttrValue::Undefined,
            },
            attr_type::SPARSE_TENSOR => match &a.sparse_tensor {
                Some(s) => AttrValue::SparseTensor(sparse_info(s)),
                None => AttrValue::Undefined,
            },
            attr_type::TYPE_PROTO => AttrValue::Type(type_info(a.tp.as_ref())),
            attr_type::FLOATS => AttrValue::Floats(a.floats),
            attr_type::INTS => AttrValue::Ints(a.ints),
            attr_type::STRINGS => AttrValue::Strings(a.strings),
            attr_type::TENSORS => {
                let mut ids = Vec::with_capacity(a.tensors.len());
                for t in a.tensors {
                    ids.push(graph.tensors.len());
                    graph.tensors.push(t);
                }
                AttrValue::Tensors(ids)
            }
            attr_type::GRAPHS => {
                let mut ids = Vec::with_capacity(a.graphs.len());
                for (k, g) in a.graphs.into_iter().enumerate() {
                    let owner = GraphOwner::Subgraph {
                        parent: gid,
                        node: ni,
                        attribute: format!("{}[{}]", a.name, k),
                    };
                    let sid = self.build_graph(g, owner, depth + 1, &opsets);
                    subgraphs.push(sid);
                    ids.push(sid);
                }
                AttrValue::Graphs(ids)
            }
            attr_type::SPARSE_TENSORS => {
                AttrValue::SparseTensors(a.sparse_tensors.iter().map(sparse_info).collect())
            }
            attr_type::TYPE_PROTOS => {
                AttrValue::Types(a.type_protos.iter().map(|t| type_info(Some(t))).collect())
            }
            _ => AttrValue::Undefined,
        };
        Attribute {
            name: a.name,
            value,
            doc: a.doc_string,
            ref_attr_name: a.ref_attr_name,
        }
    }
}

/// Attach the payload of `Constant` nodes to their output values.
fn mark_constants(graph: &mut Graph) {
    for ni in 0..graph.nodes.len() {
        let node = &graph.nodes[ni];
        if node.op_type != "Constant" || !node.is_standard() {
            continue;
        }
        let Some(out) = node.output(0) else { continue };
        let mut tensor: Option<proto::TensorProto> = None;
        let mut existing: Option<usize> = None;
        for a in &node.attributes {
            match (a.name.as_str(), &a.value) {
                ("value", AttrValue::Tensor(idx)) => existing = Some(*idx),
                ("value_float", AttrValue::Float(f)) => {
                    tensor = Some(proto::TensorProto {
                        data_type: dtype::FLOAT,
                        float_data: vec![*f],
                        ..Default::default()
                    })
                }
                ("value_floats", AttrValue::Floats(f)) => {
                    tensor = Some(proto::TensorProto {
                        data_type: dtype::FLOAT,
                        dims: vec![f.len() as i64],
                        float_data: f.clone(),
                        ..Default::default()
                    })
                }
                ("value_int", AttrValue::Int(i)) => {
                    tensor = Some(proto::TensorProto {
                        data_type: dtype::INT64,
                        int64_data: vec![*i],
                        ..Default::default()
                    })
                }
                ("value_ints", AttrValue::Ints(i)) => {
                    tensor = Some(proto::TensorProto {
                        data_type: dtype::INT64,
                        dims: vec![i.len() as i64],
                        int64_data: i.clone(),
                        ..Default::default()
                    })
                }
                ("value_string", AttrValue::String(s)) => {
                    tensor = Some(proto::TensorProto {
                        data_type: dtype::STRING,
                        string_data: vec![s.clone()],
                        ..Default::default()
                    })
                }
                ("value_strings", AttrValue::Strings(s)) => {
                    tensor = Some(proto::TensorProto {
                        data_type: dtype::STRING,
                        dims: vec![s.len() as i64],
                        string_data: s.clone(),
                        ..Default::default()
                    })
                }
                _ => {}
            }
        }
        let idx = match (existing, tensor) {
            (Some(idx), _) => idx,
            (None, Some(t)) => {
                graph.tensors.push(t);
                graph.tensors.len() - 1
            }
            (None, None) => continue,
        };
        let t = &graph.tensors[idx];
        let decl = TypeInfo::tensor(t.data_type, Some(shape::from_i64s(&t.dims)));
        let v = &mut graph.values[out];
        v.kind = ValueKind::Constant;
        v.tensor = Some(idx);
        if v.declared.is_none() {
            v.declared = Some(decl);
        }
    }
}

/// Stable topological sort (Kahn's algorithm with a min-heap on node index).
fn topo_sort(graph: &mut Graph) {
    let n = graph.nodes.len();
    let mut indeg = vec![0usize; n];
    let mut succ: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for (ni, node) in graph.nodes.iter().enumerate() {
        let mut deps: HashSet<NodeId> = HashSet::new();
        for vid in node
            .inputs
            .iter()
            .flatten()
            .chain(node.implicit_inputs.iter())
        {
            if let Some((p, _)) = graph.values[*vid].producer {
                if p != ni {
                    deps.insert(p);
                }
            }
        }
        indeg[ni] = deps.len();
        for p in deps {
            succ[p].push(ni);
        }
    }
    let mut heap: BinaryHeap<std::cmp::Reverse<NodeId>> = (0..n)
        .filter(|&i| indeg[i] == 0)
        .map(std::cmp::Reverse)
        .collect();
    let mut order = Vec::with_capacity(n);
    while let Some(std::cmp::Reverse(u)) = heap.pop() {
        order.push(u);
        for &v in &succ[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                heap.push(std::cmp::Reverse(v));
            }
        }
    }
    if order.len() < n {
        graph.has_cycle = true;
        let mut seen = vec![false; n];
        for &u in &order {
            seen[u] = true;
        }
        order.extend((0..n).filter(|&i| !seen[i]));
    }
    // The file order is valid if every node only depends on earlier nodes.
    let mut sorted = true;
    'outer: for (ni, node) in graph.nodes.iter().enumerate() {
        for vid in node.inputs.iter().flatten() {
            if let Some((p, _)) = graph.values[*vid].producer {
                if p >= ni {
                    sorted = false;
                    break 'outer;
                }
            }
        }
    }
    graph.sorted = sorted;
    graph.topo = order;
}

/// Hierarchical module path derived from a node name such as
/// `/encoder/layer.0/attention/MatMul` → `encoder/layer.0/attention`.
pub fn module_path(name: &str) -> String {
    if !name.contains('/') {
        return String::new();
    }
    let parts: Vec<&str> = name.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() <= 1 {
        return String::new();
    }
    parts[..parts.len() - 1].join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_paths() {
        assert_eq!(module_path("/encoder/layer.0/attention/MatMul"), "encoder/layer.0/attention");
        assert_eq!(module_path("/Conv"), "");
        assert_eq!(module_path("Conv_12"), "");
        assert_eq!(module_path("model/conv2d/BiasAdd"), "model/conv2d");
    }
}
