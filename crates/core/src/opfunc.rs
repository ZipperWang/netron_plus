//! Operator function definitions: what an operator computes, written in TeX,
//! with the symbols of each formula bound to a node's attributes, shapes and
//! constant inputs.
//!
//! Definitions live in `opdefs/*.md` and are compiled into the binary. Each
//! file is a list of entries, one per operator (and opset range):
//!
//! - `## Op` starts an entry; `## domain::Op` for non-standard domains.
//! - `since: N` is the first opset the entry describes (default 1). An
//!   operator may have several entries; a node uses the newest one whose
//!   `since` is not above the node's opset.
//! - `en:` / `zh:` give a one-line summary, `note.en:` / `note.zh:` a note.
//! - A fenced block opened with three backticks and `tex` holds a formula;
//!   text after `tex` on the opening line is its label (`shape` marks the
//!   output shape rule).
//! - `sym: <tex> = <source>, <source> ?= <default>` binds a symbol. Sources
//!   are tried in order: `attr:<name>`, `dtype:<attr>` (an int attribute
//!   holding a data type), or `in:<slot>.<prop>` / `out:<slot>.<prop>` where
//!   `<prop>` is `shape`, `shape[i]`, `shape[a:b]`, `rank`, `dtype` or
//!   `value` (contents of small constant or shape-derived tensors). The
//!   optional default is TeX shown when no source is available.

use crate::analysis::Analysis;
use crate::api::SubgraphRef;
use crate::dtype;
use crate::format;
use crate::infer::VState;
use crate::model::{AttrValue, Graph, GraphId, Model, Node};
use crate::ops;
use crate::tensor;
use crate::tex;
use anyhow::{anyhow, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::OnceLock;

static SOURCES: &[(&str, &str)] = &[
    ("activation.md", include_str!("opdefs/activation.md")),
    ("math.md", include_str!("opdefs/math.md")),
    ("layer.md", include_str!("opdefs/layer.md")),
    ("normalization.md", include_str!("opdefs/normalization.md")),
    ("pool.md", include_str!("opdefs/pool.md")),
    ("reduce.md", include_str!("opdefs/reduce.md")),
    ("tensor.md", include_str!("opdefs/tensor.md")),
    ("quantization.md", include_str!("opdefs/quantization.md")),
    ("contrib.md", include_str!("opdefs/contrib.md")),
];

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TexBlock {
    pub label: String,
    pub tex: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Prop {
    Shape,
    Dim(i64),
    Dims(i64, Option<i64>),
    Rank,
    Dtype,
    Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Source {
    Attr(String),
    AttrDtype(String),
    Slot { side: Side, slot: String, prop: Prop },
}

impl Source {
    fn label(&self) -> String {
        match self {
            Source::Attr(n) | Source::AttrDtype(n) => n.clone(),
            Source::Slot { slot, prop, .. } => match prop {
                Prop::Shape => format!("{}.shape", slot),
                Prop::Dim(i) => format!("{}.shape[{}]", slot, i),
                Prop::Dims(a, None) => format!("{}.shape[{}:]", slot, a),
                Prop::Dims(a, Some(b)) => format!("{}.shape[{}:{}]", slot, a, b),
                Prop::Rank => format!("rank({})", slot),
                Prop::Dtype => format!("{}.dtype", slot),
                Prop::Value => slot.clone(),
            },
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Source::Attr(_) | Source::AttrDtype(_) => "attr",
            Source::Slot { side: Side::Input, .. } => "input",
            Source::Slot { side: Side::Output, .. } => "output",
        }
    }
}

#[derive(Clone, Debug)]
struct SymbolDef {
    tex: String,
    sources: Vec<Source>,
    default: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct OpDef {
    pub domain: String,
    pub op: String,
    pub since: i64,
    pub en: String,
    pub zh: String,
    pub note_en: String,
    pub note_zh: String,
    pub formulas: Vec<TexBlock>,
    symbols: Vec<SymbolDef>,
    /// `file:line` of the entry, for error messages.
    pub origin: String,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Localized {
    pub en: String,
    pub zh: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolValue {
    pub tex: String,
    /// What the symbol is bound to, e.g. `alpha` or `X.shape[1]`.
    pub binding: String,
    /// "attr" | "input" | "output"
    pub kind: &'static str,
    /// TeX for the value; None when it is not known.
    pub value: Option<String>,
    /// "set" (attribute present in the model), "default", "derived" (from
    /// shapes or tensor contents) or "unknown".
    pub source: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OpFunction {
    pub op: String,
    pub domain: String,
    pub opset: Option<i64>,
    /// First opset of the definition used; None when there is none.
    pub since: Option<i64>,
    pub summary: Localized,
    pub note: Localized,
    pub formulas: Vec<TexBlock>,
    pub symbols: Vec<SymbolValue>,
    pub doc_url: Option<String>,
    /// Body of the model-local function this operator refers to.
    pub function: Option<SubgraphRef>,
    /// Node the symbols are bound to; None for the generic definition.
    pub node: Option<usize>,
}

// ------------------------------------------------------------------ parsing

fn norm_domain(d: &str) -> &str {
    if ops::is_standard_domain(d) {
        ""
    } else {
        d
    }
}

fn name(s: &str) -> Result<String, String> {
    let s = s.trim();
    if s.is_empty() || s.contains(char::is_whitespace) {
        return Err(format!("invalid name '{}'", s));
    }
    Ok(s.to_string())
}

fn parse_index(s: &str, all: &str) -> Result<i64, String> {
    s.trim().parse().map_err(|_| format!("bad index in '{}'", all))
}

fn parse_source(s: &str) -> Result<Source, String> {
    let s = s.trim();
    if let Some(n) = s.strip_prefix("attr:") {
        return name(n).map(Source::Attr);
    }
    if let Some(n) = s.strip_prefix("dtype:") {
        return name(n).map(Source::AttrDtype);
    }
    let (side, rest) = if let Some(r) = s.strip_prefix("in:") {
        (Side::Input, r)
    } else if let Some(r) = s.strip_prefix("out:") {
        (Side::Output, r)
    } else {
        return Err(format!("unknown source '{}'", s));
    };
    let (slot, prop) = rest
        .split_once('.')
        .ok_or_else(|| format!("missing property in '{}'", s))?;
    let prop = match prop {
        "shape" => Prop::Shape,
        "rank" => Prop::Rank,
        "dtype" => Prop::Dtype,
        "value" => Prop::Value,
        p => {
            let inner = p
                .strip_prefix("shape[")
                .and_then(|x| x.strip_suffix(']'))
                .ok_or_else(|| format!("unknown property in '{}'", s))?;
            match inner.split_once(':') {
                Some((a, b)) => Prop::Dims(
                    if a.trim().is_empty() { 0 } else { parse_index(a, s)? },
                    if b.trim().is_empty() { None } else { Some(parse_index(b, s)?) },
                ),
                None => Prop::Dim(parse_index(inner, s)?),
            }
        }
    };
    Ok(Source::Slot {
        side,
        slot: name(slot)?,
        prop,
    })
}

fn parse_symbol(s: &str) -> Result<SymbolDef, String> {
    let (main, default) = match s.split_once("?=") {
        Some((m, d)) => (m.trim(), Some(d.trim().to_string()).filter(|d| !d.is_empty())),
        None => (s.trim(), None),
    };
    let (tex, sources) = main
        .rsplit_once(" = ")
        .ok_or_else(|| format!("symbol without a source: '{}'", s))?;
    let sources = sources
        .split(',')
        .map(parse_source)
        .collect::<Result<Vec<_>, _>>()?;
    let tex = tex.trim();
    if tex.is_empty() {
        return Err(format!("symbol without TeX: '{}'", s));
    }
    Ok(SymbolDef {
        tex: tex.to_string(),
        sources,
        default,
    })
}

fn parse(file: &str, text: &str, out: &mut Vec<OpDef>, errors: &mut Vec<String>) {
    let mut cur: Option<OpDef> = None;
    let mut block: Option<(String, Vec<String>)> = None;
    for (i, raw) in text.lines().enumerate() {
        let at = format!("{}:{}", file, i + 1);
        let t = raw.trim();
        if let Some(b) = block.as_mut() {
            if t == "```" {
                let (label, lines) = block.take().unwrap_or_default();
                let tex = lines.join("\n").trim().to_string();
                match cur.as_mut() {
                    Some(d) if !tex.is_empty() => d.formulas.push(TexBlock { label, tex }),
                    Some(_) => errors.push(format!("{}: empty formula", at)),
                    None => errors.push(format!("{}: formula outside an entry", at)),
                }
            } else {
                b.1.push(raw.trim_end().to_string());
            }
            continue;
        }
        if t.is_empty() || t.starts_with("<!--") || t.starts_with("# ") {
            continue;
        }
        if let Some(h) = t.strip_prefix("## ") {
            out.extend(cur.take());
            let h = h.trim();
            let (domain, op) = match h.split_once("::") {
                Some((d, o)) => (d.trim(), o.trim()),
                None => ("", h),
            };
            cur = Some(OpDef {
                domain: norm_domain(domain).to_string(),
                op: op.to_string(),
                since: 1,
                origin: at,
                ..Default::default()
            });
            continue;
        }
        if let Some(label) = t.strip_prefix("```tex") {
            block = Some((label.trim().to_string(), Vec::new()));
            continue;
        }
        let Some(d) = cur.as_mut() else {
            errors.push(format!("{}: text outside an entry", at));
            continue;
        };
        let Some((key, value)) = t.split_once(':') else {
            errors.push(format!("{}: expected 'key: value'", at));
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "since" => match value.parse() {
                Ok(v) => d.since = v,
                Err(_) => errors.push(format!("{}: bad opset '{}'", at, value)),
            },
            "en" => d.en = value.to_string(),
            "zh" => d.zh = value.to_string(),
            "note.en" => d.note_en = value.to_string(),
            "note.zh" => d.note_zh = value.to_string(),
            "sym" => match parse_symbol(value) {
                Ok(s) => d.symbols.push(s),
                Err(e) => errors.push(format!("{}: {}", at, e)),
            },
            k => errors.push(format!("{}: unknown key '{}'", at, k)),
        }
    }
    if block.is_some() {
        errors.push(format!("{}: unterminated formula", file));
    }
    out.extend(cur.take());
}

type Index = HashMap<(String, String), Vec<OpDef>>;

struct Library {
    index: Index,
    errors: Vec<String>,
}

fn library() -> &'static Library {
    static LIB: OnceLock<Library> = OnceLock::new();
    LIB.get_or_init(|| {
        let mut defs = Vec::new();
        let mut errors = Vec::new();
        for (file, text) in SOURCES {
            parse(file, text, &mut defs, &mut errors);
        }
        let mut index: Index = HashMap::new();
        for d in defs {
            index.entry((d.domain.clone(), d.op.clone())).or_default().push(d);
        }
        for list in index.values_mut() {
            list.sort_by_key(|d| d.since);
        }
        Library { index, errors }
    })
}

/// Problems found in the definition files (empty when they are valid).
pub fn errors() -> &'static [String] {
    &library().errors
}

/// Every definition, in no particular order.
pub fn all() -> impl Iterator<Item = &'static OpDef> {
    library().index.values().flatten()
}

/// The entry describing `opset`, or the newest one when the opset is unknown.
fn pick(list: &[OpDef], opset: Option<i64>) -> Option<&OpDef> {
    match opset {
        Some(v) => list.iter().rev().find(|d| d.since <= v).or(list.first()),
        None => list.last(),
    }
}

pub fn definition(domain: &str, op: &str, opset: Option<i64>) -> Option<&'static OpDef> {
    let index = &library().index;
    let d = norm_domain(domain);
    if let Some(list) = index.get(&(d.to_string(), op.to_string())) {
        return pick(list, opset);
    }
    // ORT contrib ops reuse many standard operator definitions.
    if d == "com.microsoft" {
        if let Some(list) = index.get(&(String::new(), op.to_string())) {
            return pick(list, None);
        }
    }
    None
}

/// Every TeX snippet the definitions can produce without a model, as
/// `(origin, tex, display)`; symbol rows are rendered as `sym = default`.
pub fn snippets() -> Vec<(String, String, bool)> {
    let mut out = Vec::new();
    for d in all() {
        for f in &d.formulas {
            out.push((d.origin.clone(), f.tex.clone(), true));
        }
        for s in &d.symbols {
            out.push((d.origin.clone(), s.tex.clone(), false));
            if let Some(v) = &s.default {
                out.push((d.origin.clone(), format!("{} = {}", s.tex, v), false));
            }
        }
    }
    out
}

// ------------------------------------------------------------------ binding

fn function_ref(model: &Model, gid: GraphId) -> SubgraphRef {
    let g = &model.graphs[gid];
    SubgraphRef {
        id: gid,
        label: g.label(&model.graphs),
        nodes: g.nodes.len(),
    }
}

fn build(
    def: Option<&OpDef>,
    domain: &str,
    op: &str,
    opset: Option<i64>,
    mut bind: impl FnMut(&SymbolDef) -> SymbolValue,
) -> OpFunction {
    OpFunction {
        op: op.to_string(),
        domain: domain.to_string(),
        opset,
        since: def.map(|d| d.since),
        summary: def
            .map(|d| Localized {
                en: d.en.clone(),
                zh: d.zh.clone(),
            })
            .unwrap_or_default(),
        note: def
            .map(|d| Localized {
                en: d.note_en.clone(),
                zh: d.note_zh.clone(),
            })
            .unwrap_or_default(),
        formulas: def.map(|d| d.formulas.clone()).unwrap_or_default(),
        symbols: def
            .map(|d| d.symbols.iter().map(&mut bind).collect::<Vec<_>>())
            .unwrap_or_default(),
        doc_url: ops::doc_url(domain, op),
        function: None,
        node: None,
    }
}

fn unbound(s: &SymbolDef) -> SymbolValue {
    let first = &s.sources[0];
    SymbolValue {
        tex: s.tex.clone(),
        binding: first.label(),
        kind: first.kind(),
        value: s.default.clone(),
        source: if s.default.is_some() { "default" } else { "unknown" },
    }
}

/// The generic definition of an operator type (no node).
pub fn describe_op(model: &Model, domain: &str, op: &str) -> OpFunction {
    let opset = model.opset(domain);
    let mut out = build(definition(domain, op, opset), domain, op, opset, unbound);
    out.function = model.function(domain, op).map(|f| function_ref(model, f));
    out
}

/// The definition of a node's operator with its symbols bound to the node.
pub fn describe_node(model: &Model, an: &Analysis, gid: GraphId, ni: usize) -> Result<OpFunction> {
    let g = model.graph(gid)?;
    let node = g
        .nodes
        .get(ni)
        .ok_or_else(|| anyhow!("node {} does not exist", ni))?;
    let opset = g.opset(&node.domain);
    let ctx = Binder {
        model,
        g,
        states: &an.shapes[gid].states,
        node,
    };
    let def = definition(&node.domain, &node.op_type, opset);
    let mut out = build(def, &node.domain, &node.op_type, opset, |s| ctx.symbol(s));
    out.function = model.function_of(node).map(|f| function_ref(model, f));
    out.node = Some(ni);
    Ok(out)
}

struct Binder<'a> {
    model: &'a Model,
    g: &'a Graph,
    states: &'a [VState],
    node: &'a Node,
}

fn index_of(i: i64, len: usize) -> Option<usize> {
    let j = if i < 0 { len as i64 + i } else { i };
    (0..len as i64).contains(&j).then_some(j as usize)
}

fn bound_of(i: i64, len: usize) -> usize {
    let j = if i < 0 { len as i64 + i } else { i };
    j.clamp(0, len as i64) as usize
}

fn value_tex(st: &VState) -> Option<String> {
    let items: Vec<String> = if let Some(v) = &st.ints {
        v.iter().map(tex::dim).collect()
    } else if let Some(v) = &st.floats {
        v.iter().map(|&x| tex::num(x)).collect()
    } else {
        return None;
    };
    let scalar = st.shape.as_ref().map_or(false, |s| s.is_empty());
    if scalar && items.len() == 1 {
        return items.into_iter().next();
    }
    Some(tex::tuple(items))
}

fn attr_tex(model: &Model, g: &Graph, v: &AttrValue) -> Option<String> {
    Some(match v {
        AttrValue::Undefined => return None,
        AttrValue::Int(i) => i.to_string(),
        AttrValue::Float(f) => tex::num(*f as f64),
        AttrValue::Ints(v) => tex::tuple(v.iter().map(|i| i.to_string())),
        AttrValue::Floats(v) => tex::tuple(v.iter().map(|&f| tex::num(f as f64))),
        AttrValue::String(s) => tex::texttt(&String::from_utf8_lossy(s)),
        AttrValue::Strings(v) => tex::tuple(v.iter().map(|s| tex::texttt(&String::from_utf8_lossy(s)))),
        AttrValue::Tensor(ti) => {
            let t = &g.tensors[*ti];
            match tensor::small_floats(t, 8) {
                Some(f) if t.dims.is_empty() && f.len() == 1 => tex::num(f[0]),
                Some(f) => tex::tuple(f.iter().map(|&x| tex::num(x))),
                None => tex::texttt(&format::tensor_text(t)),
            }
        }
        other => tex::texttt(&format::attr_full(model, g, other)),
    })
}

/// Result of evaluating one symbol source against a node.
enum Eval {
    Found(String, &'static str),
    /// The attribute or slot is there but its value is not known.
    Unknown,
    /// The attribute or slot is absent.
    Absent,
}

fn slot_tex(st: &VState, prop: &Prop) -> Option<String> {
    Some(match prop {
        Prop::Shape => tex::shape(st.shape.as_deref()?),
        Prop::Dim(i) => {
            let s = st.shape.as_deref()?;
            tex::dim(&s[index_of(*i, s.len())?])
        }
        Prop::Dims(a, b) => {
            let s = st.shape.as_deref()?;
            let a = bound_of(*a, s.len());
            let b = b.map_or(s.len(), |b| bound_of(b, s.len())).max(a);
            tex::shape(&s[a..b])
        }
        Prop::Rank => st.shape.as_ref()?.len().to_string(),
        Prop::Dtype => {
            let n = dtype::name(st.dtype);
            if n.is_empty() {
                return None;
            }
            tex::texttt(n)
        }
        Prop::Value => value_tex(st)?,
    })
}

impl Binder<'_> {
    fn symbol(&self, s: &SymbolDef) -> SymbolValue {
        let mut unknown: Option<&Source> = None;
        for src in &s.sources {
            match self.eval(src) {
                Eval::Found(value, source) => {
                    return SymbolValue {
                        tex: s.tex.clone(),
                        binding: src.label(),
                        kind: src.kind(),
                        value: Some(value),
                        source,
                    }
                }
                Eval::Unknown => {
                    unknown.get_or_insert(src);
                }
                Eval::Absent => {}
            }
        }
        match unknown {
            // Present in the model but not computable: the default would be wrong.
            Some(src) => SymbolValue {
                tex: s.tex.clone(),
                binding: src.label(),
                kind: src.kind(),
                value: None,
                source: "unknown",
            },
            None => unbound(s),
        }
    }

    fn slot_value(&self, side: Side, slot: &str) -> Option<usize> {
        let n = self.node;
        let list = match side {
            Side::Input => &n.inputs,
            Side::Output => &n.outputs,
        };
        let idx = match slot.parse::<usize>() {
            Ok(i) => i,
            Err(_) => (0..list.len()).find(|&i| {
                let label = match side {
                    Side::Input => ops::input_name(&n.domain, &n.op_type, i),
                    Side::Output => ops::output_name(&n.domain, &n.op_type, i),
                };
                label == slot
            })?,
        };
        list.get(idx).copied().flatten()
    }

    fn eval(&self, src: &Source) -> Eval {
        let found = |v: Option<String>, source: &'static str| match v {
            Some(v) => Eval::Found(v, source),
            None => Eval::Unknown,
        };
        match src {
            Source::Attr(name) => match self.node.attr(name) {
                Some(v) => found(attr_tex(self.model, self.g, v), "set"),
                None => Eval::Absent,
            },
            Source::AttrDtype(name) => match self.node.attr(name) {
                Some(AttrValue::Int(v)) => {
                    let n = dtype::name(*v as i32);
                    found((!n.is_empty()).then(|| tex::texttt(n)), "set")
                }
                Some(_) => Eval::Unknown,
                None => Eval::Absent,
            },
            Source::Slot { side, slot, prop } => {
                match self.slot_value(*side, slot).and_then(|v| self.states.get(v)) {
                    Some(st) => found(slot_tex(st, prop), "derived"),
                    None => Eval::Absent,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_slot(names: &[&str], slot: &str) -> bool {
        slot.parse::<usize>().is_ok()
            || names
                .iter()
                .any(|n| *n == slot || n.strip_suffix('*') == Some(slot))
    }

    #[test]
    fn definition_files_parse() {
        assert!(errors().is_empty(), "definition errors:\n{}", errors().join("\n"));
        assert!(all().count() > 80, "only {} definitions", all().count());
    }

    #[test]
    fn definitions_are_consistent() {
        let mut problems = Vec::new();
        for d in all() {
            let Some(schema) = ops::schema(&d.domain, &d.op) else {
                problems.push(format!("{}: unknown operator {}::{}", d.origin, d.domain, d.op));
                continue;
            };
            if d.en.is_empty() || d.zh.is_empty() {
                problems.push(format!("{}: {} needs en and zh summaries", d.origin, d.op));
            }
            if d.formulas.is_empty() {
                problems.push(format!("{}: {} has no formula", d.origin, d.op));
            }
            for f in &d.formulas {
                if !tex::balanced(&f.tex) {
                    problems.push(format!("{}: unbalanced braces in {}", d.origin, d.op));
                }
            }
            for s in &d.symbols {
                if !tex::balanced(&s.tex) || !s.default.as_deref().map_or(true, tex::balanced) {
                    problems.push(format!("{}: unbalanced braces in symbol {}", d.origin, s.tex));
                }
                for src in &s.sources {
                    if let Source::Slot { side, slot, .. } = src {
                        let names = if *side == Side::Input { schema.inputs } else { schema.outputs };
                        if !has_slot(names, slot) {
                            problems.push(format!("{}: {} has no slot '{}'", d.origin, d.op, slot));
                        }
                    }
                }
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    }

    #[test]
    fn common_operators_are_covered() {
        let required = [
            "Conv", "ConvTranspose", "Gemm", "MatMul", "Relu", "LeakyRelu", "Sigmoid", "HardSigmoid",
            "HardSwish", "Tanh", "Gelu", "Softmax", "Clip", "Add", "Mul", "Div", "Pow", "Sqrt",
            "BatchNormalization", "LayerNormalization", "InstanceNormalization", "GroupNormalization",
            "MaxPool", "AveragePool", "GlobalAveragePool", "ReduceMean", "ReduceSum", "Reshape",
            "Transpose", "Concat", "Split", "Slice", "Gather", "Squeeze", "Unsqueeze", "Flatten",
            "Resize", "Pad", "Cast", "Where", "TopK", "QuantizeLinear", "DequantizeLinear",
        ];
        let missing: Vec<&str> = required
            .iter()
            .copied()
            .filter(|op| definition("", op, None).is_none())
            .collect();
        assert!(missing.is_empty(), "missing definitions: {:?}", missing);
    }

    #[test]
    fn picks_entry_for_opset() {
        assert_eq!(definition("", "Softmax", Some(11)).unwrap().since, 1);
        assert_eq!(definition("", "Softmax", Some(13)).unwrap().since, 13);
        assert_eq!(definition("", "Softmax", None).unwrap().since, 13);
        assert_eq!(definition("ai.onnx", "Clip", Some(9)).unwrap().since, 6);
        assert_eq!(definition("", "Clip", Some(17)).unwrap().since, 11);
        // Contrib ops fall back to the standard definition.
        assert_eq!(definition("com.microsoft", "Relu", Some(1)).unwrap().op, "Relu");
        assert!(definition("my.domain", "Relu", None).is_none());
    }

    #[test]
    fn parses_symbols() {
        let s = parse_symbol(r"\mathbf{K} = attr:kernel_shape, in:W.shape[2:] ?= (1, \ldots)").unwrap();
        assert_eq!(s.tex, r"\mathbf{K}");
        assert_eq!(s.default.as_deref(), Some(r"(1, \ldots)"));
        assert_eq!(s.sources[0], Source::Attr("kernel_shape".into()));
        assert_eq!(
            s.sources[1],
            Source::Slot {
                side: Side::Input,
                slot: "W".into(),
                prop: Prop::Dims(2, None)
            }
        );
        assert_eq!(s.sources[1].label(), "W.shape[2:]");
        let d = parse_symbol("C = in:X.shape[-1]").unwrap();
        assert_eq!(d.sources[0], Source::Slot { side: Side::Input, slot: "X".into(), prop: Prop::Dim(-1) });
        assert!(parse_symbol(r"\alpha").is_err());
        assert!(parse_symbol(r"\alpha = attr:").is_err());
        assert!(parse_symbol(r"\alpha = in:X.size").is_err());
    }
}
