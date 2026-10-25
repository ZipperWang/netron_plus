//! Shape and type inference for the common ONNX operators.
//!
//! Besides dtypes and shapes, small integer tensors carry their contents
//! (possibly symbolic) so that the usual exporter patterns such as
//! `Shape → Gather → Unsqueeze → Concat → Reshape` resolve to concrete shapes.
//! Declared `value_info` always wins over inferred information, and user
//! supplied dimension overrides are applied to declared symbolic dimensions.

use crate::dtype;
use crate::model::{AttrValue, Graph, Model, Node, TypeInfo, TypeKind, ValueKind};
use crate::shape::{self, norm_axis, Dim, Shape};
use crate::tensor;
use serde::Serialize;
use std::collections::HashMap;

/// Contents are only tracked for tensors up to this many elements.
const MAX_TRACKED: usize = 64;
/// Slice ends at or beyond this value mean "until the end".
const LARGE_END: i64 = (1i64 << 31) - 1;

#[derive(Clone, Debug, Default)]
pub struct VState {
    pub dtype: i32,
    pub shape: Option<Shape>,
    /// Contents of small integer tensors (known or symbolic).
    pub ints: Option<Vec<Dim>>,
    /// Contents of small float tensors.
    pub floats: Option<Vec<f64>>,
}

impl VState {
    fn new(dtype: i32, shape: Option<Shape>) -> VState {
        VState {
            dtype,
            shape,
            ints: None,
            floats: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ShapeSource {
    #[default]
    Unknown,
    Declared,
    Inferred,
    Tensor,
}

#[derive(Clone, Debug, Default)]
pub struct Overrides {
    /// Symbolic dimension name → value.
    pub params: HashMap<String, i64>,
    /// Main graph input name → per-axis override.
    pub inputs: HashMap<String, Vec<Option<i64>>>,
}

#[derive(Clone, Debug, Default)]
pub struct GraphShapes {
    pub states: Vec<VState>,
    pub sources: Vec<ShapeSource>,
}

pub fn infer(model: &Model, ov: &Overrides) -> Vec<GraphShapes> {
    let mut out: Vec<GraphShapes> = Vec::with_capacity(model.graphs.len());
    for gid in 0..model.graphs.len() {
        let gs = infer_graph(model, gid, ov, &out);
        out.push(gs);
    }
    out
}

fn apply_overrides(shape: &[Dim], ov: &Overrides, per_input: Option<&Vec<Option<i64>>>) -> Shape {
    shape
        .iter()
        .enumerate()
        .map(|(i, d)| {
            if let Some(Some(v)) = per_input.and_then(|p| p.get(i)) {
                return Dim::Known(*v);
            }
            match d {
                Dim::Param(p) => ov
                    .params
                    .get(p)
                    .map(|v| Dim::Known(*v))
                    .unwrap_or_else(|| d.clone()),
                _ => d.clone(),
            }
        })
        .collect()
}

fn from_decl(d: &TypeInfo, ov: &Overrides, per_input: Option<&Vec<Option<i64>>>) -> VState {
    let shape = if d.kind == TypeKind::Tensor || d.kind == TypeKind::SparseTensor {
        d.shape.as_ref().map(|s| apply_overrides(s, ov, per_input))
    } else {
        None
    };
    VState::new(d.dtype, shape)
}

fn merge_dim(a: Dim, b: Dim) -> Dim {
    let take_b = matches!(b, Dim::Known(_))
        || (!matches!(a, Dim::Known(_)) && (matches!(b, Dim::Param(_)) || !matches!(a, Dim::Param(_))));
    if take_b {
        b
    } else {
        a
    }
}

/// Combine an inferred state with the declared type (declared wins).
fn merge(inf: VState, decl: Option<VState>) -> (VState, ShapeSource) {
    let inferred_source = if inf.shape.is_some() {
        ShapeSource::Inferred
    } else {
        ShapeSource::Unknown
    };
    let d = match decl {
        Some(d) if d.shape.is_some() || d.dtype != 0 => d,
        _ => return (inf, inferred_source),
    };
    let dtype = if d.dtype != 0 { d.dtype } else { inf.dtype };
    let source = if d.shape.is_some() {
        ShapeSource::Declared
    } else {
        inferred_source
    };
    let shape = match (inf.shape, d.shape) {
        (Some(i), Some(ds)) if i.len() == ds.len() => {
            Some(i.into_iter().zip(ds).map(|(a, b)| merge_dim(a, b)).collect())
        }
        (_, Some(ds)) => Some(ds),
        (i, None) => i,
    };
    (
        VState {
            dtype,
            shape,
            ints: inf.ints,
            floats: inf.floats,
        },
        source,
    )
}

fn infer_graph(model: &Model, gid: usize, ov: &Overrides, done: &[GraphShapes]) -> GraphShapes {
    let g = &model.graphs[gid];
    let n = g.values.len();
    let mut st: Vec<VState> = vec![VState::default(); n];
    let mut src = vec![ShapeSource::Unknown; n];
    let is_main = gid == 0;

    for (vid, v) in g.values.iter().enumerate() {
        match v.kind {
            ValueKind::Initializer | ValueKind::Constant => {
                if let Some(ti) = v.tensor {
                    let t = &g.tensors[ti];
                    st[vid] = VState {
                        dtype: t.data_type,
                        shape: Some(shape::from_i64s(&t.dims)),
                        ints: tensor::small_ints(t, MAX_TRACKED)
                            .map(|x| x.into_iter().map(Dim::Known).collect()),
                        floats: tensor::small_floats(t, MAX_TRACKED),
                    };
                    src[vid] = ShapeSource::Tensor;
                } else if let Some(d) = &v.declared {
                    st[vid] = from_decl(d, ov, None);
                    src[vid] = ShapeSource::Declared;
                }
            }
            ValueKind::Input => {
                if let Some(d) = &v.declared {
                    let per_input = if is_main { ov.inputs.get(&v.name) } else { None };
                    st[vid] = from_decl(d, ov, per_input);
                    src[vid] = if st[vid].shape.is_some() {
                        ShapeSource::Declared
                    } else {
                        ShapeSource::Unknown
                    };
                }
            }
            ValueKind::OuterScope => {
                if let Some((pg, pv)) = v.outer {
                    if let Some(ps) = done.get(pg) {
                        if let Some(s) = ps.states.get(pv) {
                            st[vid] = s.clone();
                            src[vid] = ps.sources[pv];
                        }
                    }
                }
                if src[vid] == ShapeSource::Unknown {
                    if let Some(d) = &v.declared {
                        st[vid] = from_decl(d, ov, None);
                        src[vid] = ShapeSource::Declared;
                    }
                }
            }
            _ => {}
        }
    }

    for &ni in &g.topo {
        let node = &g.nodes[ni];
        if node.op_type == "Constant" && node.is_standard() {
            if let Some(o) = node.output(0) {
                if src[o] == ShapeSource::Tensor {
                    continue;
                }
            }
        }
        let results = {
            let ins: Vec<Option<&VState>> = node.inputs.iter().map(|i| i.map(|v| &st[v])).collect();
            let ctx = Ctx { model, graph: g, ov };
            infer_node(&ctx, node, &ins)
        };
        for (slot, out) in node.outputs.iter().enumerate() {
            let Some(vid) = *out else { continue };
            if src[vid] == ShapeSource::Tensor {
                continue;
            }
            let inferred = results.get(slot).cloned().unwrap_or_default();
            let decl = g.values[vid]
                .declared
                .as_ref()
                .filter(|d| d.kind != TypeKind::Unknown)
                .map(|d| from_decl(d, ov, None));
            let (state, source) = merge(inferred, decl);
            st[vid] = state;
            src[vid] = source;
        }
    }

    for (vid, v) in g.values.iter().enumerate() {
        if src[vid] == ShapeSource::Unknown && st[vid].dtype == 0 {
            if let Some(d) = &v.declared {
                if d.kind != TypeKind::Unknown {
                    st[vid] = from_decl(d, ov, None);
                    src[vid] = if st[vid].shape.is_some() {
                        ShapeSource::Declared
                    } else {
                        ShapeSource::Unknown
                    };
                }
            }
        }
    }

    GraphShapes {
        states: st,
        sources: src,
    }
}

struct Ctx<'a> {
    model: &'a Model,
    graph: &'a Graph,
    ov: &'a Overrides,
}

fn inp<'a>(ins: &[Option<&'a VState>], i: usize) -> Option<&'a VState> {
    ins.get(i).copied().flatten()
}

fn shp<'a>(ins: &[Option<&'a VState>], i: usize) -> Option<&'a [Dim]> {
    inp(ins, i)?.shape.as_deref()
}

fn dt(ins: &[Option<&VState>], i: usize) -> i32 {
    inp(ins, i).map(|s| s.dtype).unwrap_or(0)
}

/// Symbolic contents of a small integer input.
fn sym<'a>(ins: &[Option<&'a VState>], i: usize) -> Option<&'a [Dim]> {
    inp(ins, i)?.ints.as_deref()
}

/// Fully known integer contents of an input.
fn ints(ins: &[Option<&VState>], i: usize) -> Option<Vec<i64>> {
    sym(ins, i)?.iter().map(|d| d.known()).collect()
}

fn floats(ins: &[Option<&VState>], i: usize) -> Option<Vec<f64>> {
    let s = inp(ins, i)?;
    if let Some(f) = &s.floats {
        return Some(f.clone());
    }
    s.ints
        .as_ref()?
        .iter()
        .map(|d| d.known().map(|v| v as f64))
        .collect()
}

fn same_as(ins: &[Option<&VState>], i: usize) -> VState {
    match inp(ins, i) {
        Some(s) => VState::new(s.dtype, s.shape.clone()),
        None => VState::default(),
    }
}

fn unknown_rank(dtype: i32, rank: usize) -> VState {
    VState::new(dtype, Some(vec![Dim::Unknown; rank]))
}

/// Attribute ints, or the contents of input `idx` (opset dependent).
fn attr_or_input(node: &Node, attr: &str, ins: &[Option<&VState>], idx: usize) -> Option<Vec<i64>> {
    if let Some(v) = node.attr_ints(attr) {
        return Some(v.to_vec());
    }
    if node.input(idx).is_some() {
        return ints(ins, idx);
    }
    None
}

fn broadcast_inputs(ins: &[Option<&VState>], idxs: &[usize]) -> Option<Shape> {
    let mut shapes: Vec<&[Dim]> = Vec::new();
    for &i in idxs {
        shapes.push(shp(ins, i)?);
    }
    if shapes.is_empty() {
        return None;
    }
    Some(shape::broadcast(&shapes))
}

fn ints_binop(a: &[Dim], b: &[Dim], f: fn(&Dim, &Dim) -> Dim) -> Option<Vec<Dim>> {
    if a.len() == b.len() {
        Some(a.iter().zip(b).map(|(x, y)| f(x, y)).collect())
    } else if a.len() == 1 {
        Some(b.iter().map(|y| f(&a[0], y)).collect())
    } else if b.len() == 1 {
        Some(a.iter().map(|x| f(x, &b[0])).collect())
    } else {
        None
    }
}

fn dim_max(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Known(x), Dim::Known(y)) => Dim::Known(*x.max(y)),
        _ => Dim::Unknown,
    }
}

fn dim_min(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Known(x), Dim::Known(y)) => Dim::Known(*x.min(y)),
        _ => Dim::Unknown,
    }
}

fn dim_mod(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Known(x), Dim::Known(y)) if *y != 0 => Dim::Known(x.rem_euclid(*y)),
        _ => Dim::Unknown,
    }
}

fn dim_eq(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Known(x), Dim::Known(y)) => Dim::Known((x == y) as i64),
        (Dim::Param(p), Dim::Param(q)) if p == q => Dim::Known(1),
        (Dim::Param(_), Dim::Known(y)) | (Dim::Known(y), Dim::Param(_)) if *y < 0 => {
            Dim::Known(0)
        }
        _ => Dim::Unknown,
    }
}

fn conv_spatial(node: &Node, in_sp: &[Dim], kernel: &[i64], is_pool: bool) -> Vec<Dim> {
    let r = in_sp.len();
    let strides = node.attr_ints("strides").map(|v| v.to_vec()).unwrap_or_default();
    let dil = node.attr_ints("dilations").map(|v| v.to_vec()).unwrap_or_default();
    let pads = node.attr_ints("pads").map(|v| v.to_vec()).unwrap_or_default();
    let auto_pad = node.attr_s("auto_pad").unwrap_or_default();
    let ceil = is_pool && node.attr_i("ceil_mode").unwrap_or(0) == 1;
    (0..r)
        .map(|i| {
            let s = strides.get(i).copied().unwrap_or(1).max(1);
            let d = dil.get(i).copied().unwrap_or(1).max(1);
            let k = kernel.get(i).copied().unwrap_or(1);
            let ke = (k - 1) * d + 1;
            let pb = pads.get(i).copied().unwrap_or(0);
            let pe = pads.get(i + r).copied().unwrap_or(0);
            let same = auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER";
            let Some(x) = in_sp[i].known() else {
                if s == 1 && (same || (auto_pad != "VALID" && pb + pe == ke - 1)) {
                    return in_sp[i].clone();
                }
                if s > 1 && same {
                    return shape::div(&in_sp[i], &Dim::Known(s));
                }
                return Dim::Unknown;
            };
            if same {
                return Dim::Known((x + s - 1) / s);
            }
            if auto_pad == "VALID" {
                return Dim::Known(if x >= ke { (x - ke) / s + 1 } else { 0 });
            }
            let num = x + pb + pe - ke;
            if num < 0 {
                return Dim::Known(0);
            }
            let mut o = if ceil { (num + s - 1) / s + 1 } else { num / s + 1 };
            if ceil && (o - 1) * s >= x + pb {
                o -= 1;
            }
            Dim::Known(o)
        })
        .collect()
}

fn conv_like(node: &Node, x: Option<&[Dim]>, w: Option<&[Dim]>, dtype: i32) -> VState {
    let kernel: Vec<i64> = match node.attr_ints("kernel_shape") {
        Some(k) => k.to_vec(),
        None => match w {
            Some(w) if w.len() > 2 => w[2..].iter().map(|d| d.known().unwrap_or(1)).collect(),
            _ => Vec::new(),
        },
    };
    let Some(x) = x else {
        return match w {
            Some(w) if w.len() >= 2 => unknown_rank(dtype, w.len()),
            _ => VState::new(dtype, None),
        };
    };
    if x.len() < 2 {
        return VState::new(dtype, None);
    }
    let m = w.and_then(|w| w.first().cloned()).unwrap_or(Dim::Unknown);
    let mut out = vec![x[0].clone(), m];
    let kernel = if kernel.is_empty() {
        vec![1; x.len() - 2]
    } else {
        kernel
    };
    out.extend(conv_spatial(node, &x[2..], &kernel, false));
    VState::new(dtype, Some(out))
}

fn conv_transpose(node: &Node, x: Option<&[Dim]>, w: Option<&[Dim]>, dtype: i32) -> VState {
    let Some(x) = x else {
        return VState::new(dtype, None);
    };
    if x.len() < 2 {
        return VState::new(dtype, None);
    }
    let r = x.len() - 2;
    let group = node.attr_i("group").unwrap_or(1).max(1);
    let m = match w.and_then(|w| w.get(1)) {
        Some(d) => shape::mul(d, &Dim::Known(group)),
        None => Dim::Unknown,
    };
    let mut out = vec![x[0].clone(), m];
    if let Some(os) = node.attr_ints("output_shape") {
        let os = if os.len() == r + 2 { &os[2..] } else { os };
        out.extend(os.iter().map(|&v| Dim::Known(v)));
        return VState::new(dtype, Some(out));
    }
    let kernel: Vec<i64> = match node.attr_ints("kernel_shape") {
        Some(k) => k.to_vec(),
        None => match w {
            Some(w) if w.len() > 2 => w[2..].iter().map(|d| d.known().unwrap_or(1)).collect(),
            _ => vec![1; r],
        },
    };
    let strides = node.attr_ints("strides").map(|v| v.to_vec()).unwrap_or_default();
    let dil = node.attr_ints("dilations").map(|v| v.to_vec()).unwrap_or_default();
    let pads = node.attr_ints("pads").map(|v| v.to_vec()).unwrap_or_default();
    let opad = node.attr_ints("output_padding").map(|v| v.to_vec()).unwrap_or_default();
    let auto_pad = node.attr_s("auto_pad").unwrap_or_default();
    for i in 0..r {
        let s = strides.get(i).copied().unwrap_or(1).max(1);
        let Some(xi) = x[2 + i].known() else {
            out.push(if s == 1 { x[2 + i].clone() } else { shape::mul(&x[2 + i], &Dim::Known(s)) });
            continue;
        };
        if auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" {
            out.push(Dim::Known(xi * s));
            continue;
        }
        let d = dil.get(i).copied().unwrap_or(1).max(1);
        let k = kernel.get(i).copied().unwrap_or(1);
        let pb = pads.get(i).copied().unwrap_or(0);
        let pe = pads.get(i + r).copied().unwrap_or(0);
        let op = opad.get(i).copied().unwrap_or(0);
        out.push(Dim::Known(s * (xi - 1) + op + ((k - 1) * d + 1) - pb - pe));
    }
    VState::new(dtype, Some(out))
}

fn pool(node: &Node, x: Option<&[Dim]>, dtype: i32) -> VState {
    let Some(x) = x else {
        return VState::new(dtype, None);
    };
    if x.len() < 2 {
        return VState::new(dtype, None);
    }
    let kernel = node
        .attr_ints("kernel_shape")
        .map(|v| v.to_vec())
        .unwrap_or_else(|| vec![1; x.len() - 2]);
    let mut out = vec![x[0].clone(), x[1].clone()];
    out.extend(conv_spatial(node, &x[2..], &kernel, true));
    VState::new(dtype, Some(out))
}

fn global_pool(x: Option<&[Dim]>, dtype: i32) -> VState {
    match x {
        Some(x) if x.len() >= 2 => {
            let mut out = vec![x[0].clone(), x[1].clone()];
            out.extend(std::iter::repeat(Dim::Known(1)).take(x.len() - 2));
            VState::new(dtype, Some(out))
        }
        _ => VState::new(dtype, None),
    }
}

pub fn matmul_shape(a: &[Dim], b: &[Dim]) -> Option<Shape> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let a2: Vec<Dim> = if a.len() == 1 {
        vec![Dim::Known(1), a[0].clone()]
    } else {
        a.to_vec()
    };
    let b2: Vec<Dim> = if b.len() == 1 {
        vec![b[0].clone(), Dim::Known(1)]
    } else {
        b.to_vec()
    };
    let mut out = shape::broadcast(&[&a2[..a2.len() - 2], &b2[..b2.len() - 2]]);
    if a.len() != 1 {
        out.push(a2[a2.len() - 2].clone());
    }
    if b.len() != 1 {
        out.push(b2[b2.len() - 1].clone());
    }
    Some(out)
}

fn reshape(input: Option<&[Dim]>, target: &[Dim], allowzero: bool) -> Option<Shape> {
    let mut out: Vec<Dim> = Vec::with_capacity(target.len());
    let mut neg: Option<usize> = None;
    for (i, d) in target.iter().enumerate() {
        match d {
            Dim::Known(0) if !allowzero => {
                out.push(input.and_then(|s| s.get(i)).cloned().unwrap_or(Dim::Unknown))
            }
            Dim::Known(-1) => {
                if neg.is_some() {
                    return None;
                }
                neg = Some(i);
                out.push(Dim::Unknown);
            }
            Dim::Known(v) if *v < -1 => return None,
            other => out.push(other.clone()),
        }
    }
    if let (Some(ni), Some(inp)) = (neg, input) {
        let mut rem_in: Vec<Dim> = inp.to_vec();
        let mut rem_out: Vec<Dim> = Vec::new();
        for (i, d) in out.iter().enumerate() {
            if i == ni {
                continue;
            }
            if let Some(p) = rem_in.iter().position(|x| x == d) {
                rem_in.remove(p);
            } else {
                rem_out.push(d.clone());
            }
        }
        let num = shape::product(&rem_in);
        let den = shape::product(&rem_out);
        out[ni] = match (&num, &den) {
            (Dim::Known(a), Dim::Known(b)) if *b != 0 => Dim::Known(a / b),
            _ => shape::div(&num, &den),
        };
    }
    Some(out)
}

fn slice_dim(dim: &Dim, start: i64, end: i64, step: i64) -> Dim {
    if step == 0 {
        return Dim::Unknown;
    }
    match dim {
        Dim::Known(d) => {
            let d = *d;
            let (mut s, mut e) = (start, end);
            if s < 0 {
                s += d;
            }
            if e < 0 {
                e += d;
            }
            let len = if step > 0 {
                let s = s.clamp(0, d);
                let e = e.clamp(0, d);
                if e > s {
                    (e - s + step - 1) / step
                } else {
                    0
                }
            } else {
                let s = s.clamp(-1, d - 1);
                let e = e.clamp(-1, d - 1);
                if s > e {
                    (s - e + (-step) - 1) / (-step)
                } else {
                    0
                }
            };
            Dim::Known(len)
        }
        _ => {
            if start == 0 && step == 1 && end >= LARGE_END {
                dim.clone()
            } else if step == 1 && start >= 0 && end >= LARGE_END {
                shape::sub(dim, &Dim::Known(start))
            } else if step == 1 && start < 0 && end >= LARGE_END {
                Dim::Known(-start)
            } else {
                Dim::Unknown
            }
        }
    }
}

fn reduce_shape(x: &[Dim], axes: Option<&[i64]>, keepdims: bool, noop_empty: bool) -> Option<Shape> {
    let r = x.len();
    let axes: Vec<usize> = match axes {
        Some(a) if !a.is_empty() => {
            let mut v = Vec::new();
            for &ax in a {
                v.push(norm_axis(ax, r)?);
            }
            v
        }
        _ => {
            if noop_empty {
                return Some(x.to_vec());
            }
            (0..r).collect()
        }
    };
    let mut out = Vec::new();
    for (i, d) in x.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out.push(Dim::Known(1));
            }
        } else {
            out.push(d.clone());
        }
    }
    Some(out)
}

fn einsum_shape(eq: &str, shapes: &[Option<&[Dim]>]) -> Option<Shape> {
    let eq: String = eq.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs, rhs) = match eq.split_once("->") {
        Some((l, r)) => (l.to_string(), Some(r.to_string())),
        None => (eq.clone(), None),
    };
    let terms: Vec<&str> = lhs.split(',').collect();
    if terms.len() != shapes.len() {
        return None;
    }
    let mut dims: HashMap<char, Dim> = HashMap::new();
    let mut counts: HashMap<char, usize> = HashMap::new();
    let mut ell: Vec<Dim> = Vec::new();
    let mut has_ellipsis = false;
    for (term, s) in terms.iter().zip(shapes) {
        let s = (*s)?;
        let (before, after, has_ell) = match term.find("...") {
            Some(p) => (&term[..p], &term[p + 3..], true),
            None => (&term[..], "", false),
        };
        let nletters = before.chars().count() + after.chars().count();
        if s.len() < nletters || (!has_ell && s.len() != nletters) {
            return None;
        }
        let ell_rank = s.len() - nletters;
        let mut idx = 0;
        let mut put = |c: char, d: &Dim| {
            *counts.entry(c).or_insert(0) += 1;
            let e = dims.entry(c).or_insert(Dim::Unknown);
            if matches!(e, Dim::Unknown | Dim::Known(1)) {
                *e = d.clone();
            }
        };
        for c in before.chars() {
            put(c, &s[idx]);
            idx += 1;
        }
        if has_ell {
            has_ellipsis = true;
            ell = shape::broadcast(&[&ell, &s[idx..idx + ell_rank]]);
            idx += ell_rank;
        }
        for c in after.chars() {
            put(c, &s[idx]);
            idx += 1;
        }
    }
    let out_term = match rhs {
        Some(r) => r,
        None => {
            let mut letters: Vec<char> = counts
                .iter()
                .filter(|(_, &n)| n == 1)
                .map(|(&c, _)| c)
                .collect();
            letters.sort_unstable();
            let s: String = letters.into_iter().collect();
            if has_ellipsis {
                format!("...{}", s)
            } else {
                s
            }
        }
    };
    let mut out = Vec::new();
    let (before, after, has_ell) = match out_term.find("...") {
        Some(p) => (out_term[..p].to_string(), out_term[p + 3..].to_string(), true),
        None => (out_term.clone(), String::new(), false),
    };
    for c in before.chars() {
        out.push(dims.get(&c).cloned().unwrap_or(Dim::Unknown));
    }
    if has_ell {
        out.extend(ell.iter().cloned());
    }
    for c in after.chars() {
        out.push(dims.get(&c).cloned().unwrap_or(Dim::Unknown));
    }
    Some(out)
}

fn subgraph_outputs(ctx: &Ctx, node: &Node, attr: &str) -> Vec<VState> {
    let Some(AttrValue::Graph(sid)) = node.attr(attr) else {
        return Vec::new();
    };
    let Some(sg) = ctx.model.graphs.get(*sid) else {
        return Vec::new();
    };
    sg.outputs
        .iter()
        .map(|&vid| match &sg.values[vid].declared {
            Some(d) => from_decl(d, ctx.ov, None),
            None => VState::default(),
        })
        .collect()
}

fn infer_node(ctx: &Ctx, node: &Node, ins: &[Option<&VState>]) -> Vec<VState> {
    let op = node.op_type.as_str();
    let ms = node.domain == "com.microsoft";
    let std_or_ms = node.is_standard() || ms;
    if !std_or_ms {
        return Vec::new();
    }
    let x_dt = dt(ins, 0);
    let x = shp(ins, 0);
    match op {
        "Identity" => match inp(ins, 0) {
            Some(s) => vec![s.clone()],
            None => vec![],
        },
        "Abs" | "Acos" | "Acosh" | "Asin" | "Asinh" | "Atan" | "Atanh" | "Ceil" | "Celu"
        | "Cos" | "Cosh" | "Elu" | "Erf" | "Exp" | "Floor" | "Gelu" | "HardSigmoid"
        | "HardSwish" | "Hardmax" | "LeakyRelu" | "Log" | "LogSoftmax" | "LpNormalization"
        | "LRN" | "MeanVarianceNormalization" | "Mish" | "Reciprocal" | "Relu" | "Round"
        | "Selu" | "Shrink" | "Sigmoid" | "Sign" | "Sin" | "Sinh" | "Softmax" | "Softplus"
        | "Softsign" | "Sqrt" | "Tan" | "Tanh" | "ThresholdedRelu" | "Swish" | "BitwiseNot"
        | "Clip" | "InstanceNormalization" | "GroupNormalization" | "RMSNormalization"
        | "CumSum" | "Trilu" | "ReverseSequence" | "PRelu" | "FastGelu" | "BiasGelu"
        | "QuickGelu" | "BiasSoftmax" | "ScatterND" | "ScatterElements" | "Scatter"
        | "RotaryEmbedding" | "BiasAdd" | "ImageScaler" | "ScaledTanh" | "ParametricSoftplus"
        | "Scale" | "QLinearSigmoid" | "QLinearLeakyRelu" | "QLinearSoftmax"
        | "TensorScatter" => vec![same_as(ins, 0)],
        "Neg" => {
            let mut s = same_as(ins, 0);
            s.ints = sym(ins, 0).map(|v| {
                v.iter()
                    .map(|d| match d {
                        Dim::Known(k) => Dim::Known(-k),
                        _ => Dim::Unknown,
                    })
                    .collect()
            });
            vec![s]
        }
        "Not" | "IsNaN" | "IsInf" => vec![VState::new(dtype::BOOL, x.map(|s| s.to_vec()))],
        "Dropout" => vec![
            same_as(ins, 0),
            VState::new(dtype::BOOL, x.map(|s| s.to_vec())),
        ],
        "Add" | "Sub" | "Mul" | "Div" | "Mod" | "Pow" | "BitShift" | "BitwiseAnd"
        | "BitwiseOr" | "BitwiseXor" | "Max" | "Min" | "Sum" | "Mean" => {
            let idxs: Vec<usize> = (0..node.inputs.len()).collect();
            let mut s = VState::new(x_dt, broadcast_inputs(ins, &idxs));
            if node.inputs.len() == 2 {
                if let (Some(a), Some(b)) = (sym(ins, 0), sym(ins, 1)) {
                    let f: Option<fn(&Dim, &Dim) -> Dim> = match op {
                        "Add" => Some(shape::add),
                        "Sub" => Some(shape::sub),
                        "Mul" => Some(shape::mul),
                        "Div" => Some(shape::div),
                        "Mod" => Some(dim_mod),
                        "Max" => Some(dim_max),
                        "Min" => Some(dim_min),
                        _ => None,
                    };
                    if let Some(f) = f {
                        s.ints = ints_binop(a, b, f);
                    }
                }
            }
            vec![s]
        }
        "Equal" | "Less" | "Greater" | "LessOrEqual" | "GreaterOrEqual" | "And" | "Or"
        | "Xor" => {
            let mut s = VState::new(dtype::BOOL, broadcast_inputs(ins, &[0, 1]));
            if op == "Equal" {
                if let (Some(a), Some(b)) = (sym(ins, 0), sym(ins, 1)) {
                    s.ints = ints_binop(a, b, dim_eq);
                }
            }
            vec![s]
        }
        "Where" => {
            let dtype = if dt(ins, 1) != 0 { dt(ins, 1) } else { dt(ins, 2) };
            let mut s = VState::new(dtype, broadcast_inputs(ins, &[0, 1, 2]));
            if let (Some(c), Some(a), Some(b)) = (ints(ins, 0), sym(ins, 1), sym(ins, 2)) {
                let n = c.len().max(a.len()).max(b.len());
                let pick = |v: &[Dim], i: usize| -> Option<Dim> {
                    if v.len() == 1 {
                        Some(v[0].clone())
                    } else {
                        v.get(i).cloned()
                    }
                };
                let mut out = Vec::with_capacity(n);
                let mut ok = true;
                for i in 0..n {
                    let ci = if c.len() == 1 { c[0] } else { c.get(i).copied().unwrap_or(0) };
                    match pick(if ci != 0 { a } else { b }, i) {
                        Some(d) => out.push(d),
                        None => ok = false,
                    }
                }
                if ok {
                    s.ints = Some(out);
                }
            }
            vec![s]
        }
        "Cast" => {
            let to = node.attr_i("to").unwrap_or(0) as i32;
            let src_state = inp(ins, 0);
            let mut s = VState::new(to, x.map(|v| v.to_vec()));
            if let Some(st) = src_state {
                if dtype::is_integer(to) {
                    s.ints = st.ints.clone().or_else(|| {
                        st.floats
                            .as_ref()
                            .map(|f| f.iter().map(|v| Dim::Known(v.trunc() as i64)).collect())
                    });
                } else {
                    s.floats = floats(ins, 0);
                    s.ints = st.ints.clone();
                }
            }
            vec![s]
        }
        "CastLike" => {
            let mut s = VState::new(dt(ins, 1), x.map(|v| v.to_vec()));
            s.ints = sym(ins, 0).map(|v| v.to_vec());
            vec![s]
        }
        "Constant" => {
            // Constant nodes are normally seeded from their tensor; this only
            // handles malformed ones.
            vec![VState::default()]
        }
        "ConstantOfShape" => {
            let (vdt, val) = match node.attr("value") {
                Some(AttrValue::Tensor(ti)) => {
                    let t = &ctx.graph.tensors[*ti];
                    (t.data_type, tensor::small_ints(t, 1).and_then(|v| v.first().copied()))
                }
                _ => (dtype::FLOAT, Some(0)),
            };
            let target = sym(ins, 0).map(|v| v.to_vec());
            let mut s = match &target {
                Some(t) => VState::new(vdt, Some(t.clone())),
                None => match shp(ins, 0) {
                    Some([Dim::Known(r)]) => unknown_rank(vdt, *r as usize),
                    _ => VState::new(vdt, None),
                },
            };
            if dtype::is_integer(vdt) {
                if let (Some(t), Some(v)) = (&target, val) {
                    if let Some(n) = shape::numel(t) {
                        if n as usize <= MAX_TRACKED {
                            s.ints = Some(vec![Dim::Known(v); n as usize]);
                        }
                    }
                }
            }
            vec![s]
        }
        "Shape" => {
            let mut s = VState::new(dtype::INT64, None);
            if let Some(x) = x {
                let r = x.len() as i64;
                let clampi = |v: i64| -> usize {
                    let v = if v < 0 { v + r } else { v };
                    v.clamp(0, r) as usize
                };
                let start = clampi(node.attr_i("start").unwrap_or(0));
                let end = clampi(node.attr_i("end").unwrap_or(r));
                let dims: Vec<Dim> = if end > start { x[start..end].to_vec() } else { Vec::new() };
                s.shape = Some(vec![Dim::Known(dims.len() as i64)]);
                s.ints = Some(dims);
            }
            vec![s]
        }
        "Size" => {
            let mut s = VState::new(dtype::INT64, Some(Vec::new()));
            if let Some(x) = x {
                s.ints = Some(vec![shape::product(x)]);
            }
            vec![s]
        }
        "Reshape" => {
            let allowzero = node.attr_i("allowzero").unwrap_or(0) == 1;
            let target: Option<Vec<Dim>> = match node.attr_ints("shape") {
                Some(v) => Some(shape::from_i64s(v)),
                None => sym(ins, 1).map(|v| v.to_vec()),
            };
            let mut s = match target {
                Some(t) => VState::new(x_dt, reshape(x, &t, allowzero)),
                None => match shp(ins, 1) {
                    Some([Dim::Known(r)]) => unknown_rank(x_dt, *r as usize),
                    _ => VState::new(x_dt, None),
                },
            };
            s.ints = sym(ins, 0).map(|v| v.to_vec());
            vec![s]
        }
        "Flatten" => {
            let mut s = VState::new(x_dt, None);
            if let Some(x) = x {
                let r = x.len() as i64;
                let mut axis = node.attr_i("axis").unwrap_or(1);
                if axis < 0 {
                    axis += r;
                }
                let axis = axis.clamp(0, r) as usize;
                s.shape = Some(vec![shape::product(&x[..axis]), shape::product(&x[axis..])]);
            }
            s.ints = sym(ins, 0).map(|v| v.to_vec());
            vec![s]
        }
        "Squeeze" => {
            let axes = attr_or_input(node, "axes", ins, 1);
            let mut s = VState::new(x_dt, None);
            if let Some(x) = x {
                let r = x.len();
                let out: Option<Vec<Dim>> = match &axes {
                    Some(a) if !a.is_empty() => {
                        let ax: Option<Vec<usize>> = a.iter().map(|&v| norm_axis(v, r)).collect();
                        ax.map(|ax| {
                            x.iter()
                                .enumerate()
                                .filter(|(i, _)| !ax.contains(i))
                                .map(|(_, d)| d.clone())
                                .collect()
                        })
                    }
                    Some(_) => Some(x.iter().filter(|d| **d != Dim::Known(1)).cloned().collect()),
                    None => {
                        if node.input(1).is_some() {
                            None
                        } else {
                            Some(x.iter().filter(|d| **d != Dim::Known(1)).cloned().collect())
                        }
                    }
                };
                s.shape = out;
            }
            s.ints = sym(ins, 0).map(|v| v.to_vec());
            vec![s]
        }
        "Unsqueeze" => {
            let axes = attr_or_input(node, "axes", ins, 1);
            let mut s = VState::new(x_dt, None);
            if let (Some(x), Some(a)) = (x, &axes) {
                let out_r = x.len() + a.len();
                let ax: Option<Vec<usize>> = a.iter().map(|&v| norm_axis(v, out_r)).collect();
                if let Some(ax) = ax {
                    let mut it = x.iter();
                    let mut out = Vec::with_capacity(out_r);
                    for i in 0..out_r {
                        if ax.contains(&i) {
                            out.push(Dim::Known(1));
                        } else {
                            out.push(it.next().cloned().unwrap_or(Dim::Unknown));
                        }
                    }
                    s.shape = Some(out);
                }
            }
            s.ints = sym(ins, 0).map(|v| v.to_vec());
            vec![s]
        }
        "Transpose" => {
            let mut s = VState::new(x_dt, None);
            if let Some(x) = x {
                let perm: Vec<i64> = node
                    .attr_ints("perm")
                    .map(|p| p.to_vec())
                    .unwrap_or_else(|| (0..x.len() as i64).rev().collect());
                let out: Option<Vec<Dim>> = perm
                    .iter()
                    .map(|&p| norm_axis(p, x.len()).map(|i| x[i].clone()))
                    .collect();
                s.shape = out;
            }
            vec![s]
        }
        "Concat" => {
            let axis = node.attr_i("axis").unwrap_or(0);
            let shapes: Vec<Option<&[Dim]>> = (0..node.inputs.len()).map(|i| shp(ins, i)).collect();
            let mut s = VState::new(x_dt, None);
            if let Some(Some(first)) = shapes.iter().find(|s| s.is_some()) {
                let r = first.len();
                if let Some(ax) = norm_axis(axis, r) {
                    let mut out: Vec<Dim> = first.to_vec();
                    let mut total = Dim::Known(0);
                    for sh in &shapes {
                        match sh {
                            Some(sh) if sh.len() == r => {
                                total = shape::add(&total, &sh[ax]);
                                for (i, d) in sh.iter().enumerate() {
                                    if i != ax && !out[i].is_known() && d.is_known() {
                                        out[i] = d.clone();
                                    }
                                }
                            }
                            _ => total = Dim::Unknown,
                        }
                    }
                    out[ax] = total;
                    s.shape = Some(out);
                }
            }
            let parts: Option<Vec<&[Dim]>> = (0..node.inputs.len()).map(|i| sym(ins, i)).collect();
            if let Some(parts) = parts {
                let joined: Vec<Dim> = parts.concat();
                if joined.len() <= MAX_TRACKED {
                    s.ints = Some(joined);
                }
            }
            let fparts: Option<Vec<Vec<f64>>> = (0..node.inputs.len())
                .map(|i| inp(ins, i).and_then(|s| s.floats.clone()))
                .collect();
            if let Some(f) = fparts {
                s.floats = Some(f.concat());
            }
            vec![s]
        }
        "Split" => {
            let nout = node.outputs.len();
            let axis = node.attr_i("axis").unwrap_or(0);
            let Some(x) = x else {
                return vec![VState::new(x_dt, None); nout];
            };
            let Some(ax) = norm_axis(axis, x.len()) else {
                return vec![VState::new(x_dt, None); nout];
            };
            let split: Option<Vec<i64>> = match node.attr_ints("split") {
                Some(v) => Some(v.to_vec()),
                None => {
                    if node.input(1).is_some() {
                        ints(ins, 1)
                    } else {
                        None
                    }
                }
            };
            let sizes: Vec<Dim> = match split {
                Some(v) => v.into_iter().map(Dim::Known).collect(),
                None => {
                    if node.input(1).is_some() {
                        vec![Dim::Unknown; nout]
                    } else {
                        match x[ax].known() {
                            Some(d) => {
                                let n = node.attr_i("num_outputs").unwrap_or(nout as i64).max(1);
                                let chunk = (d + n - 1) / n;
                                (0..nout as i64)
                                    .map(|i| Dim::Known(chunk.min(d - chunk * i).max(0)))
                                    .collect()
                            }
                            None => vec![shape::div(&x[ax], &Dim::Known(nout as i64)); nout],
                        }
                    }
                }
            };
            (0..nout)
                .map(|i| {
                    let mut out = x.to_vec();
                    out[ax] = sizes.get(i).cloned().unwrap_or(Dim::Unknown);
                    VState::new(x_dt, Some(out))
                })
                .collect()
        }
        "Slice" => {
            let (starts, ends, axes, steps) = if node.attr_ints("starts").is_some() {
                (
                    node.attr_ints("starts").map(|v| v.to_vec()),
                    node.attr_ints("ends").map(|v| v.to_vec()),
                    node.attr_ints("axes").map(|v| v.to_vec()),
                    None,
                )
            } else {
                (
                    ints(ins, 1),
                    ints(ins, 2),
                    if node.input(3).is_some() { ints(ins, 3) } else { None },
                    if node.input(4).is_some() { ints(ins, 4) } else { None },
                )
            };
            let axes_given = node.attr_ints("axes").is_some() || node.input(3).is_some();
            let steps_given = node.input(4).is_some();
            let mut s = VState::new(x_dt, None);
            if let Some(x) = x {
                let r = x.len();
                if let (Some(st), Some(en)) = (&starts, &ends) {
                    let axes: Option<Vec<i64>> = match axes {
                        Some(a) => Some(a),
                        None if !axes_given => Some((0..st.len() as i64).collect()),
                        None => None,
                    };
                    let steps: Option<Vec<i64>> = match steps {
                        Some(v) => Some(v),
                        None if !steps_given => Some(vec![1; st.len()]),
                        None => None,
                    };
                    if let (Some(axes), Some(steps)) = (axes, steps) {
                        let mut out = x.to_vec();
                        let mut ok = true;
                        for k in 0..st.len() {
                            let Some(ax) = axes.get(k).and_then(|&a| norm_axis(a, r)) else {
                                ok = false;
                                break;
                            };
                            let e = en.get(k).copied().unwrap_or(i64::MAX);
                            let stp = steps.get(k).copied().unwrap_or(1);
                            out[ax] = slice_dim(&x[ax], st[k], e, stp);
                        }
                        if ok {
                            s.shape = Some(out);
                        }
                        // Slice of a small 1-D integer tensor.
                        if ok && r == 1 && !st.is_empty() {
                            if let Some(v) = sym(ins, 0) {
                                let d = v.len() as i64;
                                let mut a = st[0];
                                let mut b = en.first().copied().unwrap_or(i64::MAX);
                                let stp = steps.first().copied().unwrap_or(1);
                                if stp == 1 {
                                    if a < 0 {
                                        a += d;
                                    }
                                    if b < 0 {
                                        b += d;
                                    }
                                    let a = a.clamp(0, d) as usize;
                                    let b = b.clamp(0, d) as usize;
                                    s.ints = Some(if b > a { v[a..b].to_vec() } else { Vec::new() });
                                }
                            }
                        }
                    }
                }
                if s.shape.is_none() {
                    s.shape = Some(vec![Dim::Unknown; r]);
                }
            }
            vec![s]
        }
        "Gather" => {
            let axis = node.attr_i("axis").unwrap_or(0);
            let mut s = VState::new(x_dt, None);
            if let (Some(d), Some(ix)) = (x, shp(ins, 1)) {
                if let Some(ax) = norm_axis(axis, d.len()) {
                    let mut out = d[..ax].to_vec();
                    out.extend(ix.iter().cloned());
                    out.extend(d[ax + 1..].iter().cloned());
                    s.shape = Some(out);
                }
            }
            if let (Some(data), Some(idx)) = (sym(ins, 0), ints(ins, 1)) {
                if x.map(|v| v.len()) == Some(1) {
                    let n = data.len() as i64;
                    let picked: Option<Vec<Dim>> = idx
                        .iter()
                        .map(|&i| {
                            let i = if i < 0 { i + n } else { i };
                            data.get(i as usize).cloned()
                        })
                        .collect();
                    s.ints = picked;
                }
            }
            vec![s]
        }
        "GatherElements" => vec![VState::new(x_dt, shp(ins, 1).map(|v| v.to_vec()))],
        "GatherND" => {
            let b = node.attr_i("batch_dims").unwrap_or(0).max(0) as usize;
            let mut s = VState::new(x_dt, None);
            if let (Some(d), Some(ix)) = (x, shp(ins, 1)) {
                if let Some(Dim::Known(k)) = ix.last() {
                    let k = *k as usize;
                    if b + k <= d.len() {
                        let mut out = ix[..ix.len() - 1].to_vec();
                        out.extend(d[b + k..].iter().cloned());
                        s.shape = Some(out);
                    }
                }
            }
            vec![s]
        }
        "Expand" => {
            let mut s = VState::new(x_dt, None);
            if let Some(x) = x {
                if let Some(t) = sym(ins, 1) {
                    s.shape = Some(shape::broadcast(&[x, t]));
                } else if let Some([Dim::Known(k)]) = shp(ins, 1) {
                    s.shape = Some(vec![Dim::Unknown; (*k as usize).max(x.len())]);
                }
            }
            vec![s]
        }
        "Tile" => {
            let mut s = VState::new(x_dt, None);
            if let Some(x) = x {
                s.shape = Some(match sym(ins, 1) {
                    Some(r) if r.len() == x.len() => {
                        x.iter().zip(r).map(|(a, b)| shape::mul(a, b)).collect()
                    }
                    _ => vec![Dim::Unknown; x.len()],
                });
            }
            vec![s]
        }
        "Pad" => {
            let pads = attr_or_input(node, "pads", ins, 1);
            let mut s = VState::new(x_dt, None);
            if let Some(x) = x {
                let r = x.len();
                let axes: Option<Vec<usize>> = if node.input(3).is_some() {
                    ints(ins, 3).and_then(|a| a.iter().map(|&v| norm_axis(v, r)).collect())
                } else {
                    Some((0..r).collect())
                };
                s.shape = Some(match (pads, axes) {
                    (Some(p), Some(axes)) if p.len() == 2 * axes.len() => {
                        let mut out = x.to_vec();
                        let na = axes.len();
                        for (k, &ax) in axes.iter().enumerate() {
                            out[ax] = shape::add(&x[ax], &Dim::Known(p[k] + p[k + na]));
                        }
                        out
                    }
                    _ => vec![Dim::Unknown; r],
                });
            }
            vec![s]
        }
        "Range" => {
            let mut s = VState::new(x_dt, Some(vec![Dim::Unknown]));
            if let (Some(a), Some(b), Some(c)) = (floats(ins, 0), floats(ins, 1), floats(ins, 2)) {
                if let (Some(&a), Some(&b), Some(&c)) = (a.first(), b.first(), c.first()) {
                    if c != 0.0 {
                        let n = ((b - a) / c).ceil().max(0.0) as i64;
                        s.shape = Some(vec![Dim::Known(n)]);
                        if dtype::is_integer(x_dt) && (n as usize) <= MAX_TRACKED {
                            s.ints = Some(
                                (0..n)
                                    .map(|i| Dim::Known((a + c * i as f64) as i64))
                                    .collect(),
                            );
                        }
                    }
                }
            }
            vec![s]
        }
        "Resize" | "Upsample" => {
            let mut s = VState::new(x_dt, None);
            if let Some(x) = x {
                let r = x.len();
                let sizes = if op == "Resize" && node.input(3).is_some() {
                    sym(ins, 3).map(|v| v.to_vec())
                } else {
                    None
                };
                let scales: Option<Vec<f64>> = if op == "Upsample" {
                    node.attr_floats("scales")
                        .map(|v| v.iter().map(|&f| f as f64).collect())
                        .or_else(|| floats(ins, 1))
                } else if node.inputs.len() == 2 {
                    floats(ins, 1)
                } else if node.input(2).is_some() {
                    floats(ins, 2)
                } else {
                    None
                };
                let axes: Vec<usize> = node
                    .attr_ints("axes")
                    .map(|a| a.iter().filter_map(|&v| norm_axis(v, r)).collect())
                    .unwrap_or_else(|| (0..r).collect());
                let mut out = x.to_vec();
                if let Some(sz) = sizes.filter(|v| v.len() == axes.len()) {
                    for (k, &ax) in axes.iter().enumerate() {
                        out[ax] = sz[k].clone();
                    }
                    s.shape = Some(out);
                } else if let Some(sc) = scales.filter(|v| v.len() == axes.len() && !v.is_empty()) {
                    for (k, &ax) in axes.iter().enumerate() {
                        let f = sc[k];
                        out[ax] = match &x[ax] {
                            Dim::Known(d) => Dim::Known(((*d as f64) * f).floor() as i64),
                            d if f == 1.0 => d.clone(),
                            d if f.fract() == 0.0 && f > 0.0 => shape::mul(d, &Dim::Known(f as i64)),
                            _ => Dim::Unknown,
                        };
                    }
                    s.shape = Some(out);
                } else {
                    s.shape = Some(vec![Dim::Unknown; r]);
                }
            }
            vec![s]
        }
        "DepthToSpace" | "SpaceToDepth" => {
            let b = node.attr_i("blocksize").unwrap_or(1).max(1);
            let mut s = VState::new(x_dt, None);
            if let Some(x) = x {
                if x.len() == 4 {
                    let bb = Dim::Known(b * b);
                    let bd = Dim::Known(b);
                    s.shape = Some(if op == "DepthToSpace" {
                        vec![x[0].clone(), shape::div(&x[1], &bb), shape::mul(&x[2], &bd), shape::mul(&x[3], &bd)]
                    } else {
                        vec![x[0].clone(), shape::mul(&x[1], &bb), shape::div(&x[2], &bd), shape::div(&x[3], &bd)]
                    });
                }
            }
            vec![s]
        }
        "ReduceL1" | "ReduceL2" | "ReduceLogSum" | "ReduceLogSumExp" | "ReduceMax"
        | "ReduceMean" | "ReduceMin" | "ReduceProd" | "ReduceSum" | "ReduceSumSquare" => {
            let keep = node.attr_i("keepdims").unwrap_or(1) == 1;
            let noop = node.attr_i("noop_with_empty_axes").unwrap_or(0) == 1;
            let axes: Option<Vec<i64>> = match node.attr_ints("axes") {
                Some(a) => Some(a.to_vec()),
                None => {
                    if node.input(1).is_some() {
                        match ints(ins, 1) {
                            Some(a) => Some(a),
                            None => {
                                return vec![match x {
                                    Some(x) if keep => unknown_rank(x_dt, x.len()),
                                    _ => VState::new(x_dt, None),
                                }]
                            }
                        }
                    } else {
                        None
                    }
                }
            };
            let mut s = VState::new(x_dt, x.and_then(|x| reduce_shape(x, axes.as_deref(), keep, noop)));
            if (op == "ReduceProd" || op == "ReduceSum") && axes.is_none() {
                if let Some(v) = sym(ins, 0) {
                    let f: fn(&Dim, &Dim) -> Dim = if op == "ReduceProd" { shape::mul } else { shape::add };
                    let init = if op == "ReduceProd" { Dim::Known(1) } else { Dim::Known(0) };
                    s.ints = Some(vec![v.iter().fold(init, |acc, d| f(&acc, d))]);
                }
            }
            vec![s]
        }
        "ArgMax" | "ArgMin" => {
            let keep = node.attr_i("keepdims").unwrap_or(1) == 1;
            let axis = node.attr_i("axis").unwrap_or(0);
            vec![VState::new(
                dtype::INT64,
                x.and_then(|x| reduce_shape(x, Some(&[axis]), keep, false)),
            )]
        }
        "TopK" => {
            let axis = node.attr_i("axis").unwrap_or(-1);
            let k: Dim = match node.attr_i("k") {
                Some(k) => Dim::Known(k),
                None => sym(ins, 1).and_then(|v| v.first().cloned()).unwrap_or(Dim::Unknown),
            };
            let shape = x.and_then(|x| {
                let ax = norm_axis(axis, x.len())?;
                let mut out = x.to_vec();
                out[ax] = k.clone();
                Some(out)
            });
            vec![VState::new(x_dt, shape.clone()), VState::new(dtype::INT64, shape)]
        }
        "MatMul" | "MatMulInteger" | "MatMulInteger16" | "FusedMatMul" | "QLinearMatMul"
        | "MatMulIntegerToFloat" | "DynamicQuantizeMatMul" => {
            let (a, b) = if op == "QLinearMatMul" { (x, shp(ins, 3)) } else { (x, shp(ins, 1)) };
            let (a, b): (Option<Vec<Dim>>, Option<Vec<Dim>>) = if op == "FusedMatMul" {
                let ta = node.attr_i("transA").unwrap_or(0) == 1;
                let tb = node.attr_i("transB").unwrap_or(0) == 1;
                let swap = |s: &[Dim], t: bool| -> Vec<Dim> {
                    let mut v = s.to_vec();
                    let n = v.len();
                    if t && n >= 2 {
                        v.swap(n - 1, n - 2);
                    }
                    v
                };
                (a.map(|s| swap(s, ta)), b.map(|s| swap(s, tb)))
            } else {
                (a.map(|s| s.to_vec()), b.map(|s| s.to_vec()))
            };
            let dtype = match op {
                "MatMulInteger" | "MatMulInteger16" => dtype::INT32,
                "QLinearMatMul" => {
                    let d = dt(ins, 7);
                    if d != 0 { d } else { dtype::UINT8 }
                }
                "MatMulIntegerToFloat" | "DynamicQuantizeMatMul" => {
                    let d = dt(ins, 2);
                    if d != 0 { d } else { dtype::FLOAT }
                }
                _ => x_dt,
            };
            let shape = match (a, b) {
                (Some(a), Some(b)) => matmul_shape(&a, &b),
                _ => None,
            };
            vec![VState::new(dtype, shape)]
        }
        "MatMulNBits" => {
            let n = node.attr_i("N").map(Dim::Known).unwrap_or(Dim::Unknown);
            vec![VState::new(
                x_dt,
                x.map(|x| {
                    let mut v = x.to_vec();
                    if let Some(last) = v.last_mut() {
                        *last = n;
                    }
                    v
                }),
            )]
        }
        "Gemm" | "FusedGemm" | "QGemm" => {
            let (ai, bi) = if op == "QGemm" { (0, 3) } else { (0, 1) };
            let ta = node.attr_i("transA").unwrap_or(0) == 1;
            let tb = node.attr_i("transB").unwrap_or(0) == 1;
            let m = shp(ins, ai).and_then(|a| if a.len() == 2 { Some(a[if ta { 1 } else { 0 }].clone()) } else { None });
            let n = shp(ins, bi).and_then(|b| if b.len() == 2 { Some(b[if tb { 0 } else { 1 }].clone()) } else { None });
            let dtype = if op == "QGemm" && node.input(8).is_some() { dt(ins, 8) } else { dt(ins, ai) };
            vec![VState::new(
                dtype,
                Some(vec![m.unwrap_or(Dim::Unknown), n.unwrap_or(Dim::Unknown)]),
            )]
        }
        "Conv" | "FusedConv" => vec![conv_like(node, x, shp(ins, 1), x_dt)],
        "ConvInteger" => vec![conv_like(node, x, shp(ins, 1), dtype::INT32)],
        "QLinearConv" => {
            let d = dt(ins, 7);
            vec![conv_like(node, x, shp(ins, 3), if d != 0 { d } else { dtype::UINT8 })]
        }
        "DeformConv" => vec![conv_like(node, x, shp(ins, 1), x_dt)],
        "ConvTranspose" => vec![conv_transpose(node, x, shp(ins, 1), x_dt)],
        "MaxPool" => {
            let p = pool(node, x, x_dt);
            let idx = VState::new(dtype::INT64, p.shape.clone());
            vec![p, idx]
        }
        "AveragePool" | "LpPool" | "QLinearAveragePool" => vec![pool(node, x, x_dt)],
        "GlobalAveragePool" | "GlobalMaxPool" | "GlobalLpPool" | "QLinearGlobalAveragePool" => {
            vec![global_pool(x, x_dt)]
        }
        "MaxRoiPool" | "RoiAlign" => {
            let rois = shp(ins, 1).and_then(|r| r.first().cloned()).unwrap_or(Dim::Unknown);
            let c = x.and_then(|x| x.get(1).cloned()).unwrap_or(Dim::Unknown);
            let (h, w) = if op == "MaxRoiPool" {
                let p = node.attr_ints("pooled_shape").unwrap_or(&[]);
                (
                    p.first().map(|&v| Dim::Known(v)).unwrap_or(Dim::Unknown),
                    p.get(1).map(|&v| Dim::Known(v)).unwrap_or(Dim::Unknown),
                )
            } else {
                (
                    Dim::Known(node.attr_i("output_height").unwrap_or(1)),
                    Dim::Known(node.attr_i("output_width").unwrap_or(1)),
                )
            };
            vec![VState::new(x_dt, Some(vec![rois, c, h, w]))]
        }
        "GridSample" => {
            let mut s = VState::new(x_dt, None);
            if let (Some(x), Some(g)) = (x, shp(ins, 1)) {
                if x.len() >= 3 && g.len() == x.len() {
                    let mut out = vec![x[0].clone(), x[1].clone()];
                    out.extend(g[1..g.len() - 1].iter().cloned());
                    s.shape = Some(out);
                }
            }
            vec![s]
        }
        "AffineGrid" => {
            let mut s = VState::new(dt(ins, 0), None);
            if let Some(sz) = sym(ins, 1) {
                if sz.len() >= 3 {
                    let mut out = vec![sz[0].clone()];
                    out.extend(sz[2..].iter().cloned());
                    out.push(Dim::Known((sz.len() - 2) as i64));
                    s.shape = Some(out);
                }
            }
            vec![s]
        }
        "BatchNormalization" => {
            let c = x.and_then(|x| x.get(1).cloned()).unwrap_or(Dim::Unknown);
            vec![
                same_as(ins, 0),
                VState::new(dt(ins, 3), Some(vec![c.clone()])),
                VState::new(dt(ins, 4), Some(vec![c])),
            ]
        }
        "LayerNormalization" | "SimplifiedLayerNormalization" => {
            let axis = node.attr_i("axis").unwrap_or(-1);
            let stash = node.attr_i("stash_type").unwrap_or(1) as i32;
            let stat = x.and_then(|x| {
                let ax = norm_axis(axis, x.len())?;
                let mut v = x.to_vec();
                for d in v.iter_mut().skip(ax) {
                    *d = Dim::Known(1);
                }
                Some(v)
            });
            let mut outs = vec![same_as(ins, 0)];
            if op == "LayerNormalization" {
                outs.push(VState::new(stash, stat.clone()));
            }
            outs.push(VState::new(stash, stat));
            outs
        }
        "SkipLayerNormalization" | "SkipSimplifiedLayerNormalization" => {
            let stat = x.map(|x| {
                let mut v = x.to_vec();
                if let Some(l) = v.last_mut() {
                    *l = Dim::Known(1);
                }
                v
            });
            vec![
                same_as(ins, 0),
                VState::new(dtype::FLOAT, stat.clone()),
                VState::new(dtype::FLOAT, stat),
                same_as(ins, 0),
            ]
        }
        "EmbedLayerNormalization" => {
            let hidden = shp(ins, 2).and_then(|w| w.get(1).cloned()).unwrap_or(Dim::Unknown);
            let (b, sq) = match x {
                Some(x) if x.len() == 2 => (x[0].clone(), x[1].clone()),
                _ => (Dim::Unknown, Dim::Unknown),
            };
            vec![
                VState::new(dt(ins, 2), Some(vec![b.clone(), sq, hidden])),
                VState::new(dtype::INT32, Some(vec![b])),
            ]
        }
        "LSTM" | "GRU" | "RNN" => {
            let layout = node.attr_i("layout").unwrap_or(0);
            let dirs: i64 = if node.attr_s("direction").as_deref() == Some("bidirectional") { 2 } else { 1 };
            let h = node
                .attr_i("hidden_size")
                .map(Dim::Known)
                .or_else(|| shp(ins, 2).and_then(|r| r.get(2).cloned()))
                .unwrap_or(Dim::Unknown);
            let Some(x) = x.filter(|x| x.len() == 3) else {
                return vec![VState::new(x_dt, None); node.outputs.len()];
            };
            let (seq, batch) = if layout == 0 { (x[0].clone(), x[1].clone()) } else { (x[1].clone(), x[0].clone()) };
            let d = Dim::Known(dirs);
            let y = if layout == 0 {
                vec![seq, d.clone(), batch.clone(), h.clone()]
            } else {
                vec![batch.clone(), seq, d.clone(), h.clone()]
            };
            let yh = if layout == 0 { vec![d, batch, h] } else { vec![batch, d, h] };
            let mut outs = vec![VState::new(x_dt, Some(y)), VState::new(x_dt, Some(yh.clone()))];
            if op == "LSTM" {
                outs.push(VState::new(x_dt, Some(yh)));
            }
            outs
        }
        "Einsum" => {
            let eq = node.attr_s("equation").unwrap_or_default();
            let shapes: Vec<Option<&[Dim]>> = (0..node.inputs.len()).map(|i| shp(ins, i)).collect();
            vec![VState::new(x_dt, einsum_shape(&eq, &shapes))]
        }
        "NonZero" => vec![VState::new(
            dtype::INT64,
            x.map(|x| vec![Dim::Known(x.len() as i64), Dim::Unknown]),
        )],
        "NonMaxSuppression" => vec![VState::new(dtype::INT64, Some(vec![Dim::Unknown, Dim::Known(3)]))],
        "Compress" => {
            let s = match (x, node.attr_i("axis")) {
                (Some(x), Some(axis)) => norm_axis(axis, x.len()).map(|ax| {
                    let mut v = x.to_vec();
                    v[ax] = Dim::Unknown;
                    v
                }),
                _ => Some(vec![Dim::Unknown]),
            };
            vec![VState::new(x_dt, s)]
        }
        "Unique" => {
            let y = match (x, node.attr_i("axis")) {
                (Some(x), Some(axis)) => norm_axis(axis, x.len()).map(|ax| {
                    let mut v = x.to_vec();
                    v[ax] = Dim::Unknown;
                    v
                }),
                _ => Some(vec![Dim::Unknown]),
            };
            vec![
                VState::new(x_dt, y),
                VState::new(dtype::INT64, Some(vec![Dim::Unknown])),
                VState::new(dtype::INT64, Some(vec![Dim::Unknown])),
                VState::new(dtype::INT64, Some(vec![Dim::Unknown])),
            ]
        }
        "OneHot" => {
            let depth = sym(ins, 1).and_then(|v| v.first().cloned()).unwrap_or(Dim::Unknown);
            let axis = node.attr_i("axis").unwrap_or(-1);
            let s = x.and_then(|x| {
                let r = x.len() + 1;
                let ax = norm_axis(axis, r)?;
                let mut v = x.to_vec();
                v.insert(ax, depth.clone());
                Some(v)
            });
            vec![VState::new(dt(ins, 2), s)]
        }
        "QuantizeLinear" => {
            let d = if node.input(2).is_some() && dt(ins, 2) != 0 {
                dt(ins, 2)
            } else {
                match node.attr_i("output_dtype") {
                    Some(v) if v != 0 => v as i32,
                    _ => dtype::UINT8,
                }
            };
            vec![VState::new(d, x.map(|v| v.to_vec()))]
        }
        "DequantizeLinear" => {
            let d = match node.attr_i("output_dtype") {
                Some(v) if v != 0 => v as i32,
                _ => {
                    let s = dt(ins, 1);
                    if s != 0 { s } else { dtype::FLOAT }
                }
            };
            vec![VState::new(d, x.map(|v| v.to_vec()))]
        }
        "DynamicQuantizeLinear" => vec![
            VState::new(dtype::UINT8, x.map(|v| v.to_vec())),
            VState::new(dtype::FLOAT, Some(Vec::new())),
            VState::new(dtype::UINT8, Some(Vec::new())),
        ],
        "QLinearAdd" | "QLinearMul" => {
            let d = dt(ins, 7);
            vec![VState::new(if d != 0 { d } else { x_dt }, broadcast_inputs(ins, &[0, 3]))]
        }
        "QLinearConcat" => {
            let axis = node.attr_i("axis").unwrap_or(0);
            let mut idx = 2;
            let mut shapes = Vec::new();
            while idx < node.inputs.len() {
                shapes.push(shp(ins, idx));
                idx += 3;
            }
            let d = dt(ins, 1);
            let mut s = VState::new(if d != 0 { d } else { dt(ins, 2) }, None);
            if let Some(Some(first)) = shapes.first() {
                if let Some(ax) = norm_axis(axis, first.len()) {
                    let mut out = first.to_vec();
                    let mut total = Dim::Known(0);
                    for sh in &shapes {
                        total = match sh {
                            Some(sh) if sh.len() == out.len() => shape::add(&total, &sh[ax]),
                            _ => Dim::Unknown,
                        };
                    }
                    out[ax] = total;
                    s.shape = Some(out);
                }
            }
            vec![s]
        }
        "Attention" if ms => {
            let w = shp(ins, 1);
            let hidden = match node.attr_ints("qkv_hidden_sizes") {
                Some(v) if v.len() == 3 => Dim::Known(v[2]),
                _ => w
                    .and_then(|w| w.get(1))
                    .map(|d| shape::div(d, &Dim::Known(3)))
                    .unwrap_or(Dim::Unknown),
            };
            let out = x.filter(|x| x.len() == 3).map(|x| vec![x[0].clone(), x[1].clone(), hidden]);
            vec![VState::new(x_dt, out)]
        }
        "Attention" => {
            let q = x;
            let v = shp(ins, 2);
            let out = match (q, v) {
                (Some(q), Some(v)) if q.len() == 4 && v.len() == 4 => {
                    Some(vec![q[0].clone(), q[1].clone(), q[2].clone(), v[3].clone()])
                }
                (Some(q), Some(v)) if q.len() == 3 && v.len() == 3 => {
                    let qh = node.attr_i("q_num_heads");
                    let kvh = node.attr_i("kv_num_heads");
                    let last = match (qh, kvh) {
                        (Some(qh), Some(kvh)) if kvh > 0 => {
                            shape::mul(&shape::div(&v[2], &Dim::Known(kvh)), &Dim::Known(qh))
                        }
                        _ => Dim::Unknown,
                    };
                    Some(vec![q[0].clone(), q[1].clone(), last])
                }
                _ => None,
            };
            vec![VState::new(x_dt, out)]
        }
        "MultiHeadAttention" | "GroupQueryAttention" => {
            let out = x.filter(|x| x.len() == 3).map(|x| {
                let mut v = x.to_vec();
                if op == "MultiHeadAttention" {
                    if let Some(val) = shp(ins, 2).filter(|v| v.len() == 3) {
                        v[2] = val[2].clone();
                    }
                } else if node.input(1).is_none() {
                    v[2] = Dim::Unknown;
                }
                v
            });
            vec![VState::new(x_dt, out)]
        }
        "If" => {
            let outs = subgraph_outputs(ctx, node, "then_branch");
            if outs.iter().all(|s| s.shape.is_none() && s.dtype == 0) {
                subgraph_outputs(ctx, node, "else_branch")
            } else {
                outs
            }
        }
        "Loop" => {
            let body = subgraph_outputs(ctx, node, "body");
            let ncarried = node.inputs.len().saturating_sub(2);
            let mut outs = Vec::new();
            for i in 0..node.outputs.len() {
                if i < ncarried {
                    let b = body.get(1 + i).cloned().unwrap_or_default();
                    if b.shape.is_some() {
                        outs.push(b);
                    } else {
                        outs.push(same_as(ins, 2 + i));
                    }
                } else {
                    let b = body.get(1 + i).cloned().unwrap_or_default();
                    let shape = b.shape.map(|s| {
                        let mut v = vec![Dim::Unknown];
                        v.extend(s);
                        v
                    });
                    outs.push(VState::new(b.dtype, shape));
                }
            }
            outs
        }
        "Scan" => {
            let body = subgraph_outputs(ctx, node, "body");
            body.into_iter()
                .map(|b| VState::new(b.dtype, None))
                .collect()
        }
        "EyeLike" | "RandomNormalLike" | "RandomUniformLike" | "Bernoulli" => {
            let d = node.attr_i("dtype").map(|v| v as i32).unwrap_or(x_dt);
            vec![VState::new(d, x.map(|v| v.to_vec()))]
        }
        "RandomNormal" | "RandomUniform" => {
            let d = node.attr_i("dtype").map(|v| v as i32).unwrap_or(dtype::FLOAT);
            vec![VState::new(d, node.attr_ints("shape").map(shape::from_i64s))]
        }
        "Multinomial" => {
            let d = node.attr_i("dtype").map(|v| v as i32).unwrap_or(dtype::INT32);
            let n = node.attr_i("sample_size").unwrap_or(1);
            vec![VState::new(
                d,
                x.and_then(|x| x.first().cloned()).map(|b| vec![b, Dim::Known(n)]),
            )]
        }
        "Det" => vec![VState::new(
            x_dt,
            x.filter(|x| x.len() >= 2).map(|x| x[..x.len() - 2].to_vec()),
        )],
        "CenterCropPad" => {
            let mut s = VState::new(x_dt, None);
            if let (Some(x), Some(sz)) = (x, sym(ins, 1)) {
                let r = x.len();
                let axes: Vec<usize> = node
                    .attr_ints("axes")
                    .map(|a| a.iter().filter_map(|&v| norm_axis(v, r)).collect())
                    .unwrap_or_else(|| (0..r).collect());
                if axes.len() == sz.len() {
                    let mut out = x.to_vec();
                    for (k, &ax) in axes.iter().enumerate() {
                        out[ax] = sz[k].clone();
                    }
                    s.shape = Some(out);
                }
            }
            vec![s]
        }
        "MaxUnpool" => {
            if let Some(osh) = sym(ins, 2) {
                return vec![VState::new(x_dt, Some(osh.to_vec()))];
            }
            let mut s = VState::new(x_dt, None);
            if let Some(x) = x.filter(|x| x.len() >= 3) {
                let k = node.attr_ints("kernel_shape").unwrap_or(&[]);
                let strides = node.attr_ints("strides").unwrap_or(&[]);
                let pads = node.attr_ints("pads").unwrap_or(&[]);
                let r = x.len() - 2;
                let mut out = vec![x[0].clone(), x[1].clone()];
                for i in 0..r {
                    out.push(match x[2 + i].known() {
                        Some(d) => {
                            let st = strides.get(i).copied().unwrap_or(1);
                            let kk = k.get(i).copied().unwrap_or(1);
                            let p = pads.get(i).copied().unwrap_or(0) + pads.get(i + r).copied().unwrap_or(0);
                            Dim::Known((d - 1) * st + kk - p)
                        }
                        None => Dim::Unknown,
                    });
                }
                s.shape = Some(out);
            }
            vec![s]
        }
        "SequenceConstruct" | "SequenceEmpty" | "SequenceInsert" | "SequenceErase"
        | "SplitToSequence" | "Optional" => vec![VState::new(x_dt, None)],
        "SequenceAt" | "OptionalGetElement" | "ConcatFromSequence" => vec![VState::new(x_dt, None)],
        "SequenceLength" => vec![VState::new(dtype::INT64, Some(Vec::new()))],
        "OptionalHasElement" => vec![VState::new(dtype::BOOL, Some(Vec::new()))],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::from_i64s;

    #[test]
    fn reshape_rules() {
        let x = from_i64s(&[2, 3, 4]);
        assert_eq!(reshape(Some(&x[..]), &from_i64s(&[0, -1]), false), Some(from_i64s(&[2, 12])));
        let n = Dim::Param("N".into());
        let x = vec![n.clone(), Dim::Known(512), Dim::Known(1), Dim::Known(1)];
        assert_eq!(
            reshape(Some(&x[..]), &[n.clone(), Dim::Known(-1)], false),
            Some(vec![n.clone(), Dim::Known(512)])
        );
        let x = vec![Dim::Known(1), Dim::Param("seq".into()), Dim::Known(768)];
        assert_eq!(
            reshape(
                Some(&x[..]),
                &[Dim::Known(1), Dim::Param("seq".into()), Dim::Known(12), Dim::Known(-1)],
                false
            ),
            Some(vec![Dim::Known(1), Dim::Param("seq".into()), Dim::Known(12), Dim::Known(64)])
        );
    }

    #[test]
    fn matmul_rules() {
        assert_eq!(
            matmul_shape(&from_i64s(&[2, 3, 4]), &from_i64s(&[4, 5])),
            Some(from_i64s(&[2, 3, 5]))
        );
        assert_eq!(matmul_shape(&from_i64s(&[4]), &from_i64s(&[4, 5])), Some(from_i64s(&[5])));
        assert_eq!(matmul_shape(&from_i64s(&[3, 4]), &from_i64s(&[4])), Some(from_i64s(&[3])));
    }

    #[test]
    fn slice_rules() {
        assert_eq!(slice_dim(&Dim::Known(10), 2, 8, 2), Dim::Known(3));
        assert_eq!(slice_dim(&Dim::Known(10), -1, i64::MIN, -1), Dim::Known(10));
        assert_eq!(slice_dim(&Dim::Known(10), 0, i64::MAX, 1), Dim::Known(10));
        let n = Dim::Param("N".into());
        assert_eq!(slice_dim(&n, 0, i64::MAX, 1), n);
    }

    #[test]
    fn einsum_rules() {
        let a = from_i64s(&[2, 3, 4]);
        let b = from_i64s(&[2, 4, 5]);
        assert_eq!(
            einsum_shape("bij,bjk->bik", &[Some(&a[..]), Some(&b[..])]),
            Some(from_i64s(&[2, 3, 5]))
        );
        let c = from_i64s(&[7, 3, 4]);
        assert_eq!(einsum_shape("...ij->...ji", &[Some(&c[..])]), Some(from_i64s(&[7, 4, 3])));
        let d = from_i64s(&[3, 4]);
        let e = from_i64s(&[4, 6]);
        assert_eq!(einsum_shape("ij,jk", &[Some(&d[..]), Some(&e[..])]), Some(from_i64s(&[3, 6])));
    }
}
