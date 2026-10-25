//! Per-node compute cost estimates (MACs / FLOPs), parameter counts and
//! memory traffic. All numbers are estimates for a single inference.

use crate::dtype;
use crate::infer::VState;
use crate::model::{Graph, Node, ValueKind};
use crate::shape::{numel, Dim};
use crate::tensor;
use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeCost {
    pub params: u64,
    pub param_bytes: u64,
    pub macs: Option<u64>,
    pub flops: Option<u64>,
    pub read_bytes: Option<u64>,
    pub write_bytes: Option<u64>,
}

fn state_numel(st: &VState) -> Option<u64> {
    st.shape.as_deref().and_then(numel)
}

fn state_bytes(st: &VState) -> Option<u64> {
    dtype::bytes_for(st.dtype, state_numel(st)?)
}

fn dimv(d: &Dim) -> Option<u64> {
    d.known().filter(|&v| v >= 0).map(|v| v as u64)
}

/// FLOPs per output element for simple element-wise style operators.
fn elementwise_factor(op: &str) -> Option<u64> {
    Some(match op {
        "Add" | "Sub" | "Mul" | "Div" | "Neg" | "Abs" | "Relu" | "LeakyRelu" | "PRelu" | "Max"
        | "Min" | "Sum" | "Mean" | "Clip" | "Sign" | "Floor" | "Ceil" | "Round" | "Reciprocal"
        | "Equal" | "Less" | "Greater" | "LessOrEqual" | "GreaterOrEqual" | "And" | "Or"
        | "Xor" | "Not" | "Where" | "Mod" | "BitShift" | "BitwiseAnd" | "BitwiseOr"
        | "BitwiseXor" | "BitwiseNot" | "ThresholdedRelu" | "Shrink" | "IsNaN" | "IsInf"
        | "CumSum" | "BiasAdd" | "Pow" | "Sqrt" => 1,
        "HardSigmoid" | "Elu" | "Selu" | "Celu" | "Softsign" | "QuantizeLinear"
        | "DequantizeLinear" | "BatchNormalization" => 2,
        "HardSwish" | "LpNormalization" => 3,
        "Sigmoid" | "Tanh" | "Exp" | "Log" | "Erf" | "Softplus" | "Sin" | "Cos" | "Tan"
        | "Asin" | "Acos" | "Atan" | "Sinh" | "Cosh" | "Asinh" | "Acosh" | "Atanh" | "RMSNormalization"
        | "SimplifiedLayerNormalization" => 4,
        "Swish" | "QuickGelu" | "Softmax" | "LogSoftmax" | "Hardmax" | "InstanceNormalization"
        | "LayerNormalization" | "GroupNormalization" | "MeanVarianceNormalization"
        | "SkipLayerNormalization" | "SkipSimplifiedLayerNormalization" | "BiasSoftmax" => 5,
        "Mish" => 6,
        "Gelu" | "FastGelu" | "BiasGelu" => 8,
        "DynamicQuantizeLinear" => 4,
        "QLinearSigmoid" | "QLinearLeakyRelu" | "QLinearAdd" | "QLinearMul" | "QLinearSoftmax" => 4,
        _ => return None,
    })
}

/// Operators that only move or reinterpret data.
fn is_data_movement(op: &str) -> bool {
    matches!(
        op,
        "Reshape" | "Transpose" | "Concat" | "Gather" | "GatherElements" | "GatherND" | "Slice"
            | "Split" | "Squeeze" | "Unsqueeze" | "Flatten" | "Expand" | "Tile" | "Pad"
            | "Identity" | "Shape" | "Size" | "Cast" | "CastLike" | "Constant" | "ConstantOfShape"
            | "Dropout" | "DepthToSpace" | "SpaceToDepth" | "ScatterND" | "ScatterElements"
            | "Scatter" | "Range" | "NonZero" | "Compress" | "OneHot" | "EyeLike" | "Trilu"
            | "ReverseSequence" | "SequenceAt" | "SequenceConstruct" | "SequenceInsert"
            | "SequenceEmpty" | "SequenceLength" | "SplitToSequence" | "ConcatFromSequence"
            | "QLinearConcat" | "Optional" | "OptionalGetElement" | "OptionalHasElement"
            | "TensorScatter" | "CenterCropPad"
    )
}

fn einsum_macs(eq: &str, shapes: &[Option<&[Dim]>]) -> Option<u64> {
    if eq.contains("...") {
        return None;
    }
    let lhs = eq.split("->").next()?;
    let mut sizes: std::collections::HashMap<char, u64> = std::collections::HashMap::new();
    for (term, s) in lhs.split(',').zip(shapes) {
        let s = (*s)?;
        let letters: Vec<char> = term.chars().filter(|c| c.is_alphabetic()).collect();
        if letters.len() != s.len() {
            return None;
        }
        for (c, d) in letters.into_iter().zip(s) {
            let v = dimv(d)?;
            let e = sizes.entry(c).or_insert(v);
            if *e == 1 {
                *e = v;
            }
        }
    }
    Some(sizes.values().product())
}

pub fn node_cost(g: &Graph, node: &Node, st: &[VState]) -> NodeCost {
    let mut c = NodeCost::default();
    let mut seen: Vec<usize> = Vec::new();
    for &vid in node.inputs.iter().flatten() {
        let v = &g.values[vid];
        if !matches!(v.kind, ValueKind::Initializer | ValueKind::Constant) || seen.contains(&vid) {
            continue;
        }
        seen.push(vid);
        if let Some(ti) = v.tensor {
            let t = &g.tensors[ti];
            if dtype::is_weight_type(t.data_type) {
                c.params += tensor::element_count(t);
                c.param_bytes += tensor::byte_size(t).unwrap_or(0);
            }
        } else if let Some(sp) = &v.sparse {
            if dtype::is_weight_type(sp.dtype) {
                c.params += sp.nnz;
            }
        }
    }

    // Memory traffic.
    let mut read: Option<u64> = Some(0);
    for &vid in node.inputs.iter().flatten() {
        read = match (read, state_bytes(&st[vid])) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }
    let mut write: Option<u64> = Some(0);
    for &vid in node.outputs.iter().flatten() {
        write = match (write, state_bytes(&st[vid])) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }
    c.read_bytes = read;
    c.write_bytes = write;

    if !(node.is_standard() || node.domain == "com.microsoft") {
        return c;
    }
    let op = node.op_type.as_str();
    let shape = |i: usize| node.input(i).and_then(|v| st[v].shape.as_deref());
    let out0 = node.output(0).and_then(|v| state_numel(&st[v]));
    let in0 = node.input(0).and_then(|v| state_numel(&st[v]));

    let (macs, flops): (Option<u64>, Option<u64>) = match op {
        "Conv" | "ConvInteger" | "FusedConv" | "DeformConv" | "QLinearConv" => {
            let (wi, bi) = if op == "QLinearConv" { (3, 8) } else { (1, 2) };
            match (out0, shape(wi)) {
                (Some(on), Some(w)) if w.len() >= 2 => match numel(&w[1..]) {
                    Some(k) => {
                        let m = on.saturating_mul(k);
                        let bias = if node.input(bi).is_some() { on } else { 0 };
                        (Some(m), Some(2 * m + bias))
                    }
                    None => (None, None),
                },
                _ => (None, None),
            }
        }
        "ConvTranspose" => match (in0, shape(1)) {
            (Some(xn), Some(w)) if w.len() >= 2 => match numel(&w[1..]) {
                Some(k) => {
                    let m = xn.saturating_mul(k);
                    let bias = if node.input(2).is_some() { out0.unwrap_or(0) } else { 0 };
                    (Some(m), Some(2 * m + bias))
                }
                None => (None, None),
            },
            _ => (None, None),
        },
        "MatMul" | "MatMulInteger" | "MatMulInteger16" | "QLinearMatMul" | "FusedMatMul"
        | "MatMulIntegerToFloat" | "DynamicQuantizeMatMul" => {
            let a = shape(0);
            let ta = op == "FusedMatMul" && node.attr_i("transA").unwrap_or(0) == 1;
            let k = a.and_then(|a| match a.len() {
                0 => None,
                1 => dimv(&a[0]),
                n => dimv(&a[if ta { n - 2 } else { n - 1 }]),
            });
            match (out0, k) {
                (Some(on), Some(k)) => {
                    let m = on.saturating_mul(k);
                    (Some(m), Some(2 * m))
                }
                _ => (None, None),
            }
        }
        "MatMulNBits" => match (out0, node.attr_i("K")) {
            (Some(on), Some(k)) if k > 0 => {
                let m = on.saturating_mul(k as u64);
                (Some(m), Some(2 * m))
            }
            _ => (None, None),
        },
        "Gemm" | "FusedGemm" | "QGemm" => {
            let ta = node.attr_i("transA").unwrap_or(0) == 1;
            let k = shape(0).and_then(|a| if a.len() == 2 { dimv(&a[if ta { 0 } else { 1 }]) } else { None });
            match (out0, k) {
                (Some(on), Some(k)) => {
                    let m = on.saturating_mul(k);
                    let ci = if op == "QGemm" { 6 } else { 2 };
                    let bias = if node.input(ci).is_some() { on } else { 0 };
                    (Some(m), Some(2 * m + bias))
                }
                _ => (None, None),
            }
        }
        "Einsum" => {
            let eq = node.attr_s("equation").unwrap_or_default();
            let shapes: Vec<Option<&[Dim]>> = (0..node.inputs.len()).map(&shape).collect();
            match einsum_macs(&eq.replace(' ', ""), &shapes) {
                Some(m) => (Some(m), Some(2 * m)),
                None => (None, None),
            }
        }
        "LSTM" | "GRU" | "RNN" => {
            let gates: u64 = match op {
                "LSTM" => 4,
                "GRU" => 3,
                _ => 1,
            };
            let layout = node.attr_i("layout").unwrap_or(0);
            let dirs: u64 = if node.attr_s("direction").as_deref() == Some("bidirectional") { 2 } else { 1 };
            let h = node
                .attr_i("hidden_size")
                .map(|v| v as u64)
                .or_else(|| shape(2).and_then(|r| r.get(2)).and_then(dimv));
            match (shape(0), h) {
                (Some(x), Some(h)) if x.len() == 3 => {
                    let (s, b) = if layout == 0 { (&x[0], &x[1]) } else { (&x[1], &x[0]) };
                    match (dimv(s), dimv(b), dimv(&x[2])) {
                        (Some(s), Some(b), Some(i)) => {
                            let m = s * b * dirs * gates * h * (i + h);
                            (Some(m), Some(2 * m + s * b * dirs * h * gates * 4))
                        }
                        _ => (None, None),
                    }
                }
                _ => (None, None),
            }
        }
        "Attention" if node.domain == "com.microsoft" => match (shape(0), shape(1)) {
            (Some(x), Some(w)) if x.len() == 3 && w.len() == 2 => {
                match (dimv(&x[0]), dimv(&x[1]), dimv(&w[0]), dimv(&w[1])) {
                    (Some(b), Some(s), Some(d), Some(d3)) => {
                        let proj = b * s * d * d3;
                        let att = 2 * b * s * s * (d3 / 3);
                        (Some(proj + att), Some(2 * (proj + att)))
                    }
                    _ => (None, None),
                }
            }
            _ => (None, None),
        },
        "Attention" => match (shape(0), shape(1), shape(2)) {
            (Some(q), Some(k), Some(v)) if q.len() == 4 && k.len() == 4 && v.len() == 4 => {
                match (numel(&q[..3]), dimv(&q[3]), dimv(&k[2]), dimv(&v[3])) {
                    (Some(bhs), Some(e), Some(l), Some(ev)) => {
                        let m = bhs * l * (e + ev);
                        (Some(m), Some(2 * m + 5 * bhs * l))
                    }
                    _ => (None, None),
                }
            }
            _ => (None, None),
        },
        "MultiHeadAttention" => match (shape(0), shape(1)) {
            (Some(q), Some(k)) if q.len() == 3 && k.len() == 3 => {
                match (dimv(&q[0]), dimv(&q[1]), dimv(&q[2]), dimv(&k[1])) {
                    (Some(b), Some(s), Some(d), Some(l)) => {
                        let m = 2 * b * s * l * d;
                        (Some(m), Some(2 * m))
                    }
                    _ => (None, None),
                }
            }
            _ => (None, None),
        },
        "MaxPool" | "AveragePool" | "LpPool" | "QLinearAveragePool" => {
            let k: u64 = node
                .attr_ints("kernel_shape")
                .map(|k| k.iter().map(|&v| v.max(1) as u64).product())
                .unwrap_or(1);
            (Some(0), out0.map(|o| o * k))
        }
        "GlobalAveragePool" | "GlobalMaxPool" | "GlobalLpPool" | "QLinearGlobalAveragePool"
        | "ReduceMax" | "ReduceMean" | "ReduceMin" | "ReduceProd" | "ReduceSum" | "ReduceL1"
        | "ArgMax" | "ArgMin" | "TopK" => (Some(0), in0),
        "ReduceL2" | "ReduceLogSum" | "ReduceLogSumExp" | "ReduceSumSquare" => (Some(0), in0.map(|v| 2 * v)),
        "LRN" => {
            let size = node.attr_i("size").unwrap_or(1).max(1) as u64;
            (Some(0), out0.map(|o| o * (size + 2)))
        }
        "Resize" | "Upsample" => {
            let mode = node.attr_s("mode").unwrap_or_else(|| "nearest".into());
            let f = match mode.as_str() {
                "linear" | "bilinear" => 6,
                "cubic" => 20,
                _ => 1,
            };
            (Some(0), out0.map(|o| o * f))
        }
        "GridSample" | "RoiAlign" => (Some(0), out0.map(|o| o * 8)),
        "Det" => (Some(0), shape(0).and_then(|x| {
            let n = dimv(x.last()?)?;
            let batch = numel(&x[..x.len().saturating_sub(2)])?;
            Some(batch * n * n * n)
        })),
        _ if is_data_movement(op) => (Some(0), Some(0)),
        _ => match elementwise_factor(op) {
            Some(f) => {
                let n = if matches!(op, "Sum" | "Mean" | "Max" | "Min") {
                    out0.map(|o| o * (node.inputs.len().max(2) as u64 - 1))
                } else {
                    out0
                };
                (Some(0), n.map(|v| v * f))
            }
            None => (None, None),
        },
    };
    c.macs = macs;
    c.flops = flops;
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::from_i64s;

    #[test]
    fn einsum_cost() {
        let a = from_i64s(&[2, 3, 4]);
        let b = from_i64s(&[2, 4, 5]);
        assert_eq!(einsum_macs("bij,bjk->bik", &[Some(&a[..]), Some(&b[..])]), Some(2 * 3 * 4 * 5));
    }
}
