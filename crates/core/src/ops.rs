//! Operator metadata: display category and input/output slot names for the
//! standard ONNX operators and common ONNX Runtime contrib operators.

use serde::Serialize;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    Layer,
    Activation,
    Pool,
    Normalization,
    Dropout,
    Shape,
    Tensor,
    Data,
    Quantization,
    Logic,
    Math,
    Reduce,
    Control,
    Sequence,
    Ml,
    Custom,
    Other,
}

impl Category {
    pub fn as_str(self) -> &'static str {
        match self {
            Category::Layer => "layer",
            Category::Activation => "activation",
            Category::Pool => "pool",
            Category::Normalization => "normalization",
            Category::Dropout => "dropout",
            Category::Shape => "shape",
            Category::Tensor => "tensor",
            Category::Data => "data",
            Category::Quantization => "quantization",
            Category::Logic => "logic",
            Category::Math => "math",
            Category::Reduce => "reduce",
            Category::Control => "control",
            Category::Sequence => "sequence",
            Category::Ml => "ml",
            Category::Custom => "custom",
            Category::Other => "other",
        }
    }
}

pub struct OpSchema {
    pub name: &'static str,
    pub category: Category,
    /// Input slot names. A trailing `*` marks a variadic slot.
    pub inputs: &'static [&'static str],
    pub outputs: &'static [&'static str],
}

macro_rules! op {
    ($name:literal, $cat:ident, [$($i:literal),*], [$($o:literal),*]) => {
        OpSchema { name: $name, category: Category::$cat, inputs: &[$($i),*], outputs: &[$($o),*] }
    };
}

static STANDARD: &[OpSchema] = &[
    op!("Abs", Math, ["X"], ["Y"]),
    op!("Acos", Math, ["input"], ["output"]),
    op!("Acosh", Math, ["input"], ["output"]),
    op!("Add", Math, ["A", "B"], ["C"]),
    op!("AffineGrid", Tensor, ["theta", "size"], ["grid"]),
    op!("And", Logic, ["A", "B"], ["C"]),
    op!("ArgMax", Reduce, ["data"], ["reduced"]),
    op!("ArgMin", Reduce, ["data"], ["reduced"]),
    op!("Asin", Math, ["input"], ["output"]),
    op!("Asinh", Math, ["input"], ["output"]),
    op!("Atan", Math, ["input"], ["output"]),
    op!("Atanh", Math, ["input"], ["output"]),
    op!("Attention", Layer, ["Q", "K", "V", "attn_mask", "past_key", "past_value", "nonpad_kv_seqlen"], ["Y", "present_key", "present_value", "qk_matmul_output"]),
    op!("AveragePool", Pool, ["X"], ["Y"]),
    op!("BatchNormalization", Normalization, ["X", "scale", "B", "input_mean", "input_var"], ["Y", "running_mean", "running_var"]),
    op!("Bernoulli", Data, ["input"], ["output"]),
    op!("BitShift", Logic, ["X", "Y"], ["Z"]),
    op!("BitwiseAnd", Logic, ["A", "B"], ["C"]),
    op!("BitwiseNot", Logic, ["X"], ["Y"]),
    op!("BitwiseOr", Logic, ["A", "B"], ["C"]),
    op!("BitwiseXor", Logic, ["A", "B"], ["C"]),
    op!("BlackmanWindow", Data, ["size"], ["output"]),
    op!("Cast", Data, ["input"], ["output"]),
    op!("CastLike", Data, ["input", "target_type"], ["output"]),
    op!("Ceil", Math, ["X"], ["Y"]),
    op!("Celu", Activation, ["X"], ["Y"]),
    op!("CenterCropPad", Shape, ["input_data", "shape"], ["output_data"]),
    op!("Clip", Activation, ["input", "min", "max"], ["output"]),
    op!("Col2Im", Shape, ["input", "image_shape", "block_shape"], ["output"]),
    op!("Compress", Tensor, ["input", "condition"], ["output"]),
    op!("Concat", Tensor, ["inputs*"], ["concat_result"]),
    op!("ConcatFromSequence", Sequence, ["input_sequence"], ["concat_result"]),
    op!("Constant", Data, [], ["output"]),
    op!("ConstantOfShape", Data, ["input"], ["output"]),
    op!("Conv", Layer, ["X", "W", "B"], ["Y"]),
    op!("ConvInteger", Layer, ["x", "w", "x_zero_point", "w_zero_point"], ["y"]),
    op!("ConvTranspose", Layer, ["X", "W", "B"], ["Y"]),
    op!("Cos", Math, ["input"], ["output"]),
    op!("Cosh", Math, ["input"], ["output"]),
    op!("CumSum", Math, ["x", "axis"], ["y"]),
    op!("DFT", Math, ["input", "dft_length", "axis"], ["output"]),
    op!("DeformConv", Layer, ["X", "W", "offset", "B", "mask"], ["Y"]),
    op!("DepthToSpace", Shape, ["input"], ["output"]),
    op!("DequantizeLinear", Quantization, ["x", "x_scale", "x_zero_point"], ["y"]),
    op!("Det", Math, ["X"], ["Y"]),
    op!("Div", Math, ["A", "B"], ["C"]),
    op!("Dropout", Dropout, ["data", "ratio", "training_mode"], ["output", "mask"]),
    op!("DynamicQuantizeLinear", Quantization, ["x"], ["y", "y_scale", "y_zero_point"]),
    op!("Einsum", Layer, ["Inputs*"], ["Output"]),
    op!("Elu", Activation, ["X"], ["Y"]),
    op!("Equal", Logic, ["A", "B"], ["C"]),
    op!("Erf", Math, ["input"], ["output"]),
    op!("Exp", Math, ["input"], ["output"]),
    op!("Expand", Shape, ["input", "shape"], ["output"]),
    op!("EyeLike", Data, ["input"], ["output"]),
    op!("Flatten", Shape, ["input"], ["output"]),
    op!("Floor", Math, ["X"], ["Y"]),
    op!("GRU", Layer, ["X", "W", "R", "B", "sequence_lens", "initial_h"], ["Y", "Y_h"]),
    op!("Gather", Tensor, ["data", "indices"], ["output"]),
    op!("GatherElements", Tensor, ["data", "indices"], ["output"]),
    op!("GatherND", Tensor, ["data", "indices"], ["output"]),
    op!("Gelu", Activation, ["X"], ["Y"]),
    op!("Gemm", Layer, ["A", "B", "C"], ["Y"]),
    op!("GlobalAveragePool", Pool, ["X"], ["Y"]),
    op!("GlobalLpPool", Pool, ["X"], ["Y"]),
    op!("GlobalMaxPool", Pool, ["X"], ["Y"]),
    op!("Greater", Logic, ["A", "B"], ["C"]),
    op!("GreaterOrEqual", Logic, ["A", "B"], ["C"]),
    op!("GridSample", Tensor, ["X", "grid"], ["Y"]),
    op!("GroupNormalization", Normalization, ["X", "scale", "bias"], ["Y"]),
    op!("HammingWindow", Data, ["size"], ["output"]),
    op!("HannWindow", Data, ["size"], ["output"]),
    op!("HardSigmoid", Activation, ["X"], ["Y"]),
    op!("HardSwish", Activation, ["X"], ["Y"]),
    op!("Hardmax", Activation, ["input"], ["output"]),
    op!("Identity", Tensor, ["input"], ["output"]),
    op!("If", Control, ["cond"], ["outputs*"]),
    op!("ImageDecoder", Data, ["encoded_stream"], ["image"]),
    op!("InstanceNormalization", Normalization, ["input", "scale", "B"], ["output"]),
    op!("IsInf", Logic, ["X"], ["Y"]),
    op!("IsNaN", Logic, ["X"], ["Y"]),
    op!("LRN", Normalization, ["X"], ["Y"]),
    op!("LSTM", Layer, ["X", "W", "R", "B", "sequence_lens", "initial_h", "initial_c", "P"], ["Y", "Y_h", "Y_c"]),
    op!("LayerNormalization", Normalization, ["X", "Scale", "B"], ["Y", "Mean", "InvStdDev"]),
    op!("LeakyRelu", Activation, ["X"], ["Y"]),
    op!("Less", Logic, ["A", "B"], ["C"]),
    op!("LessOrEqual", Logic, ["A", "B"], ["C"]),
    op!("Log", Math, ["input"], ["output"]),
    op!("LogSoftmax", Activation, ["input"], ["output"]),
    op!("Loop", Control, ["M", "cond", "v_initial*"], ["v_final_and_scan_outputs*"]),
    op!("LpNormalization", Normalization, ["input"], ["output"]),
    op!("LpPool", Pool, ["X"], ["Y"]),
    op!("MatMul", Layer, ["A", "B"], ["Y"]),
    op!("MatMulInteger", Layer, ["A", "B", "a_zero_point", "b_zero_point"], ["Y"]),
    op!("Max", Math, ["data_0*"], ["max"]),
    op!("MaxPool", Pool, ["X"], ["Y", "Indices"]),
    op!("MaxRoiPool", Pool, ["X", "rois"], ["Y"]),
    op!("MaxUnpool", Pool, ["X", "I", "output_shape"], ["output"]),
    op!("Mean", Math, ["data_0*"], ["mean"]),
    op!("MeanVarianceNormalization", Normalization, ["X"], ["Y"]),
    op!("MelWeightMatrix", Data, ["num_mel_bins", "dft_length", "sample_rate", "lower_edge_hertz", "upper_edge_hertz"], ["output"]),
    op!("Min", Math, ["data_0*"], ["min"]),
    op!("Mish", Activation, ["X"], ["Y"]),
    op!("Mod", Math, ["A", "B"], ["C"]),
    op!("Mul", Math, ["A", "B"], ["C"]),
    op!("Multinomial", Data, ["input"], ["output"]),
    op!("Neg", Math, ["X"], ["Y"]),
    op!("NegativeLogLikelihoodLoss", Math, ["input", "target", "weight"], ["loss"]),
    op!("NonMaxSuppression", Tensor, ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"], ["selected_indices"]),
    op!("NonZero", Tensor, ["X"], ["Y"]),
    op!("Not", Logic, ["X"], ["Y"]),
    op!("OneHot", Tensor, ["indices", "depth", "values"], ["output"]),
    op!("Optional", Sequence, ["input"], ["output"]),
    op!("OptionalGetElement", Sequence, ["input"], ["output"]),
    op!("OptionalHasElement", Sequence, ["input"], ["output"]),
    op!("Or", Logic, ["A", "B"], ["C"]),
    op!("PRelu", Activation, ["X", "slope"], ["Y"]),
    op!("Pad", Shape, ["data", "pads", "constant_value", "axes"], ["output"]),
    op!("Pow", Math, ["X", "Y"], ["Z"]),
    op!("QLinearConv", Layer, ["x", "x_scale", "x_zero_point", "w", "w_scale", "w_zero_point", "y_scale", "y_zero_point", "B"], ["y"]),
    op!("QLinearMatMul", Layer, ["a", "a_scale", "a_zero_point", "b", "b_scale", "b_zero_point", "y_scale", "y_zero_point"], ["y"]),
    op!("QuantizeLinear", Quantization, ["x", "y_scale", "y_zero_point"], ["y"]),
    op!("RMSNormalization", Normalization, ["X", "scale"], ["Y"]),
    op!("RNN", Layer, ["X", "W", "R", "B", "sequence_lens", "initial_h"], ["Y", "Y_h"]),
    op!("RandomNormal", Data, [], ["output"]),
    op!("RandomNormalLike", Data, ["input"], ["output"]),
    op!("RandomUniform", Data, [], ["output"]),
    op!("RandomUniformLike", Data, ["input"], ["output"]),
    op!("Range", Data, ["start", "limit", "delta"], ["output"]),
    op!("Reciprocal", Math, ["X"], ["Y"]),
    op!("ReduceL1", Reduce, ["data", "axes"], ["reduced"]),
    op!("ReduceL2", Reduce, ["data", "axes"], ["reduced"]),
    op!("ReduceLogSum", Reduce, ["data", "axes"], ["reduced"]),
    op!("ReduceLogSumExp", Reduce, ["data", "axes"], ["reduced"]),
    op!("ReduceMax", Reduce, ["data", "axes"], ["reduced"]),
    op!("ReduceMean", Reduce, ["data", "axes"], ["reduced"]),
    op!("ReduceMin", Reduce, ["data", "axes"], ["reduced"]),
    op!("ReduceProd", Reduce, ["data", "axes"], ["reduced"]),
    op!("ReduceSum", Reduce, ["data", "axes"], ["reduced"]),
    op!("ReduceSumSquare", Reduce, ["data", "axes"], ["reduced"]),
    op!("RegexFullMatch", Logic, ["X"], ["Y"]),
    op!("Relu", Activation, ["X"], ["Y"]),
    op!("Reshape", Shape, ["data", "shape"], ["reshaped"]),
    op!("Resize", Shape, ["X", "roi", "scales", "sizes"], ["Y"]),
    op!("ReverseSequence", Tensor, ["input", "sequence_lens"], ["Y"]),
    op!("RoiAlign", Pool, ["X", "rois", "batch_indices"], ["Y"]),
    op!("RotaryEmbedding", Layer, ["X", "cos_cache", "sin_cache", "position_ids"], ["Y"]),
    op!("Round", Math, ["X"], ["Y"]),
    op!("STFT", Math, ["signal", "frame_step", "window", "frame_length"], ["output"]),
    op!("Scan", Control, ["initial_state_and_scan_inputs*"], ["final_state_and_scan_outputs*"]),
    op!("Scatter", Tensor, ["data", "indices", "updates"], ["output"]),
    op!("ScatterElements", Tensor, ["data", "indices", "updates"], ["output"]),
    op!("ScatterND", Tensor, ["data", "indices", "updates"], ["output"]),
    op!("Selu", Activation, ["X"], ["Y"]),
    op!("SequenceAt", Sequence, ["input_sequence", "position"], ["tensor"]),
    op!("SequenceConstruct", Sequence, ["inputs*"], ["output_sequence"]),
    op!("SequenceEmpty", Sequence, [], ["output"]),
    op!("SequenceErase", Sequence, ["input_sequence", "position"], ["output_sequence"]),
    op!("SequenceInsert", Sequence, ["input_sequence", "tensor", "position"], ["output_sequence"]),
    op!("SequenceLength", Sequence, ["input_sequence"], ["length"]),
    op!("SequenceMap", Control, ["input_sequence", "additional_inputs*"], ["out_sequence*"]),
    op!("Shape", Shape, ["data"], ["shape"]),
    op!("Shrink", Activation, ["input"], ["output"]),
    op!("Sigmoid", Activation, ["X"], ["Y"]),
    op!("Sign", Math, ["input"], ["output"]),
    op!("Sin", Math, ["input"], ["output"]),
    op!("Sinh", Math, ["input"], ["output"]),
    op!("Size", Shape, ["data"], ["size"]),
    op!("Slice", Tensor, ["data", "starts", "ends", "axes", "steps"], ["output"]),
    op!("Softmax", Activation, ["input"], ["output"]),
    op!("SoftmaxCrossEntropyLoss", Math, ["scores", "labels", "weights"], ["output", "log_prob"]),
    op!("Softplus", Activation, ["X"], ["Y"]),
    op!("Softsign", Activation, ["input"], ["output"]),
    op!("SpaceToDepth", Shape, ["input"], ["output"]),
    op!("Split", Tensor, ["input", "split"], ["outputs*"]),
    op!("SplitToSequence", Sequence, ["input", "split"], ["output_sequence"]),
    op!("Sqrt", Math, ["X"], ["Y"]),
    op!("Squeeze", Shape, ["data", "axes"], ["squeezed"]),
    op!("StringConcat", Data, ["X", "Y"], ["Z"]),
    op!("StringNormalizer", Data, ["X"], ["Y"]),
    op!("StringSplit", Data, ["X"], ["Y", "Z"]),
    op!("Sub", Math, ["A", "B"], ["C"]),
    op!("Sum", Math, ["data_0*"], ["sum"]),
    op!("Swish", Activation, ["X"], ["Y"]),
    op!("Tan", Math, ["input"], ["output"]),
    op!("Tanh", Activation, ["input"], ["output"]),
    op!("TensorScatter", Tensor, ["past_cache", "update", "write_indices"], ["present_cache"]),
    op!("TfIdfVectorizer", Data, ["X"], ["Y"]),
    op!("ThresholdedRelu", Activation, ["X"], ["Y"]),
    op!("Tile", Shape, ["input", "repeats"], ["output"]),
    op!("TopK", Tensor, ["X", "K"], ["Values", "Indices"]),
    op!("Transpose", Shape, ["data"], ["transposed"]),
    op!("Trilu", Tensor, ["input", "k"], ["output"]),
    op!("Unique", Tensor, ["X"], ["Y", "indices", "inverse_indices", "counts"]),
    op!("Unsqueeze", Shape, ["data", "axes"], ["expanded"]),
    op!("Upsample", Shape, ["X", "scales"], ["Y"]),
    op!("Where", Logic, ["condition", "X", "Y"], ["output"]),
    op!("Xor", Logic, ["A", "B"], ["C"]),
    // Deprecated / experimental ops still found in older models.
    op!("ImageScaler", Data, ["input"], ["output"]),
    op!("Crop", Shape, ["input"], ["output"]),
    op!("Scale", Math, ["input"], ["output"]),
    op!("ScaledTanh", Activation, ["input"], ["output"]),
    op!("ParametricSoftplus", Activation, ["X"], ["Y"]),
    op!("GivenTensorFill", Data, ["shape"], ["X"]),
    op!("ATen", Custom, ["input*"], ["output*"]),
];

static MICROSOFT: &[OpSchema] = &[
    op!("Attention", Layer, ["input", "weights", "bias", "mask_index", "past", "attention_bias", "past_sequence_length"], ["output", "present"]),
    op!("BiasAdd", Math, ["X", "bias", "skip"], ["Y"]),
    op!("BiasGelu", Activation, ["A", "B"], ["C"]),
    op!("BiasSoftmax", Activation, ["data", "bias"], ["output"]),
    op!("DequantizeLinear", Quantization, ["x", "x_scale", "x_zero_point"], ["y"]),
    op!("DynamicQuantizeMatMul", Layer, ["A", "B", "b_scale", "b_zero_point", "bias"], ["Y"]),
    op!("EmbedLayerNormalization", Normalization, ["input_ids", "segment_ids", "word_embedding", "position_embedding", "segment_embedding", "gamma", "beta", "mask", "position_ids"], ["output", "mask_index", "embedding_sum"]),
    op!("FastGelu", Activation, ["X", "bias"], ["Y"]),
    op!("FusedConv", Layer, ["X", "W", "B", "Z"], ["Y"]),
    op!("FusedGemm", Layer, ["A", "B", "C"], ["Y"]),
    op!("FusedMatMul", Layer, ["A", "B"], ["Y"]),
    op!("Gelu", Activation, ["X"], ["Y"]),
    op!("GatherBlockQuantized", Quantization, ["data", "indices", "scales", "zero_points"], ["output"]),
    op!("GroupQueryAttention", Layer, ["query", "key", "value", "past_key", "past_value", "seqlens_k", "total_sequence_length", "cos_cache", "sin_cache"], ["output", "present_key", "present_value"]),
    op!("LayerNormalization", Normalization, ["X", "Scale", "B"], ["Y", "Mean", "InvStdDev"]),
    op!("MatMulInteger16", Layer, ["A", "B"], ["Y"]),
    op!("MatMulIntegerToFloat", Layer, ["A", "B", "a_scale", "b_scale", "a_zero_point", "b_zero_point", "bias"], ["Y"]),
    op!("MatMulNBits", Layer, ["A", "B", "scales", "zero_points", "g_idx", "bias"], ["Y"]),
    op!("MultiHeadAttention", Layer, ["query", "key", "value", "bias", "key_padding_mask", "attention_bias", "past_key", "past_value"], ["output", "present_key", "present_value"]),
    op!("NhwcConv", Layer, ["X", "W", "B"], ["Y"]),
    op!("NhwcMaxPool", Pool, ["x"], ["y"]),
    op!("QAttention", Layer, ["input", "weight", "bias", "input_scale", "weight_scale", "mask_index", "input_zero_point", "weight_zero_point", "past"], ["output", "present"]),
    op!("QGemm", Layer, ["A", "a_scale", "a_zero_point", "B", "b_scale", "b_zero_point", "C", "y_scale", "y_zero_point"], ["Y"]),
    op!("QLinearAdd", Quantization, ["A", "A_scale", "A_zero_point", "B", "B_scale", "B_zero_point", "C_scale", "C_zero_point"], ["C"]),
    op!("QLinearAveragePool", Pool, ["X", "x_scale", "x_zero_point", "y_scale", "y_zero_point"], ["Y"]),
    op!("QLinearConcat", Quantization, ["Y_scale", "Y_zero_point", "inputs*"], ["Y"]),
    op!("QLinearConv", Layer, ["x", "x_scale", "x_zero_point", "w", "w_scale", "w_zero_point", "y_scale", "y_zero_point", "B"], ["y"]),
    op!("QLinearGlobalAveragePool", Pool, ["X", "x_scale", "x_zero_point", "y_scale", "y_zero_point"], ["Y"]),
    op!("QLinearLeakyRelu", Activation, ["X", "X_scale", "X_zero_point", "Y_scale", "Y_zero_point"], ["Y"]),
    op!("QLinearMul", Quantization, ["A", "A_scale", "A_zero_point", "B", "B_scale", "B_zero_point", "C_scale", "C_zero_point"], ["C"]),
    op!("QLinearSigmoid", Activation, ["X", "X_scale", "X_zero_point", "Y_scale", "Y_zero_point"], ["Y"]),
    op!("QLinearSoftmax", Activation, ["X", "X_scale", "x_zero_point", "y_scale", "y_zero_point"], ["Y"]),
    op!("QuantizeLinear", Quantization, ["x", "y_scale", "y_zero_point"], ["y"]),
    op!("QuickGelu", Activation, ["X"], ["Y"]),
    op!("RotaryEmbedding", Layer, ["input", "position_ids", "cos_cache", "sin_cache"], ["output"]),
    op!("SimplifiedLayerNormalization", Normalization, ["X", "scale"], ["Y", "inv_std_var"]),
    op!("SkipLayerNormalization", Normalization, ["input", "skip", "gamma", "beta", "bias"], ["output", "mean", "inv_std_var", "input_skip_bias_sum"]),
    op!("SkipSimplifiedLayerNormalization", Normalization, ["input", "skip", "gamma", "bias"], ["output", "mean", "inv_std_var", "input_skip_bias_sum"]),
];

type Index = HashMap<&'static str, HashMap<&'static str, &'static OpSchema>>;

fn index() -> &'static Index {
    static INDEX: OnceLock<Index> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut m: Index = HashMap::new();
        let std_ops = m.entry("").or_default();
        for s in STANDARD {
            std_ops.insert(s.name, s);
        }
        let ms_ops = m.entry("com.microsoft").or_default();
        for s in MICROSOFT {
            ms_ops.insert(s.name, s);
        }
        m
    })
}

fn lookup(domain: &str, op: &str) -> Option<&'static OpSchema> {
    index().get(domain).and_then(|m| m.get(op)).copied()
}

pub fn is_standard_domain(domain: &str) -> bool {
    domain.is_empty() || domain == "ai.onnx"
}

pub fn schema(domain: &str, op: &str) -> Option<&'static OpSchema> {
    let d = if is_standard_domain(domain) { "" } else { domain };
    if let Some(s) = lookup(d, op) {
        return Some(s);
    }
    // ORT contrib ops reuse many standard operator signatures.
    if d == "com.microsoft" {
        return lookup("", op);
    }
    None
}

pub fn category(domain: &str, op: &str) -> Category {
    if let Some(s) = schema(domain, op) {
        return s.category;
    }
    if domain == "ai.onnx.ml" {
        return Category::Ml;
    }
    if domain.starts_with("ai.onnx.preview") {
        return Category::Other;
    }
    if is_standard_domain(domain) {
        Category::Other
    } else {
        Category::Custom
    }
}

fn slot_name(names: &[&'static str], idx: usize, fallback: &str) -> String {
    if let Some(n) = names.get(idx) {
        if let Some(base) = n.strip_suffix('*') {
            return if idx + 1 == names.len() {
                format!("{}[0]", base)
            } else {
                base.to_string()
            };
        }
        return n.to_string();
    }
    if let Some(last) = names.last() {
        if let Some(base) = last.strip_suffix('*') {
            return format!("{}[{}]", base, idx + 1 - names.len());
        }
    }
    format!("{}{}", fallback, idx)
}

pub fn input_name(domain: &str, op: &str, idx: usize) -> String {
    match schema(domain, op) {
        Some(s) => slot_name(s.inputs, idx, "input "),
        None => format!("input {}", idx),
    }
}

pub fn output_name(domain: &str, op: &str, idx: usize) -> String {
    match schema(domain, op) {
        Some(s) => slot_name(s.outputs, idx, "output "),
        None => format!("output {}", idx),
    }
}

/// Link to the operator documentation, if one is known.
pub fn doc_url(domain: &str, op: &str) -> Option<String> {
    if is_standard_domain(domain) {
        if lookup("", op).is_some() {
            return Some(format!("https://onnx.ai/onnx/operators/onnx__{}.html", op));
        }
        return None;
    }
    match domain {
        "ai.onnx.ml" => Some(format!(
            "https://onnx.ai/onnx/operators/onnx_aionnxml_{}.html",
            op
        )),
        "com.microsoft" => Some(format!(
            "https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.{}",
            op
        )),
        _ => None,
    }
}

/// Operators that ONNX marks as deprecated, with the opset that removed them.
pub fn deprecated_since(op: &str) -> Option<i64> {
    match op {
        "Upsample" => Some(10),
        "Scatter" => Some(11),
        "ImageScaler" | "Crop" | "Scale" | "ScaledTanh" | "ParametricSoftplus"
        | "GivenTensorFill" => Some(1),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_names() {
        assert_eq!(input_name("", "Conv", 1), "W");
        assert_eq!(input_name("", "Concat", 0), "inputs[0]");
        assert_eq!(input_name("", "Concat", 3), "inputs[3]");
        assert_eq!(input_name("", "Loop", 2), "v_initial[0]");
        assert_eq!(input_name("", "Loop", 4), "v_initial[2]");
        assert_eq!(input_name("custom", "Foo", 2), "input 2");
        assert_eq!(category("", "Relu"), Category::Activation);
        assert_eq!(category("my.domain", "Relu2"), Category::Custom);
    }
}
