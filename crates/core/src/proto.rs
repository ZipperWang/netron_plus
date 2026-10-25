//! Protobuf message definitions for the ONNX schema (`onnx.proto`, IR version 11).
//!
//! The structs are written by hand with `prost` derives so the build needs no
//! `protoc`. Field numbers follow the upstream schema exactly; enum fields are
//! kept as plain `i32` values. `raw_data` uses `Bytes` so decoding from a
//! `Bytes` buffer does not copy weight payloads.
#![allow(clippy::derive_partial_eq_without_eq)]

use prost::bytes::Bytes;

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(string, tag = "4")]
    pub domain: String,
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "14")]
    pub metadata_props: Vec<StringStringEntryProto>,
    #[prost(message, repeated, tag = "20")]
    pub training_info: Vec<TrainingInfoProto>,
    #[prost(message, repeated, tag = "25")]
    pub functions: Vec<FunctionProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TrainingInfoProto {
    #[prost(message, optional, tag = "1")]
    pub initialization: Option<GraphProto>,
    #[prost(message, optional, tag = "2")]
    pub algorithm: Option<GraphProto>,
    #[prost(message, repeated, tag = "3")]
    pub initialization_binding: Vec<StringStringEntryProto>,
    #[prost(message, repeated, tag = "4")]
    pub update_binding: Vec<StringStringEntryProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "15")]
    pub sparse_initializer: Vec<SparseTensorProto>,
    #[prost(string, tag = "10")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "14")]
    pub quantization_annotation: Vec<TensorAnnotation>,
    #[prost(message, repeated, tag = "16")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(string, tag = "7")]
    pub domain: String,
    #[prost(string, tag = "8")]
    pub overload: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "9")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// Attribute type codes (`AttributeProto.AttributeType`).
pub mod attr_type {
    pub const UNDEFINED: i32 = 0;
    pub const FLOAT: i32 = 1;
    pub const INT: i32 = 2;
    pub const STRING: i32 = 3;
    pub const TENSOR: i32 = 4;
    pub const GRAPH: i32 = 5;
    pub const FLOATS: i32 = 6;
    pub const INTS: i32 = 7;
    pub const STRINGS: i32 = 8;
    pub const TENSORS: i32 = 9;
    pub const GRAPHS: i32 = 10;
    pub const SPARSE_TENSOR: i32 = 11;
    pub const SPARSE_TENSORS: i32 = 12;
    pub const TYPE_PROTO: i32 = 13;
    pub const TYPE_PROTOS: i32 = 14;
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(string, tag = "21")]
    pub ref_attr_name: String,
    #[prost(string, tag = "13")]
    pub doc_string: String,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes = "vec", tag = "4")]
    pub s: Vec<u8>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(message, optional, tag = "6")]
    pub g: Option<GraphProto>,
    #[prost(message, optional, tag = "22")]
    pub sparse_tensor: Option<SparseTensorProto>,
    #[prost(message, optional, tag = "14")]
    pub tp: Option<TypeProto>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(bytes = "vec", repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    #[prost(message, repeated, tag = "10")]
    pub tensors: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub graphs: Vec<GraphProto>,
    #[prost(message, repeated, tag = "23")]
    pub sparse_tensors: Vec<SparseTensorProto>,
    #[prost(message, repeated, tag = "15")]
    pub type_protos: Vec<TypeProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
    #[prost(string, tag = "3")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "4")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(message, optional, tag = "3")]
    pub segment: Option<tensor_proto::Segment>,
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(bytes = "vec", repeated, tag = "6")]
    pub string_data: Vec<Vec<u8>>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(string, tag = "12")]
    pub doc_string: String,
    #[prost(bytes = "bytes", tag = "9")]
    pub raw_data: Bytes,
    #[prost(message, repeated, tag = "13")]
    pub external_data: Vec<StringStringEntryProto>,
    #[prost(int32, tag = "14")]
    pub data_location: i32,
    #[prost(double, repeated, tag = "10")]
    pub double_data: Vec<f64>,
    #[prost(uint64, repeated, tag = "11")]
    pub uint64_data: Vec<u64>,
    #[prost(message, repeated, tag = "16")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

pub mod tensor_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Segment {
        #[prost(int64, tag = "1")]
        pub begin: i64,
        #[prost(int64, tag = "2")]
        pub end: i64,
    }

    /// `TensorProto.DataLocation.EXTERNAL`
    pub const EXTERNAL: i32 = 1;
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SparseTensorProto {
    #[prost(message, optional, tag = "1")]
    pub values: Option<TensorProto>,
    #[prost(message, optional, tag = "2")]
    pub indices: Option<TensorProto>,
    #[prost(int64, repeated, tag = "3")]
    pub dims: Vec<i64>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<tensor_shape_proto::Dimension>,
}

pub mod tensor_shape_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Dimension {
        #[prost(string, tag = "3")]
        pub denotation: String,
        #[prost(oneof = "dimension::Value", tags = "1, 2")]
        pub value: Option<dimension::Value>,
    }

    pub mod dimension {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Value {
            #[prost(int64, tag = "1")]
            DimValue(i64),
            #[prost(string, tag = "2")]
            DimParam(String),
        }
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TypeProto {
    #[prost(string, tag = "6")]
    pub denotation: String,
    #[prost(oneof = "type_proto::Value", tags = "1, 4, 5, 8, 9")]
    pub value: Option<type_proto::Value>,
}

pub mod type_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Tensor {
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        #[prost(message, optional, tag = "2")]
        pub shape: Option<super::TensorShapeProto>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Sequence {
        #[prost(message, optional, boxed, tag = "1")]
        pub elem_type: Option<Box<super::TypeProto>>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Map {
        #[prost(int32, tag = "1")]
        pub key_type: i32,
        #[prost(message, optional, boxed, tag = "2")]
        pub value_type: Option<Box<super::TypeProto>>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Optional {
        #[prost(message, optional, boxed, tag = "1")]
        pub elem_type: Option<Box<super::TypeProto>>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct SparseTensor {
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        #[prost(message, optional, tag = "2")]
        pub shape: Option<super::TensorShapeProto>,
    }

    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        #[prost(message, tag = "1")]
        TensorType(Tensor),
        #[prost(message, tag = "4")]
        SequenceType(Sequence),
        #[prost(message, tag = "5")]
        MapType(Map),
        #[prost(message, tag = "8")]
        SparseTensorType(SparseTensor),
        #[prost(message, tag = "9")]
        OptionalType(Optional),
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorAnnotation {
    #[prost(string, tag = "1")]
    pub tensor_name: String,
    #[prost(message, repeated, tag = "2")]
    pub quant_parameter_tensor_names: Vec<StringStringEntryProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FunctionProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(string, repeated, tag = "4")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "5")]
    pub output: Vec<String>,
    #[prost(string, repeated, tag = "6")]
    pub attribute: Vec<String>,
    #[prost(message, repeated, tag = "11")]
    pub attribute_proto: Vec<AttributeProto>,
    #[prost(message, repeated, tag = "7")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "8")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "9")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = "10")]
    pub domain: String,
    #[prost(string, tag = "13")]
    pub overload: String,
    #[prost(message, repeated, tag = "12")]
    pub value_info: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "14")]
    pub metadata_props: Vec<StringStringEntryProto>,
}
