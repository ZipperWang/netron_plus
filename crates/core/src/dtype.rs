//! ONNX tensor element types (`TensorProto.DataType`).

pub const UNDEFINED: i32 = 0;
pub const FLOAT: i32 = 1;
pub const UINT8: i32 = 2;
pub const INT8: i32 = 3;
pub const UINT16: i32 = 4;
pub const INT16: i32 = 5;
pub const INT32: i32 = 6;
pub const INT64: i32 = 7;
pub const STRING: i32 = 8;
pub const BOOL: i32 = 9;
pub const FLOAT16: i32 = 10;
pub const DOUBLE: i32 = 11;
pub const UINT32: i32 = 12;
pub const UINT64: i32 = 13;
pub const COMPLEX64: i32 = 14;
pub const COMPLEX128: i32 = 15;
pub const BFLOAT16: i32 = 16;
pub const FLOAT8E4M3FN: i32 = 17;
pub const FLOAT8E4M3FNUZ: i32 = 18;
pub const FLOAT8E5M2: i32 = 19;
pub const FLOAT8E5M2FNUZ: i32 = 20;
pub const UINT4: i32 = 21;
pub const INT4: i32 = 22;
pub const FLOAT4E2M1: i32 = 23;
pub const FLOAT8E8M0: i32 = 24;

pub fn name(dt: i32) -> &'static str {
    match dt {
        UNDEFINED => "",
        FLOAT => "float32",
        UINT8 => "uint8",
        INT8 => "int8",
        UINT16 => "uint16",
        INT16 => "int16",
        INT32 => "int32",
        INT64 => "int64",
        STRING => "string",
        BOOL => "bool",
        FLOAT16 => "float16",
        DOUBLE => "float64",
        UINT32 => "uint32",
        UINT64 => "uint64",
        COMPLEX64 => "complex64",
        COMPLEX128 => "complex128",
        BFLOAT16 => "bfloat16",
        FLOAT8E4M3FN => "float8e4m3fn",
        FLOAT8E4M3FNUZ => "float8e4m3fnuz",
        FLOAT8E5M2 => "float8e5m2",
        FLOAT8E5M2FNUZ => "float8e5m2fnuz",
        UINT4 => "uint4",
        INT4 => "int4",
        FLOAT4E2M1 => "float4e2m1",
        FLOAT8E8M0 => "float8e8m0",
        _ => "unknown",
    }
}

/// Storage size of one element in bits, `None` for strings / unknown types.
pub fn bits(dt: i32) -> Option<u64> {
    Some(match dt {
        FLOAT | INT32 | UINT32 => 32,
        UINT8 | INT8 | BOOL | FLOAT8E4M3FN | FLOAT8E4M3FNUZ | FLOAT8E5M2 | FLOAT8E5M2FNUZ
        | FLOAT8E8M0 => 8,
        UINT16 | INT16 | FLOAT16 | BFLOAT16 => 16,
        INT64 | UINT64 | DOUBLE | COMPLEX64 => 64,
        COMPLEX128 => 128,
        UINT4 | INT4 | FLOAT4E2M1 => 4,
        _ => return None,
    })
}

pub fn bytes_for(dt: i32, elements: u64) -> Option<u64> {
    bits(dt).map(|b| (elements.saturating_mul(b)).div_ceil(8))
}

pub fn is_integer(dt: i32) -> bool {
    matches!(
        dt,
        UINT8 | INT8 | UINT16 | INT16 | INT32 | INT64 | UINT32 | UINT64 | UINT4 | INT4 | BOOL
    )
}

pub fn is_float(dt: i32) -> bool {
    matches!(
        dt,
        FLOAT
            | FLOAT16
            | DOUBLE
            | BFLOAT16
            | FLOAT8E4M3FN
            | FLOAT8E4M3FNUZ
            | FLOAT8E5M2
            | FLOAT8E5M2FNUZ
            | FLOAT4E2M1
            | FLOAT8E8M0
    )
}

pub fn is_numeric(dt: i32) -> bool {
    is_integer(dt) || is_float(dt)
}

/// Element types that usually hold learned weights (used when counting
/// parameters). Shape-like types such as int64 and bool are excluded.
pub fn is_weight_type(dt: i32) -> bool {
    is_float(dt) || matches!(dt, UINT8 | INT8 | UINT4 | INT4 | INT32 | INT16 | UINT16)
}
