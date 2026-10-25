//! Tensor payload decoding, statistics, value paging and export.

use crate::dtype::*;
use crate::proto::{tensor_proto, TensorProto};
use crate::shape::numel_i64;
use anyhow::{anyhow, bail, Result};
use serde::Serialize;
use std::borrow::Cow;

/// Provides access to external tensor data files.
pub trait DataSource: Send + Sync {
    fn read_external(&self, location: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>>;
    /// `Some(exists)` when the source can check files, `None` otherwise.
    fn external_exists(&self, _location: &str) -> Option<bool> {
        None
    }
}

/// Used when no file system is available (e.g. the web build).
pub struct NoSource;

impl DataSource for NoSource {
    fn read_external(&self, location: &str, _offset: u64, _length: Option<u64>) -> Result<Vec<u8>> {
        bail!(
            "External data file '{}' is not available here; open the model from disk in the desktop app",
            location
        )
    }
}

/// Reads external data relative to the directory of the model file.
pub struct FileSource {
    pub dir: std::path::PathBuf,
}

impl FileSource {
    fn resolve(&self, location: &str) -> Result<std::path::PathBuf> {
        let rel = std::path::Path::new(location);
        if rel.is_absolute()
            || rel
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            bail!(
                "Refusing to read external data outside the model directory: {}",
                location
            );
        }
        Ok(self.dir.join(rel))
    }
}

impl DataSource for FileSource {
    fn read_external(&self, location: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let path = self.resolve(location)?;
        let mut f = std::fs::File::open(&path)
            .map_err(|e| anyhow!("Cannot open external data file {}: {}", path.display(), e))?;
        f.seek(SeekFrom::Start(offset))?;
        let mut buf = Vec::new();
        match length {
            Some(len) => {
                buf.resize(len as usize, 0);
                f.read_exact(&mut buf).map_err(|e| {
                    anyhow!("External data file {} is truncated: {}", path.display(), e)
                })?;
            }
            None => {
                f.read_to_end(&mut buf)?;
            }
        }
        Ok(buf)
    }

    fn external_exists(&self, location: &str) -> Option<bool> {
        Some(self.resolve(location).map(|p| p.is_file()).unwrap_or(false))
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExternalInfo {
    pub location: String,
    pub offset: u64,
    pub length: Option<u64>,
}

pub fn external_info(t: &TensorProto) -> Option<ExternalInfo> {
    if t.data_location != tensor_proto::EXTERNAL {
        return None;
    }
    let mut info = ExternalInfo {
        location: String::new(),
        offset: 0,
        length: None,
    };
    for kv in &t.external_data {
        match kv.key.as_str() {
            "location" => info.location = kv.value.clone(),
            "offset" => info.offset = kv.value.trim().parse().unwrap_or(0),
            "length" => info.length = kv.value.trim().parse().ok(),
            _ => {}
        }
    }
    if info.location.is_empty() {
        None
    } else {
        Some(info)
    }
}

pub fn element_count(t: &TensorProto) -> u64 {
    numel_i64(&t.dims)
}

pub fn byte_size(t: &TensorProto) -> Option<u64> {
    if t.data_type == STRING {
        return Some(t.string_data.iter().map(|s| s.len() as u64).sum());
    }
    bytes_for(t.data_type, element_count(t))
}

/// Where the elements of a tensor live.
enum Payload<'a> {
    Raw(Cow<'a, [u8]>),
    Typed,
}

fn payload<'a>(t: &'a TensorProto, src: &dyn DataSource) -> Result<Payload<'a>> {
    if let Some(ext) = external_info(t) {
        let data = src.read_external(&ext.location, ext.offset, ext.length)?;
        return Ok(Payload::Raw(Cow::Owned(data)));
    }
    if !t.raw_data.is_empty() {
        return Ok(Payload::Raw(Cow::Borrowed(&t.raw_data[..])));
    }
    Ok(Payload::Typed)
}

#[derive(Clone, Debug)]
pub enum Elem {
    F(f64),
    I(i64),
    U(u64),
    B(bool),
    S(String),
    C(f64, f64),
}

impl Elem {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Elem::F(v) => Some(*v),
            Elem::I(v) => Some(*v as f64),
            Elem::U(v) => Some(*v as f64),
            Elem::B(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Elem::I(v) => Some(*v),
            Elem::U(v) => Some(*v as i64),
            Elem::B(b) => Some(*b as i64),
            Elem::F(v) if v.fract() == 0.0 && v.is_finite() => Some(*v as i64),
            _ => None,
        }
    }

    pub fn text(&self) -> String {
        match self {
            Elem::F(v) => fmt_float(*v),
            Elem::I(v) => v.to_string(),
            Elem::U(v) => v.to_string(),
            Elem::B(b) => b.to_string(),
            Elem::S(s) => format!("{:?}", s),
            Elem::C(re, im) => {
                if *im < 0.0 {
                    format!("{}-{}i", fmt_float(*re), fmt_float(-im))
                } else {
                    format!("{}+{}i", fmt_float(*re), fmt_float(*im))
                }
            }
        }
    }
}

/// Compact float formatting used across the UI.
pub fn fmt_float(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let a = v.abs();
    if !(1e-4..1e7).contains(&a) {
        let s = format!("{:.4e}", v);
        // 1.2340e-5 -> 1.234e-5
        if let Some((m, e)) = s.split_once('e') {
            let m = if m.contains('.') {
                m.trim_end_matches('0').trim_end_matches('.')
            } else {
                m
            };
            return format!("{}e{}", m, e);
        }
        return s;
    }
    let s = format!("{:.6}", v);
    let s = if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    };
    if s == "-0" {
        "0".into()
    } else {
        s
    }
}

pub fn f8e4m3fn(b: u8) -> f64 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as f64;
    if e == 15 && (b & 0x07) == 7 {
        return f64::NAN;
    }
    if e == 0 {
        s * (m / 8.0) * 2f64.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

pub fn f8e4m3fnuz(b: u8) -> f64 {
    if b == 0x80 {
        return f64::NAN;
    }
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as f64;
    if e == 0 {
        s * (m / 8.0) * 2f64.powi(-7)
    } else {
        s * (1.0 + m / 8.0) * 2f64.powi(e - 8)
    }
}

pub fn f8e5m2(b: u8) -> f64 {
    half::f16::from_bits((b as u16) << 8).to_f64()
}

pub fn f8e5m2fnuz(b: u8) -> f64 {
    if b == 0x80 {
        return f64::NAN;
    }
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 2) & 0x1F) as i32;
    let m = (b & 0x03) as f64;
    if e == 0 {
        s * (m / 4.0) * 2f64.powi(-15)
    } else {
        s * (1.0 + m / 4.0) * 2f64.powi(e - 16)
    }
}

pub fn f8e8m0(b: u8) -> f64 {
    if b == 0xFF {
        f64::NAN
    } else {
        2f64.powi(b as i32 - 127)
    }
}

pub fn f4e2m1(n: u8) -> f64 {
    let s = if n & 0x8 != 0 { -1.0 } else { 1.0 };
    let e = ((n >> 1) & 0x3) as i32;
    let m = (n & 0x1) as f64;
    if e == 0 {
        s * 0.5 * m
    } else {
        s * 2f64.powi(e - 1) * (1.0 + 0.5 * m)
    }
}

fn nibble(dt: i32, n: u8) -> Elem {
    match dt {
        INT4 => {
            let v = (n & 0x0F) as i64;
            Elem::I(if v >= 8 { v - 16 } else { v })
        }
        UINT4 => Elem::U((n & 0x0F) as u64),
        _ => Elem::F(f4e2m1(n & 0x0F)),
    }
}

fn byte_elem(dt: i32, b: u8) -> Elem {
    match dt {
        INT8 => Elem::I(b as i8 as i64),
        UINT8 => Elem::U(b as u64),
        BOOL => Elem::B(b != 0),
        FLOAT8E4M3FN => Elem::F(f8e4m3fn(b)),
        FLOAT8E4M3FNUZ => Elem::F(f8e4m3fnuz(b)),
        FLOAT8E5M2 => Elem::F(f8e5m2(b)),
        FLOAT8E5M2FNUZ => Elem::F(f8e5m2fnuz(b)),
        FLOAT8E8M0 => Elem::F(f8e8m0(b)),
        _ => Elem::U(b as u64),
    }
}

/// Bytes per element for fixed-width raw encodings (`None` for 4-bit / strings).
fn raw_width(dt: i32) -> Option<usize> {
    match dt {
        INT8 | UINT8 | BOOL | FLOAT8E4M3FN | FLOAT8E4M3FNUZ | FLOAT8E5M2 | FLOAT8E5M2FNUZ
        | FLOAT8E8M0 => Some(1),
        INT16 | UINT16 | FLOAT16 | BFLOAT16 => Some(2),
        FLOAT | INT32 | UINT32 => Some(4),
        DOUBLE | INT64 | UINT64 | COMPLEX64 => Some(8),
        COMPLEX128 => Some(16),
        _ => None,
    }
}

fn raw_elem(dt: i32, c: &[u8]) -> Elem {
    match dt {
        FLOAT => Elem::F(f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64),
        DOUBLE => Elem::F(f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])),
        INT16 => Elem::I(i16::from_le_bytes([c[0], c[1]]) as i64),
        UINT16 => Elem::U(u16::from_le_bytes([c[0], c[1]]) as u64),
        INT32 => Elem::I(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64),
        UINT32 => Elem::U(u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64),
        INT64 => Elem::I(i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])),
        UINT64 => Elem::U(u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])),
        FLOAT16 => Elem::F(half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64()),
        BFLOAT16 => Elem::F(half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64()),
        COMPLEX64 => Elem::C(
            f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64,
            f32::from_le_bytes([c[4], c[5], c[6], c[7]]) as f64,
        ),
        COMPLEX128 => Elem::C(
            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]),
            f64::from_le_bytes([c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]]),
        ),
        _ => byte_elem(dt, c[0]),
    }
}

fn int32_elem(dt: i32, v: i32) -> Elem {
    match dt {
        FLOAT16 => Elem::F(half::f16::from_bits(v as u16).to_f64()),
        BFLOAT16 => Elem::F(half::bf16::from_bits(v as u16).to_f64()),
        BOOL => Elem::B(v != 0),
        UINT8 | UINT16 => Elem::U(v as u32 as u64),
        FLOAT8E4M3FN | FLOAT8E4M3FNUZ | FLOAT8E5M2 | FLOAT8E5M2FNUZ | FLOAT8E8M0 => {
            byte_elem(dt, v as u8)
        }
        _ => Elem::I(v as i64),
    }
}

/// Decode elements `[start, end)` and pass them to `f`.
fn decode(
    t: &TensorProto,
    p: &Payload,
    start: u64,
    end: u64,
    f: &mut dyn FnMut(Elem),
) -> Result<()> {
    let dt = t.data_type;
    let (start, end) = (start as usize, end as usize);
    if end <= start {
        return Ok(());
    }
    match p {
        Payload::Raw(b) => {
            let b: &[u8] = b;
            if let Some(w) = raw_width(dt) {
                let avail = b.len() / w;
                let end = end.min(avail);
                for i in start..end.max(start) {
                    f(raw_elem(dt, &b[i * w..i * w + w]));
                }
            } else if matches!(dt, INT4 | UINT4 | FLOAT4E2M1) {
                let avail = b.len() * 2;
                for i in start..end.min(avail).max(start) {
                    let byte = b[i / 2];
                    let n = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    f(nibble(dt, n));
                }
            } else if dt == STRING {
                bail!("Raw string tensors are not supported");
            } else {
                bail!("Unsupported data type {}", name(dt));
            }
        }
        Payload::Typed => match dt {
            FLOAT => {
                for v in t.float_data.iter().skip(start).take(end - start) {
                    f(Elem::F(*v as f64));
                }
            }
            COMPLEX64 => {
                for c in t.float_data.chunks_exact(2).skip(start).take(end - start) {
                    f(Elem::C(c[0] as f64, c[1] as f64));
                }
            }
            DOUBLE => {
                for v in t.double_data.iter().skip(start).take(end - start) {
                    f(Elem::F(*v));
                }
            }
            COMPLEX128 => {
                for c in t.double_data.chunks_exact(2).skip(start).take(end - start) {
                    f(Elem::C(c[0], c[1]));
                }
            }
            INT64 => {
                for v in t.int64_data.iter().skip(start).take(end - start) {
                    f(Elem::I(*v));
                }
            }
            UINT32 | UINT64 => {
                for v in t.uint64_data.iter().skip(start).take(end - start) {
                    f(Elem::U(*v));
                }
            }
            INT4 | UINT4 | FLOAT4E2M1 => {
                let avail = t.int32_data.len() * 2;
                for i in start..end.min(avail).max(start) {
                    let byte = t.int32_data[i / 2] as u8;
                    let n = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    f(nibble(dt, n));
                }
            }
            STRING => {
                for s in t.string_data.iter().skip(start).take(end - start) {
                    f(Elem::S(String::from_utf8_lossy(s).into_owned()));
                }
            }
            INT8 | UINT8 | INT16 | UINT16 | INT32 | BOOL | FLOAT16 | BFLOAT16 | FLOAT8E4M3FN
            | FLOAT8E4M3FNUZ | FLOAT8E5M2 | FLOAT8E5M2FNUZ | FLOAT8E8M0 => {
                for v in t.int32_data.iter().skip(start).take(end - start) {
                    f(int32_elem(dt, *v));
                }
            }
            _ => bail!("Unsupported data type {}", name(dt)),
        },
    }
    Ok(())
}

/// Small integer tensors (shape constants, axes...) for shape inference.
pub fn small_ints(t: &TensorProto, limit: usize) -> Option<Vec<i64>> {
    if !is_integer(t.data_type) || external_info(t).is_some() {
        return None;
    }
    let n = element_count(t);
    if n as usize > limit {
        return None;
    }
    let p = payload(t, &NoSource).ok()?;
    let mut out = Vec::with_capacity(n as usize);
    decode(t, &p, 0, n, &mut |e| {
        if let Some(v) = e.as_i64() {
            out.push(v)
        }
    })
    .ok()?;
    if out.len() as u64 == n {
        Some(out)
    } else {
        None
    }
}

/// Small float tensors (scales, ...) for shape inference.
pub fn small_floats(t: &TensorProto, limit: usize) -> Option<Vec<f64>> {
    if !is_numeric(t.data_type) || external_info(t).is_some() {
        return None;
    }
    let n = element_count(t);
    if n as usize > limit {
        return None;
    }
    let p = payload(t, &NoSource).ok()?;
    let mut out = Vec::with_capacity(n as usize);
    decode(t, &p, 0, n, &mut |e| {
        if let Some(v) = e.as_f64() {
            out.push(v)
        }
    })
    .ok()?;
    if out.len() as u64 == n {
        Some(out)
    } else {
        None
    }
}

/// A short inline rendering such as `[1, 2, 3]` for tiny tensors.
pub fn inline_values(t: &TensorProto, limit: usize) -> Option<String> {
    if external_info(t).is_some() {
        return None;
    }
    let n = element_count(t);
    if n as usize > limit {
        return None;
    }
    let p = payload(t, &NoSource).ok()?;
    let mut out = Vec::new();
    decode(t, &p, 0, n, &mut |e| out.push(e.text())).ok()?;
    if out.len() as u64 != n {
        return None;
    }
    if t.dims.is_empty() && out.len() == 1 {
        return out.pop();
    }
    Some(format!("[{}]", out.join(", ")))
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Histogram {
    pub min: f64,
    pub max: f64,
    pub counts: Vec<u64>,
    pub integer_bins: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TensorStats {
    pub dtype: String,
    pub dims: Vec<i64>,
    pub count: u64,
    pub finite: u64,
    pub nan: u64,
    pub pos_inf: u64,
    pub neg_inf: u64,
    pub zeros: u64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub abs_mean: Option<f64>,
    pub abs_max: Option<f64>,
    pub l2_norm: Option<f64>,
    pub fp16_overflow: u64,
    pub fp16_underflow: u64,
    pub subnormal: u64,
    pub integer: bool,
    pub histogram: Option<Histogram>,
    /// `[percent, value]` pairs.
    pub percentiles: Vec<[f64; 2]>,
    pub sampled: bool,
}

#[derive(Default)]
struct Acc {
    count: u64,
    finite: u64,
    nan: u64,
    pinf: u64,
    ninf: u64,
    zeros: u64,
    min: f64,
    max: f64,
    mean: f64,
    m2: f64,
    abs_sum: f64,
    abs_max: f64,
    sq_sum: f64,
    fp16_over: u64,
    fp16_under: u64,
    subnormal: u64,
}

impl Acc {
    fn push(&mut self, v: f64) {
        self.count += 1;
        if v.is_nan() {
            self.nan += 1;
            return;
        }
        if v.is_infinite() {
            if v > 0.0 {
                self.pinf += 1;
            } else {
                self.ninf += 1;
            }
            return;
        }
        if self.finite == 0 {
            self.min = v;
            self.max = v;
        } else {
            if v < self.min {
                self.min = v;
            }
            if v > self.max {
                self.max = v;
            }
        }
        self.finite += 1;
        if v == 0.0 {
            self.zeros += 1;
        }
        let delta = v - self.mean;
        self.mean += delta / self.finite as f64;
        self.m2 += delta * (v - self.mean);
        let a = v.abs();
        self.abs_sum += a;
        if a > self.abs_max {
            self.abs_max = a;
        }
        self.sq_sum += v * v;
        if a > 65504.0 {
            self.fp16_over += 1;
        } else if a > 0.0 && a < 6.103_515_625e-5 {
            self.fp16_under += 1;
        }
        if a > 0.0 && a < f32::MIN_POSITIVE as f64 {
            self.subnormal += 1;
        }
    }
}

pub fn compute_stats(t: &TensorProto, src: &dyn DataSource, bins: usize) -> Result<TensorStats> {
    let dt = t.data_type;
    if !is_numeric(dt) {
        bail!(
            "Statistics are only available for numeric tensors (this tensor is {})",
            name(dt)
        );
    }
    let n = element_count(t);
    let p = payload(t, src)?;
    let mut acc = Acc::default();
    decode(t, &p, 0, n, &mut |e| {
        if let Some(v) = e.as_f64() {
            acc.push(v)
        }
    })?;

    let integer = is_integer(dt);
    let mut stats = TensorStats {
        dtype: name(dt).to_string(),
        dims: t.dims.clone(),
        count: acc.count,
        finite: acc.finite,
        nan: acc.nan,
        pos_inf: acc.pinf,
        neg_inf: acc.ninf,
        zeros: acc.zeros,
        fp16_overflow: if dt == FLOAT || dt == DOUBLE { acc.fp16_over } else { 0 },
        fp16_underflow: if dt == FLOAT || dt == DOUBLE { acc.fp16_under } else { 0 },
        subnormal: acc.subnormal,
        integer,
        ..Default::default()
    };
    if acc.finite == 0 {
        return Ok(stats);
    }
    stats.min = Some(acc.min);
    stats.max = Some(acc.max);
    stats.mean = Some(acc.mean);
    stats.std = Some((acc.m2 / acc.finite as f64).max(0.0).sqrt());
    stats.abs_mean = Some(acc.abs_sum / acc.finite as f64);
    stats.abs_max = Some(acc.abs_max);
    stats.l2_norm = Some(acc.sq_sum.sqrt());

    // Second pass: histogram and a strided sample for percentiles.
    let (lo, hi) = (acc.min, acc.max);
    let range = hi - lo;
    let integer_bins = integer && range <= 255.0;
    let nbins = if range == 0.0 {
        1
    } else if integer_bins {
        range as usize + 1
    } else {
        bins.clamp(4, 512)
    };
    let mut counts = vec![0u64; nbins];
    let step = acc.finite.div_ceil(100_000).max(1);
    let mut sample: Vec<f64> = Vec::with_capacity((acc.finite / step + 1) as usize);
    let mut k: u64 = 0;
    decode(t, &p, 0, n, &mut |e| {
        let Some(v) = e.as_f64() else { return };
        if !v.is_finite() {
            return;
        }
        let idx = if range == 0.0 {
            0
        } else if integer_bins {
            (v - lo).round() as usize
        } else {
            (((v - lo) / range) * nbins as f64) as usize
        };
        counts[idx.min(nbins - 1)] += 1;
        if k % step == 0 {
            sample.push(v);
        }
        k += 1;
    })?;
    sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if !sample.is_empty() {
        for pct in [0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9] {
            let i = ((pct / 100.0) * (sample.len() - 1) as f64).round() as usize;
            stats.percentiles.push([pct, sample[i.min(sample.len() - 1)]]);
        }
    }
    stats.sampled = step > 1;
    stats.histogram = Some(Histogram {
        min: lo,
        max: hi,
        counts,
        integer_bins,
    });
    Ok(stats)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValuesPage {
    pub total: u64,
    pub offset: u64,
    pub values: Vec<String>,
}

pub fn values_page(
    t: &TensorProto,
    src: &dyn DataSource,
    offset: u64,
    count: u64,
) -> Result<ValuesPage> {
    let total = if t.data_type == STRING {
        t.string_data.len() as u64
    } else {
        element_count(t)
    };
    let start = offset.min(total);
    let end = start.saturating_add(count).min(total);
    let mut values = Vec::with_capacity((end - start) as usize);
    // Read only the needed byte range of external data when possible.
    if let (Some(ext), Some(w)) = (external_info(t), raw_width(t.data_type)) {
        let w = w as u64;
        let bytes = src.read_external(&ext.location, ext.offset + start * w, Some((end - start) * w))?;
        let p = Payload::Raw(Cow::Owned(bytes));
        decode(t, &p, 0, end - start, &mut |e| values.push(e.text()))?;
    } else {
        let p = payload(t, src)?;
        decode(t, &p, start, end, &mut |e| values.push(e.text()))?;
    }
    Ok(ValuesPage {
        total,
        offset: start,
        values,
    })
}

fn npy_header(descr: &str, dims: &[i64]) -> Vec<u8> {
    let shape = match dims.len() {
        0 => "()".to_string(),
        1 => format!("({},)", dims[0]),
        _ => format!(
            "({})",
            dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        ),
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape
    );
    let total = 10 + header.len() + 1;
    let pad = (64 - total % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out
}

/// Serialize a tensor as a NumPy `.npy` file. Types without a NumPy
/// equivalent (bfloat16, float8, float4) are converted to float32; 4-bit
/// integers are widened to 8 bits.
pub fn to_npy(t: &TensorProto, src: &dyn DataSource) -> Result<Vec<u8>> {
    let dt = t.data_type;
    let n = element_count(t);
    let p = payload(t, src)?;
    let (descr, body): (&str, Vec<u8>) = match (dt, &p) {
        (STRING, _) => bail!("String tensors cannot be exported as .npy"),
        (FLOAT | DOUBLE | INT8 | UINT8 | INT16 | UINT16 | INT32 | UINT32 | INT64 | UINT64
        | FLOAT16 | BOOL | COMPLEX64 | COMPLEX128, Payload::Raw(b)) => {
            let w = raw_width(dt).unwrap_or(1) as u64;
            let len = ((n * w) as usize).min(b.len());
            (npy_descr(dt), b[..len].to_vec())
        }
        _ => {
            let mut body = Vec::with_capacity(n as usize * 4);
            let descr = match dt {
                FLOAT | BFLOAT16 | FLOAT8E4M3FN | FLOAT8E4M3FNUZ | FLOAT8E5M2 | FLOAT8E5M2FNUZ
                | FLOAT8E8M0 | FLOAT4E2M1 => "<f4",
                INT4 => "|i1",
                UINT4 => "|u1",
                _ => npy_descr(dt),
            };
            let mut err: Option<anyhow::Error> = None;
            decode(t, &p, 0, n, &mut |e| {
                if err.is_some() {
                    return;
                }
                match (descr, &e) {
                    ("<f4", _) => body.extend_from_slice(&(e.as_f64().unwrap_or(0.0) as f32).to_le_bytes()),
                    ("<f8", Elem::F(v)) => body.extend_from_slice(&v.to_le_bytes()),
                    ("<f2", Elem::F(v)) => body.extend_from_slice(&half::f16::from_f64(*v).to_bits().to_le_bytes()),
                    ("|i1", _) => body.push(e.as_i64().unwrap_or(0) as i8 as u8),
                    ("|u1", _) => body.push(e.as_i64().unwrap_or(0) as u8),
                    ("|b1", _) => body.push(e.as_i64().unwrap_or(0) as u8),
                    ("<i2", _) => body.extend_from_slice(&(e.as_i64().unwrap_or(0) as i16).to_le_bytes()),
                    ("<u2", _) => body.extend_from_slice(&(e.as_i64().unwrap_or(0) as u16).to_le_bytes()),
                    ("<i4", _) => body.extend_from_slice(&(e.as_i64().unwrap_or(0) as i32).to_le_bytes()),
                    ("<u4", _) => body.extend_from_slice(&(e.as_i64().unwrap_or(0) as u32).to_le_bytes()),
                    ("<i8", _) => body.extend_from_slice(&e.as_i64().unwrap_or(0).to_le_bytes()),
                    ("<u8", Elem::U(v)) => body.extend_from_slice(&v.to_le_bytes()),
                    ("<u8", _) => body.extend_from_slice(&(e.as_i64().unwrap_or(0) as u64).to_le_bytes()),
                    ("<c8", Elem::C(re, im)) => {
                        body.extend_from_slice(&(*re as f32).to_le_bytes());
                        body.extend_from_slice(&(*im as f32).to_le_bytes());
                    }
                    ("<c16", Elem::C(re, im)) => {
                        body.extend_from_slice(&re.to_le_bytes());
                        body.extend_from_slice(&im.to_le_bytes());
                    }
                    _ => err = Some(anyhow!("Cannot convert element for .npy export")),
                }
            })?;
            if let Some(e) = err {
                return Err(e);
            }
            (descr, body)
        }
    };
    let mut out = npy_header(descr, &t.dims);
    out.extend_from_slice(&body);
    Ok(out)
}

fn npy_descr(dt: i32) -> &'static str {
    match dt {
        FLOAT => "<f4",
        DOUBLE => "<f8",
        FLOAT16 => "<f2",
        INT8 => "|i1",
        UINT8 => "|u1",
        BOOL => "|b1",
        INT16 => "<i2",
        UINT16 => "<u2",
        INT32 => "<i4",
        UINT32 => "<u4",
        INT64 => "<i8",
        UINT64 => "<u8",
        COMPLEX64 => "<c8",
        COMPLEX128 => "<c16",
        _ => "<f4",
    }
}

/// CSV with one row per innermost vector (or one value per line for 0/1-D).
pub fn to_csv(t: &TensorProto, src: &dyn DataSource) -> Result<Vec<u8>> {
    let n = if t.data_type == STRING {
        t.string_data.len() as u64
    } else {
        element_count(t)
    };
    let p = payload(t, src)?;
    let cols = if t.dims.len() >= 2 {
        (*t.dims.last().unwrap()).max(1) as u64
    } else {
        1
    };
    let mut out = String::with_capacity(n as usize * 8);
    let mut k: u64 = 0;
    decode(t, &p, 0, n, &mut |e| {
        let s = match &e {
            Elem::S(s) => format!("\"{}\"", s.replace('"', "\"\"")),
            Elem::F(v) => format!("{}", v),
            _ => e.text(),
        };
        out.push_str(&s);
        k += 1;
        out.push(if k % cols == 0 { '\n' } else { ',' });
    })?;
    Ok(out.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_format() {
        assert_eq!(fmt_float(1.0), "1");
        assert_eq!(fmt_float(0.5), "0.5");
        assert_eq!(fmt_float(1e-7), "1e-7");
        assert_eq!(fmt_float(-2.5e10), "-2.5e10");
        assert_eq!(fmt_float(0.1f32 as f64), "0.1");
    }

    #[test]
    fn fp8_conversions() {
        assert_eq!(f8e4m3fn(0x38), 1.0);
        assert_eq!(f8e4m3fn(0x7E), 448.0);
        assert!(f8e4m3fn(0x7F).is_nan());
        assert_eq!(f8e5m2(0x3C), 1.0);
        assert_eq!(f8e4m3fnuz(0x40), 1.0);
        assert_eq!(f8e5m2fnuz(0x40), 1.0);
        assert_eq!(f4e2m1(0x7), 6.0);
        assert_eq!(f4e2m1(0x2), 1.0);
        assert_eq!(f8e8m0(127), 1.0);
    }

    #[test]
    fn stats_and_values() {
        let t = TensorProto {
            dims: vec![2, 3],
            data_type: FLOAT,
            float_data: vec![1.0, -2.0, 0.0, 4.0, f32::NAN, 3.0],
            ..Default::default()
        };
        let s = compute_stats(&t, &NoSource, 16).unwrap();
        assert_eq!(s.count, 6);
        assert_eq!(s.nan, 1);
        assert_eq!(s.zeros, 1);
        assert_eq!(s.min, Some(-2.0));
        assert_eq!(s.max, Some(4.0));
        assert!((s.mean.unwrap() - 1.2).abs() < 1e-9);
        let page = values_page(&t, &NoSource, 1, 2).unwrap();
        assert_eq!(page.values, vec!["-2", "0"]);
        let npy = to_npy(&t, &NoSource).unwrap();
        assert_eq!(&npy[..6], b"\x93NUMPY");
        assert_eq!((npy.len() - 6 * 4) % 64, 0);
    }

    #[test]
    fn raw_int4() {
        let t = TensorProto {
            dims: vec![3],
            data_type: INT4,
            raw_data: prost::bytes::Bytes::from(vec![0xF1u8, 0x07]),
            ..Default::default()
        };
        let page = values_page(&t, &NoSource, 0, 10).unwrap();
        assert_eq!(page.values, vec!["1", "-1", "7"]);
        assert_eq!(small_ints(&t, 8), Some(vec![1, -1, 7]));
    }
}
