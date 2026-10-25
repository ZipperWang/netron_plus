//! Tensor dimensions, including symbolic dimensions and small symbolic
//! arithmetic used by shape inference.

use serde::{Serialize, Serializer};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Known(i64),
    Param(String),
    Unknown,
}

impl Serialize for Dim {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Dim::Known(v) => s.serialize_i64(*v),
            Dim::Param(p) => s.serialize_str(p),
            Dim::Unknown => s.serialize_none(),
        }
    }
}

impl Dim {
    pub fn known(&self) -> Option<i64> {
        match self {
            Dim::Known(v) => Some(*v),
            _ => None,
        }
    }

    pub fn is_known(&self) -> bool {
        matches!(self, Dim::Known(_))
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, Dim::Unknown)
    }

    pub fn label(&self) -> String {
        match self {
            Dim::Known(v) => v.to_string(),
            Dim::Param(p) => p.clone(),
            Dim::Unknown => "?".to_string(),
        }
    }
}

pub type Shape = Vec<Dim>;

pub fn from_i64s(v: &[i64]) -> Shape {
    v.iter().map(|&d| Dim::Known(d)).collect()
}

/// Number of elements, if every dimension is known.
pub fn numel(s: &[Dim]) -> Option<u64> {
    let mut n: u64 = 1;
    for d in s {
        match d {
            Dim::Known(v) if *v >= 0 => n = n.saturating_mul(*v as u64),
            _ => return None,
        }
    }
    Some(n)
}

pub fn numel_i64(dims: &[i64]) -> u64 {
    dims.iter()
        .fold(1u64, |acc, &d| acc.saturating_mul(d.max(0) as u64))
}

/// "1×3×224×224"; scalars render as "scalar" and unknown rank as "?".
pub fn dims_label(s: Option<&[Dim]>) -> String {
    match s {
        None => "?".to_string(),
        Some([]) => "scalar".to_string(),
        Some(s) => s.iter().map(Dim::label).collect::<Vec<_>>().join("×"),
    }
}

/// "float32[1,3,224,224]"
pub fn type_label(dtype: &str, s: Option<&[Dim]>) -> String {
    match s {
        None => {
            if dtype.is_empty() {
                "?".to_string()
            } else {
                dtype.to_string()
            }
        }
        Some(s) => format!(
            "{}[{}]",
            if dtype.is_empty() { "?" } else { dtype },
            s.iter().map(Dim::label).collect::<Vec<_>>().join(",")
        ),
    }
}

fn wrap(p: &str) -> String {
    if p.contains('+') || p.contains('-') {
        format!("({})", p)
    } else {
        p.to_string()
    }
}

pub fn mul(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Known(x), Dim::Known(y)) => Dim::Known(x.saturating_mul(*y)),
        (Dim::Known(0), _) | (_, Dim::Known(0)) => Dim::Known(0),
        (Dim::Known(1), d) | (d, Dim::Known(1)) => d.clone(),
        (Dim::Known(x), Dim::Param(p)) | (Dim::Param(p), Dim::Known(x)) => {
            Dim::Param(format!("{}*{}", x, wrap(p)))
        }
        (Dim::Param(p), Dim::Param(q)) => Dim::Param(format!("{}*{}", wrap(p), wrap(q))),
        _ => Dim::Unknown,
    }
}

pub fn add(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Known(x), Dim::Known(y)) => Dim::Known(x.saturating_add(*y)),
        (Dim::Known(0), d) | (d, Dim::Known(0)) => d.clone(),
        (Dim::Param(p), Dim::Known(x)) | (Dim::Known(x), Dim::Param(p)) => {
            if *x < 0 {
                Dim::Param(format!("{}-{}", p, -x))
            } else {
                Dim::Param(format!("{}+{}", p, x))
            }
        }
        (Dim::Param(p), Dim::Param(q)) => Dim::Param(format!("{}+{}", p, q)),
        _ => Dim::Unknown,
    }
}

pub fn sub(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Known(x), Dim::Known(y)) => Dim::Known(x.saturating_sub(*y)),
        (d, Dim::Known(0)) => d.clone(),
        (Dim::Param(p), Dim::Known(x)) => Dim::Param(format!("{}-{}", p, x)),
        (Dim::Param(p), Dim::Param(q)) if p == q => Dim::Known(0),
        _ => Dim::Unknown,
    }
}

/// Integer division as used by shape arithmetic (floor for known values).
pub fn div(a: &Dim, b: &Dim) -> Dim {
    match (a, b) {
        (Dim::Known(x), Dim::Known(y)) if *y != 0 => Dim::Known(x.div_euclid(*y)),
        (d, Dim::Known(1)) => d.clone(),
        (Dim::Param(p), Dim::Param(q)) if p == q => Dim::Known(1),
        (Dim::Param(p), Dim::Known(y)) => {
            // Undo a previous "k*p" multiplication when possible.
            let prefix = format!("{}*", y);
            if let Some(rest) = p.strip_prefix(&prefix) {
                if !rest.contains('*') && !rest.contains('+') && !rest.contains('-') {
                    return Dim::Param(rest.to_string());
                }
            }
            Dim::Unknown
        }
        _ => Dim::Unknown,
    }
}

/// Product of a list of dims (empty = 1).
pub fn product(dims: &[Dim]) -> Dim {
    dims.iter().fold(Dim::Known(1), |acc, d| mul(&acc, d))
}

/// Multidirectional (numpy-style) broadcasting of several shapes.
pub fn broadcast(shapes: &[&[Dim]]) -> Shape {
    let rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let mut result: Option<Dim> = None;
        let mut conflict = false;
        for s in shapes {
            if s.len() + i < rank {
                continue; // implicit leading 1
            }
            let d = &s[s.len() + i - rank];
            result = Some(match (result.take(), d) {
                (None, d) => d.clone(),
                (Some(Dim::Known(1)), d) => d.clone(),
                (Some(r), Dim::Known(1)) => r,
                (Some(Dim::Known(a)), Dim::Known(b)) => {
                    if a != *b {
                        conflict = true;
                    }
                    Dim::Known(a)
                }
                (Some(Dim::Known(a)), _) => Dim::Known(a),
                (Some(_), Dim::Known(b)) => Dim::Known(*b),
                (Some(Dim::Param(p)), Dim::Param(q)) => {
                    if &p == q {
                        Dim::Param(p)
                    } else {
                        Dim::Unknown
                    }
                }
                (Some(Dim::Param(p)), Dim::Unknown) => Dim::Param(p),
                (Some(Dim::Unknown), Dim::Param(q)) => Dim::Param(q.clone()),
                (Some(Dim::Unknown), Dim::Unknown) => Dim::Unknown,
            });
        }
        let r = result.unwrap_or(Dim::Known(1));
        out.push(if conflict { Dim::Unknown } else { r });
    }
    out
}

/// Normalize a possibly negative axis against `rank`.
pub fn norm_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a >= 0 && a < r {
        Some(a as usize)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_basic() {
        let a = from_i64s(&[8, 1, 6, 1]);
        let b = from_i64s(&[7, 1, 5]);
        assert_eq!(broadcast(&[&a, &b]), from_i64s(&[8, 7, 6, 5]));
        let c = vec![Dim::Param("N".into()), Dim::Known(3)];
        let d = vec![Dim::Known(1), Dim::Known(3)];
        assert_eq!(
            broadcast(&[&c, &d]),
            vec![Dim::Param("N".into()), Dim::Known(3)]
        );
    }

    #[test]
    fn symbolic_arith() {
        let n = Dim::Param("N".into());
        assert_eq!(mul(&n, &Dim::Known(1)), n);
        assert_eq!(mul(&n, &Dim::Known(4)), Dim::Param("4*N".into()));
        assert_eq!(div(&Dim::Param("4*N".into()), &Dim::Known(4)), n);
        assert_eq!(add(&n, &Dim::Known(0)), n);
        assert_eq!(numel(&from_i64s(&[2, 3, 4])), Some(24));
        assert_eq!(dims_label(Some(&from_i64s(&[1, 3]))), "1×3");
    }
}
