//! Small helpers for building TeX snippets from model data. Everything that
//! comes from the model (names, strings, symbolic dimensions) is escaped so
//! the front end can render the result with KaTeX.

use crate::shape::Dim;
use crate::tensor;

/// Lists longer than this are cut with `\ldots`.
const MAX_ITEMS: usize = 8;
/// Strings longer than this are shortened.
const MAX_TEXT: usize = 48;

/// Escape text for use inside `\text{}` or `\texttt{}`.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '{' | '}' | '$' | '&' | '#' | '%' | '_' => {
                out.push('\\');
                out.push(c);
            }
            '^' => out.push_str("\\textasciicircum{}"),
            '~' => out.push_str("\\textasciitilde{}"),
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

/// A model string as typewriter text.
pub fn texttt(s: &str) -> String {
    format!("\\texttt{{{}}}", escape(&crate::format::truncate(s, MAX_TEXT)))
}

/// A number, using `\times10^{n}` instead of `e` notation.
pub fn num(v: f64) -> String {
    let s = tensor::fmt_float(v);
    match s.as_str() {
        "NaN" => return "\\text{NaN}".into(),
        "inf" => return "\\infty".into(),
        "-inf" => return "-\\infty".into(),
        _ => {}
    }
    match s.split_once('e') {
        Some((m, e)) => match m {
            "1" => format!("10^{{{}}}", e),
            "-1" => format!("-10^{{{}}}", e),
            _ => format!("{}\\times10^{{{}}}", m, e),
        },
        None => s,
    }
}

pub fn dim(d: &Dim) -> String {
    match d {
        Dim::Known(v) => v.to_string(),
        Dim::Param(p) => format!("\\textit{{{}}}", escape(&crate::format::truncate(p, MAX_TEXT))),
        Dim::Unknown => "?".into(),
    }
}

/// `(a,\, b,\, c)`, shortened with `\ldots` after [`MAX_ITEMS`] items.
pub fn tuple<I: IntoIterator<Item = String>>(items: I) -> String {
    let mut v: Vec<String> = Vec::new();
    let mut more = false;
    for (i, s) in items.into_iter().enumerate() {
        if i == MAX_ITEMS {
            more = true;
            break;
        }
        v.push(s);
    }
    if more {
        v.push("\\ldots".into());
    }
    format!("({})", v.join(",\\, "))
}

pub fn shape(dims: &[Dim]) -> String {
    tuple(dims.iter().map(dim))
}

/// Do braces balance (ignoring escaped ones)? Used by tests.
pub fn balanced(s: &str) -> bool {
    let mut depth = 0i64;
    let mut escaped = false;
    for c in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' => escaped = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_model_text() {
        assert_eq!(escape("a_b{c}%"), "a\\_b\\{c\\}\\%");
        assert_eq!(escape("x\\y^z~"), "x\\textbackslash{}y\\textasciicircum{}z\\textasciitilde{}");
        assert!(balanced(&texttt("weird}{name_\\")));
        assert_eq!(dim(&Dim::Param("batch_size".into())), "\\textit{batch\\_size}");
    }

    #[test]
    fn numbers() {
        assert_eq!(num(0.5), "0.5");
        assert_eq!(num(1e-5), "10^{-5}");
        assert_eq!(num(2.5e-7), "2.5\\times10^{-7}");
        assert_eq!(num(f64::INFINITY), "\\infty");
        assert_eq!(num(3.0), "3");
    }

    #[test]
    fn tuples() {
        assert_eq!(tuple(["1".to_string(), "2".to_string()]), "(1,\\, 2)");
        let long = tuple((0..20).map(|i| i.to_string()));
        assert!(long.ends_with(",\\, \\ldots)"));
        assert!(balanced("\\frac{a}{b}"));
        assert!(!balanced("\\frac{a}{b"));
        assert!(balanced("\\{"));
    }
}
