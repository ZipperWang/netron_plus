//! Collects every TeX snippet the operator function API produces: the
//! definitions themselves and their symbols bound to each node of the sample
//! models in `$NETRON_SAMPLES`. With `$NETRON_TEX_DUMP` set they are written
//! there as JSON for `scripts/check_tex.mjs`, which renders all of them with
//! KaTeX in CI.

use netron_core::{opfunc, tex, Workspace};
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Nodes checked per graph; bound values repeat a lot in big models.
const MAX_NODES: u64 = 2000;

#[test]
fn tex_snippets() {
    // (tex, display) → where it came from
    let mut all: BTreeMap<(String, bool), String> = BTreeMap::new();
    for (origin, t, display) in opfunc::snippets() {
        all.entry((t, display)).or_insert(origin);
    }
    let from_definitions = all.len();

    if let Ok(dir) = std::env::var("NETRON_SAMPLES") {
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .expect("samples dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("onnx"))
            .collect();
        paths.sort();
        for path in &paths {
            let ws = Workspace::new();
            let p = path.to_str().unwrap();
            let file = path.file_name().unwrap().to_string_lossy().into_owned();
            let summary: Value = serde_json::from_str(
                &ws.open_path(p)
                    .unwrap_or_else(|e| panic!("{}: {:#}", p, e)),
            )
            .unwrap();
            let id = summary["id"].as_u64().unwrap();
            for g in summary["graphs"].as_array().unwrap() {
                let gid = g["id"].as_u64().unwrap();
                for n in 0..g["nodes"].as_u64().unwrap().min(MAX_NODES) {
                    let out = ws
                        .call("opFunction", &json!({"model": id, "graph": gid, "node": n}))
                        .unwrap_or_else(|e| panic!("{} graph {} node {}: {:#}", file, gid, n, e));
                    let f: Value = serde_json::from_str(&out).unwrap();
                    for s in f["symbols"].as_array().unwrap() {
                        let Some(v) = s["value"].as_str() else { continue };
                        // The front end renders a symbol row as "symbol = value".
                        let row = format!("{} = {}", s["tex"].as_str().unwrap(), v);
                        all.entry((row, false))
                            .or_insert_with(|| format!("{} graph {} node {} ({})", file, gid, n, f["op"].as_str().unwrap_or("")));
                    }
                }
            }
        }
    }

    let unbalanced: Vec<String> = all
        .iter()
        .filter(|((t, _), _)| !tex::balanced(t))
        .map(|((t, _), origin)| format!("{}: {}", origin, t))
        .collect();
    assert!(unbalanced.is_empty(), "unbalanced TeX:\n{}", unbalanced.join("\n"));
    println!("{} TeX snippets ({} from definitions)", all.len(), from_definitions);

    if let Ok(out) = std::env::var("NETRON_TEX_DUMP") {
        let items: Vec<Value> = all
            .into_iter()
            .map(|((t, display), origin)| json!({"where": origin, "tex": t, "display": display}))
            .collect();
        if let Some(dir) = std::path::Path::new(&out).parent() {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(&out, serde_json::to_string_pretty(&items).unwrap()).unwrap();
        println!("wrote {}", out);
    }
}
