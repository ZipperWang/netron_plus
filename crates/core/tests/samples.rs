//! Runs every model in `$NETRON_SAMPLES` through the whole API and compares
//! the inferred shapes with ONNX's reference shape inference (written by
//! `scripts/gen_samples.py` as `<model>.shapes.json`).

use netron_core::Workspace;
use serde_json::{json, Value};
use std::collections::HashMap;

fn label(dims: &[Value]) -> String {
    if dims.is_empty() {
        return "scalar".into();
    }
    dims.iter()
        .map(|d| match d {
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
            _ => "?".into(),
        })
        .collect::<Vec<_>>()
        .join("×")
}

#[test]
fn sample_models() {
    let Ok(dir) = std::env::var("NETRON_SAMPLES") else {
        eprintln!("NETRON_SAMPLES is not set; skipping sample model tests");
        return;
    };
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("samples dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("onnx"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no sample models found in {}", dir);

    let (mut total_known, mut total_match) = (0usize, 0usize);
    for path in &paths {
        let ws = Workspace::new();
        let start = std::time::Instant::now();
        let p = path.to_str().unwrap();
        let summary: Value = serde_json::from_str(
            &ws.open_path(p)
                .unwrap_or_else(|e| panic!("{}: {:#}", p, e)),
        )
        .unwrap();
        let id = summary["id"].as_u64().unwrap();
        let ngraphs = summary["graphs"].as_array().unwrap().len();
        let mut view_nodes = 0;
        for g in 0..ngraphs {
            let req = json!({"model": id, "graph": g});
            let view: Value = serde_json::from_str(&ws.call("graph", &json!({"model": id, "graph": g, "options": {"showAttributes": true, "showNames": true}})).unwrap()).unwrap();
            view_nodes += view["nodes"].as_array().unwrap().len();
            ws.call("nodeTable", &req).unwrap();
            ws.call("valueTable", &req).unwrap();
            ws.call("hierarchy", &req).unwrap();
            let nodes = summary["graphs"][g]["nodes"].as_u64().unwrap() as usize;
            for n in 0..nodes.min(50) {
                ws.call("node", &json!({"model": id, "graph": g, "node": n})).unwrap();
            }
        }
        ws.call("checks", &json!({"model": id, "deep": true})).unwrap();
        let weights: Value = serde_json::from_str(&ws.call("weightTable", &json!({"model": id})).unwrap()).unwrap();
        for w in weights.as_array().unwrap().iter().take(20) {
            let req = json!({"model": id, "graph": w["graph"], "tensor": w["tensor"], "count": 32});
            ws.call("tensorStats", &req).unwrap();
            ws.call("tensorValues", &req).unwrap();
            ws.export(&json!({"model": id, "graph": w["graph"], "tensor": w["tensor"], "format": "npy"})).unwrap();
        }

        let expected = path.with_extension("shapes.json");
        let mut line = format!(
            "{:<36} nodes={:<6} view={:<6} {:>6.0}ms",
            path.file_name().unwrap().to_string_lossy(),
            summary["stats"]["nodesTotal"],
            view_nodes,
            start.elapsed().as_secs_f64() * 1000.0
        );
        if expected.exists() {
            let exp: Value = serde_json::from_str(&std::fs::read_to_string(&expected).unwrap()).unwrap();
            let table: Value = serde_json::from_str(&ws.call("valueTable", &json!({"model": id, "graph": 0})).unwrap()).unwrap();
            let ours: HashMap<String, String> = table
                .as_array()
                .unwrap()
                .iter()
                .map(|r| (r["name"].as_str().unwrap().to_string(), r["shape"].as_str().unwrap().to_string()))
                .collect();
            let (mut known, mut matched) = (0, 0);
            let mut misses = Vec::new();
            for (name, info) in exp.as_object().unwrap() {
                let Some(dims) = info["shape"].as_array() else { continue };
                if !dims.iter().all(|d| d.is_number()) {
                    continue;
                }
                known += 1;
                let want = label(dims);
                let got = ours.get(name).cloned().unwrap_or_else(|| "<missing>".into());
                if got == want {
                    matched += 1;
                } else if misses.len() < 12 {
                    misses.push(format!("    {}: expected {} got {}", name, want, got));
                }
            }
            total_known += known;
            total_match += matched;
            line.push_str(&format!(" shapes {}/{}", matched, known));
            println!("{}", line);
            for m in misses {
                println!("{}", m);
            }
        } else {
            println!("{}", line);
        }
    }
    println!("shape inference agreement: {}/{}", total_match, total_known);
    if total_known > 0 {
        let ratio = total_match as f64 / total_known as f64;
        assert!(ratio >= 0.9, "shape inference agreement too low: {:.1}%", ratio * 100.0);
    }
}
