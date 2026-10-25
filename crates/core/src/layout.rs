//! Layered (Sugiyama-style) graph layout.
//!
//! 1. Cycles are broken by reversing DFS back edges.
//! 2. Nodes are layered by longest path, then pulled down toward their
//!    successors so that constants and side inputs sit next to their users.
//! 3. Edges spanning several layers are split with virtual nodes.
//! 4. Crossings are reduced with alternating barycenter sweeps.
//! 5. X coordinates come from iterative weighted isotonic regression: each
//!    layer is placed optimally (Σ w·(x_u − x_v)²) given its neighbours while
//!    keeping the order and spacing, sweeping up and down until stable.
//! 6. Edges are routed through their virtual nodes and ports are spread
//!    along the node borders.

use std::cmp::Ordering;
use std::collections::VecDeque;

pub struct LNode {
    pub w: f64,
    pub h: f64,
}

pub struct LEdge {
    pub src: usize,
    pub dst: usize,
    pub weight: f64,
    /// Very long edges with this flag are not routed (drawn as stubs).
    pub detachable: bool,
}

pub struct LayoutParams {
    pub rank_sep: f64,
    pub node_sep: f64,
    pub edge_sep: f64,
    pub detach_span: usize,
}

impl Default for LayoutParams {
    fn default() -> Self {
        LayoutParams {
            rank_sep: 36.0,
            node_sep: 24.0,
            edge_sep: 10.0,
            detach_span: usize::MAX,
        }
    }
}

#[derive(Debug, Default)]
pub struct LayoutResult {
    /// Node centers.
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub rank: Vec<usize>,
    /// Polyline for every edge from source to target (empty if not routed).
    pub points: Vec<Vec<(f64, f64)>>,
    pub detached: Vec<bool>,
    pub reversed: Vec<bool>,
    pub width: f64,
    pub height: f64,
}

pub fn layout(nodes: &[LNode], edges: &[LEdge], p: &LayoutParams) -> LayoutResult {
    let n = nodes.len();
    let m = edges.len();
    let mut res = LayoutResult {
        x: vec![0.0; n],
        y: vec![0.0; n],
        rank: vec![0; n],
        points: vec![Vec::new(); m],
        detached: vec![false; m],
        reversed: vec![false; m],
        width: 0.0,
        height: 0.0,
    };
    if n == 0 {
        return res;
    }
    let valid = |e: &LEdge| e.src != e.dst && e.src < n && e.dst < n;

    // 1. Cycle breaking.
    let mut out_adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (i, e) in edges.iter().enumerate() {
        if valid(e) {
            out_adj[e.src].push((e.dst, i));
        }
    }
    let mut state = vec![0u8; n];
    for s in 0..n {
        if state[s] != 0 {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = vec![(s, 0)];
        state[s] = 1;
        loop {
            let Some(top) = stack.last_mut() else { break };
            let v = top.0;
            if top.1 < out_adj[v].len() {
                let (w, ei) = out_adj[v][top.1];
                top.1 += 1;
                match state[w] {
                    0 => {
                        state[w] = 1;
                        stack.push((w, 0));
                    }
                    1 => res.reversed[ei] = true,
                    _ => {}
                }
            } else {
                state[v] = 2;
                stack.pop();
            }
        }
    }

    let mut dag: Vec<Option<(usize, usize)>> = vec![None; m];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indeg = vec![0usize; n];
    for (i, e) in edges.iter().enumerate() {
        if !valid(e) {
            continue;
        }
        let (u, v) = if res.reversed[i] { (e.dst, e.src) } else { (e.src, e.dst) };
        dag[i] = Some((u, v));
        succ[u].push(v);
        indeg[v] += 1;
    }

    // 2. Layering.
    let mut rank = vec![0usize; n];
    let mut order = Vec::with_capacity(n);
    let mut q: VecDeque<usize> = (0..n).filter(|&v| indeg[v] == 0).collect();
    while let Some(u) = q.pop_front() {
        order.push(u);
        for &v in &succ[u] {
            if rank[u] + 1 > rank[v] {
                rank[v] = rank[u] + 1;
            }
            indeg[v] -= 1;
            if indeg[v] == 0 {
                q.push_back(v);
            }
        }
    }
    if order.len() < n {
        let mut seen = vec![false; n];
        for &u in &order {
            seen[u] = true;
        }
        order.extend((0..n).filter(|&v| !seen[v]));
    }
    for &u in order.iter().rev() {
        if let Some(mn) = succ[u].iter().map(|&v| rank[v]).min() {
            if mn >= 1 && mn - 1 > rank[u] {
                rank[u] = mn - 1;
            }
        }
    }
    for (i, e) in edges.iter().enumerate() {
        if let Some((u, v)) = dag[i] {
            if e.detachable && rank[v].saturating_sub(rank[u]) > p.detach_span {
                res.detached[i] = true;
            }
        }
    }

    // 3. Virtual nodes.
    let mut vrank: Vec<usize> = rank.clone();
    let mut vw: Vec<f64> = nodes.iter().map(|nd| nd.w).collect();
    let mut is_virtual = vec![false; n];
    let mut lsucc: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut lpred: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut chains: Vec<Vec<usize>> = vec![Vec::new(); m];
    for i in 0..m {
        let Some((u, v)) = dag[i] else { continue };
        if res.detached[i] {
            continue;
        }
        let w0 = edges[i].weight.max(0.01);
        let mut chain = vec![u];
        let mut prev = u;
        for r in rank[u] + 1..rank[v] {
            let d = vrank.len();
            vrank.push(r);
            vw.push(p.edge_sep);
            is_virtual.push(true);
            lsucc.push(Vec::new());
            lpred.push(Vec::new());
            let w = w0 * if is_virtual[prev] { 8.0 } else { 2.0 };
            lsucc[prev].push((d, w));
            lpred[d].push((prev, w));
            chain.push(d);
            prev = d;
        }
        let w = w0 * if is_virtual[prev] { 2.0 } else { 1.0 };
        lsucc[prev].push((v, w));
        lpred[v].push((prev, w));
        chain.push(v);
        chains[i] = chain;
    }
    let total = vrank.len();

    // 4. Ordering.
    let max_rank = vrank.iter().copied().max().unwrap_or(0);
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
    let mut visited = vec![false; total];
    let roots: Vec<usize> = (0..total)
        .filter(|&v| lpred[v].is_empty())
        .chain(0..total)
        .collect();
    let mut stack: Vec<usize> = Vec::new();
    for root in roots {
        if visited[root] {
            continue;
        }
        stack.push(root);
        while let Some(v) = stack.pop() {
            if visited[v] {
                continue;
            }
            visited[v] = true;
            layers[vrank[v]].push(v);
            for &(w, _) in lsucc[v].iter().rev() {
                if !visited[w] {
                    stack.push(w);
                }
            }
        }
    }
    let mut pos = vec![0usize; total];
    set_positions(&layers, &mut pos);
    let mut best = layers.clone();
    let mut best_c = total_crossings(&layers, &lsucc, &pos);
    let iters = if total > 60_000 {
        4
    } else if total > 15_000 {
        8
    } else {
        24
    };
    for it in 0..iters {
        if best_c == 0 {
            break;
        }
        if it % 2 == 0 {
            for r in 1..=max_rank {
                reorder(&mut layers[r], &lpred, &mut pos);
            }
        } else {
            for r in (0..max_rank).rev() {
                reorder(&mut layers[r], &lsucc, &mut pos);
            }
        }
        let c = total_crossings(&layers, &lsucc, &pos);
        if c < best_c {
            best_c = c;
            best = layers.clone();
        }
    }
    let layers = best;
    set_positions(&layers, &mut pos);

    // 5. X coordinates.
    let sep = |a: usize, b: usize| -> f64 {
        (vw[a] + vw[b]) / 2.0
            + match (is_virtual[a], is_virtual[b]) {
                (true, true) => p.edge_sep,
                (false, false) => p.node_sep,
                _ => (p.node_sep + p.edge_sep) / 2.0,
            }
    };
    let mut x = vec![0.0f64; total];
    for l in &layers {
        let mut cur = 0.0;
        for (i, &v) in l.iter().enumerate() {
            if i > 0 {
                cur += sep(l[i - 1], v);
            }
            x[v] = cur;
        }
        let shift = cur / 2.0;
        for &v in l {
            x[v] -= shift;
        }
    }
    let sweeps = if total > 60_000 {
        6
    } else if total > 15_000 {
        14
    } else {
        32
    };
    for s in 0..sweeps {
        if s % 2 == 0 {
            for l in layers.iter() {
                place(l, &mut x, &lpred, &lsucc, &sep);
            }
        } else {
            for l in layers.iter().rev() {
                place(l, &mut x, &lpred, &lsucc, &sep);
            }
        }
    }

    // Y coordinates.
    let mut lh = vec![0.0f64; max_rank + 1];
    for v in 0..n {
        lh[vrank[v]] = lh[vrank[v]].max(nodes[v].h);
    }
    let mut ytop = vec![0.0f64; max_rank + 1];
    let mut cur = 0.0;
    for r in 0..=max_rank {
        ytop[r] = cur;
        cur += lh[r] + p.rank_sep;
    }
    let ymid = |r: usize| ytop[r] + lh[r] / 2.0;

    let mut minx = f64::INFINITY;
    let mut maxx = f64::NEG_INFINITY;
    for v in 0..total {
        let hw = vw[v] / 2.0;
        minx = minx.min(x[v] - hw);
        maxx = maxx.max(x[v] + hw);
    }
    for v in x.iter_mut() {
        *v -= minx;
    }
    res.width = (maxx - minx).max(0.0);
    res.height = (cur - p.rank_sep).max(0.0);
    for v in 0..n {
        res.x[v] = x[v];
        res.y[v] = ymid(vrank[v]);
        res.rank[v] = vrank[v];
    }

    // 6. Routing.
    let mut top_ports: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut bot_ports: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..m {
        let chain = &chains[i];
        if chain.len() < 2 {
            continue;
        }
        let first = chain[0];
        let last = chain[chain.len() - 1];
        let mut pts: Vec<(f64, f64)> = Vec::with_capacity(chain.len());
        pts.push((x[first], ymid(vrank[first]) + nodes[first].h / 2.0));
        for &d in &chain[1..chain.len() - 1] {
            pts.push((x[d], ymid(vrank[d])));
        }
        pts.push((x[last], ymid(vrank[last]) - nodes[last].h / 2.0));
        bot_ports[first].push((i, x[chain[1]]));
        top_ports[last].push((i, x[chain[chain.len() - 2]]));
        res.points[i] = pts;
    }
    for v in 0..n {
        spread(&mut top_ports[v], x[v], nodes[v].w, &mut res.points, false);
        spread(&mut bot_ports[v], x[v], nodes[v].w, &mut res.points, true);
    }
    for (i, e) in edges.iter().enumerate() {
        if res.reversed[i] {
            res.points[i].reverse();
        }
        if e.src == e.dst && e.src < n {
            let (cx, cy) = (x[e.src], ymid(vrank[e.src]));
            let r = nodes[e.src].w / 2.0;
            res.points[i] = vec![
                (cx + r, cy - 6.0),
                (cx + r + 24.0, cy - 6.0),
                (cx + r + 24.0, cy + 6.0),
                (cx + r, cy + 6.0),
            ];
        }
    }
    res
}

fn set_positions(layers: &[Vec<usize>], pos: &mut [usize]) {
    for l in layers {
        for (i, &v) in l.iter().enumerate() {
            pos[v] = i;
        }
    }
}

fn reorder(layer: &mut [usize], nbr: &[Vec<(usize, f64)>], pos: &mut [usize]) {
    let mut keyed: Vec<(f64, usize, usize)> = layer
        .iter()
        .map(|&v| {
            let ns = &nbr[v];
            let key = if ns.is_empty() {
                pos[v] as f64
            } else {
                ns.iter().map(|&(u, _)| pos[u] as f64).sum::<f64>() / ns.len() as f64
            };
            (key, pos[v], v)
        })
        .collect();
    keyed.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    for (i, k) in keyed.iter().enumerate() {
        layer[i] = k.2;
        pos[k.2] = i;
    }
}

fn total_crossings(layers: &[Vec<usize>], lsucc: &[Vec<(usize, f64)>], pos: &[usize]) -> u64 {
    let mut total = 0u64;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for r in 0..layers.len().saturating_sub(1) {
        let lower = layers[r + 1].len();
        pairs.clear();
        for &u in &layers[r] {
            for &(v, _) in &lsucc[u] {
                pairs.push((pos[u], pos[v]));
            }
        }
        if pairs.len() < 2 {
            continue;
        }
        pairs.sort_unstable();
        let mut bit = vec![0u32; lower + 1];
        let mut seen = 0u64;
        for &(_, b) in &pairs {
            let mut i = b + 1;
            let mut le = 0u64;
            while i > 0 {
                le += bit[i] as u64;
                i -= i & i.wrapping_neg();
            }
            total += seen - le;
            let mut i = b + 1;
            while i <= lower {
                bit[i] += 1;
                i += i & i.wrapping_neg();
            }
            seen += 1;
        }
    }
    total
}

fn place(
    layer: &[usize],
    x: &mut [f64],
    lpred: &[Vec<(usize, f64)>],
    lsucc: &[Vec<(usize, f64)>],
    sep: &dyn Fn(usize, usize) -> f64,
) {
    let k = layer.len();
    if k == 0 {
        return;
    }
    let mut off = vec![0.0; k];
    for i in 1..k {
        off[i] = off[i - 1] + sep(layer[i - 1], layer[i]);
    }
    let mut t = vec![0.0; k];
    let mut w = vec![0.0; k];
    for (i, &v) in layer.iter().enumerate() {
        let mut sw = 0.0;
        let mut sx = 0.0;
        for &(u, wt) in lpred[v].iter().chain(lsucc[v].iter()) {
            sw += wt;
            sx += wt * x[u];
        }
        if sw > 0.0 {
            t[i] = sx / sw - off[i];
            w[i] = sw;
        } else {
            t[i] = x[v] - off[i];
            w[i] = 1e-3;
        }
    }
    let y = pava(&t, &w);
    for i in 0..k {
        x[layer[i]] = y[i] + off[i];
    }
}

/// Weighted isotonic (non-decreasing) regression by pool-adjacent-violators.
pub fn pava(t: &[f64], w: &[f64]) -> Vec<f64> {
    let mut bw: Vec<f64> = Vec::with_capacity(t.len());
    let mut bv: Vec<f64> = Vec::with_capacity(t.len());
    let mut bc: Vec<usize> = Vec::with_capacity(t.len());
    for i in 0..t.len() {
        bw.push(w[i]);
        bv.push(t[i]);
        bc.push(1);
        while bv.len() >= 2 && bv[bv.len() - 2] > bv[bv.len() - 1] {
            let (w2, v2, c2) = (bw.pop().unwrap(), bv.pop().unwrap(), bc.pop().unwrap());
            let (w1, v1, c1) = (bw.pop().unwrap(), bv.pop().unwrap(), bc.pop().unwrap());
            let ww = w1 + w2;
            bw.push(ww);
            bv.push(if ww > 0.0 { (w1 * v1 + w2 * v2) / ww } else { (v1 + v2) / 2.0 });
            bc.push(c1 + c2);
        }
    }
    let mut out = Vec::with_capacity(t.len());
    for k in 0..bv.len() {
        for _ in 0..bc[k] {
            out.push(bv[k]);
        }
    }
    out
}

fn spread(
    ports: &mut [(usize, f64)],
    cx: f64,
    w: f64,
    points: &mut [Vec<(f64, f64)>],
    at_start: bool,
) {
    let k = ports.len();
    if k < 2 {
        return;
    }
    ports.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let spacing = ((w - 16.0).max(0.0) / (k as f64 - 1.0)).min(14.0);
    let start = cx - spacing * (k as f64 - 1.0) / 2.0;
    for (j, &(ei, _)) in ports.iter().enumerate() {
        let px = start + spacing * j as f64;
        let pts = &mut points[ei];
        if pts.is_empty() {
            continue;
        }
        if at_start {
            pts[0].0 = px;
        } else {
            let l = pts.len() - 1;
            pts[l].0 = px;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes(k: usize) -> Vec<LNode> {
        (0..k).map(|_| LNode { w: 80.0, h: 30.0 }).collect()
    }

    fn edge(a: usize, b: usize) -> LEdge {
        LEdge {
            src: a,
            dst: b,
            weight: 1.0,
            detachable: false,
        }
    }

    #[test]
    fn pava_works() {
        let y = pava(&[1.0, 3.0, 2.0, 4.0], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(y, vec![1.0, 2.5, 2.5, 4.0]);
    }

    #[test]
    fn chain_is_straight() {
        let n = nodes(5);
        let e: Vec<LEdge> = (0..4).map(|i| edge(i, i + 1)).collect();
        let r = layout(&n, &e, &LayoutParams::default());
        for i in 1..5 {
            assert!((r.x[i] - r.x[0]).abs() < 1e-6);
            assert!(r.y[i] > r.y[i - 1]);
        }
    }

    #[test]
    fn residual_block() {
        // 0 -> 1 -> 2 -> 3, and 0 -> 3 (skip connection)
        let n = nodes(4);
        let e = vec![edge(0, 1), edge(1, 2), edge(2, 3), edge(0, 3)];
        let r = layout(&n, &e, &LayoutParams::default());
        assert_eq!(r.rank, vec![0, 1, 2, 3]);
        assert_eq!(r.points[3].len(), 4);
        // nodes in the same layer never overlap
        assert!(r.width >= 80.0);
    }

    #[test]
    fn no_overlap_and_cycles() {
        let n = nodes(6);
        let e = vec![
            edge(0, 1),
            edge(0, 2),
            edge(0, 3),
            edge(1, 4),
            edge(2, 4),
            edge(3, 5),
            edge(5, 0),
            edge(4, 4),
        ];
        let r = layout(&n, &e, &LayoutParams::default());
        assert!(r.reversed.iter().any(|&b| b));
        for a in 0..6 {
            for b in 0..6 {
                if a != b && r.rank[a] == r.rank[b] {
                    assert!((r.x[a] - r.x[b]).abs() >= 80.0 - 1e-6);
                }
            }
        }
        for (i, pts) in r.points.iter().enumerate() {
            assert!(!pts.is_empty(), "edge {} not routed", i);
        }
    }
}
