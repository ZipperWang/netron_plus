import type { GraphView, ViewEdge, ViewNode } from '../types';
import { categoryColor, type Theme } from './colors';
import { BODY_FONT, EDGE_FONT, LINE_H, PAD_X, PAD_Y, TITLE_FONT, TITLE_H, fontMetrics, textWidth } from './metrics';

const CELL = 256;
const MIN_SCALE = 0.003;
const MAX_SCALE = 8;
/** Length of the stub drawn for detached (very long) edges. */
export const STUB = 24;

function cellKey(cx: number, cy: number): number {
  return (cx + 32768) * 65536 + (cy + 32768);
}

export function roundRect(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, w / 2, h / 2);
  c.moveTo(x + rr, y);
  c.lineTo(x + w - rr, y);
  c.quadraticCurveTo(x + w, y, x + w, y + rr);
  c.lineTo(x + w, y + h - rr);
  c.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  c.lineTo(x + rr, y + h);
  c.quadraticCurveTo(x, y + h, x, y + h - rr);
  c.lineTo(x, y + rr);
  c.quadraticCurveTo(x, y, x + rr, y);
  c.closePath();
}

function topRound(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, w / 2, h);
  c.moveTo(x, y + h);
  c.lineTo(x, y + rr);
  c.quadraticCurveTo(x, y, x + rr, y);
  c.lineTo(x + w - rr, y);
  c.quadraticCurveTo(x + w, y, x + w, y + rr);
  c.lineTo(x + w, y + h);
  c.closePath();
}

function segDist2(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax;
  const dy = by - ay;
  const len = dx * dx + dy * dy;
  let t = len > 0 ? ((px - ax) * dx + (py - ay) * dy) / len : 0;
  t = Math.max(0, Math.min(1, t));
  const x = ax + t * dx - px;
  const y = ay + t * dy - py;
  return x * x + y * y;
}

export interface DrawOptions {
  /** Ignore level-of-detail thresholds (used for image export). */
  full?: boolean;
}

export class GraphRenderer {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  theme: Theme;
  view: GraphView | null = null;
  scale = 1;
  tx = 0;
  ty = 0;
  width = 0;
  height = 0;
  private dpr = 1;
  private grid = new Map<number, number[]>();
  private boxes = new Float64Array(0);
  private labelPos = new Float64Array(0);
  private stubs = new Float64Array(0);
  private widths = new Map<string, number>();
  /** Edge indices touching each node. */
  adjacency: number[][] = [];
  selected = -1;
  selectedValue = -1;
  hover = -1;
  hoverEdge = -1;
  highlight = new Set<number>();
  traceNodes: Set<number> | null = null;
  traceEdges: Set<number> | null = null;
  showLabels = true;
  onCamera: (() => void) | null = null;
  private raf = 0;
  private animId = 0;
  private full = false;

  constructor(canvas: HTMLCanvasElement, theme: Theme) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('Canvas 2D is not available');
    this.ctx = ctx;
    this.theme = theme;
  }

  resize(w: number, h: number, dpr = window.devicePixelRatio || 1) {
    this.dpr = dpr;
    this.width = w;
    this.height = h;
    this.canvas.width = Math.max(1, Math.round(w * dpr));
    this.canvas.height = Math.max(1, Math.round(h * dpr));
    this.requestDraw();
  }

  setTheme(theme: Theme) {
    this.theme = theme;
    this.requestDraw();
  }

  setView(view: GraphView | null) {
    this.view = view;
    this.grid.clear();
    this.widths.clear();
    this.hover = -1;
    this.hoverEdge = -1;
    this.selected = -1;
    this.selectedValue = -1;
    this.highlight = new Set();
    this.traceNodes = null;
    this.traceEdges = null;
    if (!view) {
      this.adjacency = [];
      this.requestDraw();
      return;
    }
    const nodes = view.nodes;
    this.adjacency = nodes.map(() => []);
    nodes.forEach((n, i) => {
      const x0 = Math.floor(n.x / CELL);
      const x1 = Math.floor((n.x + n.w) / CELL);
      const y0 = Math.floor(n.y / CELL);
      const y1 = Math.floor((n.y + n.h) / CELL);
      for (let cx = x0; cx <= x1; cx++) {
        for (let cy = y0; cy <= y1; cy++) {
          const k = cellKey(cx, cy);
          let arr = this.grid.get(k);
          if (!arr) {
            arr = [];
            this.grid.set(k, arr);
          }
          arr.push(i);
        }
      }
    });
    const m = view.edges.length;
    this.boxes = new Float64Array(m * 4);
    this.labelPos = new Float64Array(m * 2);
    this.stubs = new Float64Array(m * 2);
    const stubCount = new Map<number, number>();
    view.edges.forEach((e, i) => {
      this.adjacency[e.from]?.push(i);
      if (e.to !== e.from) this.adjacency[e.to]?.push(i);
      const p = e.points;
      let minx = Infinity;
      let miny = Infinity;
      let maxx = -Infinity;
      let maxy = -Infinity;
      if (e.detached || p.length < 4) {
        const t = nodes[e.to];
        const k = stubCount.get(e.to) ?? 0;
        stubCount.set(e.to, k + 1);
        const off = k === 0 ? 0 : (k % 2 === 1 ? 1 : -1) * Math.ceil(k / 2) * 12;
        const sx = t ? t.x + t.w / 2 + off : 0;
        const sy = t ? t.y : 0;
        this.stubs[i * 2] = sx;
        this.stubs[i * 2 + 1] = sy;
        minx = sx - 6;
        maxx = sx + 6;
        miny = sy - STUB - 14;
        maxy = sy;
        this.labelPos[i * 2] = NaN;
        this.labelPos[i * 2 + 1] = NaN;
      } else {
        for (let k = 0; k < p.length; k += 2) {
          if (p[k] < minx) minx = p[k];
          if (p[k] > maxx) maxx = p[k];
          if (p[k + 1] < miny) miny = p[k + 1];
          if (p[k + 1] > maxy) maxy = p[k + 1];
        }
        this.labelPos[i * 2] = (p[0] + p[2]) / 2;
        this.labelPos[i * 2 + 1] = (p[1] + p[3]) / 2;
      }
      this.boxes[i * 4] = minx;
      this.boxes[i * 4 + 1] = miny;
      this.boxes[i * 4 + 2] = maxx;
      this.boxes[i * 4 + 3] = maxy;
    });
    this.requestDraw();
  }

  // ---------------------------------------------------------------- camera

  private clampScale(s: number) {
    return Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));
  }

  private changed() {
    this.requestDraw();
    this.onCamera?.();
  }

  zoomAt(factor: number, sx: number, sy: number) {
    this.stopAnim();
    const s = this.clampScale(this.scale * factor);
    const wx = (sx - this.tx) / this.scale;
    const wy = (sy - this.ty) / this.scale;
    this.scale = s;
    this.tx = sx - wx * s;
    this.ty = sy - wy * s;
    this.changed();
  }

  zoomCenter(factor: number) {
    this.zoomAt(factor, this.width / 2, this.height / 2);
  }

  panBy(dx: number, dy: number) {
    this.stopAnim();
    this.tx += dx;
    this.ty += dy;
    this.changed();
  }

  stopAnim() {
    this.animId++;
  }

  toWorld(sx: number, sy: number): [number, number] {
    return [(sx - this.tx) / this.scale, (sy - this.ty) / this.scale];
  }

  toScreen(wx: number, wy: number): [number, number] {
    return [wx * this.scale + this.tx, wy * this.scale + this.ty];
  }

  /** Center the world point (cx, cy) at `scale`. */
  setCamera(scale: number, cx: number, cy: number, animate: boolean) {
    const s = this.clampScale(scale);
    if (!animate || this.width === 0) {
      this.stopAnim();
      this.scale = s;
      this.tx = this.width / 2 - cx * s;
      this.ty = this.height / 2 - cy * s;
      this.changed();
      return;
    }
    const s0 = this.scale;
    const c0x = (this.width / 2 - this.tx) / s0;
    const c0y = (this.height / 2 - this.ty) / s0;
    const start = performance.now();
    const dur = 320;
    const id = ++this.animId;
    const step = (now: number) => {
      if (id !== this.animId) return;
      const t = Math.min(1, (now - start) / dur);
      const e = 1 - Math.pow(1 - t, 3);
      const sc = Math.exp(Math.log(s0) + (Math.log(s) - Math.log(s0)) * e);
      const x = c0x + (cx - c0x) * e;
      const y = c0y + (cy - c0y) * e;
      this.scale = sc;
      this.tx = this.width / 2 - x * sc;
      this.ty = this.height / 2 - y * sc;
      this.changed();
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  fitScale(pad = 32): number {
    const v = this.view;
    if (!v) return 1;
    const sx = (this.width - 2 * pad) / Math.max(1, v.width);
    const sy = (this.height - 2 * pad) / Math.max(1, v.height);
    return this.clampScale(Math.min(sx, sy, 1.25));
  }

  fit(animate = false) {
    const v = this.view;
    if (!v) return;
    this.setCamera(this.fitScale(), v.width / 2, v.height / 2, animate);
  }

  /** Readable start view: the whole graph if it fits, else its top. */
  initialCamera() {
    const v = this.view;
    if (!v || v.nodes.length === 0) {
      this.fit();
      return;
    }
    const fs = this.fitScale();
    if (fs >= 0.6) {
      this.fit();
      return;
    }
    const s = Math.max(0.6, Math.min(1, (this.width - 64) / Math.max(1, v.width)));
    let top = v.nodes[0];
    for (const n of v.nodes) {
      if (n.y < top.y - 0.5 || (Math.abs(n.y - top.y) < 0.5 && n.kind === 'input' && top.kind !== 'input')) top = n;
    }
    const cx = v.width * s <= this.width ? v.width / 2 : top.x + top.w / 2;
    const cy = top.y + (this.height / 2 - 48) / s;
    this.setCamera(s, cx, cy, false);
  }

  focusNode(i: number, animate = true) {
    const n = this.view?.nodes[i];
    if (!n) return;
    const s = Math.min(Math.max(this.scale, 0.85), 1.6);
    this.setCamera(s, n.x + n.w / 2, n.y + n.h / 2, animate);
  }

  focusNodes(indices: number[], animate = true) {
    const v = this.view;
    if (!v || indices.length === 0) return;
    if (indices.length === 1) {
      this.focusNode(indices[0], animate);
      return;
    }
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const i of indices) {
      const n = v.nodes[i];
      if (!n) continue;
      x0 = Math.min(x0, n.x);
      y0 = Math.min(y0, n.y);
      x1 = Math.max(x1, n.x + n.w);
      y1 = Math.max(y1, n.y + n.h);
    }
    if (!isFinite(x0)) return;
    const s = Math.min((this.width - 120) / Math.max(1, x1 - x0), (this.height - 120) / Math.max(1, y1 - y0), 1.25);
    this.setCamera(s, (x0 + x1) / 2, (y0 + y1) / 2, animate);
  }

  /** Is the node (fully) inside the viewport? */
  isVisible(i: number): boolean {
    const n = this.view?.nodes[i];
    if (!n) return false;
    const [ax, ay] = this.toScreen(n.x, n.y);
    const [bx, by] = this.toScreen(n.x + n.w, n.y + n.h);
    return ax >= 0 && ay >= 0 && bx <= this.width && by <= this.height;
  }

  // ----------------------------------------------------------- hit testing

  nodeAt(sx: number, sy: number): number {
    const v = this.view;
    if (!v) return -1;
    const [wx, wy] = this.toWorld(sx, sy);
    const cell = this.grid.get(cellKey(Math.floor(wx / CELL), Math.floor(wy / CELL)));
    if (!cell) return -1;
    for (let k = cell.length - 1; k >= 0; k--) {
      const i = cell[k];
      const n = v.nodes[i];
      if (wx >= n.x && wx <= n.x + n.w && wy >= n.y && wy <= n.y + n.h) return i;
    }
    return -1;
  }

  private curveDist2(p: number[], x: number, y: number): number {
    let best = Infinity;
    for (let k = 0; k + 3 < p.length; k += 2) {
      const x0 = p[k];
      const y0 = p[k + 1];
      const x1 = p[k + 2];
      const y1 = p[k + 3];
      const dy = (y1 - y0) / 2;
      let px = x0;
      let py = y0;
      for (let st = 1; st <= 16; st++) {
        const t = st / 16;
        const mt = 1 - t;
        const bx = mt * mt * mt * x0 + 3 * mt * mt * t * x0 + 3 * mt * t * t * x1 + t * t * t * x1;
        const by = mt * mt * mt * y0 + 3 * mt * mt * t * (y0 + dy) + 3 * mt * t * t * (y1 - dy) + t * t * t * y1;
        const d = segDist2(x, y, px, py, bx, by);
        if (d < best) best = d;
        px = bx;
        py = by;
      }
    }
    return best;
  }

  edgeAt(sx: number, sy: number): number {
    const v = this.view;
    if (!v) return -1;
    const [wx, wy] = this.toWorld(sx, sy);
    const tol = 6 / this.scale;
    let best = -1;
    let bestD = tol * tol;
    for (let i = 0; i < v.edges.length; i++) {
      const b = i * 4;
      if (wx < this.boxes[b] - tol || wx > this.boxes[b + 2] + tol || wy < this.boxes[b + 1] - tol || wy > this.boxes[b + 3] + tol) {
        continue;
      }
      const e = v.edges[i];
      let d: number;
      if (e.detached || e.points.length < 4) {
        const ux = this.stubs[i * 2];
        const uy = this.stubs[i * 2 + 1];
        d = segDist2(wx, wy, ux, uy - STUB - 12, ux, uy);
      } else {
        d = this.curveDist2(e.points, wx, wy);
      }
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  }

  // --------------------------------------------------------------- drawing

  requestDraw() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.draw();
    });
  }

  private bodyWidth(s: string): number {
    let w = this.widths.get(s);
    if (w === undefined) {
      const m = fontMetrics();
      w = textWidth(m.body, m.bodyWide, s);
      this.widths.set(s, w);
    }
    return w;
  }

  draw(opts: DrawOptions = {}) {
    this.full = !!opts.full;
    const c = this.ctx;
    const th = this.theme;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.globalAlpha = 1;
    c.fillStyle = th.bg;
    c.fillRect(0, 0, this.canvas.width, this.canvas.height);
    const v = this.view;
    if (!v) return;
    const s = this.scale;
    const dpr = this.dpr;
    c.setTransform(dpr * s, 0, 0, dpr * s, dpr * this.tx, dpr * this.ty);
    const [x0, y0] = this.toWorld(0, 0);
    const [x1, y1] = this.toWorld(this.width, this.height);
    const full = this.full;
    const simple = !full && s < 0.12;
    const arrows = full || s >= 0.18;
    const titles = full || s >= 0.25;
    const lines = full || s >= 0.42;
    const labels = this.showLabels && (full || s >= 0.6);
    const lw = full ? 1.2 : Math.max(1.1 / s, 1);

    const edges = v.edges;
    const nodes = v.nodes;
    const te = this.traceEdges;
    const normal: number[] = [];
    const implicit: number[] = [];
    const dim: number[] = [];
    const hl: number[] = [];
    for (let i = 0; i < edges.length; i++) {
      const b = i * 4;
      if (this.boxes[b + 2] < x0 || this.boxes[b] > x1 || this.boxes[b + 3] < y0 || this.boxes[b + 1] > y1) continue;
      const e = edges[i];
      const isHl =
        (this.selected >= 0 && (e.from === this.selected || e.to === this.selected)) ||
        (this.selectedValue >= 0 && e.value === this.selectedValue) ||
        this.hoverEdge === i;
      if (isHl) hl.push(i);
      else if (te && !te.has(i)) dim.push(i);
      else if (e.implicit) implicit.push(i);
      else normal.push(i);
    }
    this.strokeEdges(dim, th.edge, lw, 0.14, false, simple, arrows);
    this.strokeEdges(normal, th.edge, lw, 1, false, simple, arrows);
    this.strokeEdges(implicit, th.edgeImplicit, lw, 1, true, simple, arrows);
    this.strokeEdges(hl, th.edgeHl, lw * 1.8, 1, false, simple, arrows);

    if (labels) {
      c.font = EDGE_FONT;
      c.textAlign = 'left';
      c.textBaseline = 'middle';
      c.lineJoin = 'round';
      c.lineWidth = 3;
      c.strokeStyle = th.bg;
      for (const list of [normal, implicit, hl]) {
        c.fillStyle = list === hl ? th.edgeHl : th.label;
        for (const i of list) {
          const e = edges[i];
          const lx = this.labelPos[i * 2];
          if (!e.label || Number.isNaN(lx)) continue;
          const ly = this.labelPos[i * 2 + 1];
          c.strokeText(e.label, lx + 4, ly);
          c.fillText(e.label, lx + 4, ly);
        }
      }
    }
    if (lines) {
      // Names of detached long edges above their stubs.
      c.font = EDGE_FONT;
      c.textAlign = 'center';
      c.textBaseline = 'bottom';
      c.lineWidth = 3;
      c.strokeStyle = th.bg;
      c.fillStyle = th.label;
      for (const list of [normal, implicit, hl]) {
        for (const i of list) {
          const e = edges[i];
          if (!(e.detached || e.points.length < 4)) continue;
          const name = e.name.length > 28 ? `…${e.name.slice(-27)}` : e.name;
          const sx = this.stubs[i * 2];
          const sy = this.stubs[i * 2 + 1] - STUB - 2;
          c.strokeText(name, sx, sy);
          c.fillText(name, sx, sy);
        }
      }
    }

    const tr = this.traceNodes;
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      if (n.x > x1 || n.x + n.w < x0 || n.y > y1 || n.y + n.h < y0) continue;
      c.globalAlpha = tr !== null && !tr.has(i) ? 0.2 : 1;
      this.drawNode(n, simple, titles, lines);
    }
    c.globalAlpha = 1;
    for (const i of this.highlight) this.outline(nodes[i], th.highlight, 3);
    if (this.hover >= 0 && this.hover !== this.selected) this.outline(nodes[this.hover], th.edgeHl, 1.5, 0.6);
    if (this.selected >= 0) this.outline(nodes[this.selected], th.select, 2.5);
    this.full = false;
  }

  private strokeEdges(
    list: number[],
    color: string,
    width: number,
    alpha: number,
    dashed: boolean,
    simple: boolean,
    arrows: boolean,
  ) {
    if (!list.length) return;
    const c = this.ctx;
    const v = this.view!;
    const s = this.full ? 1 : this.scale;
    c.globalAlpha = alpha;
    c.strokeStyle = color;
    c.fillStyle = color;
    c.lineWidth = width;
    c.lineCap = 'round';
    c.lineJoin = 'round';
    c.setLineDash(dashed ? [5 / Math.min(s, 1), 4 / Math.min(s, 1)] : []);
    c.beginPath();
    for (const i of list) {
      const e = v.edges[i];
      if (e.detached || e.points.length < 4) {
        const sx = this.stubs[i * 2];
        const sy = this.stubs[i * 2 + 1];
        c.moveTo(sx, sy - STUB);
        c.lineTo(sx, sy);
        continue;
      }
      const p = e.points;
      c.moveTo(p[0], p[1]);
      for (let k = 2; k < p.length; k += 2) {
        const xa = p[k - 2];
        const ya = p[k - 1];
        const xb = p[k];
        const yb = p[k + 1];
        if (simple) c.lineTo(xb, yb);
        else {
          const dy = (yb - ya) / 2;
          c.bezierCurveTo(xa, ya + dy, xb, yb - dy, xb, yb);
        }
      }
    }
    c.stroke();
    c.setLineDash([]);
    if (arrows) {
      const size = this.full ? 6 : Math.max(5, 5 / s);
      c.beginPath();
      for (const i of list) {
        const e = v.edges[i];
        let tx: number;
        let ty: number;
        let dir = 1;
        if (e.detached || e.points.length < 4) {
          tx = this.stubs[i * 2];
          ty = this.stubs[i * 2 + 1];
        } else {
          const p = e.points;
          const n = p.length;
          tx = p[n - 2];
          ty = p[n - 1];
          dir = ty >= p[n - 3] ? 1 : -1;
        }
        c.moveTo(tx, ty);
        c.lineTo(tx - size * 0.55, ty - dir * size);
        c.lineTo(tx + size * 0.55, ty - dir * size);
        c.closePath();
      }
      c.fill();
    }
    c.globalAlpha = 1;
  }

  private drawNode(n: ViewNode, simple: boolean, titles: boolean, lines: boolean) {
    const c = this.ctx;
    const th = this.theme;
    const color = n.kind === 'op' ? categoryColor(n.cat) : categoryColor(n.kind);
    if (simple) {
      c.fillStyle = color;
      c.fillRect(n.x, n.y, n.w, n.h);
      return;
    }
    const r = n.kind === 'op' ? 4 : 8;
    const hasBody = n.lines.length > 0 || n.name.length > 0;
    c.beginPath();
    roundRect(c, n.x, n.y, n.w, n.h, r);
    if (!hasBody) {
      c.fillStyle = color;
      c.fill();
    } else {
      c.fillStyle = th.nodeBg;
      c.fill();
      c.lineWidth = this.full ? 1 : Math.max(1, 0.9 / this.scale);
      c.strokeStyle = th.nodeBorder;
      c.stroke();
      c.beginPath();
      topRound(c, n.x, n.y, n.w, TITLE_H, r);
      c.fillStyle = color;
      c.fill();
    }
    if (n.kind === 'missing' || n.kind === 'outer') {
      c.setLineDash([4, 3]);
      c.lineWidth = 1.2;
      c.strokeStyle = color;
      c.beginPath();
      roundRect(c, n.x - 2, n.y - 2, n.w + 4, n.h + 4, r + 2);
      c.stroke();
      c.setLineDash([]);
    }
    if (n.sub) {
      // A small "stacked" marker on control-flow nodes that own subgraphs.
      c.fillStyle = 'rgba(255,255,255,0.85)';
      c.beginPath();
      const mx = n.x + n.w - 9;
      const my = n.y + TITLE_H / 2;
      c.moveTo(mx - 3, my - 4);
      c.lineTo(mx + 2, my);
      c.lineTo(mx - 3, my + 4);
      c.closePath();
      c.fill();
    }
    if (!titles) return;
    c.fillStyle = th.headerText;
    c.font = TITLE_FONT;
    c.textAlign = 'left';
    c.textBaseline = 'middle';
    c.fillText(n.title, n.x + PAD_X, n.y + TITLE_H / 2 + 0.5);
    if (!lines || !hasBody) return;
    c.font = BODY_FONT;
    let y = n.y + TITLE_H + PAD_Y + LINE_H / 2;
    if (n.name) {
      c.fillStyle = th.textMuted;
      c.fillText(n.name, n.x + PAD_X, y);
      y += LINE_H;
    }
    for (const l of n.lines) {
      let x = n.x + PAD_X;
      if (l.k) {
        c.fillStyle = th.textMuted;
        c.fillText(l.k, x, y);
        x += this.bodyWidth(l.k) + 6;
      }
      c.fillStyle = th.text;
      c.fillText(l.v, x, y);
      y += LINE_H;
    }
  }

  private outline(n: ViewNode | undefined, color: string, px: number, alpha = 1) {
    if (!n) return;
    const c = this.ctx;
    const s = this.full ? 1 : this.scale;
    const pad = 2.5 / s;
    c.globalAlpha = alpha;
    c.strokeStyle = color;
    c.lineWidth = px / s;
    c.beginPath();
    roundRect(c, n.x - pad, n.y - pad, n.w + 2 * pad, n.h + 2 * pad, 6 / Math.min(s, 1));
    c.stroke();
    c.globalAlpha = 1;
  }

  /** Edge indices on all paths upstream / downstream of `start`. */
  trace(start: number, up: boolean, down: boolean): { nodes: Set<number>; edges: Set<number> } {
    const v = this.view;
    const nodes = new Set<number>([start]);
    const edges = new Set<number>();
    if (!v) return { nodes, edges };
    const walk = (forward: boolean) => {
      const stack = [start];
      const seen = new Set<number>([start]);
      while (stack.length) {
        const n = stack.pop()!;
        for (const ei of this.adjacency[n] ?? []) {
          const e = v.edges[ei];
          const next = forward ? (e.from === n ? e.to : -1) : e.to === n ? e.from : -1;
          if (next < 0) continue;
          edges.add(ei);
          if (!seen.has(next)) {
            seen.add(next);
            nodes.add(next);
            stack.push(next);
          }
        }
      }
    };
    if (up) walk(false);
    if (down) walk(true);
    return { nodes, edges };
  }

  edgesOf(i: number): ViewEdge[] {
    const v = this.view;
    if (!v) return [];
    return (this.adjacency[i] ?? []).map((e) => v.edges[e]);
  }
}
