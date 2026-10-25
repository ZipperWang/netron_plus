import { categoryColor } from './colors';
import type { GraphRenderer } from './renderer';

/** Overview of the whole graph with the visible viewport. */
export class Minimap {
  private ctx: CanvasRenderingContext2D | null;
  private layer: HTMLCanvasElement | null = null;
  private s = 1;
  private ox = 0;
  private oy = 0;
  private w = 0;
  private h = 0;
  private dpr = 1;

  constructor(
    private canvas: HTMLCanvasElement,
    private r: GraphRenderer,
  ) {
    this.ctx = canvas.getContext('2d');
  }

  rebuild() {
    const v = this.r.view;
    this.dpr = window.devicePixelRatio || 1;
    this.w = this.canvas.clientWidth;
    this.h = this.canvas.clientHeight;
    this.canvas.width = Math.max(1, Math.round(this.w * this.dpr));
    this.canvas.height = Math.max(1, Math.round(this.h * this.dpr));
    this.layer = null;
    if (!v || v.nodes.length === 0 || this.w === 0) {
      this.draw();
      return;
    }
    const pad = 6;
    this.s = Math.min((this.w - 2 * pad) / Math.max(1, v.width), (this.h - 2 * pad) / Math.max(1, v.height));
    this.ox = (this.w - v.width * this.s) / 2;
    this.oy = (this.h - v.height * this.s) / 2;
    const layer = document.createElement('canvas');
    layer.width = this.canvas.width;
    layer.height = this.canvas.height;
    const c = layer.getContext('2d');
    if (c) {
      c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      for (const n of v.nodes) {
        c.fillStyle = n.kind === 'op' ? categoryColor(n.cat) : categoryColor(n.kind);
        c.fillRect(
          this.ox + n.x * this.s,
          this.oy + n.y * this.s,
          Math.max(1, n.w * this.s),
          Math.max(1, n.h * this.s),
        );
      }
    }
    this.layer = layer;
    this.draw();
  }

  draw() {
    const c = this.ctx;
    if (!c) return;
    const th = this.r.theme;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, this.canvas.width, this.canvas.height);
    c.fillStyle = th.minimapBg;
    c.globalAlpha = 0.92;
    c.fillRect(0, 0, this.canvas.width, this.canvas.height);
    c.globalAlpha = 1;
    if (!this.layer || !this.r.view) return;
    c.drawImage(this.layer, 0, 0);
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const [x0, y0] = this.r.toWorld(0, 0);
    const [x1, y1] = this.r.toWorld(this.r.width, this.r.height);
    const rx = this.ox + x0 * this.s;
    const ry = this.oy + y0 * this.s;
    const rw = Math.max(4, (x1 - x0) * this.s);
    const rh = Math.max(4, (y1 - y0) * this.s);
    c.fillStyle = th.minimapView;
    c.globalAlpha = 0.12;
    c.fillRect(rx, ry, rw, rh);
    c.globalAlpha = 1;
    c.strokeStyle = th.minimapView;
    c.lineWidth = 1.5;
    c.strokeRect(rx, ry, rw, rh);
  }

  toWorld(mx: number, my: number): [number, number] {
    return [(mx - this.ox) / this.s, (my - this.oy) / this.s];
  }
}
