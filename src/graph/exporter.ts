import type { GraphView } from '../types';
import { categoryColor, type Theme } from './colors';
import { BODY_FONT, EDGE_FONT, LINE_H, PAD_X, PAD_Y, TITLE_FONT, TITLE_H, fontMetrics, textWidth } from './metrics';
import { GraphRenderer, STUB } from './renderer';

const MAX_SIDE = 16000;
const MAX_PIXELS = 120_000_000;

/** Render the whole graph to a PNG (scaled down when it is huge). */
export async function exportPng(view: GraphView, theme: Theme): Promise<Uint8Array> {
  const pad = 24;
  const w = view.width + 2 * pad;
  const h = view.height + 2 * pad;
  let scale = 2;
  scale = Math.min(scale, MAX_SIDE / w, MAX_SIDE / h, Math.sqrt(MAX_PIXELS / (w * h)));
  const canvas = document.createElement('canvas');
  const r = new GraphRenderer(canvas, theme);
  r.resize(Math.ceil(w), Math.ceil(h), scale);
  r.setView(view);
  r.scale = 1;
  r.tx = pad;
  r.ty = pad;
  r.draw({ full: true });
  const blob: Blob = await new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encoding failed'))), 'image/png'),
  );
  return new Uint8Array(await blob.arrayBuffer());
}

function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function fontAttrs(font: string): string {
  // "600 12px family" / "11px family"
  const m = /^(?:(\d{3})\s+)?(\d+)px\s+(.*)$/.exec(font);
  if (!m) return '';
  const weight = m[1] ? ` font-weight="${m[1]}"` : '';
  return `font-size="${m[2]}" font-family="${esc(m[3])}"${weight}`;
}

/** Self-contained SVG of the graph. */
export function exportSvg(view: GraphView, theme: Theme): string {
  const pad = 24;
  const W = Math.ceil(view.width + 2 * pad);
  const H = Math.ceil(view.height + 2 * pad);
  const metrics = fontMetrics();
  const out: string[] = [];
  out.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`,
    `<rect width="100%" height="100%" fill="${theme.bg}"/>`,
    `<g transform="translate(${pad},${pad})">`,
    `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${theme.edge}"/></marker></defs>`,
  );
  out.push(`<g fill="none" stroke="${theme.edge}" stroke-width="1.2">`);
  for (const e of view.edges) {
    const p = e.points;
    if (e.detached || p.length < 4) {
      const t = view.nodes[e.to];
      if (!t) continue;
      const x = t.x + t.w / 2;
      out.push(`<path d="M${x},${t.y - STUB} L${x},${t.y}" marker-end="url(#arrow)"/>`);
      continue;
    }
    let d = `M${p[0].toFixed(1)},${p[1].toFixed(1)}`;
    for (let k = 2; k < p.length; k += 2) {
      const xa = p[k - 2];
      const ya = p[k - 1];
      const xb = p[k];
      const yb = p[k + 1];
      const dy = (yb - ya) / 2;
      d += ` C${xa.toFixed(1)},${(ya + dy).toFixed(1)} ${xb.toFixed(1)},${(yb - dy).toFixed(1)} ${xb.toFixed(1)},${yb.toFixed(1)}`;
    }
    const dash = e.implicit ? ' stroke-dasharray="5 4"' : '';
    out.push(`<path d="${d}"${dash} marker-end="url(#arrow)"/>`);
  }
  out.push('</g>');
  out.push(`<g ${fontAttrs(EDGE_FONT)} fill="${theme.label}">`);
  for (const e of view.edges) {
    const p = e.points;
    if (!e.label || e.detached || p.length < 4) continue;
    out.push(`<text x="${((p[0] + p[2]) / 2 + 4).toFixed(1)}" y="${((p[1] + p[3]) / 2).toFixed(1)}" dominant-baseline="middle">${esc(e.label)}</text>`);
  }
  out.push('</g>');
  for (const n of view.nodes) {
    const color = n.kind === 'op' ? categoryColor(n.cat) : categoryColor(n.kind);
    const r = n.kind === 'op' ? 4 : 8;
    const hasBody = n.lines.length > 0 || n.name.length > 0;
    out.push(`<g>`);
    if (!hasBody) {
      out.push(`<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${r}" fill="${color}"/>`);
    } else {
      out.push(
        `<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${r}" fill="${theme.nodeBg}" stroke="${theme.nodeBorder}"/>`,
        `<path d="M${n.x},${n.y + TITLE_H} L${n.x},${n.y + r} Q${n.x},${n.y} ${n.x + r},${n.y} L${n.x + n.w - r},${n.y} Q${n.x + n.w},${n.y} ${n.x + n.w},${n.y + r} L${n.x + n.w},${n.y + TITLE_H} Z" fill="${color}"/>`,
      );
    }
    out.push(
      `<text x="${n.x + PAD_X}" y="${n.y + TITLE_H / 2 + 0.5}" dominant-baseline="middle" fill="#ffffff" ${fontAttrs(TITLE_FONT)}>${esc(n.title)}</text>`,
    );
    let y = n.y + TITLE_H + PAD_Y + LINE_H / 2;
    if (n.name) {
      out.push(`<text x="${n.x + PAD_X}" y="${y}" dominant-baseline="middle" fill="${theme.textMuted}" ${fontAttrs(BODY_FONT)}>${esc(n.name)}</text>`);
      y += LINE_H;
    }
    for (const l of n.lines) {
      let x = n.x + PAD_X;
      if (l.k) {
        out.push(`<text x="${x}" y="${y}" dominant-baseline="middle" fill="${theme.textMuted}" ${fontAttrs(BODY_FONT)}>${esc(l.k)}</text>`);
        x += textWidth(metrics.body, metrics.bodyWide, l.k) + 6;
      }
      out.push(`<text x="${x.toFixed(1)}" y="${y}" dominant-baseline="middle" fill="${theme.text}" ${fontAttrs(BODY_FONT)}>${esc(l.v)}</text>`);
      y += LINE_H;
    }
    out.push('</g>');
  }
  out.push('</g></svg>');
  return out.join('\n');
}
