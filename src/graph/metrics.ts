// Fonts and box metrics shared by the canvas renderer and the Rust layout
// (the layout receives per-character advance widths measured here).

export const FONT_UI =
  'system-ui, -apple-system, "Segoe UI", "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", sans-serif';
export const FONT_MONO =
  '"Cascadia Mono", "Cascadia Code", Consolas, "SF Mono", Menlo, "DejaVu Sans Mono", "Microsoft YaHei UI", monospace';

export const TITLE_FONT = `600 12px ${FONT_UI}`;
export const BODY_FONT = `11px ${FONT_MONO}`;
export const EDGE_FONT = `10px ${FONT_MONO}`;
export const TITLE_H = 22;
export const LINE_H = 15;
export const PAD_X = 8;
export const PAD_Y = 4;

export interface FontMetrics {
  title: number[];
  body: number[];
  titleWide: number;
  bodyWide: number;
  titleHeight: number;
  lineHeight: number;
  padX: number;
  padY: number;
}

let cached: FontMetrics | null = null;

export function fontMetrics(): FontMetrics {
  if (cached) return cached;
  const ctx = document.createElement('canvas').getContext('2d');
  const title: number[] = [];
  const body: number[] = [];
  let titleWide = 12;
  let bodyWide = 11;
  if (ctx) {
    const measure = (font: string, out: number[]) => {
      ctx.font = font;
      for (let c = 32; c <= 126; c++) out.push(Math.ceil(ctx.measureText(String.fromCharCode(c)).width * 100) / 100);
    };
    measure(TITLE_FONT, title);
    measure(BODY_FONT, body);
    ctx.font = TITLE_FONT;
    titleWide = ctx.measureText('中').width;
    ctx.font = BODY_FONT;
    bodyWide = ctx.measureText('中').width;
  }
  cached = { title, body, titleWide, bodyWide, titleHeight: TITLE_H, lineHeight: LINE_H, padX: PAD_X, padY: PAD_Y };
  return cached;
}

export function textWidth(table: number[], wide: number, s: string): number {
  let w = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    w += c >= 32 && c <= 126 ? table[c - 32] ?? 7 : wide;
  }
  return w;
}
