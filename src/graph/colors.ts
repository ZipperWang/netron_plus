// Node header fills per operator category. White header text clears 4.5:1 on
// every fill; the op type is always printed, so color is never the only cue.
export const CATEGORY_COLORS: Record<string, string> = {
  layer: '#1f5fae',
  activation: '#b4452c',
  pool: '#2a7a4b',
  normalization: '#16707a',
  dropout: '#5b4bc4',
  shape: '#7b5a3c',
  tensor: '#6c4a91',
  data: '#5d6168',
  quantization: '#8a5a00',
  logic: '#53701f',
  math: '#434b57',
  reduce: '#2b5f8c',
  control: '#9c3570',
  sequence: '#7c4a2b',
  ml: '#2a6b5e',
  custom: '#8c2f39',
  other: '#4b4b4b',
  module: '#3d4f7c',
  input: '#2d6a8a',
  output: '#6a3d9a',
  outer: '#6f6f68',
  missing: '#c0392b',
};

export const CATEGORY_ORDER = [
  'layer',
  'activation',
  'pool',
  'normalization',
  'dropout',
  'shape',
  'tensor',
  'data',
  'quantization',
  'logic',
  'math',
  'reduce',
  'control',
  'sequence',
  'ml',
  'custom',
  'other',
];

export function categoryColor(cat: string): string {
  return CATEGORY_COLORS[cat] ?? CATEGORY_COLORS.other;
}

export interface Theme {
  bg: string;
  nodeBg: string;
  nodeBorder: string;
  text: string;
  textMuted: string;
  headerText: string;
  edge: string;
  edgeImplicit: string;
  edgeHl: string;
  label: string;
  select: string;
  highlight: string;
  minimapBg: string;
  minimapView: string;
}

export function readTheme(): Theme {
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return {
    bg: v('--graph-bg', '#fafaf8'),
    nodeBg: v('--node-bg', '#ffffff'),
    nodeBorder: v('--node-border', '#c3c2b7'),
    text: v('--text', '#1a1a19'),
    textMuted: v('--text-3', '#6f6e69'),
    headerText: '#ffffff',
    edge: v('--edge', '#8d8b84'),
    edgeImplicit: v('--edge-implicit', '#b3b1a8'),
    edgeHl: v('--accent', '#2a78d6'),
    label: v('--text-2', '#52514e'),
    select: v('--accent', '#2a78d6'),
    highlight: v('--highlight', '#e69a00'),
    minimapBg: v('--surface', '#ffffff'),
    minimapView: v('--accent', '#2a78d6'),
  };
}
