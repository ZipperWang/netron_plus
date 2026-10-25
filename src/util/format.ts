import type { Dim, Shape } from '../types';

const UNITS = ['', 'K', 'M', 'G', 'T', 'P'];

/** 1234 -> "1.23K", 4.5e9 -> "4.50G". */
export function fmtCount(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || !isFinite(n)) return '–';
  const a = Math.abs(n);
  if (a < 1000) return Number.isInteger(n) ? String(n) : n.toFixed(digits);
  let i = 0;
  let v = a;
  while (v >= 1000 && i < UNITS.length - 1) {
    v /= 1000;
    i++;
  }
  const s = v >= 100 ? v.toFixed(Math.max(0, digits - 2)) : v >= 10 ? v.toFixed(Math.max(0, digits - 1)) : v.toFixed(digits);
  return (n < 0 ? '-' : '') + s + UNITS[i];
}

export function fmtFlops(n: number | null | undefined): string {
  if (n === null || n === undefined) return '–';
  return fmtCount(n);
}

/** Binary units: 1536 -> "1.50 KB". */
export function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined || !isFinite(n)) return '–';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return i === 0 ? `${v} B` : `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)} ${u[i]}`;
}

export function fmtInt(n: number | null | undefined): string {
  if (n === null || n === undefined || !isFinite(n)) return '–';
  return Math.round(n).toLocaleString('en-US');
}

/** General purpose number formatting for statistics. */
export function fmtNum(n: number | null | undefined): string {
  if (n === null || n === undefined) return '–';
  if (Number.isNaN(n)) return 'NaN';
  if (!isFinite(n)) return n > 0 ? 'inf' : '-inf';
  if (n === 0) return '0';
  const a = Math.abs(n);
  if (Number.isInteger(n) && a < 1e15) return n.toLocaleString('en-US');
  if (a >= 1e6 || a < 1e-4) return n.toExponential(4).replace(/\.?0+e/, 'e');
  return String(Number(n.toPrecision(6)));
}

export function fmtPct(part: number, total: number, digits = 1): string {
  if (!total) return '–';
  return `${((part / total) * 100).toFixed(digits)}%`;
}

export function dimLabel(d: Dim): string {
  if (d === null || d === undefined) return '?';
  return String(d);
}

/** [1,3,224,224] -> "1×3×224×224". */
export function shapeLabel(s: Shape): string {
  if (!s) return '?';
  if (s.length === 0) return 'scalar';
  return s.map(dimLabel).join('×');
}

/** "float32[1,3,224,224]" */
export function typeLabel(dtype: string, s: Shape): string {
  const d = dtype || '?';
  if (!s) return d;
  return `${d}[${s.map(dimLabel).join(',')}]`;
}

/** Flat element index -> "[i,j,k]" for the given dims. */
export function multiIndex(flat: number, dims: number[]): string {
  if (dims.length <= 1) return `[${flat}]`;
  const idx: number[] = new Array(dims.length).fill(0);
  let r = flat;
  for (let i = dims.length - 1; i >= 0; i--) {
    const d = Math.max(1, dims[i]);
    idx[i] = r % d;
    r = Math.floor(r / d);
  }
  return `[${idx.join(',')}]`;
}

export function csvEscape(v: unknown): string {
  if (v === null || v === undefined) return '';
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(header: string[], rows: unknown[][]): string {
  const lines = [header.map(csvEscape).join(',')];
  for (const r of rows) lines.push(r.map(csvEscape).join(','));
  return '﻿' + lines.join('\r\n') + '\r\n';
}

export function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

export function stripExt(name: string): string {
  const i = name.lastIndexOf('.');
  return i > 0 ? name.slice(0, i) : name;
}

export function safeFileName(s: string): string {
  return s.replace(/[\\/:*?"<>|\s]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 120) || 'tensor';
}
