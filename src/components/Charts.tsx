import { useLayoutEffect, useRef, useState, type RefObject } from 'react';
import type { Histogram } from '../types';
import { fmtCount, fmtNum } from '../util/format';

export interface BarItem {
  key: string;
  label: string;
  value: number;
  display: string;
  /** Identity swatch shown next to the label (never used for text). */
  swatch?: string;
  title?: string;
}

/** Horizontal single-series bars with the value at the tip. */
export function BarList({
  items,
  onClick,
  selected,
  labelWidth = 150,
}: {
  items: BarItem[];
  onClick?: (item: BarItem) => void;
  selected?: string | null;
  labelWidth?: number;
}) {
  const max = Math.max(1e-12, ...items.map((i) => i.value));
  return (
    <div className="barlist">
      {items.map((it) => (
        <div
          key={it.key}
          className={'bar-row' + (selected === it.key ? ' sel' : '') + (onClick ? ' click' : '')}
          onClick={() => onClick?.(it)}
          title={it.title ?? `${it.label}: ${it.display}`}
        >
          <div className="bar-label" style={{ width: labelWidth }}>
            {it.swatch && <span className="dot" style={{ background: it.swatch }} />}
            <span className="ellipsis">{it.label}</span>
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${Math.max(0.6, (it.value / max) * 100)}%` }} />
          </div>
          <div className="bar-value">{it.display}</div>
        </div>
      ))}
    </div>
  );
}

function useWidth<T extends HTMLElement>(): [RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(320);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function columnPath(x: number, base: number, w: number, h: number): string {
  if (h <= 0) return '';
  const r = Math.min(4, w / 2, h);
  const top = base - h;
  return `M${x},${base}V${top + r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + r}V${base}Z`;
}

/** Value histogram with a hover readout per bin. */
export function HistogramChart({ hist, log, height = 150 }: { hist: Histogram; log: boolean; height?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const n = hist.counts.length;
  const padL = 40;
  const padR = 10;
  const padT = 10;
  const padB = 22;
  const W = Math.max(40, width - padL - padR);
  const H = height - padT - padB;
  const total = hist.counts.reduce((a, b) => a + b, 0);
  const maxC = Math.max(1, ...hist.counts);
  const sy = (c: number) => (log ? Math.log10(c + 1) / Math.log10(maxC + 1) : c / maxC);
  const bw = W / Math.max(1, n);
  const gap = bw > 5 ? 2 : bw > 2.5 ? 1 : 0;
  const range = hist.max - hist.min;
  const binRange = (i: number): [number, number] => {
    if (hist.integerBins) return [hist.min + i, hist.min + i];
    if (range === 0) return [hist.min, hist.max];
    return [hist.min + (range * i) / n, hist.min + (range * (i + 1)) / n];
  };
  const xOf = (v: number) => padL + (range === 0 ? W / 2 : ((v - hist.min) / range) * W);
  const zeroInside = hist.min < 0 && hist.max > 0;
  const base = padT + H;

  return (
    <div className="hist" ref={ref}>
      <svg
        width={width}
        height={height}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const i = Math.floor((e.clientX - rect.left - padL) / bw);
          setHover(i >= 0 && i < n ? i : null);
        }}
        onMouseLeave={() => setHover(null)}
      >
        <line className="grid" x1={padL} x2={padL + W} y1={padT} y2={padT} />
        <line className="axis" x1={padL} x2={padL + W} y1={base} y2={base} />
        <text className="tick" x={padL - 6} y={padT + 4} textAnchor="end">
          {log ? `${fmtCount(maxC)}` : fmtCount(maxC)}
        </text>
        <text className="tick" x={padL - 6} y={base} textAnchor="end">
          0
        </text>
        {hist.counts.map((c, i) => {
          const h = sy(c) * H;
          const d = columnPath(padL + i * bw + gap / 2, base, Math.max(0.6, bw - gap), c > 0 ? Math.max(1, h) : 0);
          return d ? <path key={i} d={d} className={'hbar' + (hover === i ? ' hov' : '')} /> : null;
        })}
        {zeroInside && <line className="zero" x1={xOf(0)} x2={xOf(0)} y1={padT} y2={base} />}
        <text className="tick" x={padL} y={height - 6} textAnchor="start">
          {fmtNum(hist.min)}
        </text>
        <text className="tick" x={padL + W} y={height - 6} textAnchor="end">
          {fmtNum(hist.max)}
        </text>
        {zeroInside && xOf(0) - padL > 40 && padL + W - xOf(0) > 40 && (
          <text className="tick" x={xOf(0)} y={height - 6} textAnchor="middle">
            0
          </text>
        )}
      </svg>
      {hover !== null && (
        <div
          className="chart-tip"
          style={{ left: Math.min(Math.max(padL + hover * bw + bw / 2, 70), width - 70), top: 0 }}
        >
          <strong>{fmtCount(hist.counts[hover])}</strong>
          <span className="muted"> ({total ? ((hist.counts[hover] / total) * 100).toFixed(2) : 0}%)</span>
          <div className="muted mono">
            {hist.integerBins
              ? `= ${fmtNum(binRange(hover)[0])}`
              : `[${fmtNum(binRange(hover)[0])}, ${fmtNum(binRange(hover)[1])})`}
          </div>
        </div>
      )}
    </div>
  );
}
