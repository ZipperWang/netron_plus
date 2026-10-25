import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react';

export interface Column<T> {
  key: string;
  title: string;
  width: number;
  align?: 'left' | 'right';
  render?: (row: T) => ReactNode;
  /** Sort key; columns without one are not sortable. */
  sort?: (row: T) => number | string | null | undefined;
  /** First sort direction when the header is clicked (numbers usually -1). */
  firstDir?: 1 | -1;
  /** Native tooltip text. */
  tip?: (row: T) => string;
}

export interface SortState {
  key: string;
  dir: 1 | -1;
}

interface Props<T> {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => number | string;
  selectedKey?: number | string | null;
  onRowClick?: (row: T) => void;
  onRowDoubleClick?: (row: T) => void;
  rowHeight?: number;
  defaultSort?: SortState | null;
  emptyText?: string;
  rowClass?: (row: T) => string;
  onSortedChange?: (rows: T[]) => void;
}

export function sortRows<T>(rows: T[], columns: Column<T>[], sort: SortState | null): T[] {
  if (!sort) return rows;
  const col = columns.find((c) => c.key === sort.key);
  const get = col?.sort;
  if (!get) return rows;
  const keyed = rows.map((r, i) => ({ r, i, v: get(r) }));
  keyed.sort((a, b) => {
    const va = a.v;
    const vb = b.v;
    const na = va === null || va === undefined || (typeof va === 'number' && Number.isNaN(va));
    const nb = vb === null || vb === undefined || (typeof vb === 'number' && Number.isNaN(vb));
    if (na && nb) return a.i - b.i;
    if (na) return 1;
    if (nb) return -1;
    if (va === vb) return a.i - b.i;
    if (typeof va === 'string' && typeof vb === 'string') return va.localeCompare(vb) * sort.dir;
    return ((va as number) < (vb as number) ? -1 : 1) * sort.dir;
  });
  return keyed.map((k) => k.r);
}

/** Sortable table that only renders the visible rows. */
export function VirtualTable<T>({
  rows,
  columns,
  rowKey,
  selectedKey,
  onRowClick,
  onRowDoubleClick,
  rowHeight = 26,
  defaultSort = null,
  emptyText,
  rowClass,
  onSortedChange,
}: Props<T>) {
  const [sort, setSort] = useState<SortState | null>(defaultSort);
  const ref = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [height, setHeight] = useState(400);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setHeight(el.clientHeight));
    ro.observe(el);
    setHeight(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  const sorted = useMemo(() => sortRows(rows, columns, sort), [rows, columns, sort]);
  useLayoutEffect(() => {
    onSortedChange?.(sorted);
  }, [sorted, onSortedChange]);

  // Keep the selected row visible when it changes from outside.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || selectedKey === undefined || selectedKey === null) return;
    const idx = sorted.findIndex((r) => rowKey(r) === selectedKey);
    if (idx < 0) return;
    const top = (idx + 1) * rowHeight;
    if (top < el.scrollTop + rowHeight || top + rowHeight > el.scrollTop + el.clientHeight) {
      el.scrollTop = Math.max(0, top - el.clientHeight / 2);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedKey]);

  const width = columns.reduce((s, c) => s + c.width, 0);
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - 8);
  const end = Math.min(sorted.length, Math.ceil((scrollTop + height) / rowHeight) + 8);

  const clickHeader = (c: Column<T>) => {
    if (!c.sort) return;
    const first = c.firstDir ?? 1;
    if (!sort || sort.key !== c.key) setSort({ key: c.key, dir: first });
    else if (sort.dir === first) setSort({ key: c.key, dir: first === 1 ? -1 : 1 });
    else setSort(null);
  };

  return (
    <div className="vt" ref={ref} onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}>
      <div className="vt-inner" style={{ width, minWidth: '100%' }}>
        <div className="vt-head" style={{ height: rowHeight }}>
          {columns.map((c) => (
            <div
              key={c.key}
              className={'vt-th' + (c.align === 'right' ? ' r' : '') + (c.sort ? ' sortable' : '')}
              style={{ width: c.width }}
              onClick={() => clickHeader(c)}
              title={c.title}
            >
              <span className="ellipsis">{c.title}</span>
              {sort?.key === c.key && <span className="sort-ind">{sort.dir > 0 ? '▲' : '▼'}</span>}
            </div>
          ))}
        </div>
        <div className="vt-body" style={{ height: sorted.length * rowHeight }}>
          {sorted.slice(start, end).map((r, i) => {
            const idx = start + i;
            const k = rowKey(r);
            return (
              <div
                key={k}
                className={
                  'vt-row' +
                  (k === selectedKey ? ' sel' : '') +
                  (idx % 2 ? ' odd' : '') +
                  (onRowClick ? ' click' : '') +
                  (rowClass ? ` ${rowClass(r)}` : '')
                }
                style={{ top: idx * rowHeight, height: rowHeight }}
                onClick={() => onRowClick?.(r)}
                onDoubleClick={() => onRowDoubleClick?.(r)}
              >
                {columns.map((c) => (
                  <div
                    key={c.key}
                    className={'vt-td' + (c.align === 'right' ? ' r' : '')}
                    style={{ width: c.width }}
                    title={c.tip ? c.tip(r) : undefined}
                  >
                    {c.render ? c.render(r) : String((r as unknown as Record<string, unknown>)[c.key] ?? '')}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
        {sorted.length === 0 && emptyText && <div className="vt-empty">{emptyText}</div>}
      </div>
    </div>
  );
}
