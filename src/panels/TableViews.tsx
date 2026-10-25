import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  ensureNodeRows,
  ensureValueRows,
  highlightNodes,
  loadWeights,
  runChecks,
  saveText,
  select,
  showNode,
  showValue,
  useStore,
  useT,
} from '../store';
import type { Issue, NodeRow, OpRow, ValueRow, WeightRow } from '../types';
import { Badge, Button, Dot, Empty, Icon, IconButton, SearchInput, Segmented, Spinner } from '../components/common';
import { BarList, type BarItem } from '../components/Charts';
import { VirtualTable, type Column } from '../components/VirtualTable';
import { categoryColor } from '../graph/colors';
import { fmtBytes, fmtCount, fmtInt, fmtNum, fmtPct, stripExt, toCsv } from '../util/format';
import { hasKey, type MsgKey, type TFunc } from '../i18n';
import { opTitle, viewNodesOfOp } from './Sidebar';

function useFileStem(): string {
  const name = useStore((s) => s.summary?.fileName ?? 'model');
  return stripExt(name);
}

function exportCsv<T>(name: string, columns: Column<T>[], rows: T[], value: (c: Column<T>, r: T) => unknown) {
  const csv = toCsv(
    columns.map((c) => c.title),
    rows.map((r) => columns.map((c) => value(c, r))),
  );
  void saveText(name, csv, { name: 'CSV', extensions: ['csv'] });
}

function TableToolbar({
  filter,
  setFilter,
  children,
  count,
  onExport,
}: {
  filter: string;
  setFilter: (v: string) => void;
  children?: ReactNode;
  count: number;
  onExport: () => void;
}) {
  const t = useT();
  return (
    <div className="table-toolbar">
      <SearchInput value={filter} onChange={setFilter} placeholder={t('common.filter')} />
      {children}
      <div className="spacer" />
      <span className="muted small">{t('common.rows', { n: fmtInt(count) })}</span>
      <Button small icon="download" onClick={onExport}>
        {t('common.exportCsv')}
      </Button>
    </div>
  );
}

// ----------------------------------------------------------------- operators

export function OperatorsView() {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const graphId = useStore((s) => s.graphId);
  const stem = useFileStem();
  const [filter, setFilter] = useState('');
  const [metric, setMetric] = useState<'count' | 'flops' | 'params'>('count');
  const [sel, setSel] = useState<string | null>(null);
  const rows = summary?.ops ?? [];
  const totalCount = rows.reduce((a, r) => a + r.count, 0);
  const totalFlops = rows.reduce((a, r) => a + r.flops, 0);
  const totalParams = rows.reduce((a, r) => a + r.params, 0);
  const q = filter.trim().toLowerCase();
  const filtered = q ? rows.filter((r) => r.op.toLowerCase().includes(q) || r.domain.toLowerCase().includes(q) || r.category.includes(q)) : rows;

  const columns = useMemo<Column<OpRow>[]>(
    () => [
      {
        key: 'op',
        title: t('ops.op'),
        width: 210,
        sort: (r) => r.op,
        render: (r) => (
          <span className="cell-op">
            <Dot color={categoryColor(r.category)} />
            {r.op}
          </span>
        ),
      },
      { key: 'domain', title: t('ops.domain'), width: 120, sort: (r) => r.domain, render: (r) => r.domain || 'ai.onnx' },
      { key: 'category', title: t('ops.category'), width: 110, sort: (r) => r.category, render: (r) => t(`cat.${r.category}` as MsgKey) },
      { key: 'count', title: t('ops.count'), width: 80, align: 'right', sort: (r) => r.count, firstDir: -1, render: (r) => fmtInt(r.count) },
      { key: 'share', title: t('ops.share'), width: 80, align: 'right', sort: (r) => r.count, firstDir: -1, render: (r) => fmtPct(r.count, totalCount) },
      { key: 'params', title: t('ops.params'), width: 100, align: 'right', sort: (r) => r.params, firstDir: -1, render: (r) => fmtCount(r.params), tip: (r) => fmtInt(r.params) },
      { key: 'flops', title: t('ops.flops'), width: 100, align: 'right', sort: (r) => r.flops, firstDir: -1, render: (r) => fmtCount(r.flops), tip: (r) => fmtInt(r.flops) },
      { key: 'flopsShare', title: t('ops.flopsShare'), width: 100, align: 'right', sort: (r) => r.flops, firstDir: -1, render: (r) => fmtPct(r.flops, totalFlops) },
      { key: 'macs', title: t('ops.macs'), width: 100, align: 'right', sort: (r) => r.macs, firstDir: -1, render: (r) => fmtCount(r.macs) },
      { key: 'costed', title: 'FLOPs ✓', width: 80, align: 'right', sort: (r) => r.costed / Math.max(1, r.count), render: (r) => `${r.costed}/${r.count}` },
    ],
    [t, totalCount, totalFlops],
  );

  if (!summary) return null;
  const chartItems: BarItem[] = [...filtered]
    .map((o) => ({ o, v: metric === 'count' ? o.count : metric === 'flops' ? o.flops : o.params }))
    .filter((x) => x.v > 0)
    .sort((a, b) => b.v - a.v)
    .slice(0, 24)
    .map(({ o, v }) => ({
      key: `${o.domain}:${o.op}`,
      label: opTitle(o.op, o.domain),
      value: v,
      display:
        metric === 'count'
          ? `${fmtInt(v)} · ${fmtPct(v, totalCount)}`
          : `${fmtCount(v)} · ${fmtPct(v, metric === 'flops' ? totalFlops : totalParams)}`,
      swatch: categoryColor(o.category),
    }));

  const pick = (r: OpRow, jump: boolean) => {
    setSel(`${r.domain}:${r.op}`);
    highlightNodes(opTitle(r.op, r.domain), graphId, viewNodesOfOp(r), jump);
  };

  return (
    <div className="page page-split">
      <div className="split-main">
        <TableToolbar
          filter={filter}
          setFilter={setFilter}
          count={filtered.length}
          onExport={() =>
            exportCsv(`${stem}.operators.csv`, columns, filtered, (c, r) =>
              c.key === 'share' ? fmtPct(r.count, totalCount) : c.key === 'flopsShare' ? fmtPct(r.flops, totalFlops) : c.key === 'costed' ? `${r.costed}/${r.count}` : (r as unknown as Record<string, unknown>)[c.key],
            )
          }
        >
          <span className="muted small">{t('ops.hint')}</span>
        </TableToolbar>
        <VirtualTable
          rows={filtered}
          columns={columns}
          rowKey={(r) => `${r.domain}:${r.op}`}
          selectedKey={sel}
          defaultSort={{ key: 'count', dir: -1 }}
          onRowClick={(r) => pick(r, false)}
          onRowDoubleClick={(r) => pick(r, true)}
        />
      </div>
      <div className="split-side card">
        <div className="card-title">
          <span>{t('ops.metric')}</span>
          <Segmented
            value={metric}
            onChange={setMetric}
            options={[
              { value: 'count', label: t('sum.byCount') },
              { value: 'flops', label: t('sum.byFlops') },
              { value: 'params', label: t('sum.byParams') },
            ]}
          />
        </div>
        <BarList
          items={chartItems}
          selected={sel}
          labelWidth={130}
          onClick={(it) => {
            const r = filtered.find((o) => `${o.domain}:${o.op}` === it.key);
            if (r) pick(r, false);
          }}
        />
      </div>
    </div>
  );
}

// --------------------------------------------------------------------- nodes

export function NodesView() {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const graphId = useStore((s) => s.graphId);
  const nodeRows = useStore((s) => s.nodeRows);
  const nodeFilter = useStore((s) => s.nodeFilter);
  const selection = useStore((s) => s.selection);
  const stem = useFileStem();
  const [filter, setFilter] = useState(nodeFilter);
  const [op, setOp] = useState('');

  useEffect(() => {
    void ensureNodeRows();
  }, [graphId, summary]);
  useEffect(() => {
    if (nodeFilter) setFilter(nodeFilter);
  }, [nodeFilter]);

  const rows = nodeRows && nodeRows.graph === graphId ? nodeRows.rows : null;
  const ops = useMemo(() => Array.from(new Set((rows ?? []).map((r) => r.op))).sort(), [rows]);
  const q = filter.trim().toLowerCase();
  const filtered = useMemo(
    () =>
      (rows ?? []).filter(
        (r) =>
          (!op || r.op === op) &&
          (!q || r.name.toLowerCase().includes(q) || r.op.toLowerCase().includes(q) || r.module.toLowerCase().includes(q)),
      ),
    [rows, op, q],
  );
  const sums = useMemo(() => {
    let params = 0;
    let flops = 0;
    for (const r of filtered) {
      params += r.params;
      flops += r.flops ?? 0;
    }
    return { params, flops };
  }, [filtered]);

  const columns = useMemo<Column<NodeRow>[]>(
    () => [
      { key: 'order', title: t('nodes.order'), width: 60, align: 'right', sort: (r) => r.order, render: (r) => r.order },
      { key: 'op', title: t('nodes.op'), width: 160, sort: (r) => r.op, render: (r) => <span className="cell-op"><Dot color={categoryColor(r.cat)} />{r.op}</span> },
      { key: 'name', title: t('nodes.name'), width: 280, sort: (r) => r.name, render: (r) => <span className="mono">{r.name}</span>, tip: (r) => r.name },
      { key: 'inputs', title: t('nodes.inputs'), width: 190, render: (r) => <span className="mono">{r.inputs}</span>, tip: (r) => r.inputs },
      { key: 'outputs', title: t('nodes.outputs'), width: 190, render: (r) => <span className="mono">{r.outputs}</span>, tip: (r) => r.outputs },
      { key: 'params', title: t('nodes.params'), width: 90, align: 'right', sort: (r) => r.params, firstDir: -1, render: (r) => (r.params ? fmtCount(r.params) : ''), tip: (r) => fmtInt(r.params) },
      { key: 'macs', title: t('nodes.macs'), width: 90, align: 'right', sort: (r) => r.macs, firstDir: -1, render: (r) => (r.macs ? fmtCount(r.macs) : r.macs === null ? '?' : '') },
      { key: 'flops', title: t('nodes.flops'), width: 90, align: 'right', sort: (r) => r.flops, firstDir: -1, render: (r) => (r.flops ? fmtCount(r.flops) : r.flops === null ? '?' : ''), tip: (r) => fmtInt(r.flops) },
      { key: 'outBytes', title: t('nodes.outBytes'), width: 100, align: 'right', sort: (r) => r.outBytes, firstDir: -1, render: (r) => fmtBytes(r.outBytes) },
      { key: 'module', title: t('nodes.module'), width: 260, sort: (r) => r.module, render: (r) => <span className="mono muted">{r.module}</span>, tip: (r) => r.module },
    ],
    [t],
  );

  if (!summary) return null;
  return (
    <div className="page page-table">
      <TableToolbar
        filter={filter}
        setFilter={(v) => {
          setFilter(v);
          if (nodeFilter) useStore.setState({ nodeFilter: '' });
        }}
        count={filtered.length}
        onExport={() => exportCsv(`${stem}.nodes.csv`, columns, filtered, (c, r) => (r as unknown as Record<string, unknown>)[c.key])}
      >
        <select className="select" value={op} onChange={(e) => setOp(e.target.value)}>
          <option value="">{t('nodes.allOps')}</option>
          {ops.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
        <span className="muted small">
          Σ {t('nodes.params')} {fmtCount(sums.params)} · Σ FLOPs {fmtCount(sums.flops)}
        </span>
      </TableToolbar>
      {!rows ? (
        <div className="loading-row">
          <Spinner />
        </div>
      ) : (
        <VirtualTable
          rows={filtered}
          columns={columns}
          rowKey={(r) => r.id}
          selectedKey={selection?.kind === 'node' && selection.graph === graphId ? selection.id : null}
          onRowClick={(r) => select({ kind: 'node', graph: graphId, id: r.id })}
          onRowDoubleClick={(r) => showNode(graphId, r.id)}
        />
      )}
    </div>
  );
}

// ------------------------------------------------------------------- tensors

const KINDS = ['all', 'input', 'output', 'initializer', 'constant', 'intermediate'] as const;
type KindFilter = (typeof KINDS)[number];

export function TensorsView() {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const graphId = useStore((s) => s.graphId);
  const valueRows = useStore((s) => s.valueRows);
  const selection = useStore((s) => s.selection);
  const stem = useFileStem();
  const [filter, setFilter] = useState('');
  const [kind, setKind] = useState<KindFilter>('all');

  useEffect(() => {
    void ensureValueRows();
  }, [graphId, summary]);

  const rows = valueRows && valueRows.graph === graphId ? valueRows.rows : null;
  const q = filter.trim().toLowerCase();
  const filtered = useMemo(
    () =>
      (rows ?? []).filter((r) => {
        if (kind === 'output' ? !r.isOutput : kind === 'input' ? r.kind !== 'input' : kind !== 'all' && r.kind !== kind) return false;
        return !q || r.name.toLowerCase().includes(q) || r.dtype.includes(q) || r.shape.toLowerCase().includes(q) || r.producerOp.toLowerCase().includes(q);
      }),
    [rows, kind, q],
  );

  const columns = useMemo<Column<ValueRow>[]>(
    () => [
      { key: 'name', title: t('tensors.name'), width: 320, sort: (r) => r.name, render: (r) => <span className="mono">{r.name}</span>, tip: (r) => r.name },
      {
        key: 'kind',
        title: t('tensors.kind'),
        width: 130,
        sort: (r) => r.kind,
        render: (r) => (
          <span>
            {t(`insp.kind.${r.kind}` as MsgKey)}
            {r.isOutput && r.kind !== 'input' ? ' · out' : ''}
          </span>
        ),
      },
      { key: 'dtype', title: t('tensors.dtype'), width: 90, sort: (r) => r.dtype, render: (r) => <span className="mono">{r.dtype}</span> },
      { key: 'shape', title: t('tensors.shape'), width: 170, sort: (r) => r.elements ?? -1, render: (r) => <span className="mono">{r.shape}</span>, tip: (r) => r.shape },
      { key: 'elements', title: t('tensors.elements'), width: 100, align: 'right', sort: (r) => r.elements, firstDir: -1, render: (r) => fmtCount(r.elements), tip: (r) => fmtInt(r.elements) },
      { key: 'bytes', title: t('tensors.bytes'), width: 100, align: 'right', sort: (r) => r.bytes, firstDir: -1, render: (r) => fmtBytes(r.bytes) },
      { key: 'producerOp', title: t('tensors.producer'), width: 130, sort: (r) => r.producerOp, render: (r) => r.producerOp },
      { key: 'consumers', title: t('tensors.consumers'), width: 90, align: 'right', sort: (r) => r.consumers, firstDir: -1, render: (r) => r.consumers },
      { key: 'source', title: t('tensors.source'), width: 110, sort: (r) => r.source, render: (r) => <span className="muted">{t(`insp.source.${r.source}` as MsgKey)}</span> },
    ],
    [t],
  );

  if (!summary) return null;
  return (
    <div className="page page-table">
      <TableToolbar
        filter={filter}
        setFilter={setFilter}
        count={filtered.length}
        onExport={() => exportCsv(`${stem}.tensors.csv`, columns, filtered, (c, r) => (r as unknown as Record<string, unknown>)[c.key])}
      >
        <Segmented
          value={kind}
          onChange={setKind}
          options={KINDS.map((k) => ({ value: k, label: k === 'all' ? t('common.all') : k === 'output' ? t('insp.graphOutput') : t(`insp.kind.${k}` as MsgKey) }))}
        />
      </TableToolbar>
      {!rows ? (
        <div className="loading-row">
          <Spinner />
        </div>
      ) : (
        <VirtualTable
          rows={filtered}
          columns={columns}
          rowKey={(r) => r.id}
          selectedKey={selection?.kind === 'value' && selection.graph === graphId ? selection.id : null}
          onRowClick={(r) => select({ kind: 'value', graph: graphId, id: r.id })}
          onRowDoubleClick={(r) => showValue(graphId, r.id)}
        />
      )}
    </div>
  );
}

// ------------------------------------------------------------------- weights

function isProblem(r: WeightRow): boolean {
  return r.nan > 0 || r.inf > 0 || r.fp16Overflow > 0 || !!r.error || (r.elements > 1 && r.zeros === r.elements);
}

export function WeightsView() {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const rows = useStore((s) => s.weightRows);
  const loading = useStore((s) => s.loading);
  const selection = useStore((s) => s.selection);
  const stem = useFileStem();
  const [filter, setFilter] = useState('');
  const [onlyProblems, setOnlyProblems] = useState(false);
  const q = filter.trim().toLowerCase();
  const filtered = useMemo(
    () => (rows ?? []).filter((r) => (!onlyProblems || isProblem(r)) && (!q || r.name.toLowerCase().includes(q) || r.consumers.toLowerCase().includes(q) || r.dtype.includes(q))),
    [rows, q, onlyProblems],
  );
  const columns = useMemo<Column<WeightRow>[]>(
    () => [
      { key: 'name', title: t('tensors.name'), width: 280, sort: (r) => r.name, render: (r) => <span className="mono">{r.name}</span>, tip: (r) => r.error ?? r.name },
      { key: 'dtype', title: t('tensors.dtype'), width: 80, sort: (r) => r.dtype, render: (r) => <span className="mono">{r.dtype}</span> },
      { key: 'shape', title: t('tensors.shape'), width: 140, sort: (r) => r.elements, render: (r) => <span className="mono">{r.shape}</span>, tip: (r) => r.shape },
      { key: 'elements', title: t('tensors.elements'), width: 90, align: 'right', sort: (r) => r.elements, firstDir: -1, render: (r) => fmtCount(r.elements), tip: (r) => fmtInt(r.elements) },
      { key: 'bytes', title: t('tensors.bytes'), width: 90, align: 'right', sort: (r) => r.bytes, firstDir: -1, render: (r) => fmtBytes(r.bytes) },
      { key: 'min', title: t('ts.min'), width: 95, align: 'right', sort: (r) => r.min, render: (r) => fmtNum(r.min) },
      { key: 'max', title: t('ts.max'), width: 95, align: 'right', sort: (r) => r.max, firstDir: -1, render: (r) => fmtNum(r.max) },
      { key: 'mean', title: t('ts.mean'), width: 95, align: 'right', sort: (r) => r.mean, render: (r) => fmtNum(r.mean) },
      { key: 'std', title: t('ts.std'), width: 95, align: 'right', sort: (r) => r.std, firstDir: -1, render: (r) => fmtNum(r.std) },
      { key: 'absMean', title: t('ts.absMean'), width: 95, align: 'right', sort: (r) => r.absMean, firstDir: -1, render: (r) => fmtNum(r.absMean) },
      { key: 'zeros', title: t('weights.zerosPct'), width: 80, align: 'right', sort: (r) => r.zeros / Math.max(1, r.elements), firstDir: -1, render: (r) => fmtPct(r.zeros, r.elements) },
      { key: 'nan', title: 'NaN', width: 60, align: 'right', sort: (r) => r.nan, firstDir: -1, render: (r) => (r.nan ? <span className="bad">{fmtInt(r.nan)}</span> : '0') },
      { key: 'inf', title: 'Inf', width: 60, align: 'right', sort: (r) => r.inf, firstDir: -1, render: (r) => (r.inf ? <span className="bad">{fmtInt(r.inf)}</span> : '0') },
      { key: 'fp16Overflow', title: t('ts.fp16Overflow'), width: 100, align: 'right', sort: (r) => r.fp16Overflow, firstDir: -1, render: (r) => (r.fp16Overflow ? <span className="bad">{fmtInt(r.fp16Overflow)}</span> : '0') },
      { key: 'consumers', title: t('weights.consumers'), width: 160, sort: (r) => r.consumers, render: (r) => r.consumers, tip: (r) => r.consumers },
    ],
    [t],
  );
  if (!summary) return null;
  if (!rows) {
    return (
      <div className="page">
        <Empty icon="cube" title={t('weights.title')}>
          <p>{t('weights.hint')}</p>
          <Button kind="primary" icon="play" disabled={!!loading} onClick={() => void loadWeights()}>
            {loading ? t('weights.running') : t('weights.run')}
          </Button>
        </Empty>
      </div>
    );
  }
  return (
    <div className="page page-table">
      <TableToolbar
        filter={filter}
        setFilter={setFilter}
        count={filtered.length}
        onExport={() => exportCsv(`${stem}.weights.csv`, columns, filtered, (c, r) => (r as unknown as Record<string, unknown>)[c.key])}
      >
        <label className="check">
          <input type="checkbox" checked={onlyProblems} onChange={(e) => setOnlyProblems(e.target.checked)} />
          {t('weights.onlyProblems')} ({rows.filter(isProblem).length})
        </label>
        <IconButton icon="refresh" title={t('weights.run')} onClick={() => void loadWeights()} />
      </TableToolbar>
      <VirtualTable
        rows={filtered}
        columns={columns}
        rowKey={(r) => `${r.graph}:${r.value}`}
        selectedKey={selection?.kind === 'value' ? `${selection.graph}:${selection.id}` : null}
        rowClass={(r) => (isProblem(r) ? 'row-warn' : '')}
        onRowClick={(r) => select({ kind: 'value', graph: r.graph, id: r.value })}
        onRowDoubleClick={(r) => showValue(r.graph, r.value)}
      />
    </div>
  );
}

// -------------------------------------------------------------------- checks

export function issueText(t: TFunc, i: Issue): string {
  const key = `issue.${i.code}`;
  return hasKey(key) ? t(key, i.params) : i.message;
}

const SEV_ICON = { error: 'error', warning: 'alert', info: 'info' } as const;

export function ChecksView() {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const issues = useStore((s) => s.issues);
  const loading = useStore((s) => s.loading);
  const [open, setOpen] = useState<Record<number, boolean>>({});

  useEffect(() => {
    if (summary && !issues) void runChecks(false);
  }, [summary, issues]);

  const toggle = useCallback((i: number) => setOpen((o) => ({ ...o, [i]: !o[i] })), []);
  if (!summary) return null;
  const list = issues?.list ?? [];
  const counts = { error: 0, warning: 0, info: 0 };
  for (const i of list) counts[i.severity]++;

  return (
    <div className="page">
      <div className="table-toolbar">
        <span className="page-title">{t('checks.title')}</span>
        <Badge tone="error">
          {t('checks.error')} {counts.error}
        </Badge>
        <Badge tone="warning">
          {t('checks.warning')} {counts.warning}
        </Badge>
        <Badge tone="info">
          {t('checks.info')} {counts.info}
        </Badge>
        <div className="spacer" />
        <Button icon="play" disabled={!!loading} onClick={() => void runChecks(true)}>
          {issues?.deep ? `✓ ${t('checks.deep')}` : t('checks.deep')}
        </Button>
      </div>
      {!issues && (
        <div className="loading-row">
          <Spinner /> {t('checks.running')}
        </div>
      )}
      {issues && list.length === 0 && <Empty icon="check" title={t('checks.none')} />}
      <div className="issues">
        {list.map((i, idx) => (
          <div key={idx} className={`issue sev-${i.severity}`}>
            <div className="issue-main">
              <Icon name={SEV_ICON[i.severity]} size={16} className="issue-icon" />
              <span className="issue-sev">{t(`checks.${i.severity}` as MsgKey)}</span>
              <span className="issue-text">{issueText(t, i)}</span>
              {i.graph !== 0 && <span className="muted small">#{i.graph}</span>}
              <div className="spacer" />
              {i.node !== null && (
                <Button small icon="target" onClick={() => showNode(i.graph, i.node!)}>
                  {t('checks.locate')}
                </Button>
              )}
              {i.node === null && i.value !== null && (
                <Button small icon="target" onClick={() => showValue(i.graph, i.value!)}>
                  {t('checks.locate')}
                </Button>
              )}
              {i.items.length > 0 && (
                <Button small kind="ghost" onClick={() => toggle(idx)}>
                  {open[idx] ? t('checks.hide') : t('checks.items', { n: i.items.length })}
                </Button>
              )}
            </div>
            {open[idx] && (
              <div className="issue-items mono">
                {i.items.map((it, k) => (
                  <div key={k}>{it}</div>
                ))}
                {i.count > i.items.length && <div className="muted">… +{i.count - i.items.length}</div>}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
