import { useEffect, useState } from 'react';
import {
  callModel,
  errorText,
  exportTensor,
  openExternal,
  select,
  setGraph,
  showNode,
  showValue,
  copyText,
  useStore,
  useT,
} from '../store';
import type { NodeDetail, NodeRef, TensorMeta, TensorStats, ValueBrief, ValueDetail, ValuesPage } from '../types';
import { Badge, Button, CopyButton, Dot, Empty, Icon, IconButton, KV, Section, Spinner } from '../components/common';
import { HistogramChart } from '../components/Charts';
import { categoryColor } from '../graph/colors';
import { fmtBytes, fmtCount, fmtInt, fmtNum, multiIndex, safeFileName, shapeLabel, typeLabel } from '../util/format';
import type { MsgKey } from '../i18n';

function useDetail<T>(method: string, params: Record<string, number>, deps: unknown[]): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    setData(null);
    setError(null);
    callModel<T>(method, params)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(errorText(e)));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error };
}

export function Inspector() {
  const t = useT();
  const selection = useStore((s) => s.selection);
  const summary = useStore((s) => s.summary);
  if (!summary) return null;
  return (
    <aside className="inspector">
      <div className="panel-title">
        <span>{t('insp.title')}</span>
        {selection && <IconButton icon="x" size={13} title={t('common.close')} onClick={() => select(null)} />}
      </div>
      <div className="panel-scroll">
        {!selection && <Empty icon="info" title={t('insp.empty')} />}
        {selection?.kind === 'node' && <NodeInspector key={`n${selection.graph}:${selection.id}`} graph={selection.graph} id={selection.id} />}
        {selection?.kind === 'value' && <ValueInspector key={`v${selection.graph}:${selection.id}`} graph={selection.graph} id={selection.id} />}
      </div>
    </aside>
  );
}

function sourceKey(s: string): MsgKey {
  return `insp.source.${s}` as MsgKey;
}

function kindKey(s: string): MsgKey {
  return `insp.kind.${s}` as MsgKey;
}

function NodeLink({ r, graph }: { r: NodeRef; graph: number }) {
  return (
    <button type="button" className="link" onClick={() => showNode(graph, r.id)} title={r.name || r.op}>
      <span className="link-op">{r.op}</span>
      {r.name && <span className="link-name">{r.name}</span>}
    </button>
  );
}

function ValueLine({ v, graph, label }: { v: ValueBrief | null; graph: number; label: string }) {
  const t = useT();
  return (
    <div className="slot">
      <div className="slot-label" title={label}>
        {label}
      </div>
      <div className="slot-body">
        {!v ? (
          <span className="muted">{t('insp.omitted')}</span>
        ) : (
          <>
            <div className="slot-row">
              <button type="button" className="link mono" title={v.name} onClick={() => select({ kind: 'value', graph, id: v.id })}>
                {v.name}
              </button>
              <IconButton icon="target" size={12} title={t('common.showInGraph')} onClick={() => showValue(graph, v.id)} />
            </div>
            <div className="slot-type mono">
              {typeLabel(v.dtype, v.shape)}
              {v.source === 'inferred' && <span className="muted"> · {t('insp.source.inferred')}</span>}
              {v.kind === 'initializer' && <Badge>{t('insp.kind.initializer')}</Badge>}
              {v.kind === 'constant' && <Badge>{t('insp.kind.constant')}</Badge>}
            </div>
            {v.preview && <div className="slot-preview mono">{v.preview}</div>}
          </>
        )}
      </div>
    </div>
  );
}

function Metric({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="metric" title={title}>
      <div className="metric-v">{value}</div>
      <div className="metric-k">{label}</div>
    </div>
  );
}

function NodeInspector({ graph, id }: { graph: number; id: number }) {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const { data: d, error } = useDetail<NodeDetail>('node', { graph, node: id }, [graph, id, summary]);
  if (error) return <div className="error-text">{error}</div>;
  if (!d) return <div className="loading-row"><Spinner /></div>;
  const c = d.cost;
  const intensity = c.flops !== null && c.readBytes !== null && c.writeBytes !== null && c.readBytes + c.writeBytes > 0 ? c.flops / (c.readBytes + c.writeBytes) : null;
  return (
    <div className="insp">
      <div className="insp-head">
        <div className="insp-op">
          <Dot color={categoryColor(d.category)} />
          <span>{d.op}</span>
          <Badge>{t(`cat.${d.category}` as MsgKey)}</Badge>
        </div>
        <div className="insp-actions">
          <IconButton icon="target" title={t('common.showInGraph')} onClick={() => showNode(graph, id)} />
          {d.docUrl && <IconButton icon="external" title={t('insp.docs')} onClick={() => void openExternal(d.docUrl!)} />}
        </div>
      </div>
      {d.name && (
        <div className="insp-name mono">
          <span>{d.name}</span>
          <CopyButton text={d.name} title={t('common.copy')} />
        </div>
      )}
      <div className="metrics">
        <Metric label={t('insp.params')} value={fmtCount(c.params)} title={fmtInt(c.params)} />
        <Metric label={t('insp.macs')} value={fmtCount(c.macs)} title={fmtInt(c.macs)} />
        <Metric label={t('insp.flops')} value={fmtCount(c.flops)} title={fmtInt(c.flops)} />
        <Metric label={t('insp.memWrite')} value={fmtBytes(c.writeBytes)} />
      </div>
      <Section title={t('insp.inputs')} count={d.inputs.length}>
        {d.inputs.map((s) => (
          <ValueLine key={s.slot} v={s.value} graph={graph} label={s.label} />
        ))}
      </Section>
      <Section title={t('insp.outputs')} count={d.outputs.length}>
        {d.outputs.map((s) => (
          <ValueLine key={s.slot} v={s.value} graph={graph} label={s.label} />
        ))}
      </Section>
      {d.implicitInputs.length > 0 && (
        <Section title={t('insp.implicit')} count={d.implicitInputs.length} defaultOpen={false}>
          {d.implicitInputs.map((v) => (
            <ValueLine key={v.id} v={v} graph={graph} label="" />
          ))}
        </Section>
      )}
      {d.attributes.length > 0 && (
        <Section title={t('insp.attributes')} count={d.attributes.length}>
          <div className="attrs">
            {d.attributes.map((a) => (
              <div key={a.name} className="attr" title={a.doc || undefined}>
                <div className="attr-k">
                  {a.name}
                  <span className="attr-type">{a.typeName}</span>
                </div>
                <div className="attr-v mono">
                  {a.graphs.length > 0 ? (
                    a.graphs.map((g) => (
                      <Button key={g} small icon="layers" onClick={() => setGraph(g)}>
                        {t('insp.open')} {summary?.graphs[g]?.label ?? g}
                      </Button>
                    ))
                  ) : (
                    <AttrValue text={a.value} />
                  )}
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}
      {d.subgraphs.length > 0 && (
        <Section title={t('insp.subgraphs')} count={d.subgraphs.length}>
          {d.subgraphs.map((s) => (
            <button key={s.id} type="button" className="list-row" onClick={() => setGraph(s.id)}>
              <Icon name="layers" size={14} />
              <span className="ellipsis">{s.label}</span>
              <span className="muted">{s.nodes}</span>
            </button>
          ))}
        </Section>
      )}
      <Section title={t('insp.cost')} defaultOpen={false}>
        <KV k={t('insp.params')}>{fmtInt(c.params)} ({fmtBytes(c.paramBytes)})</KV>
        <KV k={t('insp.macs')}>{fmtInt(c.macs)}</KV>
        <KV k={t('insp.flops')}>{fmtInt(c.flops)}</KV>
        <KV k={t('insp.memRead')}>{fmtBytes(c.readBytes)}</KV>
        <KV k={t('insp.memWrite')}>{fmtBytes(c.writeBytes)}</KV>
        <KV k={t('insp.intensity')}>{intensity === null ? '–' : intensity.toFixed(2)}</KV>
      </Section>
      <Section title={t('insp.doc')} defaultOpen={false}>
        <KV k={t('insp.domain')}>{d.domain || 'ai.onnx'}</KV>
        <KV k={t('insp.opset')}>{d.opset ?? '–'}</KV>
        <KV k={t('insp.order')}>{d.order}</KV>
        {d.module && <KV k={t('insp.module')} mono>{d.module}</KV>}
        {d.metadata.map((m) => (
          <KV key={m.key} k={m.key}>
            {m.value}
          </KV>
        ))}
        {d.doc && <div className="doc-text">{d.doc}</div>}
      </Section>
    </div>
  );
}

function AttrValue({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const long = text.length > 160;
  return (
    <span className="attr-text">
      {long && !open ? `${text.slice(0, 160)}…` : text}
      {long && (
        <button type="button" className="link small" onClick={() => setOpen(!open)}>
          {open ? '−' : '+'}
        </button>
      )}
      <CopyButton text={text} />
    </span>
  );
}

function ValueInspector({ graph, id }: { graph: number; id: number }) {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const { data: d, error } = useDetail<ValueDetail>('value', { graph, value: id }, [graph, id, summary]);
  if (error) return <div className="error-text">{error}</div>;
  if (!d) return <div className="loading-row"><Spinner /></div>;
  const declared = d.declared?.shape !== undefined && d.declared?.shape !== null ? shapeLabel(d.declared.shape) : null;
  return (
    <div className="insp">
      <div className="insp-head">
        <div className="insp-op">
          <Icon name="cube" size={15} />
          <span>{t('insp.tensor')}</span>
          <Badge>{t(kindKey(d.kind))}</Badge>
          {d.isInput && d.kind !== 'input' && <Badge>{t('insp.graphInput')}</Badge>}
          {d.isOutput && <Badge tone="accent">{t('insp.graphOutput')}</Badge>}
        </div>
        <div className="insp-actions">
          <IconButton icon="target" title={t('common.showInGraph')} onClick={() => showValue(graph, id)} />
        </div>
      </div>
      <div className="insp-name mono">
        <span>{d.name}</span>
        <CopyButton text={d.name} title={t('common.copy')} />
      </div>
      <div className="metrics">
        <Metric label={t('common.dtype')} value={d.dtype || '?'} />
        <Metric label={t('common.shape')} value={shapeLabel(d.shape)} title={typeLabel(d.dtype, d.shape)} />
        <Metric label={t('insp.elements')} value={fmtCount(d.elements)} title={fmtInt(d.elements)} />
        <Metric label={t('insp.size')} value={fmtBytes(d.bytes)} />
      </div>
      <Section title={t('insp.tensor')}>
        <KV k={t('common.type')} mono>
          {typeLabel(d.dtype, d.shape)}
        </KV>
        <KV k={t('insp.shapeSource')}>{t(sourceKey(d.source))}</KV>
        {d.declared && declared !== null && declared !== shapeLabel(d.shape) && (
          <KV k={t('insp.declared')} mono>
            {d.declared.display}
          </KV>
        )}
        {d.declared && d.declared.kind !== 'tensor' && d.declared.kind !== 'unknown' && (
          <KV k={t('insp.declared')} mono>
            {d.declared.display}
          </KV>
        )}
        {d.contents && d.contents.length > 0 && (
          <KV k={t('insp.contents')} mono>
            [{d.contents.map((x) => (x === null ? '?' : String(x))).join(', ')}]
          </KV>
        )}
        {d.outer && (
          <KV k={t('insp.outer')}>
            <button type="button" className="link" onClick={() => showValue(d.outer![0], d.outer![1])}>
              {summary?.graphs[d.outer[0]]?.label ?? d.outer[0]}
            </button>
          </KV>
        )}
        {d.doc && <div className="doc-text">{d.doc}</div>}
      </Section>
      <Section title={t('insp.producer')}>
        {d.producer ? <NodeLink r={d.producer} graph={graph} /> : <span className="muted">–</span>}
      </Section>
      <Section title={t('insp.consumers', { n: d.consumers.length })}>
        {d.consumers.length === 0 && <span className="muted">{t('insp.noConsumers')}</span>}
        {d.consumers.slice(0, 200).map((c, i) => (
          <NodeLink key={`${c.id}:${i}`} r={c} graph={graph} />
        ))}
      </Section>
      {d.quantization.length > 0 && (
        <Section title={t('insp.quant')}>
          {d.quantization.map((q) => (
            <KV key={q.key} k={q.key} mono>
              {q.value}
            </KV>
          ))}
        </Section>
      )}
      {d.tensor && <TensorPanel graph={graph} meta={d.tensor} name={d.name} />}
    </div>
  );
}

const AUTO_STATS_LIMIT = 8_000_000;

function TensorPanel({ graph, meta, name }: { graph: number; meta: TensorMeta; name: string }) {
  const t = useT();
  const [stats, setStats] = useState<TensorStats | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState(false);

  const compute = () => {
    setBusy(true);
    setError(null);
    callModel<TensorStats>('tensorStats', { graph, tensor: meta.index, bins: 64 })
      .then(setStats)
      .catch((e) => setError(errorText(e)))
      .finally(() => setBusy(false));
  };

  useEffect(() => {
    if (meta.numeric && meta.elements <= AUTO_STATS_LIMIT) compute();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph, meta.index]);

  const file = safeFileName(name);
  return (
    <>
      <Section title={t('insp.storage')}>
        <KV k={t('insp.storage')}>{t(`insp.storage.${meta.storage}` as MsgKey)}</KV>
        {meta.external && (
          <KV k="location" mono>
            {meta.external.location} @ {fmtInt(meta.external.offset)}
            {meta.external.length !== null ? ` (+${fmtInt(meta.external.length)})` : ''}
          </KV>
        )}
        <KV k={t('common.shape')} mono>
          [{meta.dims.join(', ')}]
        </KV>
        <KV k={t('insp.elements')}>{fmtInt(meta.elements)}</KV>
        <KV k={t('insp.size')}>{fmtBytes(meta.bytes)}</KV>
        <div className="btn-row">
          {meta.dtype !== 'string' && (
            <Button small icon="download" onClick={() => void exportTensor(graph, meta.index, 'npy', file)}>
              {t('ts.exportNpy')}
            </Button>
          )}
          <Button small icon="download" onClick={() => void exportTensor(graph, meta.index, 'csv', file)}>
            {t('ts.exportCsv')}
          </Button>
          {meta.dtype !== 'string' && (
            <Button small icon="download" onClick={() => void exportTensor(graph, meta.index, 'raw', file)}>
              {t('ts.exportRaw')}
            </Button>
          )}
        </div>
      </Section>
      {meta.numeric && (
        <Section title={t('ts.stats')}>
          {!stats && !busy && (
            <div className="stack">
              {meta.elements > AUTO_STATS_LIMIT && <div className="muted">{t('ts.large')}</div>}
              <Button small icon="play" onClick={compute}>
                {t('ts.compute')}
              </Button>
            </div>
          )}
          {busy && (
            <div className="loading-row">
              <Spinner /> {t('ts.computing')}
            </div>
          )}
          {error && <div className="error-text">{error}</div>}
          {stats && <StatsView stats={stats} log={log} setLog={setLog} />}
        </Section>
      )}
      <Section title={t('ts.values')}>
        <ValuePager graph={graph} meta={meta} />
      </Section>
    </>
  );
}

function StatsView({ stats: s, log, setLog }: { stats: TensorStats; log: boolean; setLog: (v: boolean) => void }) {
  const t = useT();
  const pct = (n: number) => (s.count ? `${((n / s.count) * 100).toFixed(2)}%` : '–');
  const cells: { k: string; v: string; warn?: boolean }[] = [
    { k: t('ts.min'), v: fmtNum(s.min) },
    { k: t('ts.max'), v: fmtNum(s.max) },
    { k: t('ts.mean'), v: fmtNum(s.mean) },
    { k: t('ts.std'), v: fmtNum(s.std) },
    { k: t('ts.absMean'), v: fmtNum(s.absMean) },
    { k: t('ts.absMax'), v: fmtNum(s.absMax) },
    { k: t('ts.l2'), v: fmtNum(s.l2Norm) },
    { k: t('ts.zeros'), v: `${fmtInt(s.zeros)} · ${pct(s.zeros)}` },
    { k: t('ts.nan'), v: fmtInt(s.nan), warn: s.nan > 0 },
    { k: t('ts.inf'), v: fmtInt(s.posInf + s.negInf), warn: s.posInf + s.negInf > 0 },
  ];
  if (s.dtype === 'float32' || s.dtype === 'float64') {
    cells.push({ k: t('ts.fp16Overflow'), v: fmtInt(s.fp16Overflow), warn: s.fp16Overflow > 0 });
    cells.push({ k: t('ts.fp16Underflow'), v: `${fmtInt(s.fp16Underflow)} · ${pct(s.fp16Underflow)}` });
  }
  if (s.subnormal > 0) cells.push({ k: t('ts.subnormal'), v: fmtInt(s.subnormal) });
  return (
    <div className="stats">
      <div className="stat-grid">
        {cells.map((c) => (
          <div key={c.k} className={'stat' + (c.warn ? ' warn' : '')}>
            <div className="stat-k">{c.k}</div>
            <div className="stat-v mono">{c.v}</div>
          </div>
        ))}
      </div>
      {s.histogram && (
        <div className="chart-block">
          <div className="chart-head">
            <span>{t('ts.histogram')}</span>
            <label className="check">
              <input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} />
              {t('ts.log')}
            </label>
          </div>
          <HistogramChart hist={s.histogram} log={log} />
        </div>
      )}
      {s.percentiles.length > 0 && (
        <div className="pct-table">
          <div className="chart-head">
            <span>{t('ts.percentiles')}</span>
            {s.sampled && <span className="muted small">{t('ts.sampled')}</span>}
          </div>
          <table className="mini">
            <tbody>
              {s.percentiles.map(([p, v]) => (
                <tr key={p}>
                  <td className="muted">p{p}</td>
                  <td className="mono r">{fmtNum(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const PAGE = 128;

function ValuePager({ graph, meta }: { graph: number; meta: TensorMeta }) {
  const t = useT();
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<ValuesPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    setError(null);
    callModel<ValuesPage>('tensorValues', { graph, tensor: meta.index, offset, count: PAGE })
      .then((p) => alive && setPage(p))
      .catch((e) => alive && setError(errorText(e)));
    return () => {
      alive = false;
    };
  }, [graph, meta.index, offset]);
  const total = page?.total ?? meta.elements;
  const last = Math.max(0, Math.floor((total - 1) / PAGE) * PAGE);
  return (
    <div className="pager">
      <div className="pager-bar">
        <IconButton icon="arrowLeft" size={13} title="−" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))} />
        <span className="mono small">
          {fmtInt(Math.min(offset, total))}–{fmtInt(Math.min(offset + PAGE, total))} {t('ts.of', { n: fmtInt(total) })}
        </span>
        <IconButton
          icon="chevronRight"
          size={13}
          title="+"
          disabled={offset + PAGE >= total}
          onClick={() => setOffset(Math.min(last, offset + PAGE))}
        />
        <input
          className="pager-input"
          type="number"
          min={0}
          max={Math.max(0, total - 1)}
          placeholder={t('ts.offset')}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              const v = Number((e.target as HTMLInputElement).value);
              if (Number.isFinite(v)) setOffset(Math.max(0, Math.min(last, Math.floor(v / PAGE) * PAGE)));
            }
          }}
        />
        {page && (
          <IconButton icon="copy" size={13} title={t('ts.copyValues')} onClick={() => void copyText(page.values.join('\n'))} />
        )}
      </div>
      {error && <div className="error-text">{error}</div>}
      {page && (
        <div className="values">
          {page.values.map((v, i) => (
            <div key={i} className="value-row">
              <span className="value-idx">{multiIndex(page.offset + i, meta.dims)}</span>
              <span className="value-v">{v}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
