import { useEffect, useMemo, useState } from 'react';
import { applyOverrides, setTab, showNode, showValue, useStore, useT } from '../store';
import type { ModelSummary } from '../types';
import { Button, Icon, KV, Segmented } from '../components/common';
import { BarList, type BarItem } from '../components/Charts';
import { categoryColor } from '../graph/colors';
import { fmtBytes, fmtCount, fmtInt, shapeLabel, typeLabel } from '../util/format';
import type { MsgKey } from '../i18n';

function Tile({ label, value, sub, title }: { label: string; value: string; sub?: string; title?: string }) {
  return (
    <div className="tile" title={title}>
      <div className="tile-k">{label}</div>
      <div className="tile-v">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

export function SummaryView() {
  const t = useT();
  const s = useStore((st) => st.summary);
  const [metric, setMetric] = useState<'count' | 'flops' | 'params'>('count');
  if (!s) return null;
  const tot = s.totals;
  const mem = s.memory;
  const stdOpset = s.opsets.find((o) => o.domain === 'ai.onnx' || o.domain === '')?.version;

  const opItems: BarItem[] = [...s.ops]
    .sort((a, b) => (metric === 'count' ? b.count - a.count : metric === 'flops' ? b.flops - a.flops : b.params - a.params))
    .filter((o) => (metric === 'count' ? o.count : metric === 'flops' ? o.flops : o.params) > 0)
    .slice(0, 16)
    .map((o) => {
      const value = metric === 'count' ? o.count : metric === 'flops' ? o.flops : o.params;
      return {
        key: `${o.domain}:${o.op}`,
        label: o.domain ? `${o.domain}::${o.op}` : o.op,
        value,
        display: metric === 'count' ? fmtInt(value) : fmtCount(value),
        swatch: categoryColor(o.category),
      };
    });

  const catItems: BarItem[] = s.categories.map((c) => ({
    key: c.category,
    label: t(`cat.${c.category}` as MsgKey),
    value: c.count,
    display: fmtInt(c.count),
    swatch: categoryColor(c.category),
    title: `${t(`cat.${c.category}` as MsgKey)}: ${c.count} · ${fmtCount(c.flops)} FLOPs`,
  }));

  const dtypeItems: BarItem[] = s.dtypes.map((d) => ({
    key: d.dtype,
    label: d.dtype,
    value: d.bytes,
    display: fmtBytes(d.bytes),
    title: `${d.dtype}: ${d.tensors} tensors · ${fmtCount(d.elements)} elements · ${fmtBytes(d.bytes)}`,
  }));

  return (
    <div className="page">
      <div className="tiles">
        <Tile
          label={t('sum.nodes')}
          value={fmtInt(s.stats.nodes)}
          sub={s.stats.nodesTotal !== s.stats.nodes ? `${fmtInt(s.stats.nodesTotal)} · ${t('sum.graphs')} ${s.stats.graphs}` : undefined}
        />
        <Tile label={t('sum.params')} value={fmtCount(tot.params)} sub={fmtBytes(tot.paramBytes)} title={fmtInt(tot.params)} />
        <Tile label={t('sum.flops')} value={fmtCount(tot.flops)} sub={`${t('sum.macs')} ${fmtCount(tot.macs)}`} title={fmtInt(tot.flops)} />
        <Tile label={t('sum.weights')} value={fmtBytes(mem.weightsBytes)} />
        <Tile label={t('sum.peak')} value={fmtBytes(mem.peakBytes)} sub={`${t('sum.activations')} ${fmtBytes(mem.activationBytes)}`} />
        <Tile label={t('sum.opTypes')} value={fmtInt(s.stats.opTypes)} />
        <Tile label="Opset" value={stdOpset !== undefined ? String(stdOpset) : '–'} sub={`IR v${s.irVersion}`} />
        <Tile label={t('sum.fileSize')} value={fmtBytes(s.fileSize)} />
      </div>
      <div className="note-line muted">
        {t('sum.estimate')} {t('sum.coverage', { a: tot.costedNodes, b: tot.costedNodes + tot.uncostedNodes })}
        {tot.uncostedOps.length > 0 && <> · {t('sum.uncosted', { ops: tot.uncostedOps.join(', ') })}</>}
        {s.stats.unknownShapes > 0 && <> · {t('sum.unknownShapes', { n: s.stats.unknownShapes, m: s.stats.intermediateValues })}</>}
        {mem.peakNode !== null && (
          <>
            {' · '}
            {t('sum.peak')}{' '}
            <button type="button" className="link" onClick={() => showNode(0, mem.peakNode!)}>
              {t('sum.peakAt', { node: `#${mem.peakNode}` })}
            </button>
          </>
        )}
      </div>

      <div className="cols">
        <div className="card">
          <div className="card-title">{t('sum.io')}</div>
          <table className="grid-table">
            <thead>
              <tr>
                <th />
                <th>{t('common.name')}</th>
                <th>{t('common.type')}</th>
                <th className="r">{t('insp.size')}</th>
              </tr>
            </thead>
            <tbody>
              {s.inputs.map((io) => (
                <tr key={`i${io.value}`} className="click" onClick={() => showValue(0, io.value)}>
                  <td className="muted small">{t('sum.input')}</td>
                  <td className="mono">{io.name}</td>
                  <td className="mono">{io.display || typeLabel(io.dtype, io.shape)}</td>
                  <td className="r">{fmtBytes(io.bytes)}</td>
                </tr>
              ))}
              {s.outputs.map((io) => (
                <tr key={`o${io.value}`} className="click" onClick={() => showValue(0, io.value)}>
                  <td className="muted small">{t('sum.output')}</td>
                  <td className="mono">{io.name}</td>
                  <td className="mono">{io.display || typeLabel(io.dtype, io.shape)}</td>
                  <td className="r">{fmtBytes(io.bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <SymbolEditor summary={s} />
        </div>

        <div className="card">
          <div className="card-title">{t('sum.model')}</div>
          <KV k={t('sum.file')} mono>
            {s.filePath ?? s.fileName}
          </KV>
          <KV k={t('sum.producer')}>
            {s.producerName || '–'} {s.producerVersion}
          </KV>
          <KV k={t('sum.irVersion')}>{s.irVersion}</KV>
          <KV k={t('sum.opsets')} mono>
            {s.opsets.map((o) => `${o.domain} v${o.version}`).join(', ') || '–'}
          </KV>
          {s.domain && <KV k={t('sum.domain')}>{s.domain}</KV>}
          {s.modelVersion !== 0 && <KV k={t('sum.modelVersion')}>{s.modelVersion}</KV>}
          {s.stats.subgraphs > 0 && <KV k={t('sum.graphs')}>{t('sum.subgraphs', { n: s.stats.subgraphs })}</KV>}
          {s.stats.functions > 0 && <KV k=" ">{t('sum.functions', { n: s.stats.functions })}</KV>}
          {s.externalData && <KV k=" ">{t('sum.externalData')}</KV>}
          <KV k={t('sum.inputBytes')}>{fmtBytes(mem.inputBytes)}</KV>
          <KV k={t('sum.outputBytes')}>{fmtBytes(mem.outputBytes)}</KV>
          {s.doc && <div className="doc-text">{s.doc}</div>}
          {s.metadata.length > 0 && (
            <>
              <div className="card-subtitle">{t('sum.metadata')}</div>
              {s.metadata.map((m) => (
                <KV key={m.key} k={m.key} mono>
                  {m.value}
                </KV>
              ))}
            </>
          )}
        </div>
      </div>

      <div className="cols">
        <div className="card">
          <div className="card-title">
            <span>{t('sum.opDist')}</span>
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
          <BarList items={opItems} />
        </div>
        <div className="card">
          <div className="card-title">{t('sum.catDist')}</div>
          <BarList items={catItems} labelWidth={110} />
          {dtypeItems.length > 0 && (
            <>
              <div className="card-subtitle">{t('sum.dtypeDist')}</div>
              <BarList items={dtypeItems} labelWidth={110} />
            </>
          )}
        </div>
      </div>

      <div className="cols">
        <div className="card">
          <div className="card-title">{t('sum.largest')}</div>
          <table className="grid-table">
            <thead>
              <tr>
                <th>{t('common.name')}</th>
                <th>{t('common.dtype')}</th>
                <th>{t('common.shape')}</th>
                <th className="r">{t('insp.size')}</th>
              </tr>
            </thead>
            <tbody>
              {s.largest.map((r) => (
                <tr key={`${r.graph}:${r.value}`} className="click" onClick={() => showValue(r.graph, r.value)}>
                  <td className="mono ellipsis-cell" title={r.name}>
                    {r.name}
                  </td>
                  <td className="mono">{r.dtype}</td>
                  <td className="mono">{r.shape}</td>
                  <td className="r">{fmtBytes(r.bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <div className="card-title">{t('sum.issues')}</div>
          <div className="issue-counts">
            <div className="issue-count error">
              <Icon name="error" size={18} />
              <span className="n">{s.issues.error}</span>
              <span>{t('checks.error')}</span>
            </div>
            <div className="issue-count warning">
              <Icon name="alert" size={18} />
              <span className="n">{s.issues.warning}</span>
              <span>{t('checks.warning')}</span>
            </div>
            <div className="issue-count info">
              <Icon name="info" size={18} />
              <span className="n">{s.issues.info}</span>
              <span>{t('checks.info')}</span>
            </div>
          </div>
          <Button icon="chevronRight" onClick={() => setTab('checks')}>
            {t('sum.viewChecks')}
          </Button>
        </div>
      </div>
    </div>
  );
}

function SymbolEditor({ summary }: { summary: ModelSummary }) {
  const t = useT();
  const initial = useMemo(() => {
    const p: Record<string, string> = {};
    for (const s of summary.symbols) p[s.name] = s.value !== null ? String(s.value) : '';
    const inputs: Record<string, string[]> = {};
    for (const io of summary.inputs) {
      if (!io.declared || !io.declared.some((d) => d === null)) continue;
      const ov = summary.overrides.inputs[io.name] ?? [];
      inputs[io.name] = io.declared.map((d, i) => (d === null ? (ov[i] !== null && ov[i] !== undefined ? String(ov[i]) : '') : String(d)));
    }
    return { p, inputs };
  }, [summary]);
  const [params, setParams] = useState(initial.p);
  const [inputs, setInputs] = useState(initial.inputs);
  useEffect(() => {
    setParams(initial.p);
    setInputs(initial.inputs);
  }, [initial]);

  const unnamed = summary.inputs.filter((io) => io.declared && io.declared.some((d) => d === null));
  if (summary.symbols.length === 0 && unnamed.length === 0) {
    return <div className="muted small pad-top">{t('sum.noSymbols')}</div>;
  }
  const apply = () => {
    const p: Record<string, number | null> = {};
    for (const [k, v] of Object.entries(params)) p[k] = v.trim() === '' ? null : Math.max(0, Math.floor(Number(v)));
    const ins: Record<string, (number | null)[]> = {};
    for (const io of unnamed) {
      const vals = inputs[io.name] ?? [];
      ins[io.name] = io.declared!.map((d, i) => (d === null && vals[i] && vals[i].trim() !== '' ? Math.max(0, Math.floor(Number(vals[i]))) : null));
    }
    void applyOverrides(p, ins);
  };
  return (
    <div className="symbols">
      <div className="card-subtitle">{t('sum.symbols')}</div>
      <div className="muted small">{t('sum.symbolsHint')}</div>
      {summary.symbols.map((s) => (
        <div key={s.name} className="sym-row">
          <span className="mono sym-name">{s.name}</span>
          <input
            type="number"
            min={0}
            className="num-input"
            value={params[s.name] ?? ''}
            placeholder="?"
            onChange={(e) => setParams({ ...params, [s.name]: e.target.value })}
            onKeyDown={(e) => e.key === 'Enter' && apply()}
          />
          <span className="muted small ellipsis">{t('sum.usedBy', { inputs: s.inputs.join(', ') })}</span>
        </div>
      ))}
      {unnamed.map((io) => (
        <div key={io.name} className="sym-row">
          <span className="mono sym-name" title={io.name}>
            {io.name}
          </span>
          <span className="dims-edit">
            {io.declared!.map((d, i) =>
              d === null ? (
                <input
                  key={i}
                  type="number"
                  min={0}
                  className="num-input small"
                  placeholder="?"
                  value={inputs[io.name]?.[i] ?? ''}
                  onChange={(e) => {
                    const arr = [...(inputs[io.name] ?? [])];
                    arr[i] = e.target.value;
                    setInputs({ ...inputs, [io.name]: arr });
                  }}
                  onKeyDown={(e) => e.key === 'Enter' && apply()}
                />
              ) : (
                <span key={i} className="mono dim-fixed">
                  {String(d)}
                </span>
              ),
            )}
          </span>
          <span className="muted small mono">{shapeLabel(io.shape)}</span>
        </div>
      ))}
      <div className="btn-row">
        <Button kind="primary" small icon="check" onClick={apply}>
          {t('common.apply')}
        </Button>
        <Button small icon="refresh" onClick={() => void applyOverrides({}, {})}>
          {t('common.reset')}
        </Button>
      </div>
    </div>
  );
}
