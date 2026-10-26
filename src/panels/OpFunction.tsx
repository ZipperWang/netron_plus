import { useEffect, useState } from 'react';
import {
  callModel,
  closeOpFunction,
  errorText,
  openExternal,
  openFunction,
  useStore,
  useT,
  type OpFunctionTarget,
} from '../store';
import type { FnSymbol, OpFunction, TexBlock } from '../types';
import { Badge, Button, CopyButton, Icon, IconButton, Spinner, useEscape } from '../components/common';
import type { MsgKey } from '../i18n';
import { Tex } from '../math/Tex';
import { opTitle } from './Sidebar';

const LABELS: Record<string, MsgKey> = { shape: 'fn.shape', where: 'fn.where' };

const SOURCE_TONE = {
  set: 'accent',
  default: 'neutral',
  derived: 'info',
  unknown: 'warning',
} as const;

function useOpFunction(target: OpFunctionTarget | null): { data: OpFunction | null; error: string | null } {
  const summary = useStore((s) => s.summary);
  const [data, setData] = useState<OpFunction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const key = !target ? '' : 'node' in target ? `n${target.graph}:${target.node}` : `o${target.domain}:${target.op}`;
  useEffect(() => {
    setData(null);
    setError(null);
    if (!target) return;
    let alive = true;
    const params = 'node' in target ? { graph: target.graph, node: target.node } : { op: target.op, domain: target.domain };
    callModel<OpFunction>('opFunction', params)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(errorText(e)));
    return () => {
      alive = false;
    };
    // Shapes (and so the bound values) change with dimension overrides.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, summary]);
  return { data, error };
}

function Formula({ block }: { block: TexBlock }) {
  const t = useT();
  const key = LABELS[block.label];
  const label = key ? t(key) : block.label;
  return (
    <div className="fn-formula">
      {label && <div className="fn-label">{label}</div>}
      <div className="fn-tex">
        <Tex tex={block.tex} display />
      </div>
      <CopyButton text={block.tex} title={t('fn.copyTex')} />
    </div>
  );
}

function SymbolRow({ s }: { s: FnSymbol }) {
  const t = useT();
  return (
    <div className="fn-sym">
      <div className="fn-sym-tex">
        {s.value !== null ? (
          <Tex tex={`${s.tex} = ${s.value}`} />
        ) : (
          <>
            <Tex tex={s.tex} />
            <span className="muted"> = ?</span>
          </>
        )}
      </div>
      <div className="fn-sym-meta">
        <Badge tone={SOURCE_TONE[s.source]}>{t(`fn.source.${s.source}` as MsgKey)}</Badge>
        <span className="fn-binding" title={`${t(`fn.kind.${s.kind}` as MsgKey)} ${s.binding}`}>
          {s.binding}
        </span>
      </div>
    </div>
  );
}

/** A loaded operator function: summary, formulas and bound symbols. */
function OpFunctionBody({ f, from }: { f: OpFunction; from: number }) {
  const t = useT();
  const lang = useStore((s) => s.lang);
  const pick = (v: { en: string; zh: string }) => (lang === 'zh' ? v.zh || v.en : v.en || v.zh);
  const summary = pick(f.summary);
  const note = pick(f.note);
  const fnRef = f.function;
  return (
    <div className="fn">
      {summary && <div className="fn-summary">{summary}</div>}
      {f.formulas.map((b, i) => (
        <Formula key={i} block={b} />
      ))}
      {note && (
        <div className="fn-note">
          <Icon name="info" size={13} />
          <span>{note}</span>
        </div>
      )}
      {f.symbols.length > 0 && (
        <div className="fn-symbols">
          <div className="fn-head">{f.node !== null ? t('fn.bound') : t('fn.defaults')}</div>
          {f.symbols.map((s, i) => (
            <SymbolRow key={i} s={s} />
          ))}
        </div>
      )}
      {fnRef && (
        <button type="button" className="list-row fn-body-link" onClick={() => openFunction(from, fnRef.id)}>
          <Icon name="layers" size={14} />
          <span className="ellipsis">
            {t('fn.body')}: {fnRef.label}
          </span>
          <span className="muted">{fnRef.nodes}</span>
        </button>
      )}
      {f.since === null && !fnRef && <div className="muted fn-none">{t('fn.none')}</div>}
      <div className="fn-foot">
        {f.since !== null && (
          <span className="muted small">
            {t('fn.since', { v: f.since })}
            {f.opset !== null ? ` · ${t('fn.opset', { v: f.opset })}` : ''}
          </span>
        )}
        <div className="spacer" />
        {f.docUrl && (
          <Button small icon="external" onClick={() => void openExternal(f.docUrl!)}>
            {t('insp.docs')}
          </Button>
        )}
      </div>
    </div>
  );
}

/** The operator function of a node or an operator type, loaded from the core. */
export function OpFunctionView({ target }: { target: OpFunctionTarget }) {
  const graphId = useStore((s) => s.graphId);
  const { data, error } = useOpFunction(target);
  if (error) return <div className="error-text">{error}</div>;
  if (!data) {
    return (
      <div className="loading-row">
        <Spinner />
      </div>
    );
  }
  return <OpFunctionBody f={data} from={'node' in target ? target.graph : graphId} />;
}

/** Larger view of an operator function, opened from the graph, inspector or operator table. */
export function OpFunctionDialog() {
  const t = useT();
  const target = useStore((s) => s.opFunction);
  const graphId = useStore((s) => s.graphId);
  const { data, error } = useOpFunction(target);
  useEscape(!!target, closeOpFunction);
  if (!target) return null;
  return (
    <div className="modal-backdrop" onClick={closeOpFunction}>
      <div className="modal modal-wide" onClick={(e) => e.stopPropagation()} role="dialog" aria-label={t('fn.title')}>
        <div className="modal-head">
          <span className="fn-title">
            <Icon name="sigma" size={16} />
            <span>{data ? opTitle(data.op, data.domain) : t('fn.title')}</span>
            {data && data.node !== null && <Badge>{t('fn.nodeBadge')}</Badge>}
          </span>
          <IconButton icon="x" title={t('common.close')} onClick={closeOpFunction} />
        </div>
        <div className="modal-body">
          {error ? (
            <div className="error-text">{error}</div>
          ) : !data ? (
            <div className="loading-row">
              <Spinner />
            </div>
          ) : (
            <OpFunctionBody f={data} from={'node' in target ? target.graph : graphId} />
          )}
        </div>
      </div>
    </div>
  );
}
