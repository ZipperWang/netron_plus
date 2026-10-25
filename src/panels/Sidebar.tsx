import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import {
  callModel,
  highlightNodes,
  setGraph,
  showNode,
  showValue,
  useStore,
  useT,
} from '../store';
import type { ModuleNode, OpRow, SearchHit } from '../types';
import { Dot, Icon, Section, SearchInput } from '../components/common';
import { CATEGORY_ORDER, categoryColor } from '../graph/colors';
import { fmtCount, shapeLabel } from '../util/format';
import type { MsgKey } from '../i18n';

let searchEl: HTMLInputElement | null = null;
export function focusSearch() {
  searchEl?.focus();
  searchEl?.select();
}

export function opTitle(op: string, domain: string): string {
  return domain && domain !== 'ai.onnx' && domain !== 'com.microsoft' ? `${domain}::${op}` : op;
}

/** IR node ids in the current view whose title matches the op row. */
export function viewNodesOfOp(row: Pick<OpRow, 'op' | 'domain'>): number[] {
  const v = useStore.getState().view;
  if (!v) return [];
  const title = opTitle(row.op, row.domain);
  return v.nodes.filter((n) => n.kind === 'op' && n.title === title).map((n) => n.node);
}

export function Sidebar() {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const view = useStore((s) => s.view);
  const graphId = useStore((s) => s.graphId);
  const hierarchy = useStore((s) => s.hierarchy);
  const width = useStore((s) => s.sidebarWidth);
  const [query, setQuery] = useState('');
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    searchEl = inputRef.current;
    return () => {
      searchEl = null;
    };
  }, []);

  useEffect(() => {
    const q = query.trim();
    if (!q || !summary) {
      setHits(null);
      return;
    }
    let alive = true;
    const timer = setTimeout(() => {
      callModel<SearchHit[]>('search', { query: q, limit: 300 })
        .then((h) => alive && setHits(h))
        .catch(() => alive && setHits([]));
    }, 180);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [query, summary]);

  const opsInView = useMemo(() => {
    const counts = new Map<string, number>();
    view?.nodes.forEach((n) => {
      if (n.kind === 'op') counts.set(n.title, (counts.get(n.title) ?? 0) + 1);
    });
    return counts;
  }, [view]);

  const cats = useMemo(() => {
    const m = new Map<string, number>();
    view?.nodes.forEach((n) => {
      if (n.kind === 'op') m.set(n.cat, (m.get(n.cat) ?? 0) + 1);
    });
    return CATEGORY_ORDER.filter((c) => m.has(c)).map((c) => ({ cat: c, count: m.get(c)! }));
  }, [view]);

  if (!summary) return null;
  const ops = summary.ops;

  const onSearchKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && hits && hits.length > 0) {
      const nodeHits = hits.filter((h) => h.kind === 'node' && h.graph === graphId);
      if (nodeHits.length > 1) {
        highlightNodes(`"${query.trim()}"`, graphId, nodeHits.map((h) => h.id), true);
      } else {
        const h = hits[0];
        if (h.kind === 'node') showNode(h.graph, h.id);
        else showValue(h.graph, h.id);
      }
    } else if (e.key === 'Escape') {
      setQuery('');
    }
  };

  return (
    <aside className="sidebar" style={{ width }}>
      <div className="sidebar-search">
        <SearchInput
          value={query}
          onChange={setQuery}
          placeholder={t('side.search')}
          inputRef={inputRef}
          onKeyDown={onSearchKey}
        />
      </div>
      <div className="panel-scroll">
        {hits !== null && (
          <Section title={t('side.results')} count={hits.length}>
            {hits.length === 0 && <div className="muted pad">{t('side.noResults')}</div>}
            {hits.length > 0 && <div className="muted small pad">{t('side.searchHint')}</div>}
            {hits.map((h) => (
              <button
                key={`${h.kind}:${h.graph}:${h.id}`}
                type="button"
                className="list-row"
                onClick={() => (h.kind === 'node' ? showNode(h.graph, h.id) : showValue(h.graph, h.id))}
                title={h.name}
              >
                <Icon name={h.kind === 'node' ? 'graph' : 'cube'} size={13} />
                <span className="list-main">
                  <span className="ellipsis">{h.kind === 'node' ? h.op : h.name}</span>
                  {h.kind === 'node' && h.name && <span className="list-sub ellipsis">{h.name}</span>}
                </span>
                {h.graph !== 0 && <span className="muted small">#{h.graph}</span>}
              </button>
            ))}
          </Section>
        )}
        <Section title={t('side.inputs')} count={summary.inputs.length}>
          {summary.inputs.map((io) => (
            <button key={io.value} type="button" className="list-row" onClick={() => showValue(0, io.value)} title={io.display}>
              <Dot color={categoryColor('input')} />
              <span className="list-main">
                <span className="ellipsis mono">{io.name}</span>
                <span className="list-sub mono">
                  {io.dtype} {shapeLabel(io.shape)}
                </span>
              </span>
            </button>
          ))}
        </Section>
        <Section title={t('side.outputs')} count={summary.outputs.length}>
          {summary.outputs.map((io) => (
            <button key={io.value} type="button" className="list-row" onClick={() => showValue(0, io.value)} title={io.display}>
              <Dot color={categoryColor('output')} />
              <span className="list-main">
                <span className="ellipsis mono">{io.name}</span>
                <span className="list-sub mono">
                  {io.dtype} {shapeLabel(io.shape)}
                </span>
              </span>
            </button>
          ))}
        </Section>
        <Section title={t('side.operators')} count={ops.length}>
          {ops.map((o) => {
            const title = opTitle(o.op, o.domain);
            const here = opsInView.get(title) ?? 0;
            return (
              <button
                key={`${o.domain}:${o.op}`}
                type="button"
                className={'list-row' + (here === 0 ? ' dim' : '')}
                onClick={() => highlightNodes(title, graphId, viewNodesOfOp(o), true)}
                title={o.domain ? `${o.domain}::${o.op}` : o.op}
              >
                <Dot color={categoryColor(o.category)} />
                <span className="list-main ellipsis">{title}</span>
                <span className="count">{here !== o.count && here > 0 ? `${here}/${o.count}` : o.count}</span>
              </button>
            );
          })}
        </Section>
        {hierarchy && hierarchy.graph === graphId && hierarchy.root.children.length > 0 && (
          <Section title={t('side.modules')} count={hierarchy.root.children.length}>
            <div className="tree">
              {hierarchy.root.children.map((c) => (
                <ModuleRow key={c.path} node={c} depth={0} defaultOpen={hierarchy.root.children.length <= 3} />
              ))}
            </div>
          </Section>
        )}
        {summary.graphs.length > 1 && (
          <Section title={t('side.graphs')} count={summary.graphs.length}>
            {summary.graphs.map((g) => (
              <button
                key={g.id}
                type="button"
                className={'list-row' + (g.id === graphId ? ' sel' : '')}
                style={{ paddingLeft: 10 + Math.min(g.depth, 6) * 12 }}
                onClick={() => setGraph(g.id)}
                title={g.label}
              >
                <Icon name={g.kind === 'function' ? 'cube' : 'layers'} size={13} />
                <span className="list-main ellipsis">{g.kind === 'main' ? summary.fileName : g.label}</span>
                <span className="count">{g.nodes}</span>
              </button>
            ))}
          </Section>
        )}
        {cats.length > 0 && (
          <Section title={t('side.legend')} defaultOpen={false}>
            {cats.map((c) => (
              <button
                key={c.cat}
                type="button"
                className="list-row"
                onClick={() =>
                  highlightNodes(
                    t(`cat.${c.cat}` as MsgKey),
                    graphId,
                    (view?.nodes ?? []).filter((n) => n.kind === 'op' && n.cat === c.cat).map((n) => n.node),
                    true,
                  )
                }
              >
                <span className="swatch" style={{ background: categoryColor(c.cat) }} />
                <span className="list-main">{t(`cat.${c.cat}` as MsgKey)}</span>
                <span className="count">{c.count}</span>
              </button>
            ))}
          </Section>
        )}
      </div>
    </aside>
  );
}

function ModuleRow({ node, depth, defaultOpen }: { node: ModuleNode; depth: number; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(!!defaultOpen);
  const graphId = useStore((s) => s.graphId);
  const hasChildren = node.children.length > 0;
  const select = () => {
    const v = useStore.getState().view;
    if (!v) return;
    const prefix = `${node.path}/`;
    const ids = v.nodes
      .filter((n) => n.kind === 'op' && (n.module === node.path || n.module.startsWith(prefix)))
      .map((n) => n.node);
    highlightNodes(node.path, graphId, ids, true);
  };
  return (
    <>
      <div className="tree-row" style={{ paddingLeft: 6 + depth * 12 }}>
        <button type="button" className="tree-toggle" onClick={() => setOpen(!open)} disabled={!hasChildren}>
          {hasChildren ? <Icon name={open ? 'chevronDown' : 'chevronRight'} size={12} /> : <span className="tree-leaf" />}
        </button>
        <button type="button" className="tree-label" onClick={select} title={`${node.path}\n${node.nodes} nodes · ${fmtCount(node.params)} params · ${fmtCount(node.flops)} FLOPs`}>
          <span className="ellipsis">{node.name}</span>
          <span className="count">{node.nodes}</span>
        </button>
      </div>
      {open && node.children.map((c) => <ModuleRow key={c.path} node={c} depth={depth + 1} />)}
    </>
  );
}
