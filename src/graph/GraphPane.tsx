import { useEffect, useLayoutEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent } from 'react';
import {
  clearHighlight,
  collapseLevel,
  collapseModule,
  copyText,
  expandAll,
  expandModule,
  highlightNodes,
  inModule,
  openFunction,
  select,
  setGraph,
  setTrace,
  setViewOpts,
  showOpFunction,
  stepHighlight,
  useStore,
  useT,
  type FocusRequest,
} from '../store';
import type { GraphView, ModuleNode, ViewNode } from '../types';
import { Dropdown, Icon, IconButton, MenuList, Spinner, useEscape, type MenuItem } from '../components/common';
import { readTheme } from './colors';
import { Minimap } from './minimap';
import { GraphRenderer } from './renderer';

interface Tip {
  x: number;
  y: number;
  title: string;
  lines: string[];
}

function treeDepth(n: ModuleNode): number {
  let d = 0;
  for (const c of n.children) d = Math.max(d, 1 + treeDepth(c));
  return d;
}

const IO_KINDS = new Set(['input', 'output', 'outer', 'missing']);

function isEditable(el: Element | null): boolean {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (el as HTMLElement).isContentEditable;
}

export function GraphPane({ visible }: { visible: boolean }) {
  const t = useT();
  const view = useStore((s) => s.view);
  const viewMode = useStore((s) => s.viewMode);
  const viewLoading = useStore((s) => s.viewLoading);
  const summary = useStore((s) => s.summary);
  const graphId = useStore((s) => s.graphId);
  const selection = useStore((s) => s.selection);
  const highlight = useStore((s) => s.highlight);
  const trace = useStore((s) => s.trace);
  const focus = useStore((s) => s.focus);
  const viewOpts = useStore((s) => s.viewOpts);
  const theme = useStore((s) => s.theme);
  const collapsed = useStore((s) => s.collapsed);
  const hierarchy = useStore((s) => s.hierarchy);
  const callers = useStore((s) => s.callers);

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const miniRef = useRef<HTMLCanvasElement>(null);
  const rRef = useRef<GraphRenderer | null>(null);
  const mmRef = useRef<Minimap | null>(null);
  const prevView = useRef<GraphView | null>(null);
  const pendingFocus = useRef<FocusRequest | null>(null);
  const needsCamera = useRef(false);
  const [zoom, setZoom] = useState(100);
  const [narrow, setNarrow] = useState(false);
  const [tip, setTip] = useState<Tip | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number; items: MenuItem[] } | null>(null);
  useEscape(!!menu, () => setMenu(null));

  const nodeIndex = useMemo(() => {
    const m = new Map<number, number>();
    view?.nodes.forEach((n, i) => {
      if (n.kind === 'op') m.set(n.node, i);
    });
    // Nodes hidden inside collapsed modules resolve to the module box.
    view?.nodes.forEach((n, i) => {
      if (n.kind === 'module' && n.ids) for (const id of n.ids) if (!m.has(id)) m.set(id, i);
    });
    return m;
  }, [view]);

  // Latest values for long-lived listeners.
  const live = useRef({ view, nodeIndex, visible });
  live.current = { view, nodeIndex, visible };

  const tryFocus = () => {
    const r = rRef.current;
    const f = pendingFocus.current;
    const v = live.current.view;
    if (!r || !f || !v || v.graph !== f.graph || r.width === 0 || !live.current.visible) return;
    pendingFocus.current = null;
    if (f.kind === 'fit') {
      r.fit(true);
      return;
    }
    const idx = live.current.nodeIndex;
    let targets: number[] = [];
    if (f.kind === 'node' || f.kind === 'nodes') {
      targets = Array.from(new Set(f.ids.map((id) => idx.get(id)).filter((x): x is number => x !== undefined)));
    } else if (f.kind === 'module') {
      const mi = v.nodes.findIndex((n) => n.kind === 'module' && n.module === f.path);
      if (mi >= 0) targets = [mi];
    } else {
      const id = f.ids[0];
      const io = v.nodes.findIndex((n) => IO_KINDS.has(n.kind) && n.value === id);
      if (io >= 0) targets = [io];
      else {
        const e = v.edges.find((x) => x.value === id);
        if (e) targets = [e.from, e.to];
      }
    }
    if (targets.length === 1 && f.kind === 'node' && r.isVisible(targets[0]) && r.scale >= 0.5) {
      r.requestDraw();
      return;
    }
    if (targets.length) r.focusNodes(targets, true);
  };
  const tryFocusRef = useRef(tryFocus);
  tryFocusRef.current = tryFocus;

  // Renderer setup.
  useLayoutEffect(() => {
    const canvas = canvasRef.current!;
    const wrap = wrapRef.current!;
    const r = new GraphRenderer(canvas, readTheme());
    rRef.current = r;
    const mm = new Minimap(miniRef.current!, r);
    mmRef.current = mm;
    let mmPending = false;
    r.onCamera = () => {
      setZoom(Math.round(r.scale * 100));
      if (!mmPending) {
        mmPending = true;
        requestAnimationFrame(() => {
          mmPending = false;
          mm.draw();
        });
      }
    };
    const ro = new ResizeObserver(() => {
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;
      if (w === 0 || h === 0) return;
      setNarrow(w < 620 || h < 420);
      if (w !== r.width || h !== r.height) {
        r.resize(w, h);
        mm.rebuild();
      }
      if (needsCamera.current && r.view) {
        needsCamera.current = false;
        r.initialCamera();
      }
      tryFocusRef.current();
    });
    ro.observe(wrap);
    return () => {
      ro.disconnect();
      rRef.current = null;
      mmRef.current = null;
    };
  }, []);

  // Theme.
  useEffect(() => {
    const r = rRef.current;
    if (!r) return;
    const apply = () => {
      r.setTheme(readTheme());
      mmRef.current?.rebuild();
    };
    apply();
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, [theme]);

  // New layout: keep the anchor node where it was on screen.
  useEffect(() => {
    const r = rRef.current;
    if (!r) return;
    const old = prevView.current;
    let anchor: { node: number; value: number; kind: string; module: string; sx: number; sy: number } | null = null;
    if (old && view && viewMode === 'keep' && old.graph === view.graph && r.width > 0) {
      let idx = r.selected;
      if (idx < 0) {
        const [cx, cy] = r.toWorld(r.width / 2, r.height / 2);
        let best = Infinity;
        old.nodes.forEach((n, i) => {
          const dx = n.x + n.w / 2 - cx;
          const dy = n.y + n.h / 2 - cy;
          const d = dx * dx + dy * dy;
          if (d < best) {
            best = d;
            idx = i;
          }
        });
      }
      const n = old.nodes[idx];
      if (n) {
        const [sx, sy] = r.toScreen(n.x + n.w / 2, n.y + n.h / 2);
        anchor = { node: n.node, value: n.value, kind: n.kind, module: n.module, sx, sy };
      }
    }
    r.setView(view);
    r.showLabels = viewOpts.edgeLabels;
    prevView.current = view;
    setTip(null);
    setMenu(null);
    if (view) {
      let placed = false;
      if (anchor) {
        const a = anchor;
        let idx = view.nodes.findIndex(
          (n) => n.kind === a.kind && n.node === a.node && n.value === a.value && (a.kind !== 'module' || n.module === a.module),
        );
        if (idx < 0 && a.kind === 'op' && a.module) {
          // The node was collapsed into a module.
          idx = view.nodes.findIndex((n) => n.kind === 'module' && inModule(a.module, n.module));
        }
        if (idx < 0 && a.kind === 'module') {
          // The module was expanded: anchor on its top-most member.
          view.nodes.forEach((n, i) => {
            if ((n.kind === 'op' || n.kind === 'module') && n.module && inModule(n.module, a.module)) {
              if (idx < 0 || n.y < view.nodes[idx].y) idx = i;
            }
          });
        }
        if (idx >= 0) {
          const n = view.nodes[idx];
          r.setCamera(r.scale, n.x + n.w / 2 + (r.width / 2 - a.sx) / r.scale, n.y + n.h / 2 + (r.height / 2 - a.sy) / r.scale, false);
          placed = true;
        }
      }
      if (!placed) {
        if (r.width > 0 && visible) r.initialCamera();
        else needsCamera.current = true;
      }
    }
    mmRef.current?.rebuild();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view]);

  useEffect(() => {
    const r = rRef.current;
    if (!r) return;
    r.showLabels = viewOpts.edgeLabels;
    r.requestDraw();
  }, [viewOpts.edgeLabels]);

  const showMinimap = viewOpts.minimap && !narrow;
  useEffect(() => {
    if (showMinimap) requestAnimationFrame(() => mmRef.current?.rebuild());
  }, [showMinimap]);

  // Selection.
  useEffect(() => {
    const r = rRef.current;
    if (!r) return;
    r.selected = -1;
    r.selectedValue = -1;
    if (view && selection && selection.graph === view.graph) {
      if (selection.kind === 'node') r.selected = nodeIndex.get(selection.id) ?? -1;
      else if (selection.kind === 'module') {
        r.selected = view.nodes.findIndex((n) => n.kind === 'module' && n.module === selection.path);
      } else {
        r.selectedValue = selection.id;
        r.selected = view.nodes.findIndex((n) => IO_KINDS.has(n.kind) && n.value === selection.id);
      }
    }
    r.requestDraw();
  }, [view, selection, nodeIndex]);

  // Highlight.
  useEffect(() => {
    const r = rRef.current;
    if (!r) return;
    const hs = new Set<number>();
    if (view && highlight && highlight.graph === view.graph) {
      for (const id of highlight.nodes) {
        const i = nodeIndex.get(id);
        if (i !== undefined) hs.add(i);
      }
    }
    r.highlight = hs;
    r.requestDraw();
  }, [view, highlight, nodeIndex]);

  // Trace (after selection so r.selected is current).
  useEffect(() => {
    const r = rRef.current;
    if (!r) return;
    if (trace === 'none' || r.selected < 0) {
      r.traceNodes = null;
      r.traceEdges = null;
    } else {
      const res = r.trace(r.selected, trace !== 'down', trace !== 'up');
      r.traceNodes = res.nodes;
      r.traceEdges = res.edges;
    }
    r.requestDraw();
  }, [view, trace, selection]);

  // Focus requests.
  useEffect(() => {
    if (!focus) return;
    pendingFocus.current = focus;
    tryFocus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus, view, visible]);

  useEffect(() => {
    if (visible) {
      const r = rRef.current;
      if (r && needsCamera.current && r.width > 0 && r.view) {
        needsCamera.current = false;
        r.initialCamera();
      }
    }
  }, [visible]);

  // Wheel: zoom (mouse) / pan (touchpad), pinch or Ctrl+wheel always zooms.
  useEffect(() => {
    const canvas = canvasRef.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = rRef.current;
      if (!r) return;
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1;
      const dx = e.deltaX * unit;
      const dy = e.deltaY * unit;
      if (e.ctrlKey || e.metaKey) {
        r.zoomAt(Math.exp(-dy * 0.01), sx, sy);
        return;
      }
      const touchpad = Math.abs(dx) > 0 || !Number.isInteger(dy);
      if (useStore.getState().viewOpts.wheelZoom && !touchpad && !e.shiftKey) {
        r.zoomAt(Math.exp(-dy * 0.0016), sx, sy);
      } else if (e.shiftKey && dx === 0) {
        r.panBy(-dy, 0);
      } else {
        r.panBy(-dx, -dy);
      }
      setTip(null);
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, []);

  const selectViewNode = (i: number) => {
    const v = live.current.view;
    const n = v?.nodes[i];
    if (!v || !n) return;
    if (n.kind === 'op') select({ kind: 'node', graph: v.graph, id: n.node });
    else if (n.kind === 'module') select({ kind: 'module', graph: v.graph, id: -1, path: n.module });
    else select({ kind: 'value', graph: v.graph, id: n.value });
  };

  const subgraphsOf = (n: ViewNode): number[] => {
    if (!summary || !view) return [];
    return summary.graphs.filter((g) => g.parent === view.graph && g.parentNode === n.node).map((g) => g.id);
  };

  // Keyboard navigation.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!live.current.visible || isEditable(document.activeElement)) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const r = rRef.current;
      const v = live.current.view;
      if (!r || !v) return;
      const k = e.key;
      if (k === 'f' || k === 'F' || k === '0') {
        r.fit(true);
      } else if (k === '+' || k === '=') {
        r.zoomCenter(1.25);
      } else if (k === '-' || k === '_') {
        r.zoomCenter(0.8);
      } else if (k === '1') {
        const [cx, cy] = r.toWorld(r.width / 2, r.height / 2);
        r.setCamera(1, cx, cy, true);
      } else if (k === 'Enter' && r.selected >= 0) {
        r.focusNode(r.selected);
      } else if (k.startsWith('Arrow')) {
        if (r.selected >= 0 && (k === 'ArrowUp' || k === 'ArrowDown')) {
          const up = k === 'ArrowUp';
          const cand = r
            .edgesOf(r.selected)
            .filter((ed) => (up ? ed.to === r.selected : ed.from === r.selected))
            .map((ed) => (up ? ed.from : ed.to))
            .filter((x, i, a) => a.indexOf(x) === i)
            .sort((a, b) => v.nodes[a].x - v.nodes[b].x);
          if (cand.length) {
            selectViewNode(cand[0]);
            r.focusNode(cand[0]);
          }
        } else if (r.selected >= 0 && (k === 'ArrowLeft' || k === 'ArrowRight')) {
          // Move to the neighbour in the same layer.
          const cur = v.nodes[r.selected];
          let best = -1;
          let bestD = Infinity;
          v.nodes.forEach((n, i) => {
            if (i === r.selected || Math.abs(n.y + n.h / 2 - (cur.y + cur.h / 2)) > 4) return;
            const d = n.x - cur.x;
            if ((k === 'ArrowRight' ? d > 0 : d < 0) && Math.abs(d) < bestD) {
              bestD = Math.abs(d);
              best = i;
            }
          });
          if (best >= 0) {
            selectViewNode(best);
            r.focusNode(best);
          }
        } else {
          const step = 80;
          r.panBy(k === 'ArrowLeft' ? step : k === 'ArrowRight' ? -step : 0, k === 'ArrowUp' ? step : k === 'ArrowDown' ? -step : 0);
        }
      } else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Pointer handling.
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const hoverRaf = useRef(0);
  const hoverPos = useRef({ sx: 0, sy: 0, cx: 0, cy: 0 });

  const local = (e: { clientX: number; clientY: number }) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { sx: e.clientX - rect.left, sy: e.clientY - rect.top };
  };

  const updateHover = () => {
    const r = rRef.current;
    const v = live.current.view;
    const canvas = canvasRef.current;
    if (!r || !v || !canvas) return;
    const { sx, sy } = hoverPos.current;
    const i = r.nodeAt(sx, sy);
    const ei = i < 0 ? r.edgeAt(sx, sy) : -1;
    if (i === r.hover && ei === r.hoverEdge) {
      setTip((old) => (old ? { ...old, x: sx, y: sy } : old));
      return;
    }
    r.hover = i;
    r.hoverEdge = ei;
    r.requestDraw();
    canvas.style.cursor = i >= 0 || ei >= 0 ? 'pointer' : 'grab';
    if (i >= 0) {
      const n = v.nodes[i];
      const lines: string[] = [];
      let title = n.title;
      if (n.kind === 'op') {
        if (n.name && !viewOpts.showNames) lines.push(n.name);
        if (n.sub) lines.push(t('graph.dblclickSub'));
        else if (n.func !== undefined) lines.push(t('graph.dblclickFunction'));
      } else if (n.kind === 'module') {
        title = n.module;
        lines.push(`${t('insp.members')}: ${n.members ?? 0}`);
        lines.push(t('graph.dblclickModule'));
      } else {
        lines.push(n.lines[0]?.v ?? '');
        lines.push(t(`cat.${n.kind}` as 'cat.input'));
      }
      setTip({ x: sx, y: sy, title, lines });
    } else if (ei >= 0) {
      const e = v.edges[ei];
      const lines = [e.label || '', `${v.nodes[e.from]?.title ?? '?'} → ${v.nodes[e.to]?.title ?? '?'}`];
      if (e.detached) lines.push(t('graph.longEdge', { name: v.nodes[e.from]?.title ?? '' }));
      setTip({ x: sx, y: sy, title: e.name, lines: lines.filter(Boolean) });
    } else {
      setTip(null);
    }
  };

  const onPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0 && e.button !== 1) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, moved: false };
    setMenu(null);
  };

  const onPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const r = rRef.current;
    if (!r) return;
    const d = drag.current;
    if (d) {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      if (!d.moved && Math.abs(dx) + Math.abs(dy) > 3) {
        d.moved = true;
        e.currentTarget.style.cursor = 'grabbing';
        setTip(null);
      }
      if (d.moved) {
        r.panBy(dx, dy);
        d.x = e.clientX;
        d.y = e.clientY;
      }
      return;
    }
    const p = local(e);
    hoverPos.current = { sx: p.sx, sy: p.sy, cx: e.clientX, cy: e.clientY };
    if (!hoverRaf.current) {
      hoverRaf.current = requestAnimationFrame(() => {
        hoverRaf.current = 0;
        updateHover();
      });
    }
  };

  const onPointerUp = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    drag.current = null;
    e.currentTarget.style.cursor = 'grab';
    const r = rRef.current;
    const v = live.current.view;
    if (!d || d.moved || !r || !v || e.button !== 0) return;
    const { sx, sy } = local(e);
    const i = r.nodeAt(sx, sy);
    if (i >= 0) {
      selectViewNode(i);
      return;
    }
    const ei = r.edgeAt(sx, sy);
    if (ei >= 0) select({ kind: 'value', graph: v.graph, id: v.edges[ei].value });
    else select(null);
  };

  const onDoubleClick = (e: ReactMouseEvent<HTMLCanvasElement>) => {
    const r = rRef.current;
    const v = live.current.view;
    if (!r || !v) return;
    const { sx, sy } = local(e);
    const i = r.nodeAt(sx, sy);
    if (i < 0) return;
    const n = v.nodes[i];
    if (n.kind === 'module') {
      expandModule(n.module);
      return;
    }
    const subs = n.kind === 'op' && n.sub ? subgraphsOf(n) : [];
    if (subs.length) setGraph(subs[0]);
    else if (n.kind === 'op' && n.func !== undefined) openFunction(v.graph, n.func);
    else r.focusNode(i);
  };

  const onContextMenu = (e: ReactMouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const r = rRef.current;
    const v = live.current.view;
    if (!r || !v) return;
    const { sx, sy } = local(e);
    const i = r.nodeAt(sx, sy);
    const items: MenuItem[] = [];
    if (i >= 0) {
      selectViewNode(i);
      const n = v.nodes[i];
      items.push({ label: t('graph.focus'), icon: 'target', onClick: () => r.focusNode(i) });
      if (n.kind === 'op') {
        items.push({
          label: t('graph.viewFunction'),
          icon: 'sigma',
          onClick: () => showOpFunction({ graph: v.graph, node: n.node }),
        });
        if (n.func !== undefined) {
          const fn = n.func;
          items.push({ label: t('graph.openFunction'), icon: 'layers', onClick: () => openFunction(v.graph, fn) });
        }
        items.push(
          { separator: true },
          { label: t('graph.traceUp'), icon: 'arrowUp', onClick: () => setTrace('up') },
          { label: t('graph.traceDown'), icon: 'arrowDown', onClick: () => setTrace('down') },
          { label: t('graph.traceBoth'), icon: 'trace', onClick: () => setTrace('both') },
        );
        if (trace !== 'none') items.push({ label: t('graph.clearTrace'), icon: 'x', onClick: () => setTrace('none') });
        items.push({ separator: true });
        items.push({
          label: t('graph.sameOp', { op: n.title }),
          icon: 'layers',
          onClick: () =>
            highlightNodes(
              n.title,
              v.graph,
              v.nodes.filter((m) => m.kind === 'op' && m.title === n.title).map((m) => m.node),
              false,
            ),
        });
        const subs = n.sub ? subgraphsOf(n) : [];
        for (const sid of subs) {
          const g = summary?.graphs[sid];
          items.push({ label: `${t('graph.openSub')}: ${g?.attribute ?? sid}`, icon: 'layers', onClick: () => setGraph(sid) });
        }
        if (n.module) {
          const parts = n.module.split('/');
          items.push({ separator: true });
          for (let k = 1; k <= Math.min(parts.length, 3); k++) {
            const path = parts.slice(0, k).join('/');
            items.push({ label: t('graph.collapse', { path }), icon: 'layers', onClick: () => collapseModule(path) });
          }
        }
        items.push(
          { separator: true },
          { label: t('graph.copyName'), icon: 'copy', onClick: () => void copyText(n.name || n.title) },
          { label: t('graph.copyOp'), icon: 'copy', onClick: () => void copyText(n.title) },
          {
            label: t('graph.showInTable'),
            icon: 'table',
            onClick: () => useStore.setState({ nodeFilter: n.name || n.title, tab: 'nodes' }),
          },
        );
      } else if (n.kind === 'module') {
        const parts = n.module.split('/');
        items.push({ label: t('graph.expand'), icon: 'layers', onClick: () => expandModule(n.module) });
        if (parts.length > 1) {
          const parent = parts.slice(0, -1).join('/');
          items.push({ label: t('graph.collapse', { path: parent }), icon: 'layers', onClick: () => collapseModule(parent) });
        }
        items.push(
          { separator: true },
          { label: t('graph.traceUp'), icon: 'arrowUp', onClick: () => setTrace('up') },
          { label: t('graph.traceDown'), icon: 'arrowDown', onClick: () => setTrace('down') },
          { separator: true },
          { label: t('graph.copyPath'), icon: 'copy', onClick: () => void copyText(n.module) },
          {
            label: t('graph.showInTable'),
            icon: 'table',
            onClick: () => useStore.setState({ nodeFilter: n.module, tab: 'nodes' }),
          },
        );
      } else {
        items.push({ label: t('graph.copyName'), icon: 'copy', onClick: () => void copyText(n.title) });
      }
    } else {
      items.push({ label: t('graph.fit'), icon: 'fit', onClick: () => r.fit(true) });
      if (trace !== 'none') items.push({ label: t('graph.clearTrace'), icon: 'x', onClick: () => setTrace('none') });
      if (highlight) items.push({ label: t('graph.clear'), icon: 'x', onClick: () => clearHighlight() });
      if (collapsed.length) items.push({ label: t('graph.expandAll'), icon: 'layers', onClick: () => expandAll() });
    }
    const rect = wrapRef.current!.getBoundingClientRect();
    const x = Math.min(e.clientX - rect.left, rect.width - 240);
    const y = Math.min(e.clientY - rect.top, rect.height - 30 * items.length - 20);
    setTip(null);
    setMenu({ x: Math.max(4, x), y: Math.max(4, y), items });
  };

  // Minimap navigation.
  const mmDrag = useRef(false);
  const mmGo = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const mm = mmRef.current;
    const r = rRef.current;
    if (!mm || !r) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const [wx, wy] = mm.toWorld(e.clientX - rect.left, e.clientY - rect.top);
    r.setCamera(r.scale, wx, wy, false);
  };

  // Breadcrumb from the main graph to the current one.
  const crumbs = useMemo(() => {
    if (!summary) return [];
    const out: { id: number; label: string }[] = [];
    let g = summary.graphs[graphId];
    let guard = 0;
    while (g && guard++ < 64) {
      out.unshift({ id: g.id, label: g.kind === 'main' ? summary.fileName : g.label });
      // Local functions have no parent; continue from the graph that opened them.
      const up = g.parent ?? (g.kind === 'function' ? callers[g.id] : undefined);
      if (up === undefined) break;
      g = summary.graphs[up];
    }
    return out;
  }, [summary, graphId, callers]);

  const moduleLevels = hierarchy && view && hierarchy.graph === view.graph ? Math.min(3, treeDepth(hierarchy.root)) : 0;
  const moduleItems = (): MenuItem[] => {
    if (moduleLevels === 0) return [];
    const items: MenuItem[] = [{ separator: true }];
    for (let n = 1; n <= moduleLevels; n++) {
      items.push({ label: t('graph.collapseLevel', { n }), icon: 'layers', onClick: () => collapseLevel(n) });
    }
    items.push({ label: t('graph.expandAll'), icon: 'layers', disabled: collapsed.length === 0, onClick: () => expandAll() });
    return items;
  };

  const viewItems = (): MenuItem[] => [
    { label: t('opt.showInitializers'), checked: viewOpts.showInitializers, onClick: () => setViewOpts({ showInitializers: !viewOpts.showInitializers }) },
    { label: t('opt.showAttributes'), checked: viewOpts.showAttributes, onClick: () => setViewOpts({ showAttributes: !viewOpts.showAttributes }) },
    { label: t('opt.showNames'), checked: viewOpts.showNames, onClick: () => setViewOpts({ showNames: !viewOpts.showNames }) },
    { label: t('opt.showConstants'), checked: viewOpts.showConstants, onClick: () => setViewOpts({ showConstants: !viewOpts.showConstants }) },
    { label: t('opt.edgeLabels'), checked: viewOpts.edgeLabels, onClick: () => setViewOpts({ edgeLabels: !viewOpts.edgeLabels }) },
    { label: t('opt.detachLongEdges'), checked: viewOpts.detachLongEdges, onClick: () => setViewOpts({ detachLongEdges: !viewOpts.detachLongEdges }) },
    { label: t('opt.foldShapes'), checked: viewOpts.foldShapes, onClick: () => setViewOpts({ foldShapes: !viewOpts.foldShapes }) },
    ...moduleItems(),
    { separator: true },
    { label: t('opt.minimap'), checked: viewOpts.minimap, onClick: () => setViewOpts({ minimap: !viewOpts.minimap }) },
    { label: t('opt.wheelZoom'), checked: viewOpts.wheelZoom, onClick: () => setViewOpts({ wheelZoom: !viewOpts.wheelZoom }) },
  ];

  const hlCount = highlight && view && highlight.graph === view.graph ? highlight.nodes.filter((id) => nodeIndex.has(id)).length : 0;
  const traceCount = (() => {
    const r = rRef.current;
    return trace !== 'none' && r?.traceNodes ? r.traceNodes.size : 0;
  })();

  return (
    <div className="graph-pane" ref={wrapRef} style={{ display: visible ? undefined : 'none' }}>
      <canvas
        ref={canvasRef}
        className="graph-canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => {
          const r = rRef.current;
          if (r && (r.hover >= 0 || r.hoverEdge >= 0)) {
            r.hover = -1;
            r.hoverEdge = -1;
            r.requestDraw();
          }
          setTip(null);
        }}
        onDoubleClick={onDoubleClick}
        onContextMenu={onContextMenu}
      />
      <div className="graph-top">
        <div className="crumbs">
          {crumbs.length > 1 && (
            <IconButton
              icon="arrowLeft"
              title={t('graph.back')}
              onClick={() => setGraph(crumbs[crumbs.length - 2].id)}
            />
          )}
          {crumbs.map((c, i) => (
            <span key={c.id} className="crumb-wrap">
              {i > 0 && <Icon name="chevronRight" size={12} className="muted" />}
              <button
                type="button"
                className={'crumb' + (i === crumbs.length - 1 ? ' current' : '')}
                onClick={() => setGraph(c.id)}
                title={c.label}
              >
                {c.label}
              </button>
            </span>
          ))}
        </div>
        <div className="spacer" />
        {hlCount > 0 && highlight && (
          <div className="pillbar">
            <span>{t('graph.highlight', { n: hlCount, label: highlight.label })}</span>
            <IconButton icon="arrowUp" size={13} title={t('graph.prev')} onClick={() => stepHighlight(-1)} />
            <IconButton icon="arrowDown" size={13} title={t('graph.next')} onClick={() => stepHighlight(1)} />
            <IconButton icon="x" size={13} title={t('graph.clear')} onClick={() => clearHighlight()} />
          </div>
        )}
        {collapsed.length > 0 && (
          <div className="pillbar">
            <Icon name="layers" size={13} />
            <span>{t('graph.modulesCollapsed', { n: collapsed.length })}</span>
            <IconButton icon="x" size={13} title={t('graph.expandAll')} onClick={() => expandAll()} />
          </div>
        )}
        {trace !== 'none' && (
          <div className="pillbar">
            <Icon name="trace" size={13} />
            <span>{t('graph.tracing', { n: traceCount })}</span>
            <IconButton icon="x" size={13} title={t('graph.clearTrace')} onClick={() => setTrace('none')} />
          </div>
        )}
        <Dropdown icon="eye" label={t('graph.options')} title={t('graph.options')} items={viewItems} align="right" />
      </div>
      <div className="graph-zoom">
        <IconButton icon="minus" title={t('graph.zoomOut')} onClick={() => rRef.current?.zoomCenter(0.8)} />
        <button
          type="button"
          className="zoom-label"
          title={t('graph.actual')}
          onClick={() => {
            const r = rRef.current;
            if (!r) return;
            const [cx, cy] = r.toWorld(r.width / 2, r.height / 2);
            r.setCamera(1, cx, cy, true);
          }}
        >
          {zoom}%
        </button>
        <IconButton icon="plus" title={t('graph.zoomIn')} onClick={() => rRef.current?.zoomCenter(1.25)} />
        <IconButton icon="fit" title={t('graph.fit')} onClick={() => rRef.current?.fit(true)} />
      </div>
      <canvas
        ref={miniRef}
        className="minimap"
        style={{ display: showMinimap ? undefined : 'none' }}
        onPointerDown={(e) => {
          mmDrag.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          mmGo(e);
        }}
        onPointerMove={(e) => mmDrag.current && mmGo(e)}
        onPointerUp={() => {
          mmDrag.current = false;
        }}
      />
      {view && view.nodes.length === 0 && <div className="graph-empty">{t('graph.empty')}</div>}
      {view && (view.hiddenConstants > 0 || view.foldedNodes > 0 || view.collapsedNodes > 0) && (
        <div className="graph-note">
          {[
            view.hiddenConstants > 0 ? t('graph.hiddenConstants', { n: view.hiddenConstants }) : '',
            view.foldedNodes > 0 ? t('graph.foldedNodes', { n: view.foldedNodes }) : '',
            view.collapsedNodes > 0 ? t('graph.collapsedNodes', { n: view.collapsedNodes }) : '',
          ]
            .filter(Boolean)
            .join(' · ')}
        </div>
      )}
      {viewLoading && (
        <div className="graph-busy">
          <Spinner /> {t('loading.layout')}
        </div>
      )}
      {tip && !menu && (
        <div
          className="graph-tip"
          style={{
            left: Math.min(tip.x + 16, (wrapRef.current?.clientWidth ?? 800) - 300),
            top: tip.y + 18,
          }}
        >
          <div className="graph-tip-title">{tip.title}</div>
          {tip.lines.map((l, i) => (
            <div key={i} className="graph-tip-line">
              {l}
            </div>
          ))}
        </div>
      )}
      {menu && (
        <div
          className="ctx-backdrop"
          onMouseDown={() => setMenu(null)}
          onContextMenu={(e) => {
            e.preventDefault();
            setMenu(null);
          }}
        />
      )}
      {menu && (
        <div className="ctx-menu" style={{ left: menu.x, top: menu.y }}>
          <MenuList items={menu.items} onClose={() => setMenu(null)} />
        </div>
      )}
    </div>
  );
}
