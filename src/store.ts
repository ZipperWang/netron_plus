import { useCallback } from 'react';
import { create } from 'zustand';
import { createBackend, type Backend, type SaveFilter } from './backend';
import { fontMetrics } from './graph/metrics';
import { detectLang, translate, type Lang, type MsgKey, type TFunc } from './i18n';
import type {
  GraphView,
  Issue,
  ModelSummary,
  ModuleNode,
  NodeRow,
  ValueRow,
  WeightRow,
} from './types';
import { baseName } from './util/format';

export type Tab = 'graph' | 'summary' | 'operators' | 'nodes' | 'tensors' | 'weights' | 'checks';
export const TABS: Tab[] = ['graph', 'summary', 'operators', 'nodes', 'tensors', 'weights', 'checks'];

export interface Selection {
  kind: 'node' | 'value';
  graph: number;
  id: number;
}

export interface Highlight {
  label: string;
  graph: number;
  /** IR node ids. */
  nodes: number[];
  index: number;
}

export type Trace = 'none' | 'up' | 'down' | 'both';
export type ThemePref = 'system' | 'light' | 'dark';

export interface ViewOpts {
  showInitializers: boolean;
  showAttributes: boolean;
  showNames: boolean;
  showConstants: boolean;
  edgeLabels: boolean;
  detachLongEdges: boolean;
  wheelZoom: boolean;
}

export interface Toast {
  id: number;
  kind: 'info' | 'error' | 'success';
  text: string;
}

export interface FocusRequest {
  seq: number;
  graph: number;
  kind: 'node' | 'value' | 'nodes' | 'fit';
  ids: number[];
}

interface State {
  backend: Backend | null;
  summary: ModelSummary | null;
  loading: string | null;
  graphId: number;
  view: GraphView | null;
  viewMode: 'initial' | 'keep';
  viewLoading: boolean;
  viewOpts: ViewOpts;
  selection: Selection | null;
  highlight: Highlight | null;
  trace: Trace;
  tab: Tab;
  theme: ThemePref;
  lang: Lang;
  sidebar: boolean;
  inspector: boolean;
  recent: string[];
  focus: FocusRequest | null;
  toasts: Toast[];
  nodeRows: { graph: number; rows: NodeRow[] } | null;
  valueRows: { graph: number; rows: ValueRow[] } | null;
  weightRows: WeightRow[] | null;
  issues: { deep: boolean; list: Issue[] } | null;
  hierarchy: { graph: number; root: ModuleNode } | null;
  nodeFilter: string;
  showHelp: boolean;
  dropping: boolean;
}

const DEFAULT_VIEW: ViewOpts = {
  showInitializers: true,
  showAttributes: false,
  showNames: false,
  showConstants: false,
  edgeLabels: true,
  detachLongEdges: true,
  wheelZoom: true,
};

const PREFS_KEY = 'netron-plus.prefs.v1';

interface Prefs {
  theme: ThemePref;
  lang: Lang;
  viewOpts: ViewOpts;
  sidebar: boolean;
  inspector: boolean;
  recent: string[];
}

function loadPrefs(): Prefs {
  const base: Prefs = {
    theme: 'system',
    lang: detectLang(),
    viewOpts: { ...DEFAULT_VIEW },
    sidebar: true,
    inspector: true,
    recent: [],
  };
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (!raw) return base;
    const p = JSON.parse(raw) as Partial<Prefs>;
    return {
      theme: p.theme ?? base.theme,
      lang: p.lang ?? base.lang,
      viewOpts: { ...DEFAULT_VIEW, ...(p.viewOpts ?? {}) },
      sidebar: p.sidebar ?? true,
      inspector: p.inspector ?? true,
      recent: Array.isArray(p.recent) ? p.recent.slice(0, 12) : [],
    };
  } catch {
    return base;
  }
}

function savePrefs(s: State) {
  try {
    const p: Prefs = {
      theme: s.theme,
      lang: s.lang,
      viewOpts: s.viewOpts,
      sidebar: s.sidebar,
      inspector: s.inspector,
      recent: s.recent,
    };
    localStorage.setItem(PREFS_KEY, JSON.stringify(p));
  } catch {
    // Storage may be unavailable (private mode); preferences are optional.
  }
}

const prefs = loadPrefs();

export const useStore = create<State>()(() => ({
  backend: null,
  summary: null,
  loading: null,
  graphId: 0,
  view: null,
  viewMode: 'initial',
  viewLoading: false,
  viewOpts: prefs.viewOpts,
  selection: null,
  highlight: null,
  trace: 'none',
  tab: 'graph',
  theme: prefs.theme,
  lang: prefs.lang,
  sidebar: prefs.sidebar,
  inspector: prefs.inspector,
  recent: prefs.recent,
  focus: null,
  toasts: [],
  nodeRows: null,
  valueRows: null,
  weightRows: null,
  issues: null,
  hierarchy: null,
  nodeFilter: '',
  showHelp: false,
  dropping: false,
}));

const get = useStore.getState;
const set = useStore.setState;

export function applyTheme(theme: ThemePref) {
  const root = document.documentElement;
  if (theme === 'system') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', theme);
}

useStore.subscribe((s, prev) => {
  if (
    s.theme !== prev.theme ||
    s.lang !== prev.lang ||
    s.viewOpts !== prev.viewOpts ||
    s.sidebar !== prev.sidebar ||
    s.inspector !== prev.inspector ||
    s.recent !== prev.recent
  ) {
    savePrefs(s);
  }
  if (s.theme !== prev.theme) applyTheme(s.theme);
  if (s.lang !== prev.lang) document.documentElement.lang = s.lang === 'zh' ? 'zh-CN' : 'en';
});

export function t(key: MsgKey, params?: Record<string, string | number>): string {
  return translate(get().lang, key, params);
}

export function useT(): TFunc {
  const lang = useStore((s) => s.lang);
  return useCallback((key: MsgKey, params?: Record<string, string | number>) => translate(lang, key, params), [lang]);
}

export function errorText(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof e === 'string') return e;
  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
}

let toastSeq = 0;
export function toast(kind: Toast['kind'], text: string) {
  const id = ++toastSeq;
  set({ toasts: [...get().toasts, { id, kind, text }].slice(-5) });
  setTimeout(() => dismissToast(id), kind === 'error' ? 9000 : 4000);
}

export function dismissToast(id: number) {
  set({ toasts: get().toasts.filter((x) => x.id !== id) });
}

// ------------------------------------------------------------------ backend

let filePicker: HTMLInputElement | null = null;
export function registerFilePicker(el: HTMLInputElement | null) {
  filePicker = el;
}

let started = false;

export async function initApp() {
  if (started) return;
  started = true;
  applyTheme(get().theme);
  document.documentElement.lang = get().lang === 'zh' ? 'zh-CN' : 'en';
  let backend: Backend;
  try {
    backend = await createBackend();
  } catch (e) {
    toast('error', errorText(e));
    return;
  }
  set({ backend });
  if (backend.kind === 'tauri') {
    const f = await backend.startupFile().catch(() => null);
    if (f) await openPath(f);
  } else {
    const url = new URLSearchParams(window.location.search).get('url');
    if (url) await openUrl(url);
  }
}

async function call<T>(method: string, params: object = {}): Promise<T> {
  const { backend, summary } = get();
  if (!backend || !summary) throw new Error('No model is open');
  return backend.call<T>(method, { model: summary.id, ...params });
}

export { call as callModel };

async function closeCurrent() {
  const { backend, summary } = get();
  if (backend && summary) {
    try {
      await backend.call('close', { model: summary.id });
    } catch {
      // ignore
    }
  }
}

function resetForModel(summary: ModelSummary) {
  set({
    summary,
    graphId: 0,
    view: null,
    selection: null,
    highlight: null,
    trace: 'none',
    nodeRows: null,
    valueRows: null,
    weightRows: null,
    issues: null,
    hierarchy: null,
    nodeFilter: '',
    tab: 'graph',
    focus: null,
  });
  document.title = `${summary.fileName} — Netron Plus`;
  void loadView(0, 'initial');
  void loadHierarchy(0);
}

async function openWith(name: string, f: () => Promise<ModelSummary>): Promise<boolean> {
  if (get().loading) return false;
  set({ loading: t('loading.open', { name }) });
  try {
    await closeCurrent();
    const s = await f();
    resetForModel(s);
    return true;
  } catch (e) {
    toast('error', `${t('error.open')}: ${errorText(e)}`);
    return false;
  } finally {
    set({ loading: null });
  }
}

export async function openPath(path: string) {
  const b = get().backend;
  if (!b) return;
  if (await openWith(baseName(path), () => b.openPath(path))) {
    set({ recent: [path, ...get().recent.filter((p) => p !== path)].slice(0, 12) });
  }
}

export async function openFile(file: File) {
  const b = get().backend;
  if (!b) return;
  await openWith(file.name, () => b.openFile(file));
}

export async function openUrl(url: string) {
  const b = get().backend;
  if (!b) return;
  await openWith(baseName(url.split(/[?#]/)[0]), () => b.openUrl(url));
}

/** Must be called from a user gesture so the browser allows the file dialog. */
export function pickAndOpen() {
  const b = get().backend;
  if (!b) return;
  if (b.kind === 'web') {
    filePicker?.click();
    return;
  }
  void b.pickModelPath().then((p) => {
    if (p) void openPath(p);
  });
}

export async function closeModel() {
  await closeCurrent();
  set({
    summary: null,
    view: null,
    selection: null,
    highlight: null,
    trace: 'none',
    nodeRows: null,
    valueRows: null,
    weightRows: null,
    issues: null,
    hierarchy: null,
  });
  document.title = 'Netron Plus';
}

export function clearRecent() {
  set({ recent: [] });
}

let viewReq = 0;
export async function loadView(graph: number, mode: 'initial' | 'keep' = 'keep') {
  const { backend, summary, viewOpts } = get();
  if (!backend || !summary) return;
  const req = ++viewReq;
  set({ viewLoading: true });
  try {
    const view = await backend.call<GraphView>('graph', {
      model: summary.id,
      graph,
      options: { ...viewOpts, maxAttributes: 8, metrics: fontMetrics() },
    });
    if (req !== viewReq || get().summary?.id !== summary.id) return;
    set({ view, graphId: graph, viewMode: mode });
  } catch (e) {
    if (req === viewReq) toast('error', errorText(e));
  } finally {
    if (req === viewReq) set({ viewLoading: false });
  }
}

export async function loadHierarchy(graph: number) {
  try {
    const root = await call<ModuleNode>('hierarchy', { graph });
    if (get().graphId === graph || get().view === null) set({ hierarchy: { graph, root } });
  } catch {
    set({ hierarchy: null });
  }
}

export function setGraph(graph: number, keepSelection = false) {
  if (graph === get().graphId && get().view) return;
  set({
    highlight: null,
    trace: 'none',
    selection: keepSelection ? get().selection : null,
    graphId: graph,
  });
  void loadView(graph, 'initial');
  void loadHierarchy(graph);
}

export function setViewOpts(patch: Partial<ViewOpts>) {
  const viewOpts = { ...get().viewOpts, ...patch };
  set({ viewOpts });
  const layoutKeys: (keyof ViewOpts)[] = [
    'showInitializers',
    'showAttributes',
    'showNames',
    'showConstants',
    'edgeLabels',
    'detachLongEdges',
  ];
  if (layoutKeys.some((k) => k in patch)) void loadView(get().graphId, 'keep');
}

let focusSeq = 0;
function requestFocus(graph: number, kind: FocusRequest['kind'], ids: number[]) {
  set({ focus: { seq: ++focusSeq, graph, kind, ids } });
}

export function select(sel: Selection | null) {
  set({ selection: sel });
}

/** Select a node and bring it into view in the graph tab. */
export function showNode(graph: number, id: number) {
  set({ selection: { kind: 'node', graph, id }, tab: 'graph' });
  if (graph !== get().graphId) setGraph(graph, true);
  requestFocus(graph, 'node', [id]);
}

export function showValue(graph: number, id: number) {
  set({ selection: { kind: 'value', graph, id }, tab: 'graph' });
  if (graph !== get().graphId) setGraph(graph, true);
  requestFocus(graph, 'value', [id]);
}

export function highlightNodes(label: string, graph: number, nodes: number[], jump = true) {
  if (nodes.length === 0) {
    set({ highlight: null });
    return;
  }
  set({ highlight: { label, graph, nodes, index: 0 } });
  if (graph !== get().graphId) setGraph(graph, true);
  if (jump) {
    set({ tab: 'graph' });
    requestFocus(graph, 'nodes', nodes);
  }
}

export function stepHighlight(delta: number) {
  const h = get().highlight;
  if (!h || h.nodes.length === 0) return;
  const index = (h.index + delta + h.nodes.length) % h.nodes.length;
  set({ highlight: { ...h, index }, selection: { kind: 'node', graph: h.graph, id: h.nodes[index] } });
  requestFocus(h.graph, 'node', [h.nodes[index]]);
}

export function clearHighlight() {
  set({ highlight: null, trace: 'none' });
}

export function fitGraph() {
  requestFocus(get().graphId, 'fit', []);
}

export function setTrace(trace: Trace) {
  set({ trace });
}

export function setTab(tab: Tab) {
  set({ tab });
}

export async function applyOverrides(
  params: Record<string, number | null>,
  inputs: Record<string, (number | null)[]>,
) {
  const { summary } = get();
  if (!summary) return;
  set({ loading: t('loading.generic') });
  try {
    const s = await call<ModelSummary>('setOverrides', { params, inputs });
    set({ summary: s, nodeRows: null, valueRows: null, issues: null, weightRows: get().weightRows });
    void loadView(get().graphId, 'keep');
    void loadHierarchy(get().graphId);
  } catch (e) {
    toast('error', errorText(e));
  } finally {
    set({ loading: null });
  }
}

export async function ensureNodeRows(force = false) {
  const g = get().graphId;
  const cur = get().nodeRows;
  if (!force && cur && cur.graph === g) return;
  try {
    const rows = await call<NodeRow[]>('nodeTable', { graph: g });
    set({ nodeRows: { graph: g, rows } });
  } catch (e) {
    toast('error', errorText(e));
  }
}

export async function ensureValueRows(force = false) {
  const g = get().graphId;
  const cur = get().valueRows;
  if (!force && cur && cur.graph === g) return;
  try {
    const rows = await call<ValueRow[]>('valueTable', { graph: g });
    set({ valueRows: { graph: g, rows } });
  } catch (e) {
    toast('error', errorText(e));
  }
}

export async function loadWeights() {
  set({ loading: t('weights.running') });
  try {
    const rows = await call<WeightRow[]>('weightTable');
    set({ weightRows: rows });
  } catch (e) {
    toast('error', errorText(e));
  } finally {
    set({ loading: null });
  }
}

export async function runChecks(deep: boolean) {
  if (deep) set({ loading: t('checks.running') });
  try {
    const list = await call<Issue[]>('checks', { deep });
    set({ issues: { deep, list } });
  } catch (e) {
    toast('error', errorText(e));
  } finally {
    if (deep) set({ loading: null });
  }
}

export async function saveText(name: string, content: string, filter: SaveFilter) {
  const b = get().backend;
  if (!b) return;
  try {
    const saved = await b.saveText(name, content, filter);
    if (saved && b.kind === 'tauri') toast('success', t('common.saved', { name: saved }));
  } catch (e) {
    toast('error', errorText(e));
  }
}

export async function saveBytes(name: string, data: Uint8Array, filter: SaveFilter) {
  const b = get().backend;
  if (!b) return;
  try {
    const saved = await b.saveBytes(name, data, filter);
    if (saved && b.kind === 'tauri') toast('success', t('common.saved', { name: saved }));
  } catch (e) {
    toast('error', errorText(e));
  }
}

export async function exportTensor(graph: number, tensor: number, format: 'npy' | 'csv' | 'raw', name: string) {
  const { backend, summary } = get();
  if (!backend || !summary) return;
  const ext = format === 'raw' ? 'bin' : format;
  try {
    const saved = await backend.exportTensor(
      { model: summary.id, graph, tensor, format },
      `${name}.${ext}`,
      { name: format.toUpperCase(), extensions: [ext] },
    );
    if (saved && backend.kind === 'tauri') toast('success', t('common.saved', { name: saved }));
  } catch (e) {
    toast('error', errorText(e));
  }
}

export async function openExternal(url: string) {
  const b = get().backend;
  try {
    if (b) await b.openExternal(url);
    else window.open(url, '_blank', 'noopener');
  } catch (e) {
    toast('error', errorText(e));
  }
}

export async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    toast('info', t('common.copied'));
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
      toast('info', t('common.copied'));
    } finally {
      ta.remove();
    }
  }
}
