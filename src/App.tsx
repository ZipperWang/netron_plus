import { useEffect, useRef } from 'react';
import {
  TABS,
  clearHighlight,
  initApp,
  openFile,
  openPath,
  pickAndOpen,
  registerFilePicker,
  select,
  setTab,
  useStore,
  useT,
} from './store';
import { GraphPane } from './graph/GraphPane';
import { Inspector } from './panels/Inspector';
import { Sidebar, focusSearch } from './panels/Sidebar';
import { SummaryView } from './panels/SummaryView';
import { ChecksView, NodesView, OperatorsView, TensorsView, WeightsView } from './panels/TableViews';
import { HelpDialog, StatusBar, Toasts, Toolbar, Welcome } from './panels/Chrome';
import { OpFunctionDialog } from './panels/OpFunction';
import { Icon, Resizer, Spinner } from './components/common';

function isEditable(el: Element | null): boolean {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (el as HTMLElement).isContentEditable;
}

export default function App() {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const tab = useStore((s) => s.tab);
  const sidebar = useStore((s) => s.sidebar);
  const inspector = useStore((s) => s.inspector);
  const sidebarWidth = useStore((s) => s.sidebarWidth);
  const inspectorWidth = useStore((s) => s.inspectorWidth);
  const loading = useStore((s) => s.loading);
  const dropping = useStore((s) => s.dropping);
  const backend = useStore((s) => s.backend);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    registerFilePicker(fileRef.current);
    void initApp();
    return () => registerFilePicker(null);
  }, []);

  // Native (Tauri) file drops deliver paths.
  useEffect(() => {
    if (!backend || backend.kind !== 'tauri') return;
    let unlisten: (() => void) | null = null;
    let disposed = false;
    void backend
      .onFileDrop({
        drop: (paths) => {
          const p = paths.find((x) => /\.onnx$/i.test(x)) ?? paths[0];
          if (p) void openPath(p);
        },
        hover: (over) => useStore.setState({ dropping: over }),
      })
      .then((u) => {
        if (disposed) u();
        else unlisten = u;
      });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [backend]);

  // Browser file drops deliver File objects.
  useEffect(() => {
    if (!backend || backend.kind !== 'web') return;
    let depth = 0;
    const over = (e: DragEvent) => {
      if (e.dataTransfer?.types.includes('Files')) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
      }
    };
    const enter = (e: DragEvent) => {
      if (!e.dataTransfer?.types.includes('Files')) return;
      depth++;
      useStore.setState({ dropping: true });
    };
    const leave = () => {
      depth = Math.max(0, depth - 1);
      if (depth === 0) useStore.setState({ dropping: false });
    };
    const drop = (e: DragEvent) => {
      e.preventDefault();
      depth = 0;
      useStore.setState({ dropping: false });
      const f = e.dataTransfer?.files?.[0];
      if (f) void openFile(f);
    };
    window.addEventListener('dragover', over);
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragover', over);
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('drop', drop);
    };
  }, [backend]);

  // Global shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      const k = e.key.toLowerCase();
      if (mod && k === 'o') {
        e.preventDefault();
        pickAndOpen();
        return;
      }
      if (!useStore.getState().summary) return;
      if (mod && k === 'f') {
        e.preventDefault();
        if (!useStore.getState().sidebar) useStore.setState({ sidebar: true });
        setTimeout(focusSearch, 0);
        return;
      }
      if (mod && k === 'b') {
        e.preventDefault();
        useStore.setState({ sidebar: !useStore.getState().sidebar });
        return;
      }
      if (mod && k === 'i') {
        e.preventDefault();
        useStore.setState({ inspector: !useStore.getState().inspector });
        return;
      }
      if (mod && /^[1-7]$/.test(e.key)) {
        e.preventDefault();
        setTab(TABS[Number(e.key) - 1]);
        return;
      }
      if (e.key === 'Escape' && !isEditable(document.activeElement)) {
        if (useStore.getState().showHelp) {
          useStore.setState({ showHelp: false });
          return;
        }
        clearHighlight();
        select(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div className="app">
      <Toolbar />
      <div className="main">
        {summary ? (
          <>
            {sidebar && <Sidebar />}
            {sidebar && (
              <Resizer value={sidebarWidth} min={180} max={520} onChange={(v) => useStore.setState({ sidebarWidth: v })} />
            )}
            <section className="center">
              <GraphPane visible={tab === 'graph'} />
              {tab === 'summary' && <SummaryView />}
              {tab === 'operators' && <OperatorsView />}
              {tab === 'nodes' && <NodesView />}
              {tab === 'tensors' && <TensorsView />}
              {tab === 'weights' && <WeightsView />}
              {tab === 'checks' && <ChecksView />}
            </section>
            {inspector && (
              <Resizer
                value={inspectorWidth}
                min={260}
                max={640}
                invert
                onChange={(v) => useStore.setState({ inspectorWidth: v })}
              />
            )}
            {inspector && <Inspector />}
          </>
        ) : (
          <Welcome />
        )}
        {loading && (
          <div className="busy-overlay">
            <div className="busy-card">
              <Spinner size={20} />
              <span>{loading}</span>
            </div>
          </div>
        )}
        {dropping && (
          <div className="drop-overlay">
            <Icon name="folder" size={36} />
            <span>{t('welcome.drop')}</span>
          </div>
        )}
      </div>
      <StatusBar />
      <HelpDialog />
      {summary && <OpFunctionDialog />}
      <Toasts />
      <input
        ref={fileRef}
        type="file"
        accept=".onnx,application/octet-stream"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void openFile(f);
          e.target.value = '';
        }}
      />
    </div>
  );
}
