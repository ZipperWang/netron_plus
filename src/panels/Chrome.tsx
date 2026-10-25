import { useEffect, useState } from 'react';
import {
  TABS,
  clearRecent,
  closeModel,
  dismissToast,
  openPath,
  openUrl,
  pickAndOpen,
  saveBytes,
  saveText,
  setTab,
  toast,
  errorText,
  useStore,
  useT,
} from '../store';
import { Button, Dropdown, Icon, IconButton, type MenuItem } from '../components/common';
import { readTheme } from '../graph/colors';
import { exportPng, exportSvg } from '../graph/exporter';
import { baseName, fmtBytes, fmtCount, stripExt } from '../util/format';
import type { MsgKey } from '../i18n';

export function Toolbar() {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const tab = useStore((s) => s.tab);
  const recent = useStore((s) => s.recent);
  const backend = useStore((s) => s.backend);
  const theme = useStore((s) => s.theme);
  const lang = useStore((s) => s.lang);
  const sidebar = useStore((s) => s.sidebar);
  const inspector = useStore((s) => s.inspector);
  const loading = useStore((s) => s.loading);

  const stem = summary ? stripExt(summary.fileName) : 'model';
  const recentItems = (): MenuItem[] =>
    recent.length === 0
      ? [{ label: t('tb.noRecent'), disabled: true }]
      : [
          ...recent.map((p) => ({ label: baseName(p), hint: p.length > 48 ? `…${p.slice(-46)}` : p, onClick: () => void openPath(p) })),
          { separator: true },
          { label: t('tb.clearRecent'), icon: 'x' as const, onClick: () => clearRecent() },
        ];
  const exportItems = (): MenuItem[] => [
    {
      label: t('tb.exportPng'),
      icon: 'download',
      disabled: !useStore.getState().view,
      onClick: () => {
        const v = useStore.getState().view;
        if (!v) return;
        exportPng(v, readTheme())
          .then((bytes) => saveBytes(`${stem}.png`, bytes, { name: 'PNG', extensions: ['png'] }))
          .catch((e) => toast('error', errorText(e)));
      },
    },
    {
      label: t('tb.exportSvg'),
      icon: 'download',
      disabled: !useStore.getState().view,
      onClick: () => {
        const v = useStore.getState().view;
        if (v) void saveText(`${stem}.svg`, exportSvg(v, readTheme()), { name: 'SVG', extensions: ['svg'] });
      },
    },
    {
      label: t('tb.exportSummary'),
      icon: 'download',
      onClick: () => {
        const s = useStore.getState().summary;
        if (s) void saveText(`${stem}.summary.json`, JSON.stringify(s, null, 2), { name: 'JSON', extensions: ['json'] });
      },
    },
    { separator: true },
    { label: t('tb.close'), icon: 'x', onClick: () => void closeModel() },
  ];
  const themeIcon = theme === 'dark' ? 'moon' : theme === 'light' ? 'sun' : 'monitor';
  const issues = summary?.issues;

  return (
    <header className="toolbar">
      <div className="brand" title="Netron Plus">
        <Icon name="graph" size={18} />
        <span>Netron Plus</span>
      </div>
      <Button icon="folder" kind="ghost" onClick={pickAndOpen} title={t('tb.openTitle')} disabled={!backend || !!loading}>
        {t('tb.open')}
      </Button>
      {backend?.kind === 'tauri' && <Dropdown icon="clock" title={t('tb.recent')} items={recentItems} />}
      {summary && <Dropdown icon="download" title={t('tb.export')} items={exportItems} />}
      {summary && (
        <nav className="tabs" role="tablist">
          {TABS.map((k, i) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={tab === k}
              className={'tab' + (tab === k ? ' on' : '')}
              onClick={() => setTab(k)}
              title={`${t(`tab.${k}` as MsgKey)} (Ctrl+${i + 1})`}
            >
              {t(`tab.${k}` as MsgKey)}
              {k === 'checks' && issues && issues.error > 0 && <span className="tab-badge error">{issues.error}</span>}
              {k === 'checks' && issues && issues.error === 0 && issues.warning > 0 && (
                <span className="tab-badge warning">{issues.warning}</span>
              )}
            </button>
          ))}
        </nav>
      )}
      <div className="spacer" />
      {summary && (
        <>
          <IconButton icon="panelLeft" active={sidebar} title={t('tb.sidebar')} onClick={() => useStore.setState({ sidebar: !sidebar })} />
          <IconButton icon="panelRight" active={inspector} title={t('tb.inspector')} onClick={() => useStore.setState({ inspector: !inspector })} />
        </>
      )}
      <Dropdown
        icon={themeIcon}
        title={t('tb.theme')}
        align="right"
        items={[
          { label: t('theme.system'), icon: 'monitor', checked: theme === 'system', onClick: () => useStore.setState({ theme: 'system' }) },
          { label: t('theme.light'), icon: 'sun', checked: theme === 'light', onClick: () => useStore.setState({ theme: 'light' }) },
          { label: t('theme.dark'), icon: 'moon', checked: theme === 'dark', onClick: () => useStore.setState({ theme: 'dark' }) },
        ]}
      />
      <Dropdown
        icon="globe"
        label={lang === 'zh' ? '中文' : 'EN'}
        title={t('tb.language')}
        align="right"
        items={[
          { label: '中文', checked: lang === 'zh', onClick: () => useStore.setState({ lang: 'zh' }) },
          { label: 'English', checked: lang === 'en', onClick: () => useStore.setState({ lang: 'en' }) },
        ]}
      />
      <IconButton icon="help" title={t('tb.help')} onClick={() => useStore.setState({ showHelp: true })} />
    </header>
  );
}

interface Sample {
  name: string;
  file: string;
  description?: string;
}

export function Welcome() {
  const t = useT();
  const recent = useStore((s) => s.recent);
  const backend = useStore((s) => s.backend);
  const loading = useStore((s) => s.loading);
  const [samples, setSamples] = useState<Sample[]>([]);

  useEffect(() => {
    if (backend?.kind !== 'web') return;
    let alive = true;
    fetch(new URL('samples/index.json', document.baseURI).href)
      .then((r) => (r.ok ? r.json() : []))
      .then((s: Sample[]) => alive && Array.isArray(s) && setSamples(s))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [backend]);

  return (
    <div className="welcome">
      <div className="welcome-card">
        <div className="welcome-logo">
          <Icon name="graph" size={44} />
        </div>
        <h1>Netron Plus</h1>
        <p className="muted">{t('welcome.features')}</p>
        <div className="drop-zone" onClick={() => !loading && pickAndOpen()} role="button" tabIndex={0}>
          <Icon name="folder" size={22} />
          <span>{t('welcome.drop')}</span>
          <Button kind="primary" disabled={!backend || !!loading}>
            {t('welcome.browse')}
          </Button>
        </div>
        {backend?.kind === 'tauri' && recent.length > 0 && (
          <div className="welcome-list">
            <div className="welcome-list-title">{t('welcome.recent')}</div>
            {recent.slice(0, 8).map((p) => (
              <button key={p} type="button" className="list-row" onClick={() => void openPath(p)} title={p}>
                <Icon name="clock" size={14} />
                <span className="list-main">
                  <span className="ellipsis">{baseName(p)}</span>
                  <span className="list-sub ellipsis">{p}</span>
                </span>
              </button>
            ))}
          </div>
        )}
        {samples.length > 0 && (
          <div className="welcome-list">
            <div className="welcome-list-title">{t('welcome.samples')}</div>
            {samples.map((s) => (
              <button
                key={s.file}
                type="button"
                className="list-row"
                onClick={() => void openUrl(new URL(s.file, new URL('samples/', document.baseURI)).href)}
                title={s.description ?? s.file}
              >
                <Icon name="cube" size={14} />
                <span className="list-main">
                  <span className="ellipsis">{s.name}</span>
                  {s.description && <span className="list-sub ellipsis">{s.description}</span>}
                </span>
              </button>
            ))}
          </div>
        )}
        <p className="muted small">
          {t('welcome.privacy')} {backend?.kind === 'web' ? t('welcome.webNote') : ''}
        </p>
      </div>
    </div>
  );
}

export function StatusBar({ zoomHint }: { zoomHint?: string }) {
  const t = useT();
  const summary = useStore((s) => s.summary);
  const backend = useStore((s) => s.backend);
  const view = useStore((s) => s.view);
  const loading = useStore((s) => s.loading);
  return (
    <footer className="statusbar">
      {loading ? (
        <span className="status-busy">
          <span className="spinner" style={{ width: 11, height: 11 }} /> {loading}
        </span>
      ) : (
        <span>{summary ? summary.fileName : t('status.ready')}</span>
      )}
      {summary && (
        <>
          <span>{t('status.nodes', { n: summary.stats.nodes.toLocaleString() })}</span>
          <span>{t('status.params', { n: fmtCount(summary.totals.params) })}</span>
          <span>{t('status.flops', { n: fmtCount(summary.totals.flops) })}</span>
          <span>{fmtBytes(summary.fileSize)}</span>
          {view && view.nodes.length > 0 && <span className="muted">view {view.nodes.length} / {view.edges.length}</span>}
        </>
      )}
      <span className="spacer" />
      {zoomHint && <span>{zoomHint}</span>}
      <span className="muted">{backend ? (backend.kind === 'tauri' ? t('status.desktop') : t('status.web')) : ''}</span>
    </footer>
  );
}

export function HelpDialog() {
  const t = useT();
  const show = useStore((s) => s.showHelp);
  if (!show) return null;
  const rows: [string, MsgKey][] = [
    ['Ctrl+O', 'help.open'],
    ['Ctrl+F', 'help.search'],
    ['F / 0', 'help.fit'],
    ['+ / −', 'help.zoom'],
    ['↑ / ↓  ← / →', 'help.navigate'],
    ['Ctrl+1 … 7', 'help.tabs'],
    ['Esc', 'help.escape'],
    ['Ctrl+B / Ctrl+I', 'help.panels'],
  ];
  return (
    <div className="modal-backdrop" onClick={() => useStore.setState({ showHelp: false })}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label={t('help.title')}>
        <div className="modal-head">
          <span>{t('help.title')}</span>
          <IconButton icon="x" title={t('common.close')} onClick={() => useStore.setState({ showHelp: false })} />
        </div>
        <table className="keys">
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={v}>
                <td>
                  <kbd>{k}</kbd>
                </td>
                <td>{t(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">{t('help.mouse')}</p>
      </div>
    </div>
  );
}

export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((x) => (
        <div key={x.id} className={`toast toast-${x.kind}`} onClick={() => dismissToast(x.id)}>
          <Icon name={x.kind === 'error' ? 'error' : x.kind === 'success' ? 'check' : 'info'} size={15} />
          <span>{x.text}</span>
        </div>
      ))}
    </div>
  );
}
