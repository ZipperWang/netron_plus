import { useEffect, useMemo, useState } from 'react';
import type { KatexOptions } from 'katex';
import macros from './macros.json';

interface Katex {
  renderToString(tex: string, options?: KatexOptions): string;
}

let katex: Katex | null = null;
let loading: Promise<Katex> | null = null;

function loadKatex(): Promise<Katex> {
  loading ??= import('./katex').then((m) => (katex = m.default));
  return loading;
}

type Rendered = { html: string } | { error: string };

function render(k: Katex, tex: string, display: boolean): Rendered {
  try {
    // Same options as scripts/check_tex.mjs, which validates every formula in CI.
    const html = k.renderToString(tex, {
      displayMode: display,
      throwOnError: true,
      strict: 'ignore',
      output: 'htmlAndMathml',
      macros: { ...macros },
    });
    return { html };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
}

/** TeX rendered with KaTeX; the source is shown while KaTeX loads or when it cannot be parsed. */
export function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
  const [ready, setReady] = useState(katex !== null);
  useEffect(() => {
    if (ready) return;
    let alive = true;
    loadKatex()
      .then(() => alive && setReady(true))
      .catch(() => {
        // Keep showing the source.
      });
    return () => {
      alive = false;
    };
  }, [ready]);
  const out = useMemo(() => (ready && katex ? render(katex, tex, display) : null), [ready, tex, display]);
  if (!out || 'error' in out) {
    return (
      <code className={'tex-src' + (out ? ' tex-error' : '')} title={out && 'error' in out ? out.error : undefined}>
        {tex}
      </code>
    );
  }
  return <span className={display ? 'tex tex-display' : 'tex'} dangerouslySetInnerHTML={{ __html: out.html }} />;
}
