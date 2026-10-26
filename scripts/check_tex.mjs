// Renders every TeX snippet written by `cargo test -p netron-core --test tex`
// (with NETRON_TEX_DUMP set) with KaTeX, using the macros and options of the
// app (src/math/Tex.tsx). Fails when any snippet does not parse.
//
//   node scripts/check_tex.mjs target/tex-dump.json

import { readFileSync } from 'node:fs';
import katex from 'katex';

const file = process.argv[2];
if (!file) {
  console.error('usage: node scripts/check_tex.mjs <tex-dump.json>');
  process.exit(2);
}
const macros = JSON.parse(readFileSync(new URL('../src/math/macros.json', import.meta.url), 'utf8'));
const items = JSON.parse(readFileSync(file, 'utf8'));

let failed = 0;
for (const it of items) {
  try {
    katex.renderToString(it.tex, {
      displayMode: it.display,
      throwOnError: true,
      strict: 'ignore',
      output: 'htmlAndMathml',
      macros: { ...macros },
    });
  } catch (e) {
    failed++;
    console.error(`${it.where}: ${e instanceof Error ? e.message : e}\n    ${it.tex}`);
  }
}
console.log(`KaTeX ${katex.version}: ${items.length} TeX snippets checked, ${failed} failed`);
if (items.length === 0 || failed > 0) process.exit(1);
