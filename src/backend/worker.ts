// Web Worker hosting the WebAssembly build of the Rust core (crates/wasm).

interface CoreExports {
  memory: WebAssembly.Memory;
  np_alloc(len: number): number;
  np_open(namePtr: number, nameLen: number, dataPtr: number, dataLen: number): number;
  np_call(methodPtr: number, methodLen: number, paramsPtr: number, paramsLen: number): number;
  np_export(paramsPtr: number, paramsLen: number): number;
  np_out_ptr(): number;
  np_out_len(): number;
  np_panic_ptr(): number;
  np_panic_len(): number;
}

interface WorkerScope {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent) => void) | null;
}

const scope = self as unknown as WorkerScope;
const enc = new TextEncoder();
const dec = new TextDecoder();
let wasmUrl = '';
let core: Promise<CoreExports> | null = null;

async function load(): Promise<CoreExports> {
  const resp = await fetch(wasmUrl);
  if (!resp.ok) throw new Error(`Cannot load ${wasmUrl} (HTTP ${resp.status})`);
  const bytes = await resp.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports as unknown as CoreExports;
}

function put(c: CoreExports, data: Uint8Array): [number, number] {
  const ptr = c.np_alloc(data.length);
  new Uint8Array(c.memory.buffer, ptr, data.length).set(data);
  return [ptr, data.length];
}

function output(c: CoreExports): Uint8Array {
  return new Uint8Array(c.memory.buffer, c.np_out_ptr(), c.np_out_len()).slice();
}

function panicMessage(c: CoreExports): string {
  try {
    return dec.decode(new Uint8Array(c.memory.buffer, c.np_panic_ptr(), c.np_panic_len()).slice());
  } catch {
    return '';
  }
}

function restart() {
  core = load();
  core.catch(() => undefined);
}

scope.onmessage = async (e: MessageEvent) => {
  const msg = e.data as { id?: number; type: string; [key: string]: unknown };
  if (msg.type === 'init') {
    wasmUrl = String(msg.url);
    restart();
    return;
  }
  const id = msg.id;
  let c: CoreExports | null = null;
  try {
    if (!core) throw new Error('The engine is not initialized');
    c = await core;
    let status: number;
    if (msg.type === 'open') {
      const [np, nl] = put(c, enc.encode(String(msg.name)));
      const [dp, dl] = put(c, new Uint8Array(msg.data as ArrayBuffer));
      status = c.np_open(np, nl, dp, dl);
    } else if (msg.type === 'call') {
      const [mp, ml] = put(c, enc.encode(String(msg.method)));
      const [pp, pl] = put(c, enc.encode(String(msg.params)));
      status = c.np_call(mp, ml, pp, pl);
    } else if (msg.type === 'export') {
      const [pp, pl] = put(c, enc.encode(String(msg.params)));
      status = c.np_export(pp, pl);
    } else {
      throw new Error(`Unknown request ${msg.type}`);
    }
    const out = output(c);
    if (status !== 0) throw new Error(dec.decode(out));
    if (msg.type === 'export') scope.postMessage({ id, ok: true, result: out.buffer }, [out.buffer]);
    else scope.postMessage({ id, ok: true, result: dec.decode(out) });
  } catch (err) {
    let message = err instanceof Error ? err.message : String(err);
    if (err instanceof WebAssembly.RuntimeError && c) {
      const p = panicMessage(c);
      message = `Internal error${p ? `: ${p}` : ''}. The engine was restarted; please reopen the model.`;
      restart();
    }
    scope.postMessage({ id, ok: false, error: message });
  }
};
