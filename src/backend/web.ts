import type { Backend, SaveFilter } from './index';
import type { ModelSummary } from '../types';

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
}

function download(blob: Blob, name: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(a.href);
    a.remove();
  }, 2000);
}

/** Runs the WebAssembly build of the Rust core in a Web Worker. */
export class WebBackend implements Backend {
  readonly kind = 'web' as const;
  private worker: Worker;
  private seq = 0;
  private pending = new Map<number, Pending>();

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent) => {
      const { id, ok, result, error } = e.data as { id: number; ok: boolean; result: unknown; error: string };
      const p = this.pending.get(id);
      if (!p) return;
      this.pending.delete(id);
      if (ok) p.resolve(result);
      else p.reject(new Error(error));
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(e.message || 'Worker error'));
      this.pending.clear();
    };
    this.worker.postMessage({ type: 'init', url: new URL('netron_core.wasm', document.baseURI).href });
  }

  private request<T>(msg: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
    const id = ++this.seq;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.worker.postMessage({ ...msg, id }, transfer);
    });
  }

  async pickModelPath(): Promise<string | null> {
    return null;
  }

  async openPath(_path: string): Promise<ModelSummary> {
    throw new Error('Opening files by path is only available in the desktop app.');
  }

  private async openBuffer(name: string, buf: ArrayBuffer): Promise<ModelSummary> {
    const json = await this.request<string>({ type: 'open', name, data: buf }, [buf]);
    return JSON.parse(json) as ModelSummary;
  }

  async openFile(file: File): Promise<ModelSummary> {
    return this.openBuffer(file.name, await file.arrayBuffer());
  }

  async openUrl(url: string): Promise<ModelSummary> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
    const name = decodeURIComponent(url.split(/[?#]/)[0].split('/').pop() || 'model.onnx');
    return this.openBuffer(name, await resp.arrayBuffer());
  }

  async call<T>(method: string, params: object): Promise<T> {
    const json = await this.request<string>({ type: 'call', method, params: JSON.stringify(params) });
    return JSON.parse(json) as T;
  }

  async exportTensor(params: object, defaultName: string, _filter: SaveFilter): Promise<string | null> {
    const buf = await this.request<ArrayBuffer>({ type: 'export', params: JSON.stringify(params) });
    download(new Blob([buf], { type: 'application/octet-stream' }), defaultName);
    return defaultName;
  }

  async saveText(defaultName: string, content: string, _filter: SaveFilter): Promise<string | null> {
    download(new Blob([content], { type: 'text/plain;charset=utf-8' }), defaultName);
    return defaultName;
  }

  async saveBytes(defaultName: string, data: Uint8Array, _filter: SaveFilter): Promise<string | null> {
    download(new Blob([data], { type: 'application/octet-stream' }), defaultName);
    return defaultName;
  }

  async openExternal(url: string): Promise<void> {
    window.open(url, '_blank', 'noopener');
  }

  async startupFile(): Promise<string | null> {
    return null;
  }

  async onFileDrop(): Promise<() => void> {
    return () => {};
  }
}
