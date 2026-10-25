import type { ModelSummary } from '../types';

export interface SaveFilter {
  name: string;
  extensions: string[];
}

export interface DropHandlers {
  drop: (paths: string[]) => void;
  hover: (over: boolean) => void;
}

/** The host the UI runs in: the Tauri desktop app or a browser (WebAssembly). */
export interface Backend {
  readonly kind: 'tauri' | 'web';
  /** Native open dialog; resolves to a path or null (desktop only). */
  pickModelPath(): Promise<string | null>;
  openPath(path: string): Promise<ModelSummary>;
  openFile(file: File): Promise<ModelSummary>;
  openUrl(url: string): Promise<ModelSummary>;
  call<T>(method: string, params: object): Promise<T>;
  /** All save helpers resolve to the saved file name, or null when cancelled. */
  exportTensor(params: object, defaultName: string, filter: SaveFilter): Promise<string | null>;
  saveText(defaultName: string, content: string, filter: SaveFilter): Promise<string | null>;
  saveBytes(defaultName: string, data: Uint8Array, filter: SaveFilter): Promise<string | null>;
  openExternal(url: string): Promise<void>;
  startupFile(): Promise<string | null>;
  onFileDrop(handlers: DropHandlers): Promise<() => void>;
}

export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

export async function createBackend(): Promise<Backend> {
  if (isTauri()) {
    const m = await import('./tauri');
    return new m.TauriBackend();
  }
  const m = await import('./web');
  return new m.WebBackend();
}
