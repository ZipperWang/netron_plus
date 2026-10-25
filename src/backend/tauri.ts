import { invoke } from '@tauri-apps/api/core';
import { getCurrentWebview } from '@tauri-apps/api/webview';
import { open, save } from '@tauri-apps/plugin-dialog';
import { openUrl } from '@tauri-apps/plugin-opener';
import type { Backend, DropHandlers, SaveFilter } from './index';
import type { ModelSummary } from '../types';
import { baseName } from '../util/format';

/** Commands return JSON as raw bytes; accept every shape the IPC may deliver. */
function decodeJson<T>(r: unknown): T {
  if (typeof r === 'string') return JSON.parse(r) as T;
  if (r instanceof ArrayBuffer) return JSON.parse(new TextDecoder().decode(r)) as T;
  if (ArrayBuffer.isView(r)) return JSON.parse(new TextDecoder().decode(r)) as T;
  if (Array.isArray(r)) return JSON.parse(new TextDecoder().decode(new Uint8Array(r as number[]))) as T;
  return r as T;
}

export class TauriBackend implements Backend {
  readonly kind = 'tauri' as const;

  async pickModelPath(): Promise<string | null> {
    const r: unknown = await open({
      multiple: false,
      directory: false,
      filters: [
        { name: 'ONNX', extensions: ['onnx'] },
        { name: 'All files', extensions: ['*'] },
      ],
    });
    if (typeof r === 'string') return r;
    if (Array.isArray(r) && typeof r[0] === 'string') return r[0];
    return null;
  }

  async openPath(path: string): Promise<ModelSummary> {
    return decodeJson<ModelSummary>(await invoke<unknown>('np_open_path', { path }));
  }

  async openFile(_file: File): Promise<ModelSummary> {
    throw new Error('Drop the file onto the window or use the Open button.');
  }

  async openUrl(_url: string): Promise<ModelSummary> {
    throw new Error('Opening URLs is only supported in the web version.');
  }

  async call<T>(method: string, params: object): Promise<T> {
    return decodeJson<T>(await invoke<unknown>('np_call', { method, params }));
  }

  private async pickSave(defaultName: string, filter: SaveFilter): Promise<string | null> {
    const p = await save({ defaultPath: defaultName, filters: [filter] });
    return typeof p === 'string' && p ? p : null;
  }

  async exportTensor(params: object, defaultName: string, filter: SaveFilter): Promise<string | null> {
    const path = await this.pickSave(defaultName, filter);
    if (!path) return null;
    await invoke('np_export', { params, path });
    return baseName(path);
  }

  async saveText(defaultName: string, content: string, filter: SaveFilter): Promise<string | null> {
    const path = await this.pickSave(defaultName, filter);
    if (!path) return null;
    await invoke('np_write_text', { path, content });
    return baseName(path);
  }

  async saveBytes(defaultName: string, data: Uint8Array, filter: SaveFilter): Promise<string | null> {
    const path = await this.pickSave(defaultName, filter);
    if (!path) return null;
    await invoke('np_write_bytes', { path, data: Array.from(data) });
    return baseName(path);
  }

  async openExternal(url: string): Promise<void> {
    await openUrl(url);
  }

  async startupFile(): Promise<string | null> {
    return (await invoke<string | null>('np_startup_file')) ?? null;
  }

  async onFileDrop(h: DropHandlers): Promise<() => void> {
    return getCurrentWebview().onDragDropEvent((event) => {
      const p = event.payload;
      if (p.type === 'enter' || p.type === 'over') h.hover(true);
      else if (p.type === 'leave') h.hover(false);
      else if (p.type === 'drop') {
        h.hover(false);
        h.drop(p.paths);
      }
    });
  }
}
