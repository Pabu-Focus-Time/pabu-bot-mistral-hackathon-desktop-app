import { contextBridge, ipcRenderer } from 'electron';

export interface WindowData {
  appName: string;
  windowTitle: string;
  timestamp: string;
  error?: string;
}

export interface ElectronAPI {
  captureScreenshot: () => Promise<{ image: string; timestamp: string } | { error: string; needsPermission: boolean } | null>;
  openSystemPreferences: () => Promise<void>;
  getActiveWindow: () => Promise<WindowData>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

// Expose screenshot capture to renderer
contextBridge.exposeInMainWorld('electronAPI', {
  captureScreenshot: () => ipcRenderer.invoke('capture-screenshot'),
  openSystemPreferences: () => ipcRenderer.invoke('open-system-preferences'),
  getActiveWindow: () => ipcRenderer.invoke('get-active-window'),
});
