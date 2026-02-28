import { contextBridge, ipcRenderer } from 'electron';

// Expose screenshot capture to renderer
contextBridge.exposeInMainWorld('electronAPI', {
  captureScreenshot: () => ipcRenderer.invoke('capture-screenshot'),
  openSystemPreferences: () => ipcRenderer.invoke('open-system-preferences'),
});
