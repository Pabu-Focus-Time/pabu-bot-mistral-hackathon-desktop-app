import { ActivityData, initActivityMonitor, getActivityData } from './activityMonitor';

export type { ActivityData };

export interface WindowData {
  appName: string;
  windowTitle: string;
  timestamp: string;
}

export interface ScreenshotData {
  image: string;
  timestamp: string;
}

export interface FocusData {
  timestamp: string;
  screenshot: ScreenshotData | null;
  window: WindowData | null;
  activity: ActivityData | null;
}

let cleanupActivityMonitor: (() => void) | null = null;

export const startDataCollection = () => {
  cleanupActivityMonitor = initActivityMonitor();
};

export const stopDataCollection = () => {
  if (cleanupActivityMonitor) {
    cleanupActivityMonitor();
    cleanupActivityMonitor = null;
  }
};

export const collectFocusData = async (): Promise<FocusData> => {
  // Get window data
  let windowData: WindowData | null = null;
  try {
    const api = (window as any).electronAPI;
    if (api?.getActiveWindow) {
      const result = await api.getActiveWindow();
      if (result && !result.error) {
        windowData = {
          appName: result.appName,
          windowTitle: result.windowTitle,
          timestamp: result.timestamp,
        };
      }
    }
  } catch (error) {
    console.error('Window tracking error:', error);
  }

  // Get activity data
  const activityData: ActivityData | null = windowData
    ? getActivityData(windowData.appName)
    : null;

  return {
    timestamp: new Date().toISOString(),
    screenshot: null, // Screenshot is handled separately in websocket.ts
    window: windowData,
    activity: activityData,
  };
};
