// Screenshot functionality - uses preload API

export interface ScreenshotResult {
  image: string;
  timestamp: string;
}

export interface ScreenshotError {
  error: string;
  needsPermission: boolean;
}

export async function captureScreenshot(): Promise<ScreenshotResult | ScreenshotError | null> {
  try {
    const api = (window as any).electronAPI;
    if (api?.captureScreenshot) {
      const result = await api.captureScreenshot();
      if (result) return result;
      if (result === null) {
        return { error: 'Screen Recording permission denied', needsPermission: true };
      }
    }
    return null;
  } catch (error) {
    console.error('Screenshot capture failed:', error);
    return { error: String(error), needsPermission: false };
  }
}

export async function captureAndAnalyze(apiBase: string): Promise<{
  focus_state: string;
  confidence: number;
  reason: string;
} | { error: string; needsPermission: boolean } | null> {
  const screenshot = await captureScreenshot();
  
  if (!screenshot) return null;
  
  if ('error' in screenshot) {
    return screenshot;
  }

  try {
    const response = await fetch(`${apiBase}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: screenshot.image }),
    });

    if (!response.ok) {
      console.error('Analysis failed:', response.status);
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error('Analysis error:', error);
    return null;
  }
}
