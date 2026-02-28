// Screenshot functionality - uses preload API

export interface ScreenshotResult {
  image: string;
  timestamp: string;
}

export async function captureScreenshot(): Promise<ScreenshotResult | null> {
  try {
    // Use the exposed API from preload
    const api = (window as any).electronAPI;
    if (api?.captureScreenshot) {
      const result = await api.captureScreenshot();
      if (result) return result;
    }
    return null;
  } catch (error) {
    console.error('Screenshot capture failed:', error);
    return null;
  }
}

export async function captureAndAnalyze(apiBase: string): Promise<{
  focus_state: string;
  confidence: number;
  reason: string;
} | null> {
  const screenshot = await captureScreenshot();
  if (!screenshot) return null;

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
