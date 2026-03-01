// Screenshot functionality - uses preload API

export interface ScreenshotResult {
  image: string;
  timestamp: string;
}

export interface ScreenshotError {
  error: string;
  needsPermission: boolean;
}

export interface SmartAnalysisResult {
  focus_state: string;
  confidence: number;
  reason: string;
  content_changed: boolean;
  similarity_score: number;
  analysis_source: 'llm' | 'cached' | 'rule_based' | 'error';
  dino_available: boolean;
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

/**
 * Capture screenshot and analyze via the smart endpoint.
 * Uses DINOv3 pre-filter to skip LLM calls on unchanged screens.
 */
export async function captureAndAnalyzeSmart(
  apiBase: string,
  taskContext?: { task_name: string; current_todo: string } | null,
  windowData?: { app_name: string; window_title: string } | null,
  activityData?: { idle_seconds: number; window_switch_count: number; keypress_count: number } | null,
): Promise<SmartAnalysisResult | { error: string; needsPermission: boolean } | null> {
  const screenshot = await captureScreenshot();
  
  if (!screenshot) return null;
  
  if ('error' in screenshot) {
    return screenshot;
  }

  try {
    const body: Record<string, unknown> = { image: screenshot.image };
    if (taskContext) body.task_context = taskContext;
    if (windowData) body.window_data = windowData;
    if (activityData) body.activity_data = activityData;

    const response = await fetch(`${apiBase}/api/analyze-smart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      console.error('Smart analysis failed:', response.status);
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error('Smart analysis error:', error);
    return null;
  }
}

/**
 * Legacy: Capture and analyze via original /api/analyze endpoint.
 * Kept for backward compatibility.
 */
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
