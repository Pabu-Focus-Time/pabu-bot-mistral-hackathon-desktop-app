import { useState, useEffect, useCallback, useRef } from 'react';
import { captureAndAnalyzeSmart, SmartAnalysisResult } from './screenshot';
import { startDataCollection, stopDataCollection, collectFocusData, WindowData, ActivityData } from './dataCollector';

export interface FocusState {
  focus_state: 'focused' | 'distracted' | 'unknown' | 'paused';
  confidence: number;
  reason: unknown;
  timestamp?: string;
  source?: 'desktop' | 'robot';
}

export interface ContentChangeInfo {
  contentChanged: boolean;
  similarityScore: number;
  analysisSource: 'llm' | 'cached' | 'rule_based' | 'error';
  dinoAvailable: boolean;
}

export interface FocusDataPoint {
  time: number; // seconds since session start
  score: number; // 0-100
  state: 'focused' | 'distracted' | 'unknown' | 'paused';
  timestamp: string;
}

export interface WebSocketMessage {
  type: 'focus_update' | 'robot_focus_update' | 'robot_status' | 'notification' | 'reaction' | 'session_start' | 'session_stop' | 'task_context' | 'ping' | 'pong';
  source: 'desktop' | 'robot' | 'server';
  timestamp: string;
  payload: Record<string, unknown>;
}

const WS_URL = 'ws://127.0.0.1:9800/ws/desktop';
const API_BASE = 'http://127.0.0.1:9800';
const SCREENSHOT_INTERVAL = 1000; // 1 second

export function useFocusDetection() {
  const [isConnected, setIsConnected] = useState(false);
  const [desktopFocus, setDesktopFocus] = useState<FocusState>({
    focus_state: 'unknown',
    confidence: 0,
    reason: '',
    source: 'desktop',
  });
  const [robotFocus, setRobotFocus] = useState<FocusState>({
    focus_state: 'unknown',
    confidence: 0,
    reason: '',
    source: 'robot',
  });
  const [robotConnected, setRobotConnected] = useState(false);
  const [focusHistory, setFocusHistory] = useState<FocusDataPoint[]>([]);
  const [contentChangeInfo, setContentChangeInfo] = useState<ContentChangeInfo>({
    contentChanged: true,
    similarityScore: 1.0,
    analysisSource: 'llm',
    dinoAvailable: false,
  });
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [screenshotInterval, setScreenshotInterval] = useState<number | null>(null);
  const [autoDetect, setAutoDetect] = useState(true);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [showDistraction, setShowDistraction] = useState(false);
  const [distractionResources, setDistractionResources] = useState<Array<{ title: string; action: string }>>([]);
  const [isLoadingResources, setIsLoadingResources] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const sessionStartRef = useRef<number | null>(null);
  const prevFocusStateRef = useRef<string>('unknown');
  const taskContextRef = useRef<{
    task_name: string;
    task_description: string;
    current_todo: string;
    todos: Array<{ title: string; status: string }>;
    completed_count: number;
    total_count: number;
  } | null>(null);

  const addFocusDataPoint = useCallback((state: FocusState) => {
    if (sessionStartRef.current === null) return;
    
    const elapsed = Math.floor((Date.now() - sessionStartRef.current) / 1000);
    // Focus score: high = good. Invert when distracted.
    const score = state.focus_state === 'distracted'
      ? Math.round((1 - state.confidence) * 100)
      : state.focus_state === 'focused'
      ? Math.round(state.confidence * 100)
      : Math.round(state.confidence * 50);
    const point: FocusDataPoint = {
      time: elapsed,
      score,
      state: state.focus_state,
      timestamp: new Date().toISOString(),
    };
    
    setFocusHistory(prev => [...prev, point]);
  }, []);

  const clearFocusHistory = useCallback(() => {
    setFocusHistory([]);
    sessionStartRef.current = null;
  }, []);

  const startSession = useCallback(() => {
    sessionStartRef.current = Date.now();
    setFocusHistory([]);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    console.log('Connecting to WebSocket...');
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);

        if (message.type === 'focus_update' && message.source === 'robot') {
          const payload = message.payload as unknown as FocusState;
          console.log(`[Robot] Focus update: ${payload.focus_state} (${Math.round((payload.confidence || 0) * 100)}%) — ${payload.reason}`);
          setRobotFocus({
            focus_state: payload.focus_state,
            confidence: payload.confidence,
            reason: payload.reason,
            timestamp: message.timestamp,
            source: 'robot',
          });
        } else if (message.type === 'robot_focus_update') {
          const payload = message.payload as unknown as FocusState;
          console.log(`[Robot Camera] Focus update: ${payload.focus_state} (${Math.round((payload.confidence || 0) * 100)}%) — ${payload.reason}`);
          setRobotFocus({
            focus_state: payload.focus_state,
            confidence: payload.confidence,
            reason: payload.reason,
            timestamp: message.timestamp,
            source: 'robot',
          });
        } else if (message.type === 'robot_status') {
          const connected = !!(message.payload as Record<string, unknown>).connected;
          const count = (message.payload as Record<string, unknown>).robot_count as number;
          console.log(`[Robot] ${connected ? 'Connected' : 'Disconnected'} (${count} robot(s) online)`);
          setRobotConnected(connected);
        }
      } catch (err) {
        console.error('Failed to parse message:', err);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      wsRef.current = null;

      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    wsRef.current = ws;
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  const sendFocusUpdate = useCallback((focusState: FocusState) => {
    sendMessage({
      type: 'focus_update',
      source: 'desktop',
      timestamp: new Date().toISOString(),
      payload: focusState,
    });
  }, [sendMessage]);

  const sendReaction = useCallback((reaction: string, text?: string) => {
    sendMessage({
      type: 'reaction',
      source: 'desktop',
      timestamp: new Date().toISOString(),
      payload: { reaction, text },
    });
  }, [sendMessage]);


const setTaskContext = useCallback((
    task: { name: string; description: string; todos: Array<{ title: string; status: string }> } | null,
    currentTodo?: string | null,
  ) => {
    if (task) {
      const completed = task.todos.filter(t => t.status === 'completed').length;
      taskContextRef.current = {
        task_name: task.name,
        task_description: task.description,
        current_todo: currentTodo || task.todos.find(t => t.status !== 'completed')?.title || '',
        todos: task.todos.map(t => ({ title: t.title, status: t.status })),
        completed_count: completed,
        total_count: task.todos.length,
      };
      sendMessage({
        type: 'task_context',
        source: 'desktop',
        timestamp: new Date().toISOString(),
        payload: taskContextRef.current,
      });
    } else {
      taskContextRef.current = null;
    }
  }, [sendMessage]);

  const fetchResources = useCallback(async () => {
    if (!taskContextRef.current) return;
    setIsLoadingResources(true);
    try {
      const ctx = taskContextRef.current;
      const response = await fetch(`${API_BASE}/api/suggest-resources`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_name: ctx.task_name,
          task_description: ctx.task_description,
          current_todo: ctx.current_todo,
          todos: ctx.todos,
          // elapsed_seconds and estimated_minutes will be filled by the caller if available
        }),
      });
      if (response.ok) {
        const data = await response.json();
        setDistractionResources(data.resources || []);
      }
    } catch (err) {
      console.error('Failed to fetch resources:', err);
    } finally {
      setIsLoadingResources(false);
    }
  }, []);

  const dismissDistraction = useCallback(() => {
    setShowDistraction(false);
    setDistractionResources([]);
    setIsLoadingResources(false);
    // Exit focus-block mode: restore normal window
    const api = (window as any).electronAPI;
    if (api?.exitFocusBlock) {
      api.exitFocusBlock();
    }
  }, []);

  const analyzeScreenshot = useCallback(async () => {
    if (isAnalyzing) return;
    
    setIsAnalyzing(true);
    setPermissionError(null);
    try {
      // Collect window/activity data first for the smart endpoint
      const focusData = await collectFocusData();

      const windowPayload = focusData.window ? {
        app_name: focusData.window.appName || '',
        window_title: focusData.window.windowTitle || '',
      } : null;

      const activityPayload = focusData.activity ? {
        idle_seconds: focusData.activity.idleSeconds || 0,
        window_switch_count: focusData.activity.windowSwitchCount || 0,
        keypress_count: focusData.activity.keypressCount || 0,
      } : null;

      // Use the smart endpoint with DINOv3 pre-filter
      const result = await captureAndAnalyzeSmart(
        API_BASE,
        taskContextRef.current,
        windowPayload,
        activityPayload,
      );
      
      if (!result) {
        setIsAnalyzing(false);
        return;
      }

      if ('error' in result) {
        if (result.needsPermission) {
          setPermissionError(result.error);
        } else {
          console.error('Screenshot error:', result.error);
        }
        setIsAnalyzing(false);
        return;
      }

      // Update content change info
      setContentChangeInfo({
        contentChanged: result.content_changed,
        similarityScore: result.similarity_score,
        analysisSource: result.analysis_source,
        dinoAvailable: result.dino_available,
      });

      const focusState: FocusState = {
        focus_state: result.focus_state as 'focused' | 'distracted' | 'unknown' | 'paused',
        confidence: result.confidence,
        reason: result.reason,
        timestamp: new Date().toISOString(),
        source: 'desktop',
      };
      
      setDesktopFocus(focusState);

      // "paused" = user is viewing Pabu Focus itself, skip graph/distraction/reactions
      if (focusState.focus_state !== 'paused') {
        addFocusDataPoint(focusState);

        // Detect focus→distracted transition for overlay notification
        const prevState = prevFocusStateRef.current;
        if (
          focusState.focus_state === 'distracted' &&
          prevState !== 'distracted'
        ) {
          setShowDistraction(true);
          setDistractionResources([]);
          fetchResources();
          // Enter focus-block mode: fullscreen + always-on-top
          const api = (window as any).electronAPI;
          if (api?.enterFocusBlock) {
            api.enterFocusBlock();
          }
          // Also send native notification as fallback
          if (api?.sendNotification) {
            const taskName = taskContextRef.current?.task_name || 'your task';
            api.sendNotification(
              'You seem distracted',
              `Get back to ${taskName}`,
            );
          }
        }
        prevFocusStateRef.current = focusState.focus_state;

        sendMessage({
          type: 'focus_update',
          source: 'desktop',
          timestamp: new Date().toISOString(),
          payload: {
            ...focusState,
            content_changed: result.content_changed,
            similarity_score: result.similarity_score,
            analysis_source: result.analysis_source,
            window_data: focusData.window,
            activity_data: focusData.activity,
          },
        });

        if (focusState.focus_state === 'distracted') {
          const taskName = taskContextRef.current?.task_name;
          const currentTodo = taskContextRef.current?.current_todo;
          let robotText = 'Hey! You seem distracted. Let\'s get back to work!';
          if (taskName && currentTodo) {
            robotText = `Hey! You should be working on ${currentTodo}. Let's stay focused!`;
          } else if (taskName) {
            robotText = `Hey! You should be focusing on ${taskName}. Come on, let's go!`;
          }
          sendReaction('distraction', robotText);
        }
      }
    } catch (error) {
      console.error('Screenshot analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }, [isAnalyzing, sendFocusUpdate, sendReaction, sendMessage, addFocusDataPoint, fetchResources]);

  const startAutoDetection = useCallback(() => {
    if (screenshotInterval) return;
    startSession();
    // Reset DINOv3 state for fresh session
    fetch(`${API_BASE}/api/analyze-smart/reset`, { method: 'POST' }).catch(() => {});
    sendMessage({
      type: 'session_start',
      source: 'desktop',
      timestamp: new Date().toISOString(),
      payload: taskContextRef.current || {},
    });
    const interval = window.setInterval(analyzeScreenshot, SCREENSHOT_INTERVAL);
    setScreenshotInterval(interval);
    setAutoDetect(true);
    analyzeScreenshot();
  }, [screenshotInterval, analyzeScreenshot, startSession, sendMessage]);

  const stopAutoDetection = useCallback(() => {
    if (screenshotInterval) {
      clearInterval(screenshotInterval);
      setScreenshotInterval(null);
    }
    setAutoDetect(false);
    sendMessage({
      type: 'session_stop',
      source: 'desktop',
      timestamp: new Date().toISOString(),
      payload: {},
    });
  }, [screenshotInterval, sendMessage]);

  useEffect(() => {
    startDataCollection();
    connect();
    return () => {
      stopAutoDetection();
      stopDataCollection();
      disconnect();
    };
  }, []);

  const openSystemPreferences = useCallback(() => {
    const api = (window as any).electronAPI;
    if (api?.openSystemPreferences) {
      api.openSystemPreferences();
    } else {
      window.open('x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture');
    }
  }, []);

  return {
    isConnected,
    desktopFocus,
    robotFocus,
    robotConnected,
    focusHistory,
    contentChangeInfo,
    showDistraction,
    distractionResources,
    isLoadingResources,
    isAnalyzing,
    autoDetect,
    permissionError,
    sendReaction,
    startAutoDetection,
    stopAutoDetection,
    clearFocusHistory,
    setTaskContext,
    dismissDistraction,
    connect,
    disconnect,
    openSystemPreferences,
  };
}
