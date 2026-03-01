import { useState, useEffect, useCallback, useRef } from 'react';
import { captureAndAnalyze } from './screenshot';
import { startDataCollection, stopDataCollection, collectFocusData, WindowData, ActivityData } from './dataCollector';

export interface FocusState {
  focus_state: 'focused' | 'distracted' | 'unknown';
  confidence: number;
  reason: string | Record<string, unknown>;
  timestamp?: string;
  source?: 'desktop' | 'robot';
}

export interface FocusDataPoint {
  time: number; // seconds since session start
  score: number; // 0-100
  state: 'focused' | 'distracted' | 'unknown';
  timestamp: string;
}

export interface WebSocketMessage {
  type: 'focus_update' | 'notification' | 'reaction' | 'ping' | 'pong';
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
  const [focusHistory, setFocusHistory] = useState<FocusDataPoint[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [screenshotInterval, setScreenshotInterval] = useState<number | null>(null);
  const [autoDetect, setAutoDetect] = useState(true);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const sessionStartRef = useRef<number | null>(null);

  const addFocusDataPoint = useCallback((state: FocusState) => {
    if (sessionStartRef.current === null) return;
    
    const elapsed = Math.floor((Date.now() - sessionStartRef.current) / 1000);
    const point: FocusDataPoint = {
      time: elapsed,
      score: Math.round(state.confidence * 100),
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
          setRobotFocus({
            focus_state: payload.focus_state,
            confidence: payload.confidence,
            reason: payload.reason,
            timestamp: message.timestamp,
            source: 'robot',
          });
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

  const analyzeScreenshot = useCallback(async () => {
    if (isAnalyzing) return;
    
    setIsAnalyzing(true);
    setPermissionError(null);
    try {
      const result = await captureAndAnalyze(API_BASE);
      
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

      const focusData = await collectFocusData();

      const focusState: FocusState = {
        focus_state: result.focus_state as 'focused' | 'distracted' | 'unknown',
        confidence: result.confidence,
        reason: result.reason,
        timestamp: new Date().toISOString(),
        source: 'desktop',
      };
      
      setDesktopFocus(focusState);
      addFocusDataPoint(focusState);
      
      sendMessage({
        type: 'focus_update',
        source: 'desktop',
        timestamp: new Date().toISOString(),
        payload: {
          ...focusState,
          vision_analysis: {
            focus_state: result.focus_state,
            confidence: result.confidence,
            reason: result.reason,
          },
          window_data: focusData.window,
          activity_data: focusData.activity,
        },
      });
      
      if (focusState.focus_state === 'distracted') {
        sendReaction('warning', 'Desktop detected distraction!');
      }
    } catch (error) {
      console.error('Screenshot analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }, [isAnalyzing, sendFocusUpdate, sendReaction, sendMessage, addFocusDataPoint]);

  const startAutoDetection = useCallback(() => {
    if (screenshotInterval) return;
    startSession();
    const interval = window.setInterval(analyzeScreenshot, SCREENSHOT_INTERVAL);
    setScreenshotInterval(interval);
    setAutoDetect(true);
    analyzeScreenshot();
  }, [screenshotInterval, analyzeScreenshot, startSession]);

  const stopAutoDetection = useCallback(() => {
    if (screenshotInterval) {
      clearInterval(screenshotInterval);
      setScreenshotInterval(null);
    }
    setAutoDetect(false);
  }, [screenshotInterval]);

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
    focusHistory,
    isAnalyzing,
    autoDetect,
    permissionError,
    sendReaction,
    startAutoDetection,
    stopAutoDetection,
    clearFocusHistory,
    connect,
    disconnect,
    openSystemPreferences,
  };
}
