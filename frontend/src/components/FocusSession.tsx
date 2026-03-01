import React from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, Clock, Target, Monitor, Zap } from 'lucide-react';
import { tokens, getFocusColor, getFocusBg } from '../styles/tokens';
import { FocusState } from '../websocket';

interface FocusSessionProps {
  isActive: boolean;
  focusState: FocusState;
  currentTaskName: string | null;
  currentTodoName: string | null;
  sessionDuration: number;
  windowData: {
    appName: string;
    windowTitle: string;
  } | null;
  activityData: {
    idleSeconds: number;
    windowSwitchCount: number;
  } | null;
  onStart: () => void;
  onStop: () => void;
}

const FocusSession: React.FC<FocusSessionProps> = ({
  isActive,
  focusState,
  currentTaskName,
  currentTodoName,
  sessionDuration,
  windowData,
  activityData,
  onStart,
  onStop,
}) => {
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getFocusLabel = (state: string) => {
    switch (state) {
      case 'focused': return 'Focused';
      case 'distracted': return 'Distracted';
      default: return 'Unknown';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      style={{
        background: tokens.colors.surface,
        borderRadius: tokens.radius.lg,
        boxShadow: tokens.shadows.md,
        border: `1px solid ${isActive ? tokens.colors.accent : tokens.colors.border}`,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: tokens.spacing.lg,
          borderBottom: `1px solid ${tokens.colors.border}`,
          background: isActive ? getFocusBg(focusState.focus_state) : tokens.colors.surfaceSecondary,
          transition: tokens.transitions.normal,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacing.md }}>
            <div
              style={{
                width: '48px',
                height: '48px',
                borderRadius: '50%',
                background: isActive ? getFocusColor(focusState.focus_state) : tokens.colors.textTertiary,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Target size={24} color="white" />
            </div>
            <div>
              <div
                style={{
                  fontSize: tokens.typography.fontSize.xs,
                  fontWeight: tokens.typography.fontWeight.medium,
                  color: tokens.colors.textSecondary,
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}
              >
                Focus Session
              </div>
              <div
                style={{
                  fontSize: tokens.typography.fontSize.lg,
                  fontWeight: tokens.typography.fontWeight.semibold,
                  color: tokens.colors.text,
                }}
              >
                {isActive ? getFocusLabel(focusState.focus_state) : 'Not Active'}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacing.md }}>
            <div style={{ textAlign: 'right' }}>
              <div
                style={{
                  fontSize: tokens.typography.fontSize.xxl,
                  fontWeight: tokens.typography.fontWeight.bold,
                  color: tokens.colors.text,
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {formatDuration(sessionDuration)}
              </div>
              <div
                style={{
                  fontSize: tokens.typography.fontSize.xs,
                  color: tokens.colors.textSecondary,
                }}
              >
                Session Duration
              </div>
            </div>
          </div>
        </div>
      </div>

      {isActive && (
        <div style={{ padding: tokens.spacing.lg }}>
          {currentTaskName && (
            <div
              style={{
                padding: tokens.spacing.md,
                background: tokens.colors.surfaceSecondary,
                borderRadius: tokens.radius.md,
                marginBottom: tokens.spacing.md,
              }}
            >
              <div
                style={{
                  fontSize: tokens.typography.fontSize.xs,
                  fontWeight: tokens.typography.fontWeight.medium,
                  color: tokens.colors.textSecondary,
                  marginBottom: tokens.spacing.xs,
                }}
              >
                CURRENT TASK
              </div>
              <div
                style={{
                  fontSize: tokens.typography.fontSize.md,
                  fontWeight: tokens.typography.fontWeight.medium,
                  color: tokens.colors.text,
                }}
              >
                {currentTaskName}
              </div>
              {currentTodoName && (
                <div
                  style={{
                    fontSize: tokens.typography.fontSize.sm,
                    color: tokens.colors.accent,
                    marginTop: tokens.spacing.xs,
                  }}
                >
                  → {currentTodoName}
                </div>
              )}
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: tokens.spacing.md }}>
            <div
              style={{
                padding: tokens.spacing.md,
                background: tokens.colors.surfaceSecondary,
                borderRadius: tokens.radius.md,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacing.xs, marginBottom: tokens.spacing.xs }}>
                <Monitor size={14} style={{ color: tokens.colors.textSecondary }} />
                <span
                  style={{
                    fontSize: tokens.typography.fontSize.xs,
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: tokens.colors.textSecondary,
                  }}
                >
                  Active Window
                </span>
              </div>
              <div
                style={{
                  fontSize: tokens.typography.fontSize.sm,
                  fontWeight: tokens.typography.fontWeight.medium,
                  color: tokens.colors.text,
                }}
              >
                {windowData?.appName || 'Unknown'}
              </div>
              {windowData?.windowTitle && (
                <div
                  style={{
                    fontSize: tokens.typography.fontSize.xs,
                    color: tokens.colors.textTertiary,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {windowData.windowTitle}
                </div>
              )}
            </div>

            <div
              style={{
                padding: tokens.spacing.md,
                background: tokens.colors.surfaceSecondary,
                borderRadius: tokens.radius.md,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacing.xs, marginBottom: tokens.spacing.xs }}>
                <Zap size={14} style={{ color: tokens.colors.textSecondary }} />
                <span
                  style={{
                    fontSize: tokens.typography.fontSize.xs,
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: tokens.colors.textSecondary,
                  }}
                >
                  Activity
                </span>
              </div>
              <div
                style={{
                  fontSize: tokens.typography.fontSize.sm,
                  fontWeight: tokens.typography.fontWeight.medium,
                  color: tokens.colors.text,
                }}
              >
                {activityData?.idleSeconds ? `${activityData.idleSeconds}s idle` : 'Active'}
              </div>
              <div
                style={{
                  fontSize: tokens.typography.fontSize.xs,
                  color: tokens.colors.textTertiary,
                }}
              >
                {activityData?.windowSwitchCount || 0} switches/min
              </div>
            </div>
          </div>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onStop}
            style={{
              width: '100%',
              marginTop: tokens.spacing.lg,
              padding: tokens.spacing.md,
              background: tokens.colors.danger,
              color: tokens.colors.surface,
              border: 'none',
              borderRadius: tokens.radius.md,
              fontSize: tokens.typography.fontSize.md,
              fontWeight: tokens.typography.fontWeight.medium,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: tokens.spacing.sm,
            }}
          >
            <Pause size={18} />
            End Session
          </motion.button>
        </div>
      )}

      {!isActive && (
        <div style={{ padding: tokens.spacing.lg }}>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onStart}
            style={{
              width: '100%',
              padding: tokens.spacing.md,
              background: tokens.colors.accent,
              color: tokens.colors.surface,
              border: 'none',
              borderRadius: tokens.radius.md,
              fontSize: tokens.typography.fontSize.md,
              fontWeight: tokens.typography.fontWeight.medium,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: tokens.spacing.sm,
            }}
          >
            <Play size={18} />
            Start Focus Session
          </motion.button>
        </div>
      )}
    </motion.div>
  );
};

export default FocusSession;
