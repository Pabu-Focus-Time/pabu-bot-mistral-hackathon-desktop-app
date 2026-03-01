import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Square, Clock, Target, Activity, RefreshCw, Minus } from 'lucide-react';
import { tokens, getFocusColor, getFocusBg } from '../styles/tokens';
import { FocusState, FocusDataPoint, ContentChangeInfo } from '../websocket';
import FocusGraph from './FocusGraph';

interface FocusSessionProps {
  isActive: boolean;
  focusState: FocusState;
  robotFocus: FocusState;
  focusHistory: FocusDataPoint[];
  contentChangeInfo: ContentChangeInfo;
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
  robotFocus,
  focusHistory,
  contentChangeInfo,
  currentTaskName,
  currentTodoName,
  sessionDuration,
  windowData,
  activityData,
  onStart,
  onStop,
}) => {
  const formatDuration = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Focus score: high = good. When distracted, invert confidence so it shows LOW.
  // e.g., "distracted at 95% confidence" → focus score = 5%
  const rawConfidence = focusState.confidence;
  const focusScore = focusState.focus_state === 'distracted'
    ? Math.round((1 - rawConfidence) * 100)
    : focusState.focus_state === 'focused'
    ? Math.round(rawConfidence * 100)
    : Math.round(rawConfidence * 50); // unknown → mid-range
  const stateColor = getFocusColor(focusState.focus_state);

  // Not active state - compact start button
  if (!isActive) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        style={{ marginBottom: '24px' }}
      >
        <div style={{
          background: tokens.colors.surface,
          borderRadius: tokens.radius.xl,
          border: `1px solid ${tokens.colors.border}`,
          padding: '32px',
          textAlign: 'center',
        }}>
          <div style={{
            width: '56px',
            height: '56px',
            borderRadius: '50%',
            background: tokens.colors.accentMuted,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 16px',
          }}>
            <Target size={24} style={{ color: tokens.colors.accent }} />
          </div>

          <div style={{
            fontSize: tokens.typography.fontSize.lg,
            fontWeight: tokens.typography.fontWeight.semibold,
            color: tokens.colors.text,
            marginBottom: '6px',
          }}>
            Ready to focus?
          </div>
          <div style={{
            fontSize: tokens.typography.fontSize.md,
            color: tokens.colors.textSecondary,
            marginBottom: '24px',
          }}>
            Start a session to track your focus in real-time
          </div>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onStart}
            style={{
              padding: '10px 32px',
              background: tokens.colors.accent,
              color: '#fff',
              border: 'none',
              borderRadius: tokens.radius.md,
              fontSize: tokens.typography.fontSize.base,
              fontWeight: tokens.typography.fontWeight.medium,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              transition: tokens.transitions.normal,
              fontFamily: tokens.typography.fontFamily,
            }}
          >
            <Play size={16} fill="currentColor" />
            Start Focus Session
          </motion.button>
        </div>
      </motion.div>
    );
  }

  // Active session state
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      style={{ marginBottom: '24px' }}
    >
      <div style={{
        background: tokens.colors.surface,
        borderRadius: tokens.radius.xl,
        border: `1px solid ${tokens.colors.border}`,
        overflow: 'hidden',
      }}>
        {/* Session Header */}
        <div style={{
          padding: '20px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: `1px solid ${tokens.colors.border}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {/* Focus State Indicator */}
            <div style={{ position: 'relative' }}>
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                background: getFocusBg(focusState.focus_state),
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: `2px solid ${stateColor}`,
              }}>
                <Activity size={18} style={{ color: stateColor }} />
              </div>
              {/* Pulse dot */}
              <div className="animate-pulse" style={{
                position: 'absolute',
                top: '0px',
                right: '0px',
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: stateColor,
                border: `2px solid ${tokens.colors.surface}`,
              }} />
            </div>

            <div>
              <div style={{
                fontSize: tokens.typography.fontSize.xs,
                color: tokens.colors.textTertiary,
                textTransform: 'uppercase',
                letterSpacing: tokens.typography.letterSpacing.wider,
                marginBottom: '2px',
              }}>
                Focus Session
              </div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
              }}>
                <span style={{
                  fontSize: tokens.typography.fontSize.lg,
                  fontWeight: tokens.typography.fontWeight.semibold,
                  color: tokens.colors.text,
                  textTransform: 'capitalize',
                }}>
                  {focusState.focus_state === 'unknown' ? 'Analyzing...' : focusState.focus_state}
                </span>
                <span style={{
                  fontSize: tokens.typography.fontSize.sm,
                  color: stateColor,
                  fontWeight: tokens.typography.fontWeight.medium,
                }}>
                  {focusScore}%
                </span>
                {/* Content Change Badge */}
                <ContentBadge contentChangeInfo={contentChangeInfo} />
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {/* Timer */}
            <div style={{ textAlign: 'right' }}>
              <div style={{
                fontSize: tokens.typography.fontSize.xxl,
                fontWeight: tokens.typography.fontWeight.bold,
                color: tokens.colors.text,
                fontFamily: tokens.typography.fontMono,
                fontVariantNumeric: 'tabular-nums',
                letterSpacing: tokens.typography.letterSpacing.tight,
                lineHeight: 1,
              }}>
                {formatDuration(sessionDuration)}
              </div>
              {currentTaskName && (
                <div style={{
                  fontSize: tokens.typography.fontSize.xs,
                  color: tokens.colors.textTertiary,
                  marginTop: '4px',
                  maxWidth: '160px',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  textAlign: 'right',
                }}>
                  {currentTaskName}
                </div>
              )}
            </div>

            {/* Stop Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onStop}
              style={{
                width: '36px',
                height: '36px',
                borderRadius: tokens.radius.md,
                background: tokens.colors.dangerMuted,
                color: tokens.colors.danger,
                border: `1px solid rgba(239, 68, 68, 0.2)`,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: tokens.transitions.fast,
              }}
            >
              <Square size={14} fill="currentColor" />
            </motion.button>
          </div>
        </div>

        {/* Focus Graph */}
        <div style={{ padding: '16px 20px 20px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '12px',
          }}>
            <span style={{
              fontSize: tokens.typography.fontSize.xs,
              color: tokens.colors.textTertiary,
              textTransform: 'uppercase',
              letterSpacing: tokens.typography.letterSpacing.wider,
            }}>
              Focus Timeline
            </span>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <div style={{
                  width: '8px',
                  height: '2px',
                  borderRadius: '1px',
                  background: tokens.colors.accent,
                }} />
                <span style={{
                  fontSize: tokens.typography.fontSize.xs,
                  color: tokens.colors.textTertiary,
                }}>
                  Score
                </span>
              </div>
              {focusHistory.length > 0 && (
                <span style={{
                  fontSize: tokens.typography.fontSize.xs,
                  color: tokens.colors.textTertiary,
                  fontFamily: tokens.typography.fontMono,
                }}>
                  avg {Math.round(focusHistory.reduce((s, p) => s + p.score, 0) / focusHistory.length)}%
                </span>
              )}
            </div>
          </div>

          <FocusGraph data={focusHistory} height={180} />
        </div>

        {/* Session Details Row */}
        <div style={{
          padding: '0 20px 20px',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr 1fr',
          gap: '8px',
        }}>
          <DetailChip label="Desktop" value={focusState.focus_state} state={focusState.focus_state} />
          <DetailChip label="Robot" value={robotFocus.focus_state} state={robotFocus.focus_state} />
          <DetailChip label="Confidence" value={`${focusScore}%`} />
          <DetailChip
            label="Source"
            value={
              contentChangeInfo.analysisSource === 'cached' ? 'Cached' :
              contentChangeInfo.analysisSource === 'llm' ? 'AI' :
              contentChangeInfo.analysisSource === 'rule_based' ? 'Rules' :
              'N/A'
            }
            subtitle={
              contentChangeInfo.dinoAvailable
                ? `${Math.round((1 - contentChangeInfo.similarityScore) * 100)}% similar`
                : undefined
            }
          />
        </div>
      </div>
    </motion.div>
  );
};

// -- Content Change Badge --
interface ContentBadgeProps {
  contentChangeInfo: ContentChangeInfo;
}

const ContentBadge: React.FC<ContentBadgeProps> = ({ contentChangeInfo }) => {
  const { contentChanged, similarityScore, dinoAvailable } = contentChangeInfo;

  if (!dinoAvailable) return null;

  const isChanged = contentChanged;
  const bgColor = isChanged ? tokens.colors.accentMuted : tokens.colors.surfaceSecondary;
  const textColor = isChanged ? tokens.colors.accent : tokens.colors.textTertiary;
  const borderColor = isChanged ? 'rgba(82,139,255,0.25)' : tokens.colors.borderLight;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={isChanged ? 'changed' : 'same'}
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        transition={{ duration: 0.2 }}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '4px',
          padding: '2px 8px',
          borderRadius: tokens.radius.full,
          background: bgColor,
          border: `1px solid ${borderColor}`,
          fontSize: tokens.typography.fontSize.xs,
          color: textColor,
          fontWeight: tokens.typography.fontWeight.medium,
          whiteSpace: 'nowrap',
        }}
      >
        {isChanged ? (
          <RefreshCw size={10} style={{ color: textColor }} />
        ) : (
          <Minus size={10} style={{ color: textColor }} />
        )}
        {isChanged ? 'Changed' : 'Same'}
      </motion.div>
    </AnimatePresence>
  );
};

// -- Detail Chip --
interface DetailChipProps {
  label: string;
  value: string;
  state?: string;
  subtitle?: string;
}

const DetailChip: React.FC<DetailChipProps> = ({ label, value, state, subtitle }) => {
  const stateColor = state
    ? state === 'focused'
      ? tokens.colors.success
      : state === 'distracted'
      ? tokens.colors.danger
      : tokens.colors.textTertiary
    : tokens.colors.textSecondary;

  return (
    <div style={{
      padding: '10px 12px',
      background: tokens.colors.surfaceSecondary,
      borderRadius: tokens.radius.md,
      border: `1px solid ${tokens.colors.borderLight}`,
    }}>
      <div style={{
        fontSize: tokens.typography.fontSize.xs,
        color: tokens.colors.textTertiary,
        marginBottom: '4px',
      }}>
        {label}
      </div>
      <div style={{
        fontSize: tokens.typography.fontSize.md,
        fontWeight: tokens.typography.fontWeight.medium,
        color: state ? stateColor : tokens.colors.text,
        textTransform: 'capitalize',
      }}>
        {value}
      </div>
      {subtitle && (
        <div style={{
          fontSize: tokens.typography.fontSize.xs,
          color: tokens.colors.textTertiary,
          marginTop: '2px',
          fontFamily: tokens.typography.fontMono,
        }}>
          {subtitle}
        </div>
      )}
    </div>
  );
};

export default FocusSession;
