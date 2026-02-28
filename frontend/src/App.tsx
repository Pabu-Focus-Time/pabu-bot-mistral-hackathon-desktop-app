import React from 'react';
import { useFocusDetection } from './websocket';

const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%)',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    padding: '40px 20px',
    color: '#1d1d1f',
  },
  card: {
    background: 'rgba(255, 255, 255, 0.9)',
    backdropFilter: 'blur(20px)',
    borderRadius: '20px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.08), 0 4px 12px rgba(0, 0, 0, 0.04)',
    padding: '32px',
    maxWidth: '480px',
    margin: '0 auto',
  },
  header: {
    textAlign: 'center' as const,
    marginBottom: '32px',
  },
  logo: {
    fontSize: '48px',
    marginBottom: '8px',
  },
  title: {
    fontSize: '28px',
    fontWeight: 600,
    marginBottom: '4px',
    letterSpacing: '-0.5px',
  },
  subtitle: {
    fontSize: '15px',
    color: '#86868b',
    fontWeight: 400,
  },
  statusBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '6px 14px',
    borderRadius: '20px',
    fontSize: '13px',
    fontWeight: 500,
    marginTop: '16px',
  },
  focusCard: {
    padding: '20px',
    borderRadius: '16px',
    marginBottom: '16px',
    transition: 'all 0.3s ease',
  },
  focusTitle: {
    fontSize: '13px',
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    color: '#86868b',
    marginBottom: '12px',
  },
  focusState: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  focusIcon: {
    width: '48px',
    height: '48px',
    borderRadius: '12px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '24px',
  },
  focusInfo: {
    flex: 1,
  },
  focusLabel: {
    fontSize: '20px',
    fontWeight: 600,
    textTransform: 'capitalize' as const,
  },
  confidence: {
    fontSize: '13px',
    color: '#86868b',
    marginTop: '2px',
  },
  reason: {
    fontSize: '13px',
    color: '#86868b',
    marginTop: '8px',
    fontStyle: 'italic',
  },
  divider: {
    height: '1px',
    background: 'linear-gradient(90deg, transparent, #e5e5e5, transparent)',
    margin: '24px 0',
  },
  button: {
    width: '100%',
    padding: '14px 24px',
    borderRadius: '12px',
    border: 'none',
    fontSize: '15px',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    marginTop: '8px',
  },
  stats: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '12px',
    marginTop: '24px',
  },
  statItem: {
    textAlign: 'center' as const,
    padding: '16px',
    background: '#f5f5f7',
    borderRadius: '12px',
  },
  statValue: {
    fontSize: '24px',
    fontWeight: 700,
  },
  statLabel: {
    fontSize: '12px',
    color: '#86868b',
    marginTop: '4px',
  },
  timestamp: {
    fontSize: '11px',
    color: '#a1a1a6',
    textAlign: 'center' as const,
    marginTop: '16px',
  },
  permissionBanner: {
    background: '#fff3cd',
    border: '1px solid #ffc107',
    borderRadius: '12px',
    padding: '16px',
    marginBottom: '24px',
    textAlign: 'center' as const,
  },
  permissionTitle: {
    fontSize: '15px',
    fontWeight: 600,
    color: '#856404',
    marginBottom: '8px',
  },
  permissionText: {
    fontSize: '13px',
    color: '#856404',
    marginBottom: '12px',
  },
  permissionButton: {
    background: '#ffc107',
    color: '#856404',
    border: 'none',
    borderRadius: '8px',
    padding: '8px 16px',
    fontSize: '13px',
    fontWeight: 600,
    cursor: 'pointer',
  },
};

function getFocusColor(state: string) {
  switch (state) {
    case 'focused': return '#34C759';
    case 'distracted': return '#FF3B30';
    default: return '#8E8E93';
  }
}

function getFocusIcon(state: string) {
  switch (state) {
    case 'focused': return '🎯';
    case 'distracted': return '⚠️';
    default: return '❓';
  }
}

const App: React.FC = () => {
  const {
    isConnected,
    desktopFocus,
    robotFocus,
    isAnalyzing,
    autoDetect,
    permissionError,
    startAutoDetection,
    stopAutoDetection,
    openSystemPreferences,
  } = useFocusDetection();

  const desktopColor = getFocusColor(desktopFocus.focus_state);
  const robotColor = getFocusColor(robotFocus.focus_state);

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.header}>
          <div style={styles.logo}>🤖</div>
          <h1 style={styles.title}>Pabu Focus</h1>
          <p style={styles.subtitle}>Your AI-powered productivity assistant</p>
          
          <div style={{
            ...styles.statusBadge,
            background: isConnected ? '#dafbe1' : '#ffe5e5',
            color: isConnected ? '#1a7f37' : '#c42b1c',
          }}>
            <span style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: isConnected ? '#34C759' : '#FF3B30',
            }} />
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
        </div>

        {/* Permission Error Banner */}
        {permissionError && (
          <div style={styles.permissionBanner}>
            <div style={styles.permissionTitle}>⚠️ Screen Recording Permission Required</div>
            <div style={styles.permissionText}>
              {permissionError}. Please grant access in System Settings.
            </div>
            <button 
              style={styles.permissionButton}
              onClick={openSystemPreferences}
            >
              Open System Settings
            </button>
          </div>
        )}

        {/* Desktop Focus */}
        <div style={styles.focusCard}>
          <div style={styles.focusTitle}>🖥️ Desktop</div>
          <div style={styles.focusState}>
            <div style={{
              ...styles.focusIcon,
              background: `${desktopColor}20`,
            }}>
              {getFocusIcon(desktopFocus.focus_state)}
            </div>
            <div style={styles.focusInfo}>
              <div style={{...styles.focusLabel, color: desktopColor}}>
                {desktopFocus.focus_state}
              </div>
              <div style={styles.confidence}>
                Confidence: {Math.round(desktopFocus.confidence * 100)}%
              </div>
            </div>
          </div>
          {desktopFocus.reason && (
            <div style={styles.reason}>{typeof desktopFocus.reason === 'string' ? desktopFocus.reason : JSON.stringify(desktopFocus.reason)}</div>
          )}
        </div>

        {/* Robot Focus */}
        <div style={styles.focusCard}>
          <div style={styles.focusTitle}>🤖 Reachy Mini</div>
          <div style={styles.focusState}>
            <div style={{
              ...styles.focusIcon,
              background: `${robotColor}20`,
            }}>
              {getFocusIcon(robotFocus.focus_state)}
            </div>
            <div style={styles.focusInfo}>
              <div style={{...styles.focusLabel, color: robotColor}}>
                {robotFocus.focus_state}
              </div>
              <div style={styles.confidence}>
                Confidence: {Math.round(robotFocus.confidence * 100)}%
              </div>
            </div>
          </div>
          {robotFocus.reason && (
            <div style={styles.reason}>{typeof robotFocus.reason === 'string' ? robotFocus.reason : JSON.stringify(robotFocus.reason)}</div>
          )}
        </div>

        <div style={styles.divider} />

        {/* Controls */}
        <button
          style={{
            ...styles.button,
            background: autoDetect ? '#ff375f' : '#007aff',
            color: 'white',
          }}
          onClick={autoDetect ? stopAutoDetection : startAutoDetection}
        >
          {isAnalyzing ? '⏳ Analyzing...' : autoDetect ? '⏹ Stop Detection' : '▶ Start Detection'}
        </button>

        <div style={styles.stats}>
          <div style={styles.statItem}>
            <div style={{...styles.statValue, color: desktopColor}}>
              {autoDetect ? '●' : '○'}
            </div>
            <div style={styles.statLabel}>Auto Scan</div>
          </div>
          <div style={styles.statItem}>
            <div style={{...styles.statValue, color: isAnalyzing ? '#007aff' : '#86868b'}}>
              {isAnalyzing ? '1s' : '--'}
            </div>
            <div style={styles.statLabel}>Scan Interval</div>
          </div>
        </div>

        <div style={styles.timestamp}>
          Last update: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

export default App;
