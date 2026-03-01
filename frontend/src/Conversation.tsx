import React, { useCallback, useEffect, useRef } from 'react';
import { useConversation } from '@elevenlabs/react';

const AGENT_ID = 'agent_9801kjk2p0kkf5wsrq1cc2byvyqx';

interface ConversationPanelProps {
  focusState?: string;
  confidence?: number;
  reason?: unknown;
  sendVoiceEvent?: (event: string, text: string) => void;
}

const styles = {
  wrapper: {
    marginTop: '16px',
  },
  button: {
    width: '100%',
    padding: '14px 24px',
    borderRadius: '12px',
    border: 'none',
    fontSize: '15px',
    fontWeight: 600 as const,
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
  },
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    marginTop: '12px',
    fontSize: '13px',
    color: '#86868b',
  },
  pulse: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    display: 'inline-block',
  },
};

const ConversationPanel: React.FC<ConversationPanelProps> = ({
  focusState,
  confidence,
  reason,
  sendVoiceEvent,
}) => {
  const lastSentFocus = useRef<string | null>(null);

  const conversation = useConversation({
    onConnect: () => {
      console.log('[ElevenLabs] Connected');
      sendVoiceEvent?.('connected', '');
    },
    onDisconnect: () => {
      console.log('[ElevenLabs] Disconnected');
      sendVoiceEvent?.('disconnected', '');
    },
    onMessage: (message) => {
      console.log('[ElevenLabs] Message:', message);
      const msg = message as { source?: string; message?: string };
      if (msg.source === 'ai' && msg.message) {
        sendVoiceEvent?.('agent_response', msg.message);
      } else if (msg.source === 'user' && msg.message) {
        sendVoiceEvent?.('user_transcript', msg.message);
      }
    },
    onError: (error) => console.error('[ElevenLabs] Error:', error),
  });

  const startConversation = useCallback(async () => {
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      await conversation.startSession({
        agentId: AGENT_ID,
      });
    } catch (err) {
      console.error('Failed to start conversation:', err);
    }
  }, [conversation]);

  const stopConversation = useCallback(async () => {
    await conversation.endSession();
  }, [conversation]);

  useEffect(() => {
    if (
      conversation.status !== 'connected' ||
      !focusState ||
      focusState === lastSentFocus.current
    ) {
      return;
    }

    lastSentFocus.current = focusState;
    const reasonStr = typeof reason === 'string' ? reason : JSON.stringify(reason);
    const ctx = `User focus state changed to "${focusState}" (confidence: ${(
      (confidence ?? 0) * 100
    ).toFixed(0)}%). Reason: ${reasonStr || 'N/A'}`;
    conversation.sendContextualUpdate(ctx);
  }, [focusState, confidence, reason, conversation]);

  const isConnected = conversation.status === 'connected';
  const isSpeaking = conversation.isSpeaking;

  return (
    <div style={styles.wrapper}>
      <button
        style={{
          ...styles.button,
          background: isConnected ? '#ff9500' : '#5856d6',
          color: 'white',
        }}
        onClick={isConnected ? stopConversation : startConversation}
      >
        {isConnected ? '⏹ End Conversation' : '🎙 Talk to Pabu'}
      </button>

      {isConnected && (
        <div style={styles.statusRow}>
          <span
            style={{
              ...styles.pulse,
              background: isSpeaking ? '#ff9500' : '#34C759',
              animation: isSpeaking ? undefined : 'none',
            }}
          />
          {isSpeaking ? 'Pabu is speaking…' : 'Listening…'}
        </div>
      )}
    </div>
  );
};

export default ConversationPanel;
