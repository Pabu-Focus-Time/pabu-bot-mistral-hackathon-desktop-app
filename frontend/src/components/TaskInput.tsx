import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Loader2 } from 'lucide-react';
import { tokens } from '../styles/tokens';

interface TaskInputProps {
  onGenerate: (taskName: string, taskDescription: string) => Promise<void>;
}

const TaskInput: React.FC<TaskInputProps> = ({ onGenerate }) => {
  const [taskName, setTaskName] = useState('');
  const [taskDescription, setTaskDescription] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!taskName.trim() || isGenerating) return;

    setIsGenerating(true);
    try {
      await onGenerate(taskName, taskDescription);
      setTaskName('');
      setTaskDescription('');
      setIsExpanded(false);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      style={{
        background: tokens.colors.surface,
        borderRadius: tokens.radius.lg,
        boxShadow: tokens.shadows.sm,
        border: `1px solid ${tokens.colors.border}`,
        overflow: 'hidden',
      }}
    >
      <form onSubmit={handleSubmit}>
        <div style={{ padding: tokens.spacing.lg }}>
          {!isExpanded ? (
            <motion.button
              type="button"
              onClick={() => setIsExpanded(true)}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              style={{
                width: '100%',
                padding: `${tokens.spacing.md} ${tokens.spacing.lg}`,
                background: tokens.colors.surfaceSecondary,
                border: `1px dashed ${tokens.colors.border}`,
                borderRadius: tokens.radius.md,
                color: tokens.colors.textSecondary,
                fontSize: tokens.typography.fontSize.md,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: tokens.spacing.sm,
                transition: tokens.transitions.normal,
              }}
            >
              <span style={{ opacity: 0.6 }}>+</span>
              What's your task?
            </motion.button>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: tokens.spacing.lg }}>
              <div>
                <label
                  style={{
                    display: 'block',
                    fontSize: tokens.typography.fontSize.sm,
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: tokens.colors.textSecondary,
                    marginBottom: tokens.spacing.xs,
                  }}
                >
                  Task name
                </label>
                <input
                  type="text"
                  value={taskName}
                  onChange={(e) => setTaskName(e.target.value)}
                  placeholder="Build a React application..."
                  autoFocus
                  style={{
                    width: '100%',
                    padding: `${tokens.spacing.md} ${tokens.spacing.md}`,
                    fontSize: tokens.typography.fontSize.lg,
                    fontWeight: tokens.typography.fontWeight.medium,
                    border: `1px solid ${tokens.colors.border}`,
                    borderRadius: tokens.radius.md,
                    outline: 'none',
                    background: tokens.colors.surface,
                    color: tokens.colors.text,
                    transition: tokens.transitions.fast,
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = tokens.colors.accent;
                    e.target.style.boxShadow = `0 0 0 3px ${tokens.colors.focusLight}`;
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = tokens.colors.border;
                    e.target.style.boxShadow = 'none';
                  }}
                />
              </div>

              <div>
                <label
                  style={{
                    display: 'block',
                    fontSize: tokens.typography.fontSize.sm,
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: tokens.colors.textSecondary,
                    marginBottom: tokens.spacing.xs,
                  }}
                >
                  Description (optional)
                </label>
                <textarea
                  value={taskDescription}
                  onChange={(e) => setTaskDescription(e.target.value)}
                  placeholder="Describe your task in detail so we can break it down into actionable steps..."
                  rows={3}
                  style={{
                    width: '100%',
                    padding: `${tokens.spacing.md} ${tokens.spacing.md}`,
                    fontSize: tokens.typography.fontSize.md,
                    fontFamily: tokens.typography.fontFamily,
                    border: `1px solid ${tokens.colors.border}`,
                    borderRadius: tokens.radius.md,
                    outline: 'none',
                    background: tokens.colors.surface,
                    color: tokens.colors.text,
                    resize: 'vertical',
                    minHeight: '80px',
                    transition: tokens.transitions.fast,
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = tokens.colors.accent;
                    e.target.style.boxShadow = `0 0 0 3px ${tokens.colors.focusLight}`;
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = tokens.colors.border;
                    e.target.style.boxShadow = 'none';
                  }}
                />
              </div>

              <div style={{ display: 'flex', gap: tokens.spacing.sm, justifyContent: 'flex-end' }}>
                <button
                  type="button"
                  onClick={() => {
                    setIsExpanded(false);
                    setTaskName('');
                    setTaskDescription('');
                  }}
                  style={{
                    padding: `${tokens.spacing.sm} ${tokens.spacing.lg}`,
                    fontSize: tokens.typography.fontSize.md,
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: tokens.colors.textSecondary,
                    background: 'transparent',
                    border: 'none',
                    borderRadius: tokens.radius.md,
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
                <motion.button
                  type="submit"
                  disabled={!taskName.trim() || isGenerating}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  style={{
                    padding: `${tokens.spacing.sm} ${tokens.spacing.lg}`,
                    fontSize: tokens.typography.fontSize.md,
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: tokens.colors.surface,
                    background: taskName.trim() && !isGenerating ? tokens.colors.accent : tokens.colors.border,
                    border: 'none',
                    borderRadius: tokens.radius.md,
                    cursor: taskName.trim() && !isGenerating ? 'pointer' : 'not-allowed',
                    display: 'flex',
                    alignItems: 'center',
                    gap: tokens.spacing.xs,
                    transition: tokens.transitions.fast,
                  }}
                >
                  {isGenerating ? (
                    <>
                      <Loader2 size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles size={16} />
                      Generate Todos
                    </>
                  )}
                </motion.button>
              </div>
            </div>
          )}
        </div>
      </form>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </motion.div>
  );
};

export default TaskInput;
