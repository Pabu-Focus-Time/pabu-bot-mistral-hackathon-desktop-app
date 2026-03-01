import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Loader2, Plus } from 'lucide-react';
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
    <div style={{
      background: tokens.colors.surface,
      borderRadius: tokens.radius.lg,
      border: `1px solid ${tokens.colors.border}`,
      overflow: 'hidden',
      transition: tokens.transitions.normal,
    }}>
      <form onSubmit={handleSubmit}>
        <div style={{ padding: '12px 16px' }}>
          {!isExpanded ? (
            <button
              type="button"
              onClick={() => setIsExpanded(true)}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'transparent',
                border: `1px dashed ${tokens.colors.border}`,
                borderRadius: tokens.radius.md,
                color: tokens.colors.textTertiary,
                fontSize: tokens.typography.fontSize.md,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: tokens.transitions.normal,
                fontFamily: tokens.typography.fontFamily,
              }}
            >
              <Plus size={14} />
              <span>What's your task?</span>
            </button>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <label style={{
                  display: 'block',
                  fontSize: tokens.typography.fontSize.xs,
                  fontWeight: tokens.typography.fontWeight.medium,
                  color: tokens.colors.textTertiary,
                  textTransform: 'uppercase',
                  letterSpacing: tokens.typography.letterSpacing.wider,
                  marginBottom: '6px',
                }}>
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
                    padding: '8px 10px',
                    fontSize: tokens.typography.fontSize.base,
                    fontWeight: tokens.typography.fontWeight.medium,
                    border: `1px solid ${tokens.colors.border}`,
                    borderRadius: tokens.radius.sm,
                    outline: 'none',
                    background: tokens.colors.surfaceSecondary,
                    color: tokens.colors.text,
                    transition: tokens.transitions.fast,
                    fontFamily: tokens.typography.fontFamily,
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = tokens.colors.accent;
                    e.target.style.boxShadow = `0 0 0 2px ${tokens.colors.accentMuted}`;
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = tokens.colors.border;
                    e.target.style.boxShadow = 'none';
                  }}
                />
              </div>

              <div>
                <label style={{
                  display: 'block',
                  fontSize: tokens.typography.fontSize.xs,
                  fontWeight: tokens.typography.fontWeight.medium,
                  color: tokens.colors.textTertiary,
                  textTransform: 'uppercase',
                  letterSpacing: tokens.typography.letterSpacing.wider,
                  marginBottom: '6px',
                }}>
                  Description (optional)
                </label>
                <textarea
                  value={taskDescription}
                  onChange={(e) => setTaskDescription(e.target.value)}
                  placeholder="Describe your task..."
                  rows={2}
                  style={{
                    width: '100%',
                    padding: '8px 10px',
                    fontSize: tokens.typography.fontSize.md,
                    fontFamily: tokens.typography.fontFamily,
                    border: `1px solid ${tokens.colors.border}`,
                    borderRadius: tokens.radius.sm,
                    outline: 'none',
                    background: tokens.colors.surfaceSecondary,
                    color: tokens.colors.text,
                    resize: 'vertical',
                    minHeight: '60px',
                    transition: tokens.transitions.fast,
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = tokens.colors.accent;
                    e.target.style.boxShadow = `0 0 0 2px ${tokens.colors.accentMuted}`;
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = tokens.colors.border;
                    e.target.style.boxShadow = 'none';
                  }}
                />
              </div>

              <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                <button
                  type="button"
                  onClick={() => {
                    setIsExpanded(false);
                    setTaskName('');
                    setTaskDescription('');
                  }}
                  style={{
                    padding: '6px 14px',
                    fontSize: tokens.typography.fontSize.sm,
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: tokens.colors.textSecondary,
                    background: 'transparent',
                    border: 'none',
                    borderRadius: tokens.radius.sm,
                    cursor: 'pointer',
                    fontFamily: tokens.typography.fontFamily,
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
                    padding: '6px 14px',
                    fontSize: tokens.typography.fontSize.sm,
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: taskName.trim() && !isGenerating ? '#fff' : tokens.colors.textTertiary,
                    background: taskName.trim() && !isGenerating ? tokens.colors.accent : tokens.colors.surfaceActive,
                    border: 'none',
                    borderRadius: tokens.radius.sm,
                    cursor: taskName.trim() && !isGenerating ? 'pointer' : 'not-allowed',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    transition: tokens.transitions.fast,
                    fontFamily: tokens.typography.fontFamily,
                  }}
                >
                  {isGenerating ? (
                    <>
                      <Loader2 size={14} className="animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles size={14} />
                      Generate Todos
                    </>
                  )}
                </motion.button>
              </div>
            </div>
          )}
        </div>
      </form>
    </div>
  );
};

export default TaskInput;
