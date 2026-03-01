import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronRight, Play, Trash2, Sparkles } from 'lucide-react';
import { tokens, getFocusBg } from '../styles/tokens';
import { Task, getCompletedCount, getTotalCount, flattenTodos } from '../types/tasks';
import MindMap from './MindMap';

interface TaskCardProps {
  task: Task;
  isActive: boolean;
  currentTodoId: string | null;
  onStartSession: (taskId: string) => void;
  onDeleteTask: (taskId: string) => void;
  onNodeClick: (taskId: string, nodeId: string) => void;
}

const TaskCard: React.FC<TaskCardProps> = ({
  task,
  isActive,
  currentTodoId,
  onStartSession,
  onDeleteTask,
  onNodeClick,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isHovered, setIsHovered] = useState(false);

  const completedCount = getCompletedCount(task.todos);
  const totalCount = getTotalCount(task.todos);
  const progress = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;

  const allTodos = flattenTodos(task.todos);
  const currentTodo = currentTodoId ? allTodos.find(t => t.id === currentTodoId) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      style={{
        background: tokens.colors.surface,
        borderRadius: tokens.radius.lg,
        boxShadow: tokens.shadows.sm,
        border: `1px solid ${isActive ? tokens.colors.accent : tokens.colors.border}`,
        overflow: 'hidden',
        transition: tokens.transitions.normal,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        style={{
          padding: tokens.spacing.lg,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacing.md, flex: 1 }}>
          <div style={{ color: tokens.colors.textSecondary }}>
            {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
          </div>
          
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacing.sm }}>
              <span
                style={{
                  fontSize: tokens.typography.fontSize.lg,
                  fontWeight: tokens.typography.fontWeight.semibold,
                  color: tokens.colors.text,
                }}
              >
                {task.name}
              </span>
              {isActive && (
                <span
                  style={{
                    padding: `2px ${tokens.spacing.sm}`,
                    fontSize: tokens.typography.fontSize.xs,
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: tokens.colors.surface,
                    background: tokens.colors.accent,
                    borderRadius: tokens.radius.full,
                  }}
                >
                  ACTIVE
                </span>
              )}
            </div>
            {task.description && (
              <div
                style={{
                  fontSize: tokens.typography.fontSize.sm,
                  color: tokens.colors.textSecondary,
                  marginTop: tokens.spacing.xs,
                }}
              >
                {task.description}
              </div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacing.sm }}>
          {!isActive && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={(e) => {
                e.stopPropagation();
                onStartSession(task.id);
              }}
              style={{
                padding: `${tokens.spacing.sm} ${tokens.spacing.md}`,
                background: tokens.colors.accent,
                color: tokens.colors.surface,
                border: 'none',
                borderRadius: tokens.radius.md,
                fontSize: tokens.typography.fontSize.sm,
                fontWeight: tokens.typography.fontWeight.medium,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: tokens.spacing.xs,
              }}
            >
              <Play size={14} fill="currentColor" />
              Start
            </motion.button>
          )}

          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={(e) => {
              e.stopPropagation();
              onDeleteTask(task.id);
            }}
            style={{
              padding: tokens.spacing.sm,
              background: 'transparent',
              color: tokens.colors.textSecondary,
              border: 'none',
              borderRadius: tokens.radius.sm,
              cursor: 'pointer',
              opacity: isHovered ? 1 : 0,
              transition: tokens.transitions.fast,
            }}
          >
            <Trash2 size={16} />
          </motion.button>
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div style={{ padding: `0 ${tokens.spacing.lg} ${tokens.spacing.lg}` }}>
              {task.todos.length > 0 ? (
                <>
                  <div style={{ marginBottom: tokens.spacing.md }}>
                    <MindMap
                      todos={task.todos}
                      currentTodoId={currentTodoId}
                      onNodeClick={(nodeId) => onNodeClick(task.id, nodeId)}
                    />
                  </div>

                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: tokens.spacing.md,
                      padding: tokens.spacing.md,
                      background: tokens.colors.surfaceSecondary,
                      borderRadius: tokens.radius.md,
                    }}
                  >
                    <div style={{ flex: 1 }}>
                      <div
                        style={{
                          height: '6px',
                          background: tokens.colors.border,
                          borderRadius: tokens.radius.full,
                          overflow: 'hidden',
                        }}
                      >
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${progress}%` }}
                          transition={{ duration: 0.5, ease: 'easeOut' }}
                          style={{
                            height: '100%',
                            background: tokens.colors.success,
                            borderRadius: tokens.radius.full,
                          }}
                        />
                      </div>
                    </div>
                    <span
                      style={{
                        fontSize: tokens.typography.fontSize.sm,
                        fontWeight: tokens.typography.fontWeight.medium,
                        color: tokens.colors.textSecondary,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {completedCount}/{totalCount}
                    </span>
                  </div>

                  {currentTodo && isActive && (
                    <div
                      style={{
                        marginTop: tokens.spacing.md,
                        padding: tokens.spacing.md,
                        background: getFocusBg('focused'),
                        borderRadius: tokens.radius.md,
                        border: `1px solid ${tokens.colors.success}30`,
                      }}
                    >
                      <div
                        style={{
                          fontSize: tokens.typography.fontSize.xs,
                          fontWeight: tokens.typography.fontWeight.medium,
                          color: tokens.colors.success,
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
                        {currentTodo.title}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div
                  style={{
                    padding: tokens.spacing.xl,
                    textAlign: 'center',
                    color: tokens.colors.textSecondary,
                    fontSize: tokens.typography.fontSize.sm,
                  }}
                >
                  <Sparkles size={24} style={{ marginBottom: tokens.spacing.sm, opacity: 0.5 }} />
                  <div>Generate todos for this task</div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default TaskCard;
