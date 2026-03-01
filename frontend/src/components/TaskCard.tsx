import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronDown, ChevronRight, Play, Trash2, Check } from 'lucide-react';
import { tokens } from '../styles/tokens';
import { Task, getCompletedCount, getTotalCount, flattenTodos } from '../types/tasks';

interface TaskCardProps {
  task: Task;
  isActive: boolean;
  currentTodoId: string | null;
  onStartSession: (taskId: string) => void;
  onDeleteTask: (taskId: string) => void;
  onNodeClick: (taskId: string, nodeId: string) => void;
  onToggleTodo: (taskId: string, todoId: string) => void;
}

const TaskCard: React.FC<TaskCardProps> = ({
  task,
  isActive,
  currentTodoId,
  onStartSession,
  onDeleteTask,
  onNodeClick,
  onToggleTodo,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isHovered, setIsHovered] = useState(false);

  const completedCount = getCompletedCount(task.todos);
  const totalCount = getTotalCount(task.todos);
  const progress = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;
  const allTodos = flattenTodos(task.todos);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      style={{
        background: tokens.colors.surface,
        borderRadius: tokens.radius.lg,
        border: `1px solid ${isActive ? tokens.colors.accentMuted : tokens.colors.border}`,
        overflow: 'hidden',
        transition: tokens.transitions.normal,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Card Header */}
      <div
        style={{
          padding: '14px 16px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, minWidth: 0 }}>
          <span style={{ color: tokens.colors.textTertiary, display: 'flex', flexShrink: 0 }}>
            {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </span>

          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{
                fontSize: tokens.typography.fontSize.base,
                fontWeight: tokens.typography.fontWeight.medium,
                color: tokens.colors.text,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}>
                {task.name}
              </span>
              {isActive && (
                <span style={{
                  padding: '1px 6px',
                  fontSize: '10px',
                  fontWeight: tokens.typography.fontWeight.medium,
                  color: tokens.colors.accent,
                  background: tokens.colors.accentMuted,
                  borderRadius: tokens.radius.full,
                  flexShrink: 0,
                }}>
                  ACTIVE
                </span>
              )}
              {totalCount > 0 && (
                <span style={{
                  fontSize: tokens.typography.fontSize.xs,
                  color: tokens.colors.textTertiary,
                  fontFamily: tokens.typography.fontMono,
                  flexShrink: 0,
                }}>
                  {completedCount}/{totalCount}
                </span>
              )}
            </div>
            {task.description && (
              <div style={{
                fontSize: tokens.typography.fontSize.sm,
                color: tokens.colors.textTertiary,
                marginTop: '2px',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}>
                {task.description}
              </div>
            )}
          </div>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          flexShrink: 0,
          marginLeft: '8px',
        }}>
          {!isActive && (
            <button
              onClick={(e: React.MouseEvent) => {
                e.stopPropagation();
                onStartSession(task.id);
              }}
              style={{
                padding: '5px 12px',
                background: tokens.colors.accentMuted,
                color: tokens.colors.accent,
                border: `1px solid rgba(82, 139, 255, 0.2)`,
                borderRadius: tokens.radius.sm,
                fontSize: tokens.typography.fontSize.sm,
                fontWeight: tokens.typography.fontWeight.medium,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                transition: tokens.transitions.fast,
                fontFamily: tokens.typography.fontFamily,
              }}
            >
              <Play size={12} fill="currentColor" />
              Start
            </button>
          )}

          <button
            onClick={(e: React.MouseEvent) => {
              e.stopPropagation();
              onDeleteTask(task.id);
            }}
            style={{
              padding: '5px',
              background: 'transparent',
              color: tokens.colors.textTertiary,
              border: 'none',
              borderRadius: tokens.radius.xs,
              cursor: 'pointer',
              opacity: isHovered ? 1 : 0,
              transition: tokens.transitions.fast,
              display: 'flex',
            }}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      {totalCount > 0 && (
        <div style={{
          height: '2px',
          background: tokens.colors.borderLight,
        }}>
          <div style={{
            width: `${progress}%`,
            height: '100%',
            background: progress === 100 ? tokens.colors.success : tokens.colors.accent,
            transition: 'width 0.3s ease',
          }} />
        </div>
      )}

      {/* Expanded Content - Notion-style Checklist */}
      {isExpanded && task.todos.length > 0 && (
        <div style={{ padding: '4px 0 8px' }}>
          {allTodos.map((todo) => {
            const isCompleted = todo.status === 'completed';
            const isCurrent = currentTodoId === todo.id && isActive;

            return (
              <div
                key={todo.id}
                onClick={(e) => {
                  e.stopPropagation();
                  onNodeClick(task.id, todo.id);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '10px',
                  padding: '6px 16px 6px 42px',
                  cursor: 'pointer',
                  background: isCurrent ? tokens.colors.accentMuted : 'transparent',
                  transition: tokens.transitions.fast,
                  borderLeft: isCurrent ? `2px solid ${tokens.colors.accent}` : '2px solid transparent',
                }}
              >
                {/* Checkbox */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleTodo(task.id, todo.id);
                  }}
                  style={{
                    width: '16px',
                    height: '16px',
                    borderRadius: '3px',
                    border: isCompleted
                      ? `1.5px solid ${tokens.colors.accent}`
                      : `1.5px solid ${tokens.colors.textTertiary}`,
                    background: isCompleted ? tokens.colors.accent : 'transparent',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    marginTop: '1px',
                    transition: tokens.transitions.fast,
                  }}
                >
                  {isCompleted && <Check size={10} strokeWidth={3} style={{ color: '#fff' }} />}
                </button>

                {/* Todo content */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: tokens.typography.fontSize.md,
                    color: isCompleted ? tokens.colors.textTertiary : tokens.colors.text,
                    textDecoration: isCompleted ? 'line-through' : 'none',
                    lineHeight: 1.4,
                  }}>
                    {todo.title}
                  </div>
                  {todo.description && (
                    <div style={{
                      fontSize: tokens.typography.fontSize.xs,
                      color: tokens.colors.textTertiary,
                      marginTop: '1px',
                      lineHeight: 1.4,
                    }}>
                      {todo.description}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </motion.div>
  );
};

export default TaskCard;
