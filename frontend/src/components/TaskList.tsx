import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Folder } from 'lucide-react';
import { tokens } from '../styles/tokens';
import { Task } from '../types/tasks';
import TaskCard from './TaskCard';
import TaskInput from './TaskInput';

interface TaskListProps {
  tasks: Task[];
  activeTaskId: string | null;
  currentTodoId: string | null;
  onGenerateTodos: (taskName: string, taskDescription: string) => Promise<void>;
  onStartSession: (taskId: string) => void;
  onDeleteTask: (taskId: string) => void;
  onNodeClick: (taskId: string, nodeId: string) => void;
  onToggleTodo: (taskId: string, todoId: string) => void;
}

const TaskList: React.FC<TaskListProps> = ({
  tasks,
  activeTaskId,
  currentTodoId,
  onGenerateTodos,
  onStartSession,
  onDeleteTask,
  onNodeClick,
  onToggleTodo,
}) => {
  const handleGenerate = async (taskName: string, taskDescription: string) => {
    await onGenerateTodos(taskName, taskDescription);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <TaskInput onGenerate={handleGenerate} />

      {tasks.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 4px',
          }}>
            <span style={{
              fontSize: tokens.typography.fontSize.xs,
              fontWeight: tokens.typography.fontWeight.medium,
              color: tokens.colors.textTertiary,
              textTransform: 'uppercase',
              letterSpacing: tokens.typography.letterSpacing.wider,
            }}>
              Your Tasks
            </span>
            <span style={{
              fontSize: tokens.typography.fontSize.xs,
              color: tokens.colors.textTertiary,
              fontFamily: tokens.typography.fontMono,
            }}>
              {tasks.length}
            </span>
          </div>

          <AnimatePresence mode="popLayout">
            {tasks.map((task) => (
              <TaskCard
                key={task.id}
                task={task}
                isActive={task.id === activeTaskId}
                currentTodoId={task.id === activeTaskId ? currentTodoId : null}
                onStartSession={onStartSession}
                onDeleteTask={onDeleteTask}
                onNodeClick={onNodeClick}
                onToggleTodo={onToggleTodo}
              />
            ))}
          </AnimatePresence>
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            padding: '48px 32px',
            textAlign: 'center',
          }}
        >
          <div style={{
            width: '48px',
            height: '48px',
            margin: '0 auto 16px',
            borderRadius: tokens.radius.lg,
            background: tokens.colors.surfaceSecondary,
            border: `1px solid ${tokens.colors.border}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <Folder size={20} style={{ color: tokens.colors.textTertiary }} />
          </div>
          <div style={{
            fontSize: tokens.typography.fontSize.base,
            fontWeight: tokens.typography.fontWeight.medium,
            color: tokens.colors.textSecondary,
            marginBottom: '4px',
          }}>
            No tasks yet
          </div>
          <div style={{
            fontSize: tokens.typography.fontSize.sm,
            color: tokens.colors.textTertiary,
          }}>
            Create a task above to get started
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default TaskList;
