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
  onGenerateTodos: (taskName: string, taskDescription: string) => void;
  onStartSession: (taskId: string) => void;
  onDeleteTask: (taskId: string) => void;
  onNodeClick: (taskId: string, nodeId: string) => void;
}

const TaskList: React.FC<TaskListProps> = ({
  tasks,
  activeTaskId,
  currentTodoId,
  onGenerateTodos,
  onStartSession,
  onDeleteTask,
  onNodeClick,
}) => {
  const handleGenerate = async (taskName: string, taskDescription: string) => {
    onGenerateTodos(taskName, taskDescription);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: tokens.spacing.lg }}>
      <TaskInput onGenerate={handleGenerate} />

      {tasks.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: tokens.spacing.md }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: `0 ${tokens.spacing.xs}`,
            }}
          >
            <span
              style={{
                fontSize: tokens.typography.fontSize.sm,
                fontWeight: tokens.typography.fontWeight.medium,
                color: tokens.colors.textSecondary,
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
              }}
            >
              Your Tasks
            </span>
            <span
              style={{
                fontSize: tokens.typography.fontSize.xs,
                color: tokens.colors.textTertiary,
              }}
            >
              {tasks.length} {tasks.length === 1 ? 'task' : 'tasks'}
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
              />
            ))}
          </AnimatePresence>
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            padding: tokens.spacing.xxl,
            textAlign: 'center',
          }}
        >
          <div
            style={{
              width: '64px',
              height: '64px',
              margin: `0 auto ${tokens.spacing.lg}`,
              borderRadius: tokens.radius.lg,
              background: tokens.colors.surfaceSecondary,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Folder size={28} style={{ color: tokens.colors.textTertiary }} />
          </div>
          <div
            style={{
              fontSize: tokens.typography.fontSize.lg,
              fontWeight: tokens.typography.fontWeight.medium,
              color: tokens.colors.text,
              marginBottom: tokens.spacing.xs,
            }}
          >
            No tasks yet
          </div>
          <div
            style={{
              fontSize: tokens.typography.fontSize.sm,
              color: tokens.colors.textSecondary,
            }}
          >
            Create your first task to get started
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default TaskList;
