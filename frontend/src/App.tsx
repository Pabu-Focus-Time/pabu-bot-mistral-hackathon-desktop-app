import React, { useState, useEffect } from 'react';
import { useFocusDetection } from './websocket';
import { tokens } from './styles/tokens';
import TaskList from './components/TaskList';
import FocusSession from './components/FocusSession';
import { Task, createTask, createTodoNode } from './types/tasks';
import { motion, AnimatePresence } from 'framer-motion';
import { Target, Clock, Zap, Brain, Activity, Wifi, WifiOff, AlertTriangle, Settings } from 'lucide-react';

const App: React.FC = () => {
  const {
    isConnected,
    desktopFocus,
    robotFocus,
    focusHistory,
    permissionError,
    startAutoDetection,
    stopAutoDetection,
    clearFocusHistory,
    openSystemPreferences,
  } = useFocusDetection();

  const [tasks, setTasks] = useState<Task[]>([]);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [currentTodoId, setCurrentTodoId] = useState<string | null>(null);
  const [sessionDuration, setSessionDuration] = useState(0);

  useEffect(() => {
    const saved = localStorage.getItem('pabu-tasks');
    if (saved) {
      try {
        setTasks(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to load tasks:', e);
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('pabu-tasks', JSON.stringify(tasks));
  }, [tasks]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (activeTaskId) {
      interval = setInterval(() => {
        setSessionDuration(d => d + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [activeTaskId]);

  const handleGenerateTodos = async (taskName: string, taskDescription: string) => {
    const newTask = createTask(taskName, taskDescription);

    try {
      const response = await fetch('http://127.0.0.1:9800/api/generate-todos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_name: taskName,
          task_description: taskDescription,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const todos = data.todos || [];
        newTask.todos = todos.map((t: { title: string; description: string }) =>
          createTodoNode(t.title, t.description)
        );
      } else {
        // Fallback if API fails
        newTask.todos = [
          createTodoNode('Research', 'Research the topic and gather resources'),
          createTodoNode('Plan', 'Create a plan outline'),
          createTodoNode('Implement', 'Implement the core functionality'),
          createTodoNode('Test', 'Test and verify the implementation'),
          createTodoNode('Refine', 'Refine and polish the result'),
        ];
      }
    } catch (err) {
      console.error('Failed to generate todos from API:', err);
      // Fallback if server unreachable
      newTask.todos = [
        createTodoNode('Research', 'Research the topic and gather resources'),
        createTodoNode('Plan', 'Create a plan outline'),
        createTodoNode('Implement', 'Implement the core functionality'),
        createTodoNode('Test', 'Test and verify the implementation'),
        createTodoNode('Refine', 'Refine and polish the result'),
      ];
    }

    setTasks(prev => [...prev, newTask]);
  };

  const handleDeleteTask = (taskId: string) => {
    setTasks(prev => prev.filter(t => t.id !== taskId));
    if (activeTaskId === taskId) {
      setActiveTaskId(null);
      setCurrentTodoId(null);
    }
  };

  const handleToggleTodo = (taskId: string, todoId: string) => {
    setTasks(prev => prev.map(task => {
      if (task.id !== taskId) return task;
      return {
        ...task,
        todos: task.todos.map(todo =>
          todo.id === todoId
            ? { ...todo, status: todo.status === 'completed' ? 'pending' as const : 'completed' as const }
            : todo
        ),
      };
    }));
  };

  const handleStartSession = (taskId: string) => {
    setActiveTaskId(taskId);
    setSessionDuration(0);
    startAutoDetection();
  };

  const handleStopSession = () => {
    setActiveTaskId(null);
    setCurrentTodoId(null);
    stopAutoDetection();
    clearFocusHistory();
  };

  const handleNodeClick = (taskId: string, nodeId: string) => {
    if (taskId === activeTaskId) {
      setCurrentTodoId(nodeId);
    }
  };

  const activeTask = tasks.find(t => t.id === activeTaskId);
  const totalTasks = tasks.length;

  const formatDuration = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const focusScore = Math.round(desktopFocus.confidence * 100);

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      background: tokens.colors.background,
      fontFamily: tokens.typography.fontFamily,
      color: tokens.colors.text,
      overflow: 'hidden',
    }}>
      {/* Sidebar */}
      <aside style={{
        width: tokens.layout.sidebarWidth,
        minWidth: tokens.layout.sidebarWidth,
        background: tokens.colors.sidebar,
        borderRight: `1px solid ${tokens.colors.border}`,
        display: 'flex',
        flexDirection: 'column',
        padding: '20px 0',
      } as React.CSSProperties}>
        {/* App Logo */}
        <div style={{
          padding: '0 20px',
          marginBottom: '32px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}>
          <div style={{
            width: '28px',
            height: '28px',
            borderRadius: tokens.radius.md,
            background: `linear-gradient(135deg, ${tokens.colors.accent} 0%, #7C5CFC 100%)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '14px',
            flexShrink: 0,
          }}>
            P
          </div>
          <div>
            <div style={{
              fontSize: tokens.typography.fontSize.base,
              fontWeight: tokens.typography.fontWeight.semibold,
              color: tokens.colors.text,
              letterSpacing: tokens.typography.letterSpacing.tight,
              lineHeight: 1.2,
            }}>
              Pabu Focus
            </div>
            <div style={{
              fontSize: tokens.typography.fontSize.xs,
              color: tokens.colors.textTertiary,
              lineHeight: 1.2,
            }}>
              AI-powered focus
            </div>
          </div>
        </div>

        {/* Sidebar Nav Items */}
        <nav style={{
          flex: 1,
          padding: '0 12px',
          display: 'flex',
          flexDirection: 'column',
          gap: '2px',
        }}>
          <SidebarItem icon={<Target size={16} />} label="Dashboard" active />
          <SidebarItem icon={<Activity size={16} />} label="Focus Session" badge={activeTaskId ? 'Live' : undefined} />
          <SidebarItem icon={<Zap size={16} />} label="Tasks" count={totalTasks} />
          <SidebarItem icon={<Brain size={16} />} label="Analytics" disabled />

          {/* Session Quick Stats */}
          {activeTaskId && (
            <div style={{
              marginTop: '16px',
              padding: '12px',
              background: tokens.colors.sidebarActive,
              borderRadius: tokens.radius.md,
              border: `1px solid ${tokens.colors.border}`,
            }}>
              <div style={{
                fontSize: tokens.typography.fontSize.xs,
                color: tokens.colors.textTertiary,
                textTransform: 'uppercase',
                letterSpacing: tokens.typography.letterSpacing.wider,
                marginBottom: '8px',
              }}>
                Active Session
              </div>
              <div style={{
                fontSize: tokens.typography.fontSize.xl,
                fontWeight: tokens.typography.fontWeight.bold,
                color: tokens.colors.text,
                fontFamily: tokens.typography.fontMono,
                fontVariantNumeric: 'tabular-nums',
                marginBottom: '4px',
              }}>
                {formatDuration(sessionDuration)}
              </div>
              <div style={{
                fontSize: tokens.typography.fontSize.xs,
                color: tokens.colors.textSecondary,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}>
                {activeTask?.name || 'No task'}
              </div>
            </div>
          )}
        </nav>

        {/* Sidebar Footer */}
        <div style={{
          padding: '0 12px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px',
            borderRadius: tokens.radius.sm,
          }}>
            <div style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: isConnected ? tokens.colors.success : tokens.colors.danger,
              flexShrink: 0,
            }} />
            <span style={{
              fontSize: tokens.typography.fontSize.xs,
              color: tokens.colors.textSecondary,
            }}>
              {isConnected ? 'Server connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main style={{
        flex: 1,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}>
        {/* Permission Banner */}
        <AnimatePresence>
          {permissionError && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              style={{
                padding: '10px 24px',
                background: tokens.colors.warningMuted,
                borderBottom: `1px solid rgba(240, 180, 41, 0.2)`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '12px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <AlertTriangle size={14} style={{ color: tokens.colors.warning, flexShrink: 0 }} />
                <span style={{
                  fontSize: tokens.typography.fontSize.sm,
                  color: tokens.colors.warning,
                }}>
                  Screen recording permission required for focus detection
                </span>
              </div>
              <button
                onClick={openSystemPreferences}
                style={{
                  padding: '4px 12px',
                  background: 'rgba(240, 180, 41, 0.2)',
                  color: tokens.colors.warning,
                  border: `1px solid rgba(240, 180, 41, 0.3)`,
                  borderRadius: tokens.radius.sm,
                  fontSize: tokens.typography.fontSize.xs,
                  fontWeight: tokens.typography.fontWeight.medium,
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                Open Settings
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Scrollable content */}
        <div className="scroll-container" style={{
          flex: 1,
          overflowY: 'auto',
          padding: '32px',
        }}>
          <div style={{ maxWidth: '960px', margin: '0 auto' }}>
            {/* Focus Session Panel */}
            <FocusSession
              isActive={!!activeTaskId}
              focusState={desktopFocus}
              robotFocus={robotFocus}
              focusHistory={focusHistory}
              currentTaskName={activeTask?.name || null}
              currentTodoName={currentTodoId ? 'Selected task' : null}
              sessionDuration={sessionDuration}
              windowData={null}
              activityData={null}
              onStart={() => tasks.length > 0 && handleStartSession(tasks[0].id)}
              onStop={handleStopSession}
            />

            {/* Stats Row */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '12px',
              marginBottom: '32px',
            }}>
              <StatCard
                icon={<Target size={15} />}
                label="Active Mission"
                value={activeTask?.name || 'None'}
                accent={!!activeTask}
              />
              <StatCard
                icon={<Zap size={15} />}
                label="Total Tasks"
                value={totalTasks.toString()}
              />
              <StatCard
                icon={<Clock size={15} />}
                label="Session Time"
                value={formatDuration(sessionDuration)}
                mono
              />
              <StatCard
                icon={<Brain size={15} />}
                label="Focus Score"
                value={`${focusScore}%`}
                valueColor={
                  desktopFocus.focus_state === 'focused' ? tokens.colors.success :
                  desktopFocus.focus_state === 'distracted' ? tokens.colors.danger :
                  tokens.colors.textSecondary
                }
              />
            </div>

            {/* Tasks */}
            <TaskList
              tasks={tasks}
              activeTaskId={activeTaskId}
              currentTodoId={currentTodoId}
              onGenerateTodos={handleGenerateTodos}
              onStartSession={handleStartSession}
              onDeleteTask={handleDeleteTask}
              onNodeClick={handleNodeClick}
              onToggleTodo={handleToggleTodo}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

// -- Sidebar Item Component --
interface SidebarItemProps {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  disabled?: boolean;
  badge?: string;
  count?: number;
}

const SidebarItem: React.FC<SidebarItemProps> = ({ icon, label, active, disabled, badge, count }) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '6px 8px',
    borderRadius: tokens.radius.sm,
    background: active ? tokens.colors.sidebarActive : 'transparent',
    color: disabled ? tokens.colors.textTertiary : active ? tokens.colors.text : tokens.colors.textSecondary,
    fontSize: tokens.typography.fontSize.md,
    cursor: disabled ? 'default' : 'pointer',
    transition: tokens.transitions.fast,
    opacity: disabled ? 0.5 : 1,
  }}>
    <span style={{ display: 'flex', flexShrink: 0 }}>{icon}</span>
    <span style={{ flex: 1, fontWeight: active ? tokens.typography.fontWeight.medium : tokens.typography.fontWeight.regular }}>
      {label}
    </span>
    {badge && (
      <span style={{
        padding: '1px 6px',
        fontSize: tokens.typography.fontSize.xs,
        fontWeight: tokens.typography.fontWeight.medium,
        color: tokens.colors.success,
        background: tokens.colors.successMuted,
        borderRadius: tokens.radius.full,
      }}>
        {badge}
      </span>
    )}
    {count !== undefined && count > 0 && (
      <span style={{
        fontSize: tokens.typography.fontSize.xs,
        color: tokens.colors.textTertiary,
        fontFamily: tokens.typography.fontMono,
      }}>
        {count}
      </span>
    )}
  </div>
);

// -- Stat Card Component --
interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  valueColor?: string;
  accent?: boolean;
  mono?: boolean;
}

const StatCard: React.FC<StatCardProps> = ({ icon, label, value, valueColor, accent, mono }) => (
  <div style={{
    background: tokens.colors.surface,
    borderRadius: tokens.radius.lg,
    padding: '16px',
    border: `1px solid ${accent ? tokens.colors.accentMuted : tokens.colors.border}`,
    transition: tokens.transitions.normal,
  }}>
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      marginBottom: '8px',
      color: tokens.colors.textTertiary,
    }}>
      {icon}
      <span style={{
        fontSize: tokens.typography.fontSize.xs,
        letterSpacing: tokens.typography.letterSpacing.wide,
        textTransform: 'uppercase',
      }}>
        {label}
      </span>
    </div>
    <div style={{
      fontSize: tokens.typography.fontSize.xl,
      fontWeight: tokens.typography.fontWeight.bold,
      color: valueColor || tokens.colors.text,
      fontFamily: mono ? tokens.typography.fontMono : undefined,
      fontVariantNumeric: 'tabular-nums',
      letterSpacing: tokens.typography.letterSpacing.tight,
      whiteSpace: 'nowrap',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
    }}>
      {value}
    </div>
  </div>
);

export default App;
