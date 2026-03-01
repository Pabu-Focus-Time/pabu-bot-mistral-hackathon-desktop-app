import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useFocusDetection } from './websocket';
import { tokens } from './styles/tokens';
import TaskList from './components/TaskList';
import FocusSession from './components/FocusSession';
import { Task, createTask, createTodoNode, getCompletedCount, getTotalCount } from './types/tasks';
import { motion, AnimatePresence } from 'framer-motion';
import { Target, Clock, Zap, Brain, Activity, Wifi, WifiOff, AlertTriangle, Settings, X, ArrowRight, BookOpen, Lightbulb, ExternalLink, Loader2, CheckCircle2 } from 'lucide-react';

const App: React.FC = () => {
  const {
    isConnected,
    desktopFocus,
    robotFocus,
    robotConnected,
    isAnalyzing,
    autoDetect,
    focusHistory,
    contentChangeInfo,
    showDistraction,
    distractionResources,
    isLoadingResources,
    permissionError,
    startAutoDetection,
    stopAutoDetection,
    clearFocusHistory,
    setTaskContext,
    dismissDistraction,
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

  // Per-todo timer: increment elapsedSeconds on the current todo every second
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (activeTaskId && currentTodoId) {
      interval = setInterval(() => {
        setTasks(prev => prev.map(task => {
          if (task.id !== activeTaskId) return task;
          return {
            ...task,
            todos: task.todos.map(todo =>
              todo.id === currentTodoId
                ? { ...todo, elapsedSeconds: (todo.elapsedSeconds || 0) + 1 }
                : todo
            ),
          };
        }));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [activeTaskId, currentTodoId]);

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
        newTask.todos = todos.map((t: { title: string; description: string; estimated_minutes?: number }) =>
          createTodoNode(t.title, t.description, t.estimated_minutes)
        );
      } else {
        // Fallback if API fails
        newTask.todos = [
          createTodoNode('Research', 'Research the topic and gather resources', 20),
          createTodoNode('Plan', 'Create a plan outline', 15),
          createTodoNode('Implement', 'Implement the core functionality', 45),
          createTodoNode('Test', 'Test and verify the implementation', 20),
          createTodoNode('Refine', 'Refine and polish the result', 15),
        ];
      }
    } catch (err) {
      console.error('Failed to generate todos from API:', err);
      // Fallback if server unreachable
      newTask.todos = [
        createTodoNode('Research', 'Research the topic and gather resources', 20),
        createTodoNode('Plan', 'Create a plan outline', 15),
        createTodoNode('Implement', 'Implement the core functionality', 45),
        createTodoNode('Test', 'Test and verify the implementation', 20),
        createTodoNode('Refine', 'Refine and polish the result', 15),
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
    setTasks(prev => {
      const updated = prev.map(task => {
        if (task.id !== taskId) return task;
        return {
          ...task,
          todos: task.todos.map(todo =>
            todo.id === todoId
              ? { ...todo, status: todo.status === 'completed' ? 'pending' as const : 'completed' as const }
              : todo
          ),
        };
      });

      // Auto-advance: if the toggled todo was the current one and it's now completed, move to next pending
      if (taskId === activeTaskId && todoId === currentTodoId) {
        const activeTask = updated.find(t => t.id === taskId);
        if (activeTask) {
          const toggledTodo = activeTask.todos.find(t => t.id === todoId);
          if (toggledTodo?.status === 'completed') {
            const nextPending = activeTask.todos.find(t => t.status !== 'completed' && t.id !== todoId);
            if (nextPending) {
              setCurrentTodoId(nextPending.id);
              // Mark startedAt on the new current todo
              const withStarted = updated.map(t => {
                if (t.id !== taskId) return t;
                return {
                  ...t,
                  todos: t.todos.map(td =>
                    td.id === nextPending.id
                      ? { ...td, startedAt: Date.now(), elapsedSeconds: td.elapsedSeconds || 0 }
                      : td
                  ),
                };
              });
              setTaskContext(activeTask, nextPending.title);
              return withStarted;
            }
          }
        }
      }

      // Re-sync task context if this is the active task
      if (taskId === activeTaskId) {
        const activeTask = updated.find(t => t.id === taskId);
        if (activeTask) {
          const currentTodo = currentTodoId
            ? activeTask.todos.find(t => t.id === currentTodoId)?.title
            : undefined;
          setTaskContext(activeTask, currentTodo);
        }
      }

      return updated;
    });
  };

  const handleStartSession = (taskId: string) => {
    setActiveTaskId(taskId);
    setSessionDuration(0);
    const task = tasks.find(t => t.id === taskId);
    if (task) {
      // Auto-set current todo to the first pending todo
      const firstPending = task.todos.find(t => t.status !== 'completed');
      if (firstPending) {
        setCurrentTodoId(firstPending.id);
        // Mark startedAt on the first pending todo
        setTasks(prev => prev.map(t => {
          if (t.id !== taskId) return t;
          return {
            ...t,
            todos: t.todos.map(todo =>
              todo.id === firstPending.id
                ? { ...todo, startedAt: Date.now(), elapsedSeconds: 0 }
                : todo
            ),
          };
        }));
        setTaskContext(task, firstPending.title);
      } else {
        setTaskContext(task);
      }
    }
    startAutoDetection();
  };

  const handleStopSession = () => {
    setActiveTaskId(null);
    setCurrentTodoId(null);
    setTaskContext(null);
    stopAutoDetection();
    clearFocusHistory();
  };

  const handleNodeClick = (taskId: string, nodeId: string) => {
    if (taskId === activeTaskId) {
      // Mark startedAt on the newly selected todo (elapsed keeps accumulating)
      setTasks(prev => prev.map(task => {
        if (task.id !== taskId) return task;
        return {
          ...task,
          todos: task.todos.map(todo =>
            todo.id === nodeId
              ? { ...todo, startedAt: todo.startedAt || Date.now() }
              : todo
          ),
        };
      }));

      setCurrentTodoId(nodeId);
      // Update task context with the newly selected todo
      const task = tasks.find(t => t.id === taskId);
      if (task) {
        const todoTitle = task.todos.find(t => t.id === nodeId)?.title;
        setTaskContext(task, todoTitle);
      }
    }
  };

  const activeTask = tasks.find(t => t.id === activeTaskId);
  const totalTasks = tasks.length;

  // Current todo info for the active session
  const currentTodo = activeTask && currentTodoId
    ? activeTask.todos.find(t => t.id === currentTodoId)
    : null;
  const currentTodoElapsed = currentTodo?.elapsedSeconds || 0;
  const currentTodoEstimate = currentTodo?.estimatedMinutes || 0;
  const isBehindSchedule = currentTodoEstimate > 0 && currentTodoElapsed > currentTodoEstimate * 60;

  const formatDuration = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Focus score: high = good. Invert when distracted.
  const focusScore = desktopFocus.focus_state === 'distracted'
    ? Math.round((1 - desktopFocus.confidence) * 100)
    : desktopFocus.focus_state === 'focused'
    ? Math.round(desktopFocus.confidence * 100)
    : Math.round(desktopFocus.confidence * 50);

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
              background: robotConnected ? tokens.colors.success : tokens.colors.textTertiary,
              flexShrink: 0,
            }} />
            <span style={{
              fontSize: tokens.typography.fontSize.xs,
              color: tokens.colors.textSecondary,
            }}>
              {robotConnected ? 'Robot online' : 'Robot offline'}
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
              contentChangeInfo={contentChangeInfo}
              currentTaskName={activeTask?.name || null}
              currentTodoName={currentTodo?.title || null}
              currentTodoElapsed={currentTodoElapsed}
              currentTodoEstimate={currentTodoEstimate}
              isBehindSchedule={isBehindSchedule}
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

      {/* Distraction Full-Screen Overlay */}
      <AnimatePresence>
        {showDistraction && activeTask && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 9999,
              background: 'rgba(0, 0, 0, 0.7)',
              backdropFilter: 'blur(8px)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            onClick={dismissDistraction}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ type: 'spring', stiffness: 400, damping: 30, delay: 0.05 }}
              onClick={(e: React.MouseEvent) => e.stopPropagation()}
              style={{
                width: '480px',
                maxWidth: '90vw',
                background: tokens.colors.surface,
                borderRadius: tokens.radius.xl,
                border: `1px solid ${tokens.colors.border}`,
                boxShadow: '0 24px 64px rgba(0,0,0,0.6), 0 8px 24px rgba(0,0,0,0.3)',
                overflow: 'hidden',
              }}
            >
              {/* Top accent bar */}
              <div style={{
                height: '3px',
                background: `linear-gradient(90deg, ${tokens.colors.warning}, ${tokens.colors.danger})`,
              }} />

              <div style={{ padding: '28px 28px 24px' }}>
                {/* Header */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: '20px',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                      width: '40px',
                      height: '40px',
                      borderRadius: tokens.radius.lg,
                      background: tokens.colors.warningMuted,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}>
                      <AlertTriangle size={20} style={{ color: tokens.colors.warning }} />
                    </div>
                    <div>
                      <div style={{
                        fontSize: tokens.typography.fontSize.lg,
                        fontWeight: tokens.typography.fontWeight.semibold,
                        color: tokens.colors.text,
                        lineHeight: 1.2,
                      }}>
                        You're distracted
                      </div>
                      <div style={{
                        fontSize: tokens.typography.fontSize.sm,
                        color: tokens.colors.textSecondary,
                        marginTop: '2px',
                      }}>
                        Get back to <span style={{
                          color: tokens.colors.accent,
                          fontWeight: tokens.typography.fontWeight.medium,
                        }}>{activeTask.name}</span>
                      </div>
                    </div>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={dismissDistraction}
                    style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: tokens.radius.sm,
                      background: 'transparent',
                      border: 'none',
                      color: tokens.colors.textTertiary,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      padding: 0,
                    }}
                  >
                    <X size={16} />
                  </motion.button>
                </div>

                {/* Progress Summary */}
                <div style={{
                  padding: '14px 16px',
                  background: tokens.colors.backgroundSecondary,
                  borderRadius: tokens.radius.md,
                  border: `1px solid ${tokens.colors.border}`,
                  marginBottom: '20px',
                }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: '10px',
                  }}>
                    <span style={{
                      fontSize: tokens.typography.fontSize.xs,
                      color: tokens.colors.textTertiary,
                      textTransform: 'uppercase',
                      letterSpacing: tokens.typography.letterSpacing.wider,
                    }}>
                      Session Progress
                    </span>
                    <span style={{
                      fontSize: tokens.typography.fontSize.xs,
                      color: tokens.colors.textSecondary,
                      fontFamily: tokens.typography.fontMono,
                    }}>
                      {getCompletedCount(activeTask.todos)}/{getTotalCount(activeTask.todos)} done
                    </span>
                  </div>

                  {/* Progress bar */}
                  <div style={{
                    height: '4px',
                    background: tokens.colors.surfaceActive,
                    borderRadius: tokens.radius.full,
                    overflow: 'hidden',
                    marginBottom: '12px',
                  }}>
                    <div style={{
                      height: '100%',
                      width: `${getTotalCount(activeTask.todos) > 0
                        ? (getCompletedCount(activeTask.todos) / getTotalCount(activeTask.todos)) * 100
                        : 0}%`,
                      background: tokens.colors.accent,
                      borderRadius: tokens.radius.full,
                      transition: tokens.transitions.normal,
                    }} />
                  </div>

                  {/* Current todo info */}
                  {currentTodo && (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '8px',
                    }}>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        flex: 1,
                        minWidth: 0,
                      }}>
                        <ArrowRight size={12} style={{ color: tokens.colors.accent, flexShrink: 0 }} />
                        <span style={{
                          fontSize: tokens.typography.fontSize.sm,
                          color: tokens.colors.text,
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                        }}>
                          {currentTodo.title}
                        </span>
                      </div>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        flexShrink: 0,
                      }}>
                        <span style={{
                          fontSize: tokens.typography.fontSize.xs,
                          fontFamily: tokens.typography.fontMono,
                          color: isBehindSchedule ? tokens.colors.danger : tokens.colors.textSecondary,
                        }}>
                          {formatDuration(currentTodoElapsed)}
                          {currentTodoEstimate > 0 && `/${currentTodoEstimate}m`}
                        </span>
                        {isBehindSchedule && (
                          <span style={{
                            fontSize: '9px',
                            fontWeight: tokens.typography.fontWeight.semibold,
                            color: tokens.colors.danger,
                            background: tokens.colors.dangerMuted,
                            padding: '1px 5px',
                            borderRadius: tokens.radius.full,
                            textTransform: 'uppercase',
                            letterSpacing: tokens.typography.letterSpacing.wide,
                          }}>
                            Behind
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* AI-Suggested Resources */}
                <div style={{ marginBottom: '24px' }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    marginBottom: '10px',
                  }}>
                    <Lightbulb size={13} style={{ color: tokens.colors.warning }} />
                    <span style={{
                      fontSize: tokens.typography.fontSize.xs,
                      color: tokens.colors.textTertiary,
                      textTransform: 'uppercase',
                      letterSpacing: tokens.typography.letterSpacing.wider,
                    }}>
                      Suggestions to refocus
                    </span>
                  </div>

                  {isLoadingResources ? (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '12px 14px',
                      background: tokens.colors.backgroundSecondary,
                      borderRadius: tokens.radius.md,
                      border: `1px solid ${tokens.colors.border}`,
                    }}>
                      <Loader2 size={14} style={{ color: tokens.colors.accent, animation: 'spin 1s linear infinite' }} />
                      <span style={{
                        fontSize: tokens.typography.fontSize.sm,
                        color: tokens.colors.textSecondary,
                      }}>
                        Generating suggestions...
                      </span>
                    </div>
                  ) : distractionResources.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      {distractionResources.map((resource, i) => (
                        <div
                          key={i}
                          style={{
                            padding: '10px 14px',
                            background: tokens.colors.backgroundSecondary,
                            borderRadius: tokens.radius.md,
                            border: `1px solid ${tokens.colors.border}`,
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: '10px',
                          }}
                        >
                          <CheckCircle2 size={14} style={{
                            color: tokens.colors.success,
                            marginTop: '1px',
                            flexShrink: 0,
                          }} />
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{
                              fontSize: tokens.typography.fontSize.sm,
                              fontWeight: tokens.typography.fontWeight.medium,
                              color: tokens.colors.text,
                              marginBottom: '2px',
                            }}>
                              {resource.title}
                            </div>
                            <div style={{
                              fontSize: tokens.typography.fontSize.xs,
                              color: tokens.colors.textSecondary,
                              lineHeight: tokens.typography.lineHeight.normal,
                            }}>
                              {resource.action}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{
                      padding: '12px 14px',
                      background: tokens.colors.backgroundSecondary,
                      borderRadius: tokens.radius.md,
                      border: `1px solid ${tokens.colors.border}`,
                      fontSize: tokens.typography.fontSize.sm,
                      color: tokens.colors.textSecondary,
                    }}>
                      Take a deep breath and return to your current task.
                    </div>
                  )}
                </div>

                {/* Dismiss Button */}
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={dismissDistraction}
                  style={{
                    width: '100%',
                    padding: '12px 20px',
                    background: tokens.colors.accent,
                    color: '#fff',
                    border: 'none',
                    borderRadius: tokens.radius.md,
                    fontSize: tokens.typography.fontSize.base,
                    fontWeight: tokens.typography.fontWeight.semibold,
                    cursor: 'pointer',
                    fontFamily: tokens.typography.fontFamily,
                    letterSpacing: tokens.typography.letterSpacing.tight,
                  }}
                >
                  I'm back on track
                </motion.button>
              </div>

              {/* Auto-dismiss progress bar */}
              <motion.div
                initial={{ scaleX: 1 }}
                animate={{ scaleX: 0 }}
                transition={{ duration: 30, ease: 'linear' }}
                style={{
                  height: '2px',
                  background: tokens.colors.warning,
                  transformOrigin: 'left',
                  opacity: 0.5,
                }}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
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
