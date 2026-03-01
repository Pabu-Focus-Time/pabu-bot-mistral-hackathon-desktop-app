import React, { useState, useEffect } from 'react';
import { useFocusDetection } from './websocket';
import { tokens } from './styles/tokens';
import TaskList from './components/TaskList';
import FocusSession from './components/FocusSession';
import { Task, createTask, createTodoNode } from './types/tasks';

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

  const [tasks, setTasks] = useState<Task[]>([]);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [currentTodoId, setCurrentTodoId] = useState<string | null>(null);
  const [sessionDuration, setSessionDuration] = useState(0);

  // Load tasks from localStorage on mount
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

  // Save tasks to localStorage
  useEffect(() => {
    localStorage.setItem('pabu-tasks', JSON.stringify(tasks));
  }, [tasks]);

  // Session timer
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
    // Create new task with generated todos (mock for now - will connect to Mistral later)
    const newTask = createTask(taskName, taskDescription);
    
    // Generate some mock todo items
    newTask.todos = [
      createTodoNode('Research', 'Research the topic and gather resources'),
      createTodoNode('Plan', 'Create a plan outline'),
      createTodoNode('Implement', 'Implement the core functionality'),
      createTodoNode('Test', 'Test and verify the implementation'),
      createTodoNode('Refine', 'Refine and polish the result'),
    ];
    
    setTasks(prev => [...prev, newTask]);
  };

  const handleDeleteTask = (taskId: string) => {
    setTasks(prev => prev.filter(t => t.id !== taskId));
    if (activeTaskId === taskId) {
      setActiveTaskId(null);
      setCurrentTodoId(null);
    }
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
  };

  const handleNodeClick = (taskId: string, nodeId: string) => {
    if (taskId === activeTaskId) {
      setCurrentTodoId(nodeId);
    }
  };

  const activeTask = tasks.find(t => t.id === activeTaskId);

  return (
    <div style={{
      minHeight: '100vh',
      background: tokens.colors.background,
      fontFamily: tokens.typography.fontFamily,
      color: tokens.colors.text,
      padding: tokens.spacing.xl,
    }}>
      <div style={{
        maxWidth: tokens.layout.maxWidth,
        margin: '0 auto',
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: tokens.spacing.xl,
        }}>
          <div>
            <h1 style={{
              fontSize: tokens.typography.fontSize.xxl,
              fontWeight: tokens.typography.fontWeight.bold,
              margin: 0,
            }}>
              🤖 Pabu Focus
            </h1>
            <p style={{
              fontSize: tokens.typography.fontSize.md,
              color: tokens.colors.textSecondary,
              margin: `${tokens.spacing.xs} 0 0`,
            }}>
              Your AI-powered productivity assistant
            </p>
          </div>
          
          {/* Connection Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: tokens.spacing.sm,
            padding: `${tokens.spacing.sm} ${tokens.spacing.md}`,
            background: isConnected ? tokens.colors.successLight : '#ffe5e5',
            borderRadius: tokens.radius.full,
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: isConnected ? tokens.colors.success : '#DC4C4C',
            }} />
            <span style={{
              fontSize: tokens.typography.fontSize.sm,
              fontWeight: tokens.typography.fontWeight.medium,
              color: isConnected ? tokens.colors.success : '#c42b1c',
            }}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Permission Banner */}
        {permissionError && (
          <div style={{
            padding: tokens.spacing.lg,
            background: tokens.colors.warningLight,
            border: `1px solid ${tokens.colors.warning}`,
            borderRadius: tokens.radius.md,
            marginBottom: tokens.spacing.lg,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <div>
              <div style={{
                fontWeight: tokens.typography.fontWeight.medium,
                color: '#856404',
              }}>
                ⚠️ Screen Recording Permission Required
              </div>
              <div style={{
                fontSize: tokens.typography.fontSize.sm,
                color: '#856404',
              }}>
                {permissionError}
              </div>
            </div>
            <button
              onClick={openSystemPreferences}
              style={{
                padding: `${tokens.spacing.sm} ${tokens.spacing.md}`,
                background: tokens.colors.warning,
                color: '#856404',
                border: 'none',
                borderRadius: tokens.radius.sm,
                fontWeight: tokens.typography.fontWeight.medium,
                cursor: 'pointer',
              }}
            >
              Open Settings
            </button>
          </div>
        )}

        {/* Main Content Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 320px',
          gap: tokens.spacing.lg,
        }}>
          {/* Left: Task List */}
          <div>
            <TaskList
              tasks={tasks}
              activeTaskId={activeTaskId}
              currentTodoId={currentTodoId}
              onGenerateTodos={handleGenerateTodos}
              onStartSession={handleStartSession}
              onDeleteTask={handleDeleteTask}
              onNodeClick={handleNodeClick}
            />
          </div>

          {/* Right: Focus Session */}
          <div>
            <FocusSession
              isActive={!!activeTaskId}
              focusState={desktopFocus}
              currentTaskName={activeTask?.name || null}
              currentTodoName={currentTodoId ? 'Selected task' : null}
              sessionDuration={sessionDuration}
              windowData={null}
              activityData={null}
              onStart={() => handleStartSession(tasks[0]?.id || '')}
              onStop={handleStopSession}
            />

            {/* Focus Stats */}
            <div style={{
              marginTop: tokens.spacing.lg,
              padding: tokens.spacing.lg,
              background: tokens.colors.surface,
              borderRadius: tokens.radius.lg,
              border: `1px solid ${tokens.colors.border}`,
            }}>
              <div style={{
                fontSize: tokens.typography.fontSize.sm,
                fontWeight: tokens.typography.fontWeight.medium,
                color: tokens.colors.textSecondary,
                marginBottom: tokens.spacing.md,
              }}>
                FOCUS ANALYSIS
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: tokens.spacing.sm }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: tokens.colors.textSecondary }}>Desktop</span>
                  <span style={{ 
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: desktopFocus.focus_state === 'focused' ? tokens.colors.success : 
                           desktopFocus.focus_state === 'distracted' ? tokens.colors.danger : 
                           tokens.colors.textSecondary 
                  }}>
                    {desktopFocus.focus_state}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: tokens.colors.textSecondary }}>Robot</span>
                  <span style={{ 
                    fontWeight: tokens.typography.fontWeight.medium,
                    color: robotFocus.focus_state === 'focused' ? tokens.colors.success : 
                           robotFocus.focus_state === 'distracted' ? tokens.colors.danger : 
                           tokens.colors.textSecondary 
                  }}>
                    {robotFocus.focus_state}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: tokens.colors.textSecondary }}>Confidence</span>
                  <span style={{ fontWeight: tokens.typography.fontWeight.medium }}>
                    {Math.round(desktopFocus.confidence * 100)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
