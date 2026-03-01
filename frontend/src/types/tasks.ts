export interface TodoNode {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed';
  children: TodoNode[];
}

export interface Task {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  todos: TodoNode[];
}

export interface TaskWithFocus extends Task {
  isActive: boolean;
  currentTodoId: string | null;
  focusState: {
    focus_state: 'focused' | 'distracted' | 'unknown';
    confidence: number;
    reason: string;
  };
}

export const createTask = (name: string, description: string): Task => ({
  id: `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  name,
  description,
  createdAt: new Date().toISOString(),
  todos: [],
});

export const createTodoNode = (title: string, description = ''): TodoNode => ({
  id: `todo-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  title,
  description,
  status: 'pending',
  children: [],
});

export const flattenTodos = (todos: TodoNode[]): TodoNode[] => {
  const result: TodoNode[] = [];
  const traverse = (nodes: TodoNode[]) => {
    for (const node of nodes) {
      result.push(node);
      if (node.children.length > 0) {
        traverse(node.children);
      }
    }
  };
  traverse(todos);
  return result;
};

export const getCompletedCount = (todos: TodoNode[]): number => {
  return flattenTodos(todos).filter(t => t.status === 'completed').length;
};

export const getTotalCount = (todos: TodoNode[]): number => {
  return flattenTodos(todos).length;
};
