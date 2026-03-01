import React, { useCallback, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  MiniMap,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  NodeTypes,
  Handle,
  Position,
  BackgroundVariant,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { tokens, getFocusColor, getFocusBg } from '../styles/tokens';
import { TodoNode } from '../types/tasks';

interface MindMapProps {
  todos: TodoNode[];
  currentTodoId?: string | null;
  onNodeClick?: (nodeId: string) => void;
}

interface TodoNodeComponentProps {
  data: {
    label: string;
    description: string;
    status: string;
    isCurrent: boolean;
  };
}

const TodoNodeComponent: React.FC<TodoNodeComponentProps> = ({ data }) => {
  const isCompleted = data.status === 'completed';
  const isCurrent = data.isCurrent;

  return (
    <div
      style={{
        padding: `${tokens.spacing.sm} ${tokens.spacing.md}`,
        borderRadius: tokens.radius.md,
        background: isCurrent ? getFocusBg('focused') : tokens.colors.surface,
        border: `2px solid ${isCurrent ? tokens.colors.success : isCompleted ? tokens.colors.success : tokens.colors.border}`,
        boxShadow: tokens.shadows.sm,
        minWidth: '140px',
        maxWidth: '180px',
        transition: tokens.transitions.fast,
        opacity: isCompleted ? 0.7 : 1,
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: tokens.colors.textSecondary }} />
      
      <div style={{ display: 'flex', alignItems: 'center', gap: tokens.spacing.xs }}>
        <div
          style={{
            width: '16px',
            height: '16px',
            borderRadius: '50%',
            border: `2px solid ${isCompleted ? tokens.colors.success : tokens.colors.border}`,
            background: isCompleted ? tokens.colors.success : 'transparent',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          {isCompleted && (
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <path d="M2 5L4 7L8 3" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </div>
        <span
          style={{
            fontSize: tokens.typography.fontSize.sm,
            fontWeight: tokens.typography.fontWeight.medium,
            color: tokens.colors.text,
            textDecoration: isCompleted ? 'line-through' : 'none',
          }}
        >
          {data.label}
        </span>
      </div>
      
      {data.description && (
        <div
          style={{
            fontSize: tokens.typography.fontSize.xs,
            color: tokens.colors.textSecondary,
            marginTop: tokens.spacing.xs,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {data.description}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} style={{ background: tokens.colors.textSecondary }} />
    </div>
  );
};

const nodeTypes: NodeTypes = {
  todo: TodoNodeComponent,
};

const MindMap: React.FC<MindMapProps> = ({ todos, currentTodoId, onNodeClick }) => {
  const { initialNodes, initialEdges } = useMemo(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    let nodeIndex = 0;

    const traverse = (items: TodoNode[], parentId: string | null, level: number, offsetX: number, offsetY: number) => {
      for (const item of items) {
        const levelWidth = 250;
        const verticalGap = 80;
        
        const x = level * levelWidth;
        const y = offsetY;

        nodes.push({
          id: item.id,
          type: 'todo',
          position: { x, y },
          data: {
            label: item.title,
            description: item.description,
            status: item.status,
            isCurrent: item.id === currentTodoId,
          },
        });

        if (parentId) {
          edges.push({
            id: `e-${parentId}-${item.id}`,
            source: parentId,
            target: item.id,
            type: 'smoothstep',
            style: { stroke: tokens.colors.border, strokeWidth: 2 },
            animated: item.id === currentTodoId,
          });
        }

        if (item.children.length > 0) {
          const childOffsetY = offsetY - ((item.children.length - 1) * verticalGap) / 2;
          traverse(item.children, item.id, level + 1, offsetX, childOffsetY);
          offsetY += (item.children.length - 1) * verticalGap;
        }

        offsetY += verticalGap;
        nodeIndex++;
      }
    };

    traverse(todos, null, 0, 0, 0);

    return { initialNodes: nodes, initialEdges: edges };
  }, [todos, currentTodoId]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  React.useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onNodeClickHandler = useCallback(
    (_: React.MouseEvent, node: Node) => {
      if (onNodeClick) {
        onNodeClick(node.id);
      }
    },
    [onNodeClick]
  );

  if (todos.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        width: '100%',
        height: '300px',
        borderRadius: tokens.radius.md,
        overflow: 'hidden',
        border: `1px solid ${tokens.colors.border}`,
        background: tokens.colors.surfaceSecondary,
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClickHandler}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
        proOptions={{ hideAttribution: true }}
      >
        <Controls
          style={{
            background: tokens.colors.surface,
            borderRadius: tokens.radius.sm,
            border: `1px solid ${tokens.colors.border}`,
          }}
        />
        <MiniMap
          style={{
            background: tokens.colors.surfaceSecondary,
            borderRadius: tokens.radius.sm,
          }}
          nodeColor={(node) => {
            const status = node.data?.status;
            if (status === 'completed') return tokens.colors.success;
            if (node.id === currentTodoId) return tokens.colors.accent;
            return tokens.colors.textSecondary;
          }}
        />
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color={tokens.colors.border} />
      </ReactFlow>
    </div>
  );
};

export default MindMap;
