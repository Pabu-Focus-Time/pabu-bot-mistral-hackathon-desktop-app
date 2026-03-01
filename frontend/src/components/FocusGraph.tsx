import React, { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { tokens } from '../styles/tokens';
import { FocusDataPoint } from '../websocket';

interface FocusGraphProps {
  data: FocusDataPoint[];
  height?: number;
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins === 0) return `${secs}s`;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;

  const point = payload[0].payload as FocusDataPoint;
  const stateColor =
    point.state === 'focused'
      ? tokens.colors.success
      : point.state === 'distracted'
      ? tokens.colors.danger
      : tokens.colors.textTertiary;

  return (
    <div
      style={{
        background: tokens.colors.surface,
        border: `1px solid ${tokens.colors.border}`,
        borderRadius: tokens.radius.md,
        padding: '10px 14px',
        boxShadow: tokens.shadows.lg,
      }}
    >
      <div
        style={{
          fontSize: tokens.typography.fontSize.xs,
          color: tokens.colors.textSecondary,
          marginBottom: '4px',
          fontFamily: tokens.typography.fontMono,
        }}
      >
        {formatTime(point.time)}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: stateColor,
          }}
        />
        <span
          style={{
            fontSize: tokens.typography.fontSize.md,
            fontWeight: tokens.typography.fontWeight.semibold,
            color: tokens.colors.text,
          }}
        >
          {point.score}%
        </span>
        <span
          style={{
            fontSize: tokens.typography.fontSize.xs,
            color: stateColor,
            textTransform: 'capitalize',
          }}
        >
          {point.state}
        </span>
      </div>
    </div>
  );
};

const FocusGraph: React.FC<FocusGraphProps> = ({ data, height = 200 }) => {
  // Build chart data with color segments
  const chartData = useMemo(() => {
    if (data.length === 0) return [];
    return data.map((point) => ({
      ...point,
      focusedScore: point.state === 'focused' ? point.score : undefined,
      distractedScore: point.state === 'distracted' ? point.score : undefined,
    }));
  }, [data]);

  // Calculate average score
  const avgScore = useMemo(() => {
    if (data.length === 0) return 0;
    return Math.round(data.reduce((sum, p) => sum + p.score, 0) / data.length);
  }, [data]);

  if (data.length === 0) {
    return (
      <div
        style={{
          height: `${height}px`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: tokens.colors.surfaceSecondary,
          borderRadius: tokens.radius.lg,
          border: `1px solid ${tokens.colors.border}`,
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <div
            style={{
              fontSize: tokens.typography.fontSize.md,
              color: tokens.colors.textTertiary,
              marginBottom: '4px',
            }}
          >
            Waiting for focus data...
          </div>
          <div
            style={{
              fontSize: tokens.typography.fontSize.xs,
              color: tokens.colors.textTertiary,
            }}
          >
            Graph will appear when session starts
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        background: tokens.colors.surfaceSecondary,
        borderRadius: tokens.radius.lg,
        border: `1px solid ${tokens.colors.border}`,
        padding: '16px 8px 8px 0',
      }}
    >
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData} margin={{ top: 4, right: 12, left: 12, bottom: 4 }}>
          <defs>
            <linearGradient id="focusGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tokens.colors.graphGreen} stopOpacity={0.3} />
              <stop offset="50%" stopColor={tokens.colors.graphLine} stopOpacity={0.1} />
              <stop offset="100%" stopColor={tokens.colors.graphLine} stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tokens.colors.accent} stopOpacity={0.25} />
              <stop offset="100%" stopColor={tokens.colors.accent} stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="3 3"
            stroke={tokens.colors.graphGrid}
            horizontal={true}
            vertical={false}
          />

          <XAxis
            dataKey="time"
            tickFormatter={formatTime}
            stroke={tokens.colors.textTertiary}
            tick={{ fontSize: 10, fill: tokens.colors.textTertiary }}
            axisLine={{ stroke: tokens.colors.border }}
            tickLine={false}
            interval="preserveStartEnd"
            minTickGap={50}
          />

          <YAxis
            domain={[0, 100]}
            stroke={tokens.colors.textTertiary}
            tick={{ fontSize: 10, fill: tokens.colors.textTertiary }}
            axisLine={false}
            tickLine={false}
            width={32}
            tickFormatter={(v) => `${v}%`}
            ticks={[0, 25, 50, 75, 100]}
          />

          <Tooltip content={<CustomTooltip />} />

          {avgScore > 0 && (
            <ReferenceLine
              y={avgScore}
              stroke={tokens.colors.textTertiary}
              strokeDasharray="4 4"
              strokeWidth={1}
            />
          )}

          <Area
            type="monotone"
            dataKey="score"
            stroke={tokens.colors.accent}
            strokeWidth={2}
            fill="url(#scoreGradient)"
            dot={false}
            activeDot={{
              r: 4,
              stroke: tokens.colors.accent,
              strokeWidth: 2,
              fill: tokens.colors.surface,
            }}
            animationDuration={300}
            isAnimationActive={true}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FocusGraph;
