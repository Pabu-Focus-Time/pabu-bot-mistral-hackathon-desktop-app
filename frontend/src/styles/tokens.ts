export const tokens = {
  colors: {
    // Backgrounds - warm Notion-style darks
    background: '#191919',
    backgroundSecondary: '#1E1E1E',
    
    // Surfaces - card/panel layers
    surface: '#202020',
    surfaceSecondary: '#252525',
    surfaceHover: '#2C2C2C',
    surfaceActive: '#333333',
    
    // Sidebar
    sidebar: '#1C1C1C',
    sidebarHover: '#272727',
    sidebarActive: '#2B2B2B',
    
    // Text - white with varying opacity
    text: 'rgba(255, 255, 255, 0.87)',
    textSecondary: 'rgba(255, 255, 255, 0.54)',
    textTertiary: 'rgba(255, 255, 255, 0.28)',
    textInverse: '#191919',
    
    // Accent - soft blue
    accent: '#528BFF',
    accentHover: '#6B9FFF',
    accentMuted: 'rgba(82, 139, 255, 0.15)',
    accentText: '#528BFF',
    
    // Status colors
    success: '#3ECF8E',
    successMuted: 'rgba(62, 207, 142, 0.12)',
    warning: '#F0B429',
    warningMuted: 'rgba(240, 180, 41, 0.12)',
    danger: '#EF4444',
    dangerMuted: 'rgba(239, 68, 68, 0.12)',
    
    // Borders - subtle
    border: 'rgba(255, 255, 255, 0.08)',
    borderLight: 'rgba(255, 255, 255, 0.05)',
    borderHover: 'rgba(255, 255, 255, 0.14)',
    
    // Focus graph colors
    graphGreen: '#3ECF8E',
    graphRed: '#EF4444',
    graphLine: '#528BFF',
    graphGrid: 'rgba(255, 255, 255, 0.04)',
    graphArea: 'rgba(82, 139, 255, 0.08)',
  },
  shadows: {
    sm: '0 1px 2px rgba(0, 0, 0, 0.3)',
    md: '0 4px 12px rgba(0, 0, 0, 0.4)',
    lg: '0 8px 24px rgba(0, 0, 0, 0.5)',
    xl: '0 12px 40px rgba(0, 0, 0, 0.6)',
    glow: '0 0 20px rgba(82, 139, 255, 0.15)',
  },
  radius: {
    xs: '4px',
    sm: '6px',
    md: '8px',
    lg: '12px',
    xl: '16px',
    xxl: '20px',
    full: '9999px',
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '12px',
    lg: '16px',
    xl: '24px',
    xxl: '32px',
    xxxl: '48px',
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    fontMono: '"SF Mono", "Fira Code", "JetBrains Mono", Menlo, Monaco, monospace',
    fontSize: {
      xs: '11px',
      sm: '12px',
      md: '13px',
      base: '14px',
      lg: '16px',
      xl: '20px',
      xxl: '24px',
      xxxl: '32px',
      display: '48px',
    },
    fontWeight: {
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
    lineHeight: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.7,
    },
    letterSpacing: {
      tight: '-0.02em',
      normal: '0',
      wide: '0.04em',
      wider: '0.08em',
    },
  },
  transitions: {
    fast: '0.1s ease',
    normal: '0.2s ease',
    slow: '0.35s ease',
    spring: '0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
  },
  layout: {
    maxWidth: '1200px',
    sidebarWidth: '240px',
    sidebarCollapsed: '64px',
    headerHeight: '48px',
  },
};

export const focusColors = {
  focused: tokens.colors.success,
  distracted: tokens.colors.danger,
  unknown: tokens.colors.textTertiary,
};

export const getFocusColor = (state: string) => {
  return focusColors[state as keyof typeof focusColors] || focusColors.unknown;
};

export const getFocusBg = (state: string) => {
  switch (state) {
    case 'focused': return tokens.colors.successMuted;
    case 'distracted': return tokens.colors.dangerMuted;
    default: return tokens.colors.surfaceSecondary;
  }
};
