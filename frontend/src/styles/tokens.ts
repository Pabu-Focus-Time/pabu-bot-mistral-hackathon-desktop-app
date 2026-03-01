export const tokens = {
  colors: {
    background: '#F7F7F5',
    surface: '#FFFFFF',
    surfaceSecondary: '#F7F6F3',
    surfaceHover: '#EFEFEE',
    text: '#37352F',
    textSecondary: '#9B9A97',
    textTertiary: '#EBEBE9',
    accent: '#2EAADC',
    accentHover: '#2596C8',
    success: '#0F7B6C',
    successLight: '#D4EDDA',
    warning: '#E7A62E',
    warningLight: '#FFF3CD',
    danger: '#DC4C4C',
    dangerLight: '#F8D7DA',
    border: '#E9E9E7',
    borderLight: '#F0F0EE',
    focus: '#4A90D9',
    focusLight: '#E3F2FD',
  },
  shadows: {
    sm: '0 1px 2px rgba(0,0,0,0.04)',
    md: '0 4px 12px rgba(0,0,0,0.08)',
    lg: '0 8px 24px rgba(0,0,0,0.12)',
    xl: '0 12px 32px rgba(0,0,0,0.16)',
  },
  radius: {
    sm: '6px',
    md: '10px',
    lg: '14px',
    xl: '20px',
    full: '9999px',
  },
  spacing: {
    xs: '4px',
    sm: '8px',
    md: '12px',
    lg: '16px',
    xl: '24px',
    xxl: '32px',
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    fontSize: {
      xs: '11px',
      sm: '12px',
      md: '14px',
      lg: '16px',
      xl: '20px',
      xxl: '28px',
      xxxl: '36px',
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
  },
  transitions: {
    fast: '0.15s ease',
    normal: '0.25s ease',
    slow: '0.4s ease',
  },
  layout: {
    maxWidth: '900px',
    sidebarWidth: '240px',
  },
};

export const focusColors = {
  focused: '#0F7B6C',
  distracted: '#DC4C4C',
  unknown: '#9B9A97',
};

export const getFocusColor = (state: string) => {
  return focusColors[state as keyof typeof focusColors] || focusColors.unknown;
};

export const getFocusBg = (state: string) => {
  switch (state) {
    case 'focused': return '#E6F7F5';
    case 'distracted': return '#FDECEC';
    default: return '#F5F5F5';
  }
};
