export interface ActivityData {
  keypressCount: number;
  mouseMoved: boolean;
  idleSeconds: number;
  windowSwitchCount: number;
}

let lastActivity = Date.now();
let keypressCount = 0;
let mouseMoved = false;
let previousApp = '';
let appSwitchCount = 0;
let windowSwitchTimestamps: number[] = [];

export const initActivityMonitor = () => {
  lastActivity = Date.now();
  keypressCount = 0;
  mouseMoved = false;
  previousApp = '';
  appSwitchCount = 0;
  windowSwitchTimestamps = [];

  const handleKeyDown = () => {
    keypressCount++;
    lastActivity = Date.now();
  };

  const handleMouseMove = () => {
    mouseMoved = true;
    lastActivity = Date.now();
  };

  const handleFocus = () => {
    lastActivity = Date.now();
  };

  document.addEventListener('keydown', handleKeyDown);
  document.addEventListener('mousemove', handleMouseMove);
  document.addEventListener('focus', handleFocus);

  return () => {
    document.removeEventListener('keydown', handleKeyDown);
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('focus', handleFocus);
  };
};

export const getActivityData = (currentApp: string): ActivityData => {
  const now = Date.now();
  const idleSeconds = Math.floor((now - lastActivity) / 1000);

  // Track app switches
  if (currentApp && currentApp !== previousApp) {
    appSwitchCount++;
    windowSwitchTimestamps.push(now);
    previousApp = currentApp;
  }

  // Count switches in last 60 seconds
  const oneMinuteAgo = now - 60000;
  const recentSwitches = windowSwitchTimestamps.filter(ts => ts > oneMinuteAgo).length;

  const data: ActivityData = {
    keypressCount,
    mouseMoved,
    idleSeconds,
    windowSwitchCount: recentSwitches,
  };

  // Reset counters for next interval
  keypressCount = 0;
  mouseMoved = false;

  return data;
};

export const getDwellTime = (): number => {
  return Math.floor((Date.now() - lastActivity) / 1000);
};
