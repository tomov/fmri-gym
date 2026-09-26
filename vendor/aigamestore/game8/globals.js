// globals.js - Global state and constants

export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 600;

export const BUBBLE_RADIUS = 20;
export const BUBBLE_DIAMETER = BUBBLE_RADIUS * 2;

export const SHOOTER_Y = CANVAS_HEIGHT - 60;
export const LOSE_LINE_Y = CANVAS_HEIGHT - 100;

export const BUBBLE_COLORS = [
  [255, 100, 100], // Red
  [100, 150, 255], // Blue
  [100, 255, 100], // Green
  [255, 255, 100], // Yellow
  [200, 100, 255], // Purple
  [255, 180, 100]  // Orange
];

export const gameState = {
  gamePhase: "START", // "START", "PLAYING", "GAME_OVER_WIN", "GAME_OVER_LOSE", "PAUSED", "LEVEL_TRANSITION"
  controlMode: "HUMAN", // "HUMAN", "TEST_1", "TEST_2"
  player: null,
  entities: [],
  score: 0,
  currentLevel: 1,
  shotsRemaining: 0,
  bubbleGrid: [],
  projectileBubble: null,
  nextBubble: null,
  shooterAngle: -Math.PI / 2,
  canFire: true,
  swapAvailable: true,
  levelStartTime: 0,
  transitionTimer: 0,
  testActionIndex: 0,
  testActions: [],
  autoRestartScheduled: false, // New: Flag to prevent multiple auto-restart timers
  autoRestartTimeoutId: null   // New: Stores the ID of the setTimeout for auto-restart
};

const keyLabels = new URLSearchParams(window.location.search);
const ARROWS = { LEFT: '←', RIGHT: '→', UP: '↑', DOWN: '↓' };

// Label for a game key on screen: `?label_SPACE=3` lets a host that remaps keys
// (e.g. a scanner button box) show the key the player actually presses.
export function keyLabel(name, fallback = name) {
  return keyLabels.get(`label_${name}`) ?? fallback;
}

// A hint naming several keys at once ("ARROW KEYS"): the game's own text unless
// one of them is relabeled, then each key's label.
export function keysLabel(fallback, names) {
  if (!names.some((name) => keyLabels.has(`label_${name}`))) return fallback;
  return names.map((name) => keyLabel(name, ARROWS[name] ?? name)).join(' ');
}

export function getGameState() {
  return gameState;
}

// Expose globally
if (typeof window !== 'undefined') {
  window.getGameState = getGameState;
}
