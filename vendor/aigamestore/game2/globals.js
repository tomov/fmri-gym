// Global game state object
export const gameState = {
  player: null, // Not used in this game (birds are projectiles)
  entities: [], // All game entities (birds, pigs, structures)
  score: 0,
  highScore: 0,
  currentLevel: 1,
  gamePhase: "START", // "START", "PLAYING", "PAUSED", "GAME_OVER_WIN", "GAME_OVER_LOSE", "LEVEL_COMPLETE"
  controlMode: "HUMAN", // "HUMAN", "TEST_1", "TEST_2"
  birdsRemaining: [],
  pigsRemaining: 0,
  activeBirds: [],
  slingshotPullPos: null,
  isAiming: false,
  structureBodies: [],
  particleEffects: [],
  levelStartScore: 0,
  totalLevels: 9,
  matterEngine: null,
  matterWorld: null,
  groundBody: null,
  keysPressed: {}, // Track which keys are currently held down
  autoRestartScheduled: false, // NEW: Flag to prevent multiple auto-restart timers
  autoRestartTimeoutId: null // NEW: Stores the ID of the setTimeout for auto-restart
};

// Canvas dimensions
export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 400;

// Game constants
export const GROUND_Y = 350;
export const SLINGSHOT_X = 80;
export const SLINGSHOT_Y = 320;
export const MAX_PULL_DISTANCE = 60;
export const LAUNCH_POWER_MULTIPLIER = 0.15;

// Physics constants
export const GRAVITY = 0.4;
export const BIRD_AIR_FRICTION = 0.01;

// Collision thresholds
export const PIG_DEFEAT_THRESHOLD = 3;
export const WOOD_DESTROY_THRESHOLD = 5;
export const STONE_DESTROY_THRESHOLD = 5; // Reduced from 10 to 5

// Scoring
export const SCORE_SMALL_PIG = 100;
export const SCORE_LARGE_PIG = 250;
export const SCORE_WOOD_BLOCK = 10;
export const SCORE_STONE_BLOCK = 20;
export const SCORE_UNUSED_BIRD = 500;

// Bird types
export const BIRD_TYPES = {
  RED: 'RED',
  BLUE: 'BLUE',
  YELLOW: 'YELLOW',
  BLACK: 'BLACK',
  GREEN: 'GREEN',
  WHITE: 'WHITE'
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

// Expose gameState to window for debugging and recording scripts
if (typeof window !== 'undefined') {
  window.getGameState = () => gameState;
}
