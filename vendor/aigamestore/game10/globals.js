// globals.js - Global constants and state management

export const CANVAS_WIDTH = 640;
export const CANVAS_HEIGHT = 400;

export const GAME_PHASES = {
  START: "START",
  PLAYING: "PLAYING",
  PAUSED: "PAUSED",
  GAME_OVER_WIN: "GAME_OVER_WIN",
  GAME_OVER_LOSE: "GAME_OVER_LOSE"
};

export const TILE_SIZE = 32;
export const PLAYER_SIZE = 20;
export const ANTAGONIST_SIZE = 28;

export const PLAYER_SPEED = 2.5;
export const PLAYER_SPRINT_SPEED = 4.5;
export const ANTAGONIST_BASE_SPEED = 1.5;

export const gameState = {
  player: null,
  entities: [],
  antagonists: [],
  doors: [],
  items: [],
  hidingSpots: [],
  walls: [],
  score: 0,
  level: 1,
  gamePhase: GAME_PHASES.START,
  controlMode: "HUMAN",
  discoveredRooms: new Set(),
  isRunning: false,
  hasFlashlight: false,
  flashlightOn: false,
  inChase: false,
  lastEvadeTime: 0,
  levelComplete: false,
  undetectedBonus: true,
  currentLevelData: null,
  keysPressed: {},
  inventory: [],
  messages: [],
  requiredItemIds: [], // Items that must be collected to exit level
  collectedRequiredItems: new Set() // Track which required items have been collected
};

// Expose getGameState globally
if (typeof window !== 'undefined') {
  window.getGameState = function() {
    return gameState;
  };
}