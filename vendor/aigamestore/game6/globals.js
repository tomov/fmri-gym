// globals.js - Global constants and game state

export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 400;

export const GAME_PHASES = {
  START: "START",
  PLAYING: "PLAYING",
  PAUSED: "PAUSED",
  GAME_OVER_WIN: "GAME_OVER_WIN",
  GAME_OVER_LOSE: "GAME_OVER_LOSE"
};

export const CONTROL_MODES = {
  HUMAN: "HUMAN",
  TEST_1: "TEST_1",
  TEST_2: "TEST_2"
};

// Game state object
export const gameState = {
  player: null,
  entities: [],
  score: 0,
  gamePhase: GAME_PHASES.START,
  controlMode: CONTROL_MODES.HUMAN,
  currentCheckpoint: 0,
  deaths: 0,
  cameraOffset: 0,
  worldWidth: 3000,
  interactionTarget: null,
  levelComplete: false
};

// Player constants
export const PLAYER_WIDTH = 20;
export const PLAYER_HEIGHT = 30;
export const PLAYER_SPEED = 3;
export const JUMP_FORCE = -12;
export const GRAVITY = 0.6;
export const MAX_FALL_SPEED = 15;

// Entity types
export const ENTITY_TYPES = {
  PLAYER: "PLAYER",
  CRATE: "CRATE",
  SPIKE: "SPIKE",
  LEVER: "LEVER",
  GATE: "GATE",
  CHECKPOINT: "CHECKPOINT",
  PLATFORM: "PLATFORM",
  MOVING_PLATFORM: "MOVING_PLATFORM"
};

// Expose getGameState globally
window.getGameState = function() {
  return gameState;
};