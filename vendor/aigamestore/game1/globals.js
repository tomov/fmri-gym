// globals.js - Global state and constants

export const CANVAS_WIDTH = 700;
export const CANVAS_HEIGHT = 400;

export const GAME_PHASES = {
  START: "START",
  PLAYING: "PLAYING",
  PAUSED: "PAUSED",
  GAME_OVER_WIN: "GAME_OVER_WIN",
  GAME_OVER_LOSE: "GAME_OVER_LOSE",
  LEVEL_COMPLETE: "LEVEL_COMPLETE"
};

export const gameState = {
  gamePhase: GAME_PHASES.START,
  controlMode: "HUMAN",
  
  // Level and score
  currentLevel: 1,
  score: 0,
  levelMovesMade: 0,
  levelMaxMoves: 0,
  levelUndoCount: 0,
  
  // Tubes
  tubes: [],
  selectedTubeIndex: -1,
  highlightedTubeIndex: 0,
  
  // Undo system
  previousStates: [],
  
  // Animation
  isAnimating: false,
  animationProgress: 0,
  animationDuration: 30, // frames
  animationSourceIndex: -1,
  animationDestIndex: -1,
  animationWaterColor: null,
  animationWaterAmount: 0,
  
  // Auto-restart
  autoRestartScheduled: false,
  autoRestartTimeoutId: null, // Stores the ID returned by setTimeout
  
  // Player (for logging purposes)
  player: {
    x: CANVAS_WIDTH / 2,
    y: CANVAS_HEIGHT / 2
  }
};

// Level configurations
export const LEVELS = [
  {
    level: 1,
    difficulty: 'Easy',
    maxMoves: 8,
    tubes: [
      { capacity: 4, colors: ['red', 'blue', 'red', 'blue'] },
      { capacity: 4, colors: ['blue', 'red', 'blue', 'red'] },
      { capacity: 4, colors: [] }
    ]
  },
  {
    level: 2,
    difficulty: 'Easy',
    maxMoves: 10,
    tubes: [
      { capacity: 4, colors: ['red', 'blue', 'green', 'red'] },
      { capacity: 4, colors: ['blue', 'green', 'red', 'blue'] },
      { capacity: 4, colors: ['green', 'red', 'blue', 'green'] },
      { capacity: 4, colors: [] }
    ]
  },
  {
    level: 3,
    difficulty: 'Easy',
    maxMoves: 15,
    tubes: [
      { capacity: 4, colors: ['red', 'blue', 'yellow', 'red'] },
      { capacity: 4, colors: ['blue', 'green', 'red', 'blue'] },
      { capacity: 4, colors: ['green', 'yellow', 'blue', 'green'] },
      { capacity: 4, colors: ['yellow', 'red', 'green', 'yellow'] },
      { capacity: 4, colors: [] }
    ]
  },
  {
    level: 4,
    difficulty: 'Medium',
    maxMoves: 20,
    tubes: [
      { capacity: 4, colors: ['red', 'blue', 'yellow', 'green'] },
      { capacity: 4, colors: ['blue', 'green', 'red', 'yellow'] },
      { capacity: 4, colors: ['green', 'yellow', 'blue', 'red'] },
      { capacity: 4, colors: ['yellow', 'red', 'green', 'blue'] },
      { capacity: 4, colors: [] },
      { capacity: 4, colors: [] }
    ]
  },
  {
    level: 5,
    difficulty: 'Medium',
    maxMoves: 25,
    tubes: [
      { capacity: 4, colors: ['red', 'blue', 'yellow', 'purple'] },
      { capacity: 4, colors: ['blue', 'green', 'red', 'blue'] },
      { capacity: 4, colors: ['green', 'yellow', 'purple', 'red'] },
      { capacity: 4, colors: ['yellow', 'purple', 'green', 'yellow'] },
      { capacity: 4, colors: ['purple', 'red', 'green', 'blue'] },
      { capacity: 4, colors: [] },
      { capacity: 4, colors: [] }
    ]
  },
  {
    level: 6,
    difficulty: 'Medium',
    maxMoves: 35,
    tubes: [
      { capacity: 4, colors: ['red', 'blue', 'yellow', 'purple'] },
      { capacity: 4, colors: ['blue', 'green', 'orange', 'red'] },
      { capacity: 4, colors: ['green', 'yellow', 'purple', 'orange'] },
      { capacity: 4, colors: ['yellow', 'red', 'green', 'orange'] },
      { capacity: 4, colors: ['purple', 'orange', 'green', 'yellow'] },
      { capacity: 4, colors: ['blue', 'red', 'purple', 'blue'] },
      { capacity: 4, colors: [] },
      { capacity: 4, colors: [] }
    ]
  },
  {
    level: 7,
    difficulty: 'Hard',
    maxMoves: 40,
    tubes: [
      { capacity: 4, colors: ['red', 'blue', 'yellow', 'purple'] },
      { capacity: 4, colors: ['blue', 'green', 'orange', 'red'] },
      { capacity: 4, colors: ['green', 'yellow', 'purple', 'pink'] },
      { capacity: 4, colors: ['yellow', 'red', 'green', 'orange'] },
      { capacity: 4, colors: ['purple', 'orange', 'pink', 'yellow'] },
      { capacity: 4, colors: ['blue', 'red', 'purple', 'pink'] },
      { capacity: 4, colors: ['orange', 'pink', 'blue', 'green'] },
      { capacity: 4, colors: [] },
      { capacity: 4, colors: [] }
    ]
  },
  {
    level: 8,
    difficulty: 'Hard',
    maxMoves: 50,
    tubes: [
      { capacity: 4, colors: ['red', 'blue', 'yellow', 'purple'] },
      { capacity: 4, colors: ['blue', 'green', 'orange', 'red'] },
      { capacity: 4, colors: ['green', 'yellow', 'pink', 'orange'] },
      { capacity: 4, colors: ['yellow', 'red', 'green', 'pink'] },
      { capacity: 4, colors: ['purple', 'orange', 'pink', 'yellow'] },
      { capacity: 4, colors: ['blue', 'red', 'purple', 'pink'] },
      { capacity: 4, colors: ['orange', 'purple', 'green', 'blue'] },
      { capacity: 4, colors: [] },
      { capacity: 4, colors: [] }
    ]
  },
  {
    level: 9,
    difficulty: 'Hard',
    maxMoves: 60,
    tubes: [
      { capacity: 4, colors: ['red', 'blue', 'yellow', 'purple'] },
      { capacity: 4, colors: ['blue', 'green', 'cyan', 'red'] },
      { capacity: 4, colors: ['green', 'yellow', 'pink', 'orange'] },
      { capacity: 4, colors: ['yellow', 'red', 'green', 'pink'] },
      { capacity: 4, colors: ['purple', 'orange', 'cyan', 'yellow'] },
      { capacity: 4, colors: ['blue', 'red', 'purple', 'pink'] },
      { capacity: 4, colors: ['orange', 'purple', 'cyan', 'blue'] },
      { capacity: 4, colors: ['pink', 'orange', 'cyan', 'green'] },
      { capacity: 4, colors: [] },
      { capacity: 4, colors: [] }
    ]
  }
];

export const COLOR_MAP = {
  'red': [220, 50, 50],
  'blue': [50, 120, 220],
  'green': [50, 200, 80],
  'yellow': [240, 200, 30],
  'purple': [180, 60, 200],
  'orange': [255, 140, 30],
  'pink': [255, 120, 180],
  'cyan': [0, 220, 220]
};

export function getGameState() {
  return gameState;
}

// Expose globally (window in browser, including when game runs in iframe)
const g = typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : typeof self !== 'undefined' ? self : {};
if (g && typeof g === 'object') {
  g.getGameState = getGameState;
  g.gameState = gameState;
}