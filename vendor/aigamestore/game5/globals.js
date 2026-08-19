export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 400;

export const GAME_PHASES = {
  START: "START",
  PLAYING: "PLAYING",
  PAUSED: "PAUSED",
  GAME_OVER_WIN: "GAME_OVER_WIN",
  GAME_OVER_LOSE: "GAME_OVER_LOSE"
};

export const gameState = {
  player: null,
  entities: [],
  score: 0,
  level: 1,
  lives: 3,
  gamePhase: GAME_PHASES.START,
  controlMode: "HUMAN",
  cheeseCollected: 0,
  totalCheese: 0,
  mouseHoleActive: false,
  invulnerable: false,
  invulnerableTimer: 0,
  levelTransitionTimer: 0,
  highScore: 0
};

// Load high score from localStorage
if (typeof window !== 'undefined' && window.localStorage) {
  const stored = window.localStorage.getItem('game5HighScore');
  if (stored) {
    gameState.highScore = parseInt(stored, 10) || 0;
  }
}

// Expose gameState globally
if (typeof window !== 'undefined') {
  window.getGameState = () => gameState;
}