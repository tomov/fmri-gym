// levels.js - Level management

import { gameState } from './globals.js';
import { createLevelLayout } from './grid.js';

export function getLevelShotLimit(level) {
  // Lenient shot limits increasing with difficulty/bubble count
  const shotLimits = [
    30, // Level 1 - Easy
    35, // Level 2 - Easy
    40, // Level 3 - Easy
    50, // Level 4 - Medium
    55, // Level 5 - Medium
    60, // Level 6 - Medium
    70, // Level 7 - Hard
    80, // Level 8 - Hard
    90  // Level 9 - Hard
  ];
  return shotLimits[level - 1] || 50;
}

export function initializeLevel(level, p) {
  gameState.currentLevel = level;
  gameState.bubbleGrid = createLevelLayout(level);
  gameState.shotsRemaining = getLevelShotLimit(level);
  gameState.levelStartTime = p.millis();
  gameState.canFire = true;
  gameState.swapAvailable = true;

  // Log level start
  p.logs.game_info.push({
    data: { event: 'level_start', level: level, shots: gameState.shotsRemaining },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
}

export function calculateShotBonus() {
  return gameState.shotsRemaining * 100;
}