// levelManager.js - Level loading and management

import { gameState, LEVELS, GAME_PHASES, CANVAS_WIDTH } from './globals.js';
import { Tube } from './tube.js';

export function loadLevel(levelNumber, p) {
  if (levelNumber < 1 || levelNumber > LEVELS.length) {
    return false;
  }
  
  const levelConfig = LEVELS[levelNumber - 1];
  
  gameState.currentLevel = levelNumber;
  gameState.levelMaxMoves = levelConfig.maxMoves;
  gameState.levelMovesMade = 0;
  gameState.levelUndoCount = 0;
  gameState.selectedTubeIndex = -1;
  gameState.highlightedTubeIndex = 0;
  gameState.previousStates = [];
  gameState.isAnimating = false;
  
  // Create tubes
  gameState.tubes = [];
  const numTubes = levelConfig.tubes.length;
  const tubeWidth = 50;
  const spacing = 70;
  const startX = (CANVAS_WIDTH - (numTubes * spacing - 20)) / 2;
  const tubeY = 150;
  
  for (let i = 0; i < numTubes; i++) {
    const tubeConfig = levelConfig.tubes[i];
    const tube = new Tube(
      tubeConfig.capacity,
      tubeConfig.colors,
      startX + i * spacing,
      tubeY,
      tubeWidth,
      p
    );
    gameState.tubes.push(tube);
  }
  
  return true;
}

export function checkLevelComplete() {
  // Check if all tubes are sorted
  return gameState.tubes.every(tube => tube.isSorted());
}

export function calculateLevelScore() {
  const baseScore = 100;
  const moveBonus = Math.max(0, (gameState.levelMaxMoves - gameState.levelMovesMade) * 5);
  const undoBonus = gameState.levelUndoCount === 0 ? 50 : 0;
  
  return baseScore + moveBonus + undoBonus;
}

export function advanceToNextLevel(p) {
  if (gameState.currentLevel < LEVELS.length) {
    loadLevel(gameState.currentLevel + 1, p);
    gameState.gamePhase = GAME_PHASES.PLAYING;
    return true;
  } else {
    gameState.gamePhase = GAME_PHASES.GAME_OVER_WIN;
    return false;
  }
}