import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT, GAME_PHASES } from './globals.js';

export function renderUI(p) {
  if (gameState.gamePhase === GAME_PHASES.PLAYING) {
    renderPlayingUI(p);
  } else if (gameState.gamePhase === GAME_PHASES.PAUSED) {
    renderPlayingUI(p);
    // Removed renderPauseScreen call to hide visual pause overlay
  } else if (gameState.gamePhase === GAME_PHASES.GAME_OVER_WIN || 
             gameState.gamePhase === GAME_PHASES.GAME_OVER_LOSE) {
    renderGameOverScreen(p);
  }
}

function renderPlayingUI(p) {
  // Placeholder for playing state specific UI elements.
  // Note: Main UI rendering is handled in rendering.js
}

function renderGameOverScreen(p) {
  // Placeholder for game over state specific UI elements.
  // Note: Main Game Over rendering is handled in rendering.js
}