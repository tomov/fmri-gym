// input.js - Input handling

import { gameState, GAME_PHASES, LEVELS } from './globals.js';
import { saveState, undoLastMove } from './gameLogic.js';
import { loadLevel, advanceToNextLevel } from './levelManager.js';

/**
 * Resets the game to the start screen state, clearing scores, level, and any pending auto-restart.
 * @param {object} p The p5 instance.
 */
export function resetGameToStartScreen(p) {
  // Clear any pending auto-restart
  if (gameState.autoRestartTimeoutId) {
    clearTimeout(gameState.autoRestartTimeoutId);
    gameState.autoRestartTimeoutId = null;
  }
  gameState.autoRestartScheduled = false; // Ensure this flag is reset

  gameState.gamePhase = GAME_PHASES.START;
  gameState.score = 0;
  gameState.currentLevel = 1;
  
  // Also reset other state that loadLevel doesn't touch immediately
  gameState.selectedTubeIndex = -1;
  gameState.highlightedTubeIndex = 0;
  gameState.previousStates = [];
  gameState.isAnimating = false;
  // Note: tubes and level-specific stats are reset by loadLevel when ENTER is pressed from START.

  p.logs.game_info.push({
    data: { action: 'game_restart', phase: gameState.gamePhase },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
}

export function handleKeyPressed(p) {
  const key = p.key;
  const keyCode = p.keyCode;
  
  // Log input
  p.logs.inputs.push({
    input_type: 'keyPressed',
    data: { key, keyCode },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
  
  // ENTER - Start game or resume from pause
  if (keyCode === 13) {
    if (gameState.gamePhase === GAME_PHASES.START) {
      loadLevel(1, p);
      gameState.gamePhase = GAME_PHASES.PLAYING;
      p.logs.game_info.push({
        data: { phase: gameState.gamePhase, level: gameState.currentLevel },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    } else if (gameState.gamePhase === GAME_PHASES.PAUSED) {
      gameState.gamePhase = GAME_PHASES.PLAYING;
      p.logs.game_info.push({
        data: { phase: gameState.gamePhase },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    }
    return;
  }
  
  // R - Restart to start screen (manual restart)
  if (keyCode === 82) {
    if (gameState.gamePhase === GAME_PHASES.GAME_OVER_WIN || 
        gameState.gamePhase === GAME_PHASES.GAME_OVER_LOSE) {
      resetGameToStartScreen(p); // Use the unified restart function
    }
    return;
  }
  
  // ESC - Pause/Unpause
  if (keyCode === 27) {
    if (gameState.gamePhase === GAME_PHASES.PLAYING) {
      gameState.gamePhase = GAME_PHASES.PAUSED;
      p.logs.game_info.push({
        data: { phase: gameState.gamePhase },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    } else if (gameState.gamePhase === GAME_PHASES.PAUSED) {
      gameState.gamePhase = GAME_PHASES.PLAYING;
      p.logs.game_info.push({
        data: { phase: gameState.gamePhase },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    }
    return;
  }
  
  // SPACE - Advance to next level (from LEVEL_COMPLETE phase)
  if (keyCode === 32 && gameState.gamePhase === GAME_PHASES.LEVEL_COMPLETE) {
    if (gameState.currentLevel < LEVELS.length) {
      advanceToNextLevel(p);
    } else {
      gameState.gamePhase = GAME_PHASES.GAME_OVER_WIN;
    }
    
    p.logs.game_info.push({
      data: { 
        phase: gameState.gamePhase,
        level: gameState.currentLevel
      },
      framecount: p.frameCount,
      timestamp: Date.now()
    });
    return;
  }
  
  // Game controls only work during PLAYING phase
  if (gameState.gamePhase !== GAME_PHASES.PLAYING) return;
  if (gameState.isAnimating) return;
  
  // Arrow Left - Navigate left
  if (keyCode === 37) {
    gameState.highlightedTubeIndex = (gameState.highlightedTubeIndex - 1 + gameState.tubes.length) % gameState.tubes.length;
    return;
  }
  
  // Arrow Right - Navigate right
  if (keyCode === 39) {
    gameState.highlightedTubeIndex = (gameState.highlightedTubeIndex + 1) % gameState.tubes.length;
    return;
  }
  
  // Space - Select/Pour
  if (keyCode === 32) {
    handleSpacePress(p);
    return;
  }
  
  // Z - Undo
  if (keyCode === 90 || key === 'z' || key === 'Z') {
    undoLastMove(p);
    return;
  }
}

function handleSpacePress(p) {
  if (gameState.selectedTubeIndex === -1) {
    // Select source tube
    const tube = gameState.tubes[gameState.highlightedTubeIndex];
    if (!tube.isEmpty()) {
      gameState.selectedTubeIndex = gameState.highlightedTubeIndex;
    }
  } else {
    // Attempt pour
    const sourceTube = gameState.tubes[gameState.selectedTubeIndex];
    const destTube = gameState.tubes[gameState.highlightedTubeIndex];
    
    if (gameState.selectedTubeIndex !== gameState.highlightedTubeIndex) {
      if (sourceTube.canPourInto(destTube)) {
        saveState();
        startPourAnimation(gameState.selectedTubeIndex, gameState.highlightedTubeIndex, p);
      }
    }
    
    gameState.selectedTubeIndex = -1;
  }
}

function startPourAnimation(sourceIndex, destIndex, p) {
  const sourceTube = gameState.tubes[sourceIndex];
  const layer = sourceTube.getTopContiguousLayer();
  
  gameState.isAnimating = true;
  gameState.animationProgress = 0;
  gameState.animationSourceIndex = sourceIndex;
  gameState.animationDestIndex = destIndex;
  gameState.animationWaterColor = layer.color;
  gameState.animationWaterAmount = layer.amount;
}

export { handleSpacePress, startPourAnimation };