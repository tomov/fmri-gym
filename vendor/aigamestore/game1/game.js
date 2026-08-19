// game.js - Main game file

import { gameState, GAME_PHASES, CANVAS_WIDTH, CANVAS_HEIGHT, getGameState } from "./globals.js";
import { handleKeyPressed, resetGameToStartScreen } from './input.js'; // Import resetGameToStartScreen
import { updateAnimation } from './gameLogic.js';
import { 
  renderStartScreen, 
  renderPlayingScreen, 
  renderLevelComplete,
  renderGameOverWin,
  renderGameOverLose 
} from './rendering.js';

const p5 = window.p5;

let gameInstance = new p5(p => {
  // Initialize logs
  p.logs = {
    game_info: [],
    inputs: [],
    player_info: []
  };
  
  p.setup = function() {
    p.createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    p.randomSeed(42);
    p.frameRate(60);
    
    // Initialize game state
    gameState.gamePhase = GAME_PHASES.START;
    
    // Log initial state
    p.logs.game_info.push({
      data: { phase: gameState.gamePhase },
      framecount: p.frameCount,
      timestamp: Date.now()
    });
  };
  
  p.draw = function() {
    // --- Auto-restart logic start ---
    const isGameOverPhase = gameState.gamePhase === GAME_PHASES.GAME_OVER_WIN || gameState.gamePhase === GAME_PHASES.GAME_OVER_LOSE;

    if (isGameOverPhase && !gameState.autoRestartScheduled) {
      // Schedule auto-restart if in game over phase and not already scheduled
      gameState.autoRestartScheduled = true;
      gameState.autoRestartTimeoutId = setTimeout(() => {
        resetGameToStartScreen(p); // Call the unified reset function
        // The resetGameToStartScreen function itself will clear autoRestartScheduled and autoRestartTimeoutId
      }, 1000); // 1 second delay
    } else if (!isGameOverPhase && gameState.autoRestartScheduled) {
      // If we leave a game over state (e.g., via manual 'R' key) before timeout, clear the scheduled restart
      clearTimeout(gameState.autoRestartTimeoutId);
      gameState.autoRestartScheduled = false;
      gameState.autoRestartTimeoutId = null;
    }
    // --- Auto-restart logic end ---

    // Update animations
    if (gameState.gamePhase === GAME_PHASES.PLAYING) {
      updateAnimation(p);
    }
    
    // Render based on game phase
    if (gameState.gamePhase === GAME_PHASES.START) {
      renderStartScreen(p);
    } else if (gameState.gamePhase === GAME_PHASES.PLAYING || gameState.gamePhase === GAME_PHASES.PAUSED) {
      renderPlayingScreen(p);
    } else if (gameState.gamePhase === GAME_PHASES.LEVEL_COMPLETE) {
      renderLevelComplete(p);
    } else if (gameState.gamePhase === GAME_PHASES.GAME_OVER_WIN) {
      renderGameOverWin(p);
    } else if (gameState.gamePhase === GAME_PHASES.GAME_OVER_LOSE) {
      renderGameOverLose(p);
    }
    
    // Log player info periodically
    if (p.frameCount % 60 === 0) {
      p.logs.player_info.push({
        screen_x: gameState.player.x,
        screen_y: gameState.player.y,
        game_x: gameState.player.x,
        game_y: gameState.player.y,
        framecount: p.frameCount
      });
    }
  };
  
  p.keyPressed = function() {
    handleKeyPressed(p);
    return false; // Prevent default
  };
});

// Expose game instance and state globally (ensures getGameState is available in this frame/iframe)
window.gameInstance = gameInstance;
window.getGameState = getGameState;
window.gameState = gameState;
// Expose level loading for dev mode
window.loadLevel = function(levelNum) {
  const state = window.getGameState ? window.getGameState() : (window.gameState || (window.gameInstance && window.gameInstance.gameState));
  if (state) {
    state.currentLevel = levelNum;
    // For dev mode, directly set to PLAYING phase and clear any pending auto-restart
    state.gamePhase = GAME_PHASES.PLAYING;
    if (state.autoRestartTimeoutId) {
      clearTimeout(state.autoRestartTimeoutId);
      state.autoRestartTimeoutId = null;
    }
    state.autoRestartScheduled = false;
  }
};

// Control mode switching
window.setControlMode = function(mode) {
  gameState.controlMode = mode;
  
  // Update button styles
  document.querySelectorAll('.control-button').forEach(btn => {
    btn.classList.remove('active');
  });
  
  if (mode === 'HUMAN') {
    document.getElementById('humanModeBtn').classList.add('active');
  } else if (mode === 'TEST_1') {
    document.getElementById('test_1_ModeBtn').classList.add('active');
  } else if (mode === 'TEST_2') {
    document.getElementById('test_2_ModeBtn').classList.add('active');
  }
};