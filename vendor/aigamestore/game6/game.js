// game.js - Main game file

import { 
  gameState, 
  GAME_PHASES,
  CONTROL_MODES,
  CANVAS_WIDTH,
  CANVAS_HEIGHT
} from './globals.js';
import { createLevel } from './level.js';
import { updatePhysics, handlePlayerInput } from './physics.js';
import { 
  renderGame, 
  renderStartScreen, 
  renderGameOverScreen 
} from './renderer.js';
import { 
  keys,
  handleKeyPressed, 
  handleKeyReleased 
} from './input.js';
import { updateCamera } from './utils.js';

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
    p.frameRate(60);
    p.randomSeed(42);
    
    // Initialize level
    createLevel();
    
    // Log initial state
    p.logs.game_info.push({
      data: { gamePhase: gameState.gamePhase },
      framecount: p.frameCount,
      timestamp: Date.now()
    });
  };
  
  p.draw = function() {
    // Handle different game phases
    switch (gameState.gamePhase) {
      case GAME_PHASES.START:
        renderStartScreen(p);
        break;
        
      case GAME_PHASES.PLAYING:
        // Get input based on control mode
        // Update game
        handlePlayerInput(keys);
        updatePhysics();
        
        if (gameState.player) {
          gameState.player.update();
          updateCamera(gameState.player.x);
          
          // Log player info periodically
          if (p.frameCount % 30 === 0) {
            p.logs.player_info.push({
              screen_x: gameState.player.x - gameState.cameraOffset,
              screen_y: gameState.player.y,
              game_x: gameState.player.x,
              game_y: gameState.player.y,
              framecount: p.frameCount
            });
          }
        }
        
        // Render
        renderGame(p);
        
        // Reset jump key (to prevent continuous jumping)
        keys.jump = false;
        break;
        
      case GAME_PHASES.PAUSED:
        renderGame(p);
        break;
        
      case GAME_PHASES.GAME_OVER_WIN:
      case GAME_PHASES.GAME_OVER_LOSE:
        renderGameOverScreen(p);
        
        // Log game over
        if (p.frameCount % 60 === 0) {
          p.logs.game_info.push({
            data: { 
              gamePhase: gameState.gamePhase,
              score: gameState.score,
              deaths: gameState.deaths
            },
            framecount: p.frameCount,
            timestamp: Date.now()
          });
        }
        break;
    }
  };
  
  p.keyPressed = function() {
    handleKeyPressed(p, p.keyCode);
    return false; // Prevent default
  };
  
  p.keyReleased = function() {
    handleKeyReleased(p, p.keyCode);
    return false; // Prevent default
  };
});

// Expose game instance globally
window.gameInstance = gameInstance;
// Expose level loading for dev mode
window.loadLevel = function(levelNum) {
  const state = window.getGameState ? window.getGameState() : (window.gameState || (window.gameInstance && window.gameInstance.gameState));
  if (state) {
    state.currentLevel = levelNum;
    // Try common reset/start patterns
    if (typeof resetGame === 'function') {
      resetGame();
    }
    if (typeof startGame === 'function') {
      startGame();
    } else if (state.gamePhase !== undefined) {
      state.gamePhase = "PLAYING";
    }
  }
};

// Control mode switching
window.setControlMode = function(mode) {
  gameState.controlMode = mode;
  
  // Update button states
  const buttons = ['humanModeBtn', 'test_1_ModeBtn', 'test_2_ModeBtn'];
  buttons.forEach(btnId => {
    const btn = document.getElementById(btnId);
    if (btn) {
      btn.classList.remove('active');
    }
  });
  
  const modeMap = {
    'HUMAN': 'humanModeBtn',
    'TEST_1': 'test_1_ModeBtn',
    'TEST_2': 'test_2_ModeBtn'
  };
  
  const activeBtn = document.getElementById(modeMap[mode]);
  if (activeBtn) {
    activeBtn.classList.add('active');
  }
};