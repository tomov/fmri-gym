import { gameState, GAME_PHASES, CANVAS_WIDTH, CANVAS_HEIGHT } from './globals.js';
import { handleKeyPressed, handleKeyReleased, handlePlayerMovement, startGame } from './controls.js';
import { checkCollisions } from './collision.js';
import { loadLevel, updateLevelTransition } from './levelManager.js';
import { renderGame } from './rendering.js';

const p5 = window.p5;

let gameInstance = new p5(p => {
  // Initialize logs
  p.logs = {
    game_info: [],
    inputs: [],
    player_info: []
  };

  let needsLevelLoad = false;

  p.setup = function() {
    p.createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    p.frameRate(60);
    p.randomSeed(42);
    
    p.logs.game_info.push({
      event: 'game_initialized',
      data: { canvas_width: CANVAS_WIDTH, canvas_height: CANVAS_HEIGHT },
      framecount: p.frameCount,
      timestamp: Date.now()
    });
  };

  p.draw = function() {
    // Handle level loading
    if (gameState.gamePhase === GAME_PHASES.PLAYING && !gameState.player) {
      loadLevel(p, gameState.level);
    }

    // Handle level transition
    if (gameState.levelTransitionTimer > 0) {
      updateLevelTransition(p);
      renderGame(p);
      return;
    }

    // Update game state
    if (gameState.gamePhase === GAME_PHASES.PLAYING) {
      // Handle player movement
      handlePlayerMovement(p);

      // Update all entities
      for (const entity of gameState.entities) {
        entity.update();
      }

      // Remove dead particles
      gameState.entities = gameState.entities.filter(e => {
        if (e.type === 'particle') {
          return !e.isDead();
        }
        return true;
      });

      // Check collisions
      checkCollisions(p);

      // Update invulnerability timer
      if (gameState.invulnerable) {
        gameState.invulnerableTimer--;
        if (gameState.invulnerableTimer <= 0) {
          gameState.invulnerable = false;
        }
      }

      // Log player position periodically
      if (p.frameCount % 60 === 0 && gameState.player) {
        p.logs.player_info.push({
          screen_x: gameState.player.x,
          screen_y: gameState.player.y,
          game_x: gameState.player.x,
          game_y: gameState.player.y,
          framecount: p.frameCount
        });
      }
    }

    // Render
    renderGame(p);
  };

  p.keyPressed = function() {
    handleKeyPressed(p, p.key, p.keyCode);
    return false;
  };

  p.keyReleased = function() {
    handleKeyReleased(p, p.key, p.keyCode);
    return false;
  };
});

// Expose game instance globally
window.gameInstance = gameInstance;
// Expose level loading for dev mode
window.loadLevel = function(levelNum) {
  const state = window.getGameState ? window.getGameState() : (window.gameState || (window.gameInstance && window.gameInstance.gameState));
  if (state) {
    state.currentLevel = levelNum;
    if (window.gameInstance) {
      // Try to import and call loadLevel if available
      if (typeof loadLevel === 'function') {
        loadLevel(window.gameInstance, levelNum);
      } else if (typeof initializeLevel === 'function') {
        initializeLevel(window.gameInstance, levelNum);
      }
      if (state.gamePhase !== undefined) {
        state.gamePhase = "PLAYING";
      }
    }
  }
};

// Control mode switching
window.setControlMode = function(mode) {
  // Ensure control mode is always HUMAN now
  gameState.controlMode = 'HUMAN';
  
  // Update button states
  const buttons = ['humanModeBtn']; // Only human mode button remains
  buttons.forEach(btnId => {
    const btn = document.getElementById(btnId);
    if (btn) {
      btn.classList.remove('active');
    }
  });
  
  const activeBtn = document.getElementById('humanModeBtn'); // Always activate human mode button
  if (activeBtn) {
    activeBtn.classList.add('active');
  }
  
  gameInstance.logs.game_info.push({
    event: 'control_mode_changed',
    data: { mode: gameState.controlMode }, // Log the actual mode set
    framecount: gameInstance.frameCount,
    timestamp: Date.now()
  });
};