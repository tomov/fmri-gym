import { gameState, GAME_PHASES, CONTROL_MODES, CANVAS_WIDTH, CANVAS_HEIGHT, getGameState } from './globals.js';
import { Player } from './player.js';
import { handleKeyPressed, getKeys, resetGame, loadLevel } from './input.js'; // Imported resetGame and loadLevel
import { drawStartScreen, drawGame, drawGameOver } from './rendering.js'; // Import rendering functions
import { updatePhysics, updateParticles } from './physics.js'; // Import physics update functions

let gameInstance; // Declare gameInstance at the module scope

// Ensure p5.js is fully loaded and the DOM is ready before creating the sketch
window.onload = function() {
  const p5 = window.p5; // Moved this line inside window.onload to ensure p5 is available
  gameInstance = new p5(p => {
    // Game state is imported from globals

    p.setup = function() {
      p.createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
      p.frameRate(60);
      gameState.frameRate = p.frameRate(); // Capture the actual frame rate
      p.randomSeed(42);
      
      // Initialize logs
      p.logs = {
        game_info: [],
        inputs: [],
        player_info: []
      };

      // Initialize game state
      gameState.player = new Player(100, 100);
      gameState.entities = [gameState.player];
      gameState.unlockedWeapons = ['BUSTER'];
      gameState.currentWeapon = 0;
      
      p.logs.game_info.push({
        data: { phase: GAME_PHASES.START, message: "Game initialized" },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    };

    p.draw = function() {
      // Single background call
      p.background(20, 20, 40);

      if (gameState.gamePhase === GAME_PHASES.START) {
        drawStartScreen(p);
      } else if (gameState.gamePhase === GAME_PHASES.PLAYING) {
        updateGame(p);
        drawGame(p);
      } else if (gameState.gamePhase === GAME_PHASES.PAUSED) {
        drawGame(p);
      } else if (gameState.gamePhase === GAME_PHASES.GAME_OVER_WIN || gameState.gamePhase === GAME_PHASES.GAME_OVER_LOSE) {
        drawGameOver(p, gameState.gamePhase === GAME_PHASES.GAME_OVER_WIN);

        // Auto-restart logic
        if (!gameState.autoRestartTimer) {
            gameState.autoRestartTimer = p.frameCount;
        }
        const framesToWait = gameState.frameRate; // 1 second
        if (p.frameCount - gameState.autoRestartTimer > framesToWait) {
            resetGame(p, true); // Automatically restart the game, bypassing start screen
            gameState.autoRestartTimer = null; // Clear the timer after restart
        }
      }
    };

    function updateGame(p) {
      if (!gameState.currentStage) return;

      const keys = getKeys(p);

      // Update player
      if (gameState.player) {
        gameState.player.update(p, keys);
        
        // Log player info periodically
        if (p.frameCount % 30 === 0) {
          p.logs.player_info.push({
            screen_x: gameState.player.x - gameState.camera.x,
            screen_y: gameState.player.y - gameState.camera.y,
            game_x: gameState.player.x,
            game_y: gameState.player.y,
            framecount: p.frameCount
          });
        }

        // Update camera
        const targetCameraX = gameState.player.x - CANVAS_WIDTH / 2;
        gameState.camera.x = p.constrain(targetCameraX, 0, 
          Math.max(0, gameState.currentStage.width - CANVAS_WIDTH));
      }

      // Update entities
      for (let entity of gameState.entities) {
        if (entity !== gameState.player) {
          entity.update(p);
        }
      }

      // Update stage
      gameState.currentStage.update(p);

      // Update physics
      updatePhysics(p);
      updateParticles(p);
    }

    p.keyPressed = function() {
      handleKeyPressed(p);
    };
  }, document.body); // End of new p5 instance creation

  // Expose game instance globally
  window.gameInstance = gameInstance;
  // Expose level loading for dev mode
  window.loadLevel = function(levelNum) {
    // Use the gameInstance's p (sketch instance) to call the actual loadLevel and resetGame
    const p = window.gameInstance; 
    if (p) {
      // Ensure state is reset before loading new level for dev mode
      resetGame(p); // Call the imported resetGame
      p.getGameState().currentLevel = levelNum;
      loadLevel(p, levelNum); // Call the imported loadLevel
    } else {
      console.warn("p5 instance (gameInstance) not available for dev loadLevel.");
    }
  };
}; // End of window.onload

// Control mode setter
window.setControlMode = function(mode) {
  if (CONTROL_MODES[mode]) {
    gameState.controlMode = mode;
    
    // Update button styles
    const buttons = document.querySelectorAll('.control-button');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    if (mode === 'HUMAN') {
      document.getElementById('humanModeBtn').classList.add('active');
    } else if (mode === 'TEST_1') {
      document.getElementById('test_1_ModeBtn').classList.add('active');
    } else if (mode === 'TEST_2') {
      document.getElementById('test_2_ModeBtn').classList.add('active');
    }
  }
};