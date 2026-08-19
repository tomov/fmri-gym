import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT, GROUND_Y, SCORE_UNUSED_BIRD } from './globals.js';
import { initPhysicsEngine, updatePhysics, clearPhysicsWorld } from './physics.js';
import { Bird, Pig, StructureBlock } from './entities.js';
import { createLevel } from './levels.js';
import { handleKeyPressed, handleKeyReleased, updateAiming, updateTestingControls } from './input.js';
import { renderGame } from './renderer.js';

function getCurrentLevelScore() {
  return gameState.score - gameState.levelStartScore;
}

// Centralized game control functions, exported for use by input.js and internal calls
function startGame(p) {
  clearPhysicsWorld();
  
  const levelData = createLevel(1);
  
  gameState.gamePhase = "PLAYING";
  gameState.currentLevel = 1;
  gameState.score = 0;
  gameState.levelStartScore = 0;
  gameState.birdsRemaining = [...levelData.birds];
  gameState.entities = [...levelData.pigs, ...levelData.structures];
  gameState.pigsRemaining = levelData.pigs.length;
  gameState.activeBirds = [];
  gameState.particleEffects = [];
  gameState.keysPressed = {};
  
  // Auto-start aiming when level begins
  if (gameState.birdsRemaining.length > 0) {
    gameState.isAiming = true;
    gameState.slingshotPullPos = { x: 0, y: 0 };
  } else {
    gameState.isAiming = false;
  }
  
  // Load high score
  const savedHighScore = localStorage.getItem('game2HighScore');
  if (savedHighScore) {
    gameState.highScore = parseInt(savedHighScore);
  }

  p.logs.game_info.push({
    data: { phase: "PLAYING", level: 1 },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
}

function loadNextLevel(p) {
  clearPhysicsWorld();
  
  gameState.currentLevel++;
  const levelData = createLevel(gameState.currentLevel);
  
  if (!levelData) {
    gameState.gamePhase = "GAME_OVER_WIN";
    updateHighScore();
    p.logs.game_info.push({
      data: { phase: "GAME_OVER_WIN", score: gameState.score },
      framecount: p.frameCount,
      timestamp: Date.now()
    });
    return;
  }
  
  gameState.gamePhase = "PLAYING";
  gameState.levelStartScore = gameState.score;
  gameState.birdsRemaining = [...levelData.birds];
  gameState.entities = [...levelData.pigs, ...levelData.structures];
  gameState.pigsRemaining = levelData.pigs.length;
  gameState.activeBirds = [];
  gameState.particleEffects = [];
  gameState.keysPressed = {};
  
  // Auto-start aiming when level begins
  if (gameState.birdsRemaining.length > 0) {
    gameState.isAiming = true;
    gameState.slingshotPullPos = { x: 0, y: 0 };
  } else {
    gameState.isAiming = false;
  }
  
  p.logs.game_info.push({
    data: { phase: "PLAYING", level: gameState.currentLevel },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
}

function updateHighScore() {
  if (gameState.score > gameState.highScore) {
    gameState.highScore = gameState.score;
    localStorage.setItem('game2HighScore', gameState.highScore.toString());
  }
}

function restartGame(p, returnToStart = true) { // Added returnToStart parameter, defaults to true
  // Cancel any pending auto-restart
  if (gameState.autoRestartTimeoutId) {
    clearTimeout(gameState.autoRestartTimeoutId);
    gameState.autoRestartTimeoutId = null;
  }
  gameState.autoRestartScheduled = false; // Reset flag

  if (returnToStart) {
    // Original behavior: return to start screen
    clearPhysicsWorld();
    
    gameState.gamePhase = "START";
    gameState.entities = [];
    gameState.activeBirds = [];
    gameState.birdsRemaining = [];
    gameState.isAiming = false;
    gameState.slingshotPullPos = null; // Ensure slingshot pull position is reset
    gameState.particleEffects = [];
    gameState.keysPressed = {};
    gameState.score = 0; // Reset score on full restart
    gameState.currentLevel = 1; // Reset level on full restart
    gameState.levelStartScore = 0; // Reset level score baseline on full restart
    
    p.logs.game_info.push({
      data: { phase: "START" },
      framecount: p.frameCount,
      timestamp: Date.now()
    });
  } else {
    // New behavior for auto-restart: immediately start a new game (Level 1)
    startGame(p); // startGame handles clearing physics, setting phase to PLAYING, and initializing level 1
  }
}

// Export these centralized functions so input.js can use them
export { startGame, loadNextLevel, restartGame, updateHighScore };

let gameInstance; // Declare gameInstance outside to be accessible
window.onload = function() {
  const p5 = window.p5;
  if (!p5) {
    console.error("p5.js is not loaded. Cannot initialize game.");
    return;
  }

  gameInstance = new p5(p => {
    p.setup = function() {
      p.createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
      p.frameRate(60);
      p.randomSeed(42);
      
      // Initialize Matter.js physics
      initPhysicsEngine();
      
      // Initialize logs
      p.logs = {
        game_info: [],
        inputs: [],
        player_info: []
      };
      
      // Log initial state
      p.logs.game_info.push({
        data: { phase: "START" },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    };

    p.draw = function() {
      // Clear background once
      p.background(135, 206, 235);
      
      // Update game logic
      if (gameState.gamePhase === "PLAYING") {
        updateAiming(p);
        updateTestingControls(p);
        updatePhysics();
        updateGameState(p);
        checkWinLoseConditions(p);
      }
      
      // Render
      renderGame(p);

      // NEW: Clear auto-restart timer if game phase changes away from GAME_OVER
      // This handles cases like manual restart (R key) taking precedence.
      if (gameState.gamePhase !== "GAME_OVER_WIN" && gameState.gamePhase !== "GAME_OVER_LOSE") {
          if (gameState.autoRestartTimeoutId) {
              clearTimeout(gameState.autoRestartTimeoutId);
              gameState.autoRestartTimeoutId = null;
              gameState.autoRestartScheduled = false;
          }
      }
    };

    p.keyPressed = function() {
      // Pass the p5 instance and the centralized game control functions to the input handler
      handleKeyPressed(p, p.key, p.keyCode, startGame, loadNextLevel, restartGame);
    };

    p.keyReleased = function() {
      handleKeyReleased(p, p.key, p.keyCode);
    };

    function updateGameState(p) {
      // Update trails for birds
      gameState.activeBirds.forEach(bird => {
        if (bird.active) {
          bird.updateTrail();
        }
      });
      
      // Remove inactive birds from active list
      const World = window.Matter.World;
      gameState.activeBirds = gameState.activeBirds.filter(bird => {
        if (!bird.active) {
          // Ensure body is removed from world when bird becomes inactive
          if (bird.body && gameState.matterWorld) {
            World.remove(gameState.matterWorld, bird.body);
            bird.body = null; // Clear reference to prevent stale body issues
          }
          return false;
        }
        
        // Remove birds that are too slow (settled anywhere)
        const speed = Math.sqrt(
          bird.body.velocity.x * bird.body.velocity.x +
          bird.body.velocity.y * bird.body.velocity.y
        );
        if (speed < 1) {
          bird.active = false;
          // Ensure body is removed from world
          if (bird.body && gameState.matterWorld) {
            World.remove(gameState.matterWorld, bird.body);
            bird.body = null; // Clear reference to prevent stale body issues
          }
          return false;
        }
        
        return true;
      });
      
      // Auto-start aiming for next bird if no active birds and birds remaining
      if (gameState.activeBirds.length === 0 && gameState.birdsRemaining.length > 0 && !gameState.isAiming) {
        gameState.isAiming = true;
        gameState.slingshotPullPos = { x: 0, y: 0 };
      }
      
      // Remove inactive entities and ensure their bodies are removed from world
      gameState.entities = gameState.entities.filter(entity => {
        if (!entity.active) {
          // Ensure body is removed from world when entity becomes inactive
          if (entity.body && gameState.matterWorld) {
            World.remove(gameState.matterWorld, entity.body);
            entity.body = null; // Clear reference to prevent stale body issues
          }
          return false;
        }
        return true;
      });
      
      // Update pig count
      gameState.pigsRemaining = gameState.entities.filter(e => e.type === 'pig' && e.active).length;
    }

    function checkWinLoseConditions(p) {
      // Check win condition
      if (gameState.pigsRemaining === 0) {
        // Add bonus for unused birds
        const bonusScore = gameState.birdsRemaining.length * SCORE_UNUSED_BIRD;
        gameState.score += bonusScore;
        
        if (gameState.currentLevel >= gameState.totalLevels) {
          // Game complete
          gameState.gamePhase = "GAME_OVER_WIN";
          updateHighScore();
          p.logs.game_info.push({
            data: { phase: "GAME_OVER_WIN", score: gameState.score },
            framecount: p.frameCount,
            timestamp: Date.now()
          });
          // NEW: Add auto-restart logic for GAME_OVER_WIN
          if (!gameState.autoRestartScheduled) {
              gameState.autoRestartScheduled = true;
              gameState.autoRestartTimeoutId = setTimeout(() => {
                  // Only restart if still in GAME_OVER_WIN (manual restart could have happened)
                  if (gameState.gamePhase === "GAME_OVER_WIN") {
                      restartGame(p, false); // Pass false to immediately start a new game
                  }
                  gameState.autoRestartScheduled = false;
                  gameState.autoRestartTimeoutId = null;
              }, 1000); // 1 second delay
          }
        } else {
          // Level complete
          gameState.gamePhase = "LEVEL_COMPLETE";
          p.logs.game_info.push({
            data: { phase: "LEVEL_COMPLETE", level: gameState.currentLevel, score: getCurrentLevelScore() },
            framecount: p.frameCount,
            timestamp: Date.now()
          });
          
          // Auto-advance after delay
          setTimeout(() => {
            if (gameState.gamePhase === "LEVEL_COMPLETE") {
              loadNextLevel(p); // Use the centralized loadNextLevel
            }
          }, 3000);
        }
      }
      
      // Check lose condition
      if (gameState.birdsRemaining.length === 0 && gameState.activeBirds.length === 0 && gameState.pigsRemaining > 0) {
        gameState.gamePhase = "GAME_OVER_LOSE";
        updateHighScore();
        p.logs.game_info.push({
          data: { phase: "GAME_OVER_LOSE", score: gameState.score },
          framecount: p.frameCount,
          timestamp: Date.now()
        });
        // NEW: Add auto-restart logic for GAME_OVER_LOSE
        if (!gameState.autoRestartScheduled) {
            gameState.autoRestartScheduled = true;
            gameState.autoRestartTimeoutId = setTimeout(() => {
                // Only restart if still in GAME_OVER_LOSE (manual restart could have happened)
                if (gameState.gamePhase === "GAME_OVER_LOSE") {
                    restartGame(p, false); // Pass false to immediately start a new game
                }
                gameState.autoRestartScheduled = false;
                gameState.autoRestartTimeoutId = null;
            }, 1000); // 1 second delay
        }
      }
    }

    // Expose game instance and state globally
    window.gameInstance = gameInstance;
    // Expose level loading for dev mode
    window.loadLevel = function(levelNum) {
      const state = window.getGameState ? window.getGameState() : (window.gameState || (window.gameInstance && window.gameInstance.gameState));
      if (state) {
        state.currentLevel = levelNum;
        // Use the centralized loadNextLevel
        loadNextLevel(p);
      }
    };
    window.getGameState = function() {
      return gameState;
    };

    // Control mode switching
    window.setControlMode = function(mode) {
      gameState.controlMode = mode;
      
      // Update button states
      document.querySelectorAll('.control-button').forEach(btn => {
        btn.classList.remove('active');
      });
      
      const buttonMap = {
        'HUMAN': 'humanModeBtn',
        'TEST_1': 'test_1_ModeBtn',
        'TEST_2': 'test_2_ModeBtn'
      };
      
      const activeBtn = document.getElementById(buttonMap[mode]);
      if (activeBtn) {
        activeBtn.classList.add('active');
      }
    };
  });
};

export { gameInstance };
