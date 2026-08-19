import { gameState, SLINGSHOT_X, SLINGSHOT_Y, MAX_PULL_DISTANCE, LAUNCH_POWER_MULTIPLIER } from './globals.js';
import { Bird } from './entities.js';
import { createLevel } from './levels.js';
import { clearPhysicsWorld } from './physics.js';
// NEW: Import centralized game control functions from game.js
import { startGame, loadNextLevel, restartGame } from './game.js';

const Matter = window.Matter;
const Body = Matter.Body;

// NEW: handleKeyPressed now accepts callbacks for game control functions
export function handleKeyPressed(p, key, keyCode, startGameCallback, loadNextLevelCallback, restartGameCallback) {
  // Track key state for continuous controls
  gameState.keysPressed[keyCode] = true;

  // Log input
  p.logs.inputs.push({
    input_type: 'keyPressed',
    data: { key, keyCode },
    framecount: p.frameCount,
    timestamp: Date.now()
  });

  // ENTER - Start game, advance level, or resume from pause
  if (keyCode === 13) {
    if (gameState.gamePhase === "START") {
      startGameCallback(p); // Use the provided callback
    } else if (gameState.gamePhase === "LEVEL_COMPLETE") {
      loadNextLevelCallback(p); // Use the provided callback
    } else if (gameState.gamePhase === "PAUSED") {
      gameState.gamePhase = "PLAYING";
      p.logs.game_info.push({
        data: { phase: "PLAYING" },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    }
  }

  // ESC - Pause/Unpause
  if (keyCode === 27) {
    if (gameState.gamePhase === "PLAYING") {
      gameState.gamePhase = "PAUSED";
      p.logs.game_info.push({
        data: { phase: "PAUSED" },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    } else if (gameState.gamePhase === "PAUSED") {
      gameState.gamePhase = "PLAYING";
      p.logs.game_info.push({
        data: { phase: "PLAYING" },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    }
  }

  // R - Restart
  if (keyCode === 82) {
    // Call restartGameCallback without the 'returnToStart' parameter,
    // relying on its default value of 'true' to return to the start screen.
    restartGameCallback(p); 
  }

  // SPACE - Toggle aiming or launch
  if (keyCode === 32 && gameState.gamePhase === "PLAYING") {
    if (!gameState.isAiming) {
      // Start aiming if bird is available and no active birds
      if (gameState.birdsRemaining.length > 0 && gameState.activeBirds.length === 0) {
        gameState.isAiming = true;
        gameState.slingshotPullPos = { x: 0, y: 0 };
      }
    } else {
      // Launch the bird
      launchBird(p);
    }
  }

  // Z - Activate bird ability
  if (keyCode === 90 && gameState.gamePhase === "PLAYING") {
    activateBirdAbility(p);
  }
}

export function handleKeyReleased(p, key, keyCode) {
  // Track key state for continuous controls
  gameState.keysPressed[keyCode] = false;

  // Log input
  p.logs.inputs.push({
    input_type: 'keyReleased',
    data: { key, keyCode },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
}

export function updateAiming(p) {
  // Continuous aim adjustment when aiming and arrow keys are held
  if (gameState.isAiming && gameState.slingshotPullPos) {
    const pullSpeed = 2; // Continuous speed per frame (slower than tap-based)
    let changed = false;

    if (gameState.keysPressed[37]) { // LEFT
      gameState.slingshotPullPos.x -= pullSpeed;
      changed = true;
    }
    if (gameState.keysPressed[39]) { // RIGHT
      gameState.slingshotPullPos.x += pullSpeed;
      changed = true;
    }
    if (gameState.keysPressed[38]) { // UP
      gameState.slingshotPullPos.y -= pullSpeed;
      changed = true;
    }
    if (gameState.keysPressed[40]) { // DOWN
      gameState.slingshotPullPos.y += pullSpeed;
      changed = true;
    }

    // Clamp to max pull distance if changed
    if (changed) {
      const dist = Math.sqrt(
        gameState.slingshotPullPos.x * gameState.slingshotPullPos.x +
        gameState.slingshotPullPos.y * gameState.slingshotPullPos.y
      );
      if (dist > MAX_PULL_DISTANCE) {
        const scale = MAX_PULL_DISTANCE / dist;
        gameState.slingshotPullPos.x *= scale;
        gameState.slingshotPullPos.y *= scale;
      }
    }
  }
}

export function updateTestingControls(p) {
  if (gameState.controlMode === 'HUMAN') return;

  if (gameState.controlMode === 'TEST_1') {
    // Basic testing - auto-launch at medium power
    if (gameState.gamePhase === "START" && p.frameCount % 10 === 0) {
      startGame(p); // Use imported startGame
    }
    
    if (gameState.gamePhase === "PLAYING" && !gameState.isAiming && gameState.birdsRemaining.length > 0 && gameState.activeBirds.length === 0) {
      if (p.frameCount % 120 === 0) {
        gameState.isAiming = true;
        gameState.slingshotPullPos = { x: 30, y: 20 };
      }
    }
    
    if (gameState.isAiming && p.frameCount % 10 === 0) {
      launchBird(p);
    }
  }

  if (gameState.controlMode === 'TEST_2') {
    // Test to win - smart targeting
    if (gameState.gamePhase === "START" && p.frameCount % 10 === 0) {
      startGame(p); // Use imported startGame
    }
    
    if (gameState.gamePhase === "PLAYING" && !gameState.isAiming && gameState.birdsRemaining.length > 0 && gameState.activeBirds.length === 0) {
      if (p.frameCount % 120 === 0) {
        gameState.isAiming = true;
        // Aim at structures
        gameState.slingshotPullPos = { x: 45, y: 25 };
      }
    }
    
    if (gameState.isAiming && p.frameCount % 10 === 0) {
      launchBird(p);
    }
    
    // Auto-use abilities
    if (gameState.activeBirds.length > 0 && p.frameCount % 30 === 15) {
      activateBirdAbility(p);
    }
    
    // Auto-advance
    if (gameState.gamePhase === "LEVEL_COMPLETE" && p.frameCount % 120 === 60) {
      loadNextLevel(p); // Use imported loadNextLevel
    }
  }
}

// REMOVED: startGame function (now imported from game.js)
// REMOVED: loadNextLevel function (now imported from game.js)
// REMOVED: updateHighScore function (now imported from game.js)
// REMOVED: restartGame function (now imported from game.js)

function launchBird(p) {
  if (gameState.birdsRemaining.length === 0) return;

  const birdType = gameState.birdsRemaining.shift();
  const bird = new Bird(SLINGSHOT_X, SLINGSHOT_Y, birdType);
  
  // Apply launch force
  const vx = -gameState.slingshotPullPos.x * LAUNCH_POWER_MULTIPLIER;
  const vy = -gameState.slingshotPullPos.y * LAUNCH_POWER_MULTIPLIER;
  Body.setVelocity(bird.body, { x: vx, y: vy });
  
  gameState.activeBirds.push(bird);
  gameState.entities.push(bird);
  gameState.isAiming = false;
  gameState.slingshotPullPos = null;
}

function activateBirdAbility(p) {
  if (gameState.activeBirds.length === 0) return;

  const bird = gameState.activeBirds[0];
  if (bird && !bird.abilityUsed && bird.active) {
    const newBirds = bird.useAbility(gameState, p);
    if (newBirds) {
      gameState.activeBirds.push(...newBirds);
      gameState.entities.push(...newBirds);
    }
  }
}