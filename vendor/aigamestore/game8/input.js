// input.js - Input handling

import { gameState } from './globals.js';

export function handleKeyPressed(p) {
  // Log input
  p.logs.inputs.push({
    input_type: 'keyPressed',
    data: { key: p.key, keyCode: p.keyCode },
    framecount: p.frameCount,
    timestamp: Date.now()
  });

  // Global controls - Esc = pause, Enter = resume (when paused)
  if (p.keyCode === 13) { // ENTER - start or resume from pause
    if (gameState.gamePhase === 'START') {
      startGame(p);
    } else if (gameState.gamePhase === 'PAUSED') {
      unpauseGame(p);
    }
  } else if (p.keyCode === 27) { // ESC - pause (and unpause)
    if (gameState.gamePhase === 'PLAYING') {
      pauseGame(p);
    } else if (gameState.gamePhase === 'PAUSED') {
      unpauseGame(p);
    }
  } else if (p.keyCode === 82) { // R
    if (gameState.gamePhase === 'GAME_OVER_WIN' || gameState.gamePhase === 'GAME_OVER_LOSE') {
      restartGame(p, 'START'); // Manual restart, explicitly go to START screen
    }
  }

  // Gameplay controls
  if (gameState.gamePhase === 'PLAYING' && gameState.controlMode === 'HUMAN') {
    if (p.keyCode === 32) { // SPACE
      fireProjectile(p);
    } else if (p.keyCode === 90) { // Z
      swapBubbles();
    }
  }
}

export function handleContinuousInput(p) {
  if (gameState.gamePhase !== 'PLAYING' || gameState.controlMode !== 'HUMAN') return;

  if (p.keyIsDown(37)) { // LEFT ARROW
    if (gameState.player) {
      gameState.player.rotateLeft();
    }
  }
  if (p.keyIsDown(39)) { // RIGHT ARROW
    if (gameState.player) {
      gameState.player.rotateRight();
    }
  }
}

function startGame(p) {
  gameState.gamePhase = 'PLAYING';
  gameState.score = 0;
  gameState.currentLevel = 1;

  // Initialize level will be called in game loop
  p.logs.game_info.push({
    data: { event: 'game_start', gamePhase: gameState.gamePhase },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
}

function pauseGame(p) {
  gameState.gamePhase = 'PAUSED';
  p.logs.game_info.push({
    data: { event: 'game_paused', gamePhase: gameState.gamePhase },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
}

function unpauseGame(p) {
  gameState.gamePhase = 'PLAYING';
  // Adjust level start time to account for pause duration
  const pauseDuration = p.millis() - gameState.levelStartTime;
  gameState.levelStartTime += pauseDuration;

  p.logs.game_info.push({
    data: { event: 'game_unpaused', gamePhase: gameState.gamePhase },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
}

export function restartGame(p, targetPhase = 'START') { // Exported to be callable from game.js for auto-restart
  // Clear any pending auto-restart timer if it exists
  if (gameState.autoRestartTimeoutId) {
    clearTimeout(gameState.autoRestartTimeoutId);
    gameState.autoRestartTimeoutId = null;
  }
  gameState.autoRestartScheduled = false; // Reset the flag

  gameState.gamePhase = targetPhase; // Set game phase based on parameter
  gameState.score = 0;
  gameState.currentLevel = 1;
  gameState.bubbleGrid = []; // Clear grid to trigger initializeLevel in draw loop
  gameState.projectileBubble = null;
  gameState.nextBubble = null;
  gameState.canFire = true; // Ensure firing is possible
  gameState.swapAvailable = true; // Ensure swap is available
  if (gameState.player) {
    gameState.player.angle = -Math.PI / 2; // Reset shooter angle
  }
  gameState.entities = []; // Clear any active projectiles

  p.logs.game_info.push({
    data: { event: 'game_restart', gamePhase: gameState.gamePhase },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
}

function fireProjectile(p) {
  if (!gameState.canFire || !gameState.projectileBubble) return;
  if (gameState.shotsRemaining <= 0) return;

  const shooter = gameState.player;
  const dir = shooter.getShootDirection();
  const startX = shooter.x + dir.x * 30;
  const startY = shooter.y + dir.y * 30;

  // Create projectile from current bubble
  const projectile = {
    x: startX,
    y: startY,
    vx: dir.x,
    vy: dir.y,
    colorIndex: gameState.projectileBubble.colorIndex,
    color: gameState.projectileBubble.color,
    radius: gameState.projectileBubble.radius,
    speed: 8,
    active: true
  };

  gameState.entities.push(projectile);
  gameState.canFire = false;
  gameState.shotsRemaining--;

  // Move next bubble to current
  gameState.projectileBubble = gameState.nextBubble;
  gameState.nextBubble = null;
  gameState.swapAvailable = true;
}

function swapBubbles() {
  if (!gameState.swapAvailable || !gameState.projectileBubble || !gameState.nextBubble) return;

  const temp = gameState.projectileBubble;
  gameState.projectileBubble = gameState.nextBubble;
  gameState.nextBubble = temp;
  gameState.swapAvailable = false;
}

export { startGame, pauseGame, unpauseGame, fireProjectile, swapBubbles };