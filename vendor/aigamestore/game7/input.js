import { KEY, gameState, nextLevel, resetToLevel1 } from './globals.js';
import { Player } from './entities.js';
import { generateLevel } from './level.js';
import { isReplayMode, getInputsForFrame, setCurrentFrame } from './replay_controller.js';
import { setSeed } from './rng.js'; // Import setSeed for deterministic restarts

export const keys = {
  up: false,
  down: false,
  left: false,
  right: false,
  shoot: false,
  sprint: false,
  crouch: false
};

export function setupInputHandlers(p) {
  p.keyPressed = function() {
    // In replay mode, ignore keyboard input
    if (isReplayMode()) {
      return false;
    }
    
    handleKeyPress(p, p.keyCode);
    
    p.logs.inputs.push({
      "input_type": "keyPressed",
      "data": { "key": p.key, "keyCode": p.keyCode },
      "framecount": p.frameCount,
      "timestamp": Date.now()
    });
    
    return !(Object.values(KEY).includes(p.keyCode));
  };
  
  p.keyReleased = function() {
    // In replay mode, ignore keyboard input
    if (isReplayMode()) {
      return false;
    }
    
    handleKeyRelease(p, p.keyCode);
    
    p.logs.inputs.push({
      "input_type": "keyReleased",
      "data": { "key": p.key, "keyCode": p.keyCode },
      "framecount": p.frameCount,
      "timestamp": Date.now()
    });
    
    return !(Object.values(KEY).includes(p.keyCode));
  };
}

// Track which keys are currently pressed in replay mode
let replayKeyState = {
  [KEY.UP]: false,
  [KEY.DOWN]: false,
  [KEY.LEFT]: false,
  [KEY.RIGHT]: false,
  [KEY.Z]: false,
  [KEY.SPACE]: false,
  [KEY.SHIFT]: false
};

/**
 * Reset replay key state (called when replay stops)
 */
export function resetReplayKeyState() {
  replayKeyState[KEY.UP] = false;
  replayKeyState[KEY.DOWN] = false;
  replayKeyState[KEY.LEFT] = false;
  replayKeyState[KEY.RIGHT] = false;
  replayKeyState[KEY.Z] = false;
  replayKeyState[KEY.SPACE] = false;
  replayKeyState[KEY.SHIFT] = false;
  
  keys.up = false;
  keys.down = false;
  keys.left = false;
  keys.right = false;
  keys.shoot = false;
  keys.sprint = false;
  keys.crouch = false;
}

/**
 * Apply replay inputs for the current frame
 * This should be called from the game loop before processing player input
 * @param {p5} p - p5 instance
 */
export function applyReplayInputs(p) {
  if (!isReplayMode()) {
    return;
  }
  
  // Update current frame in replay controller
  setCurrentFrame(p.frameCount);
  
  // Get inputs for this frame
  const frameInputs = getInputsForFrame(p.frameCount);
  
  // Apply pressed keys - set key state to true
  for (const keyCode of frameInputs.pressed) {
    replayKeyState[keyCode] = true;
    
    // Handle special keys that trigger actions
    switch (keyCode) {
      case KEY.ENTER:
        // Handle phase transitions
        if (gameState.gamePhase === "START") {
          startGame(p);
        }
        break;
      case KEY.ESC:
        // Handle pause
        if (gameState.gamePhase === "PLAYING") {
          pauseGame(p);
        } else if (gameState.gamePhase === "PAUSED") {
          resumeGame(p);
        }
        break;
    }
  }
  
  // Apply released keys - set key state to false
  for (const keyCode of frameInputs.released) {
    replayKeyState[keyCode] = false;
  }
  
  // Map replay key state to keys object
  keys.up = replayKeyState[KEY.UP] || false;
  keys.down = replayKeyState[KEY.DOWN] || false;
  keys.left = replayKeyState[KEY.LEFT] || false;
  keys.right = replayKeyState[KEY.RIGHT] || false;
  keys.shoot = replayKeyState[KEY.Z] || false;
  keys.sprint = replayKeyState[KEY.SPACE] || false;
  keys.crouch = replayKeyState[KEY.SHIFT] || false;
}

export function handleKeyPress(p, keyCode) {
  if (gameState.controlMode === "HUMAN") {
    switch (keyCode) {
      case KEY.UP:
        keys.up = true;
        break;
      case KEY.DOWN:
        keys.down = true;
        break;
      case KEY.LEFT:
        keys.left = true;
        break;
      case KEY.RIGHT:
        keys.right = true;
        break;
      case KEY.Z:
        keys.shoot = true;
        break;
      case KEY.SPACE:
        keys.sprint = true;
        break;
      case KEY.SHIFT:
        keys.crouch = true;
        break;
      case KEY.KEY_1:
        if (gameState.player) gameState.player.switchWeapon("pistol");
        break;
      case KEY.KEY_2:
        if (gameState.player) gameState.player.switchWeapon("rifle");
        break;
      case KEY.KEY_3:
        if (gameState.player) gameState.player.switchWeapon("shotgun");
        break;
      case KEY.KEY_4:
        if (gameState.player) gameState.player.switchWeapon("sniper");
        break;
      case KEY.ENTER:
        if (gameState.gamePhase === "START") {
          startGame(p);
        } else if (gameState.gamePhase === "PAUSED") {
          resumeGame(p);
        } else if (gameState.gamePhase === "GAME_OVER_WIN") {
          // Progress to next level
          gameState.autoRestartTimer = null; // Cancel any pending auto-restart
          nextLevel(); // Increments level, calls resetGame(), sets player=null
          
          // Re-initialize level and player
          setSeed(42); // Ensure deterministic level generation
          p.randomSeed(42);
          generateLevel(p);
          gameState.player = new Player(gameState.level.width / 2, gameState.level.height / 2);
          gameState.gamePhase = "PLAYING"; // Set phase to PLAYING
          
          p.logs.game_info.push({
            "game_status": gameState.gamePhase,
            "data": { "level": gameState.currentLevel },
            "framecount": p.frameCount,
            "timestamp": Date.now()
          });
        }
        break;
      case KEY.ESC:
        if (gameState.gamePhase === "PLAYING") {
          pauseGame(p);
        } else if (gameState.gamePhase === "PAUSED") {
          resumeGame(p);
        }
        break;
      case KEY.R:
        if (gameState.gamePhase === "GAME_OVER_WIN" || gameState.gamePhase === "GAME_OVER_LOSE") {
          gameState.autoRestartTimer = null; // Cancel any pending auto-restart
          resetToStart(p);
        }
        break;
    }
  }
}

export function handleKeyRelease(p, keyCode) {
  if (gameState.controlMode === "HUMAN") {
    switch (keyCode) {
      case KEY.UP:
        keys.up = false;
        break;
      case KEY.DOWN:
        keys.down = false;
        break;
      case KEY.LEFT:
        keys.left = false;
        break;
      case KEY.RIGHT:
        keys.right = false;
        break;
      case KEY.Z:
        keys.shoot = false;
        break;
      case KEY.SPACE:
        keys.sprint = false;
        break;
      case KEY.SHIFT:
        keys.crouch = false;
        break;
    }
  }
}

export function handleAutomatedInputs(p) {
  if (gameState.gamePhase === "PLAYING" && gameState.controlMode !== "HUMAN") {
    const action = window.game_testing_controller(gameState);
    
    keys.up = false;
    keys.down = false;
    keys.left = false;
    keys.right = false;
    keys.shoot = false;
    keys.sprint = false;
    keys.crouch = false;
    
    if (action !== null) {
      switch (action) {
        case KEY.UP:
          keys.up = true;
          break;
        case KEY.DOWN:
          keys.down = true;
          break;
        case KEY.LEFT:
          keys.left = true;
          break;
        case KEY.RIGHT:
          keys.right = true;
          break;
        case KEY.Z:
          keys.shoot = true;
          break;
        case KEY.SPACE:
          keys.sprint = true;
          break;
        case KEY.SHIFT:
          keys.crouch = true;
          break;
      }
    }
  }
}

export function startGame(p) {
  gameState.gamePhase = "PLAYING";
  gameState.autoRestartTimer = null; // Ensure auto-restart timer is cleared on manual start
  
  p.logs.game_info.push({
    "game_status": gameState.gamePhase,
    "data": {},
    "framecount": p.frameCount,
    "timestamp": Date.now()
  });
}

export function pauseGame(p) {
  gameState.gamePhase = "PAUSED";
  
  p.logs.game_info.push({
    "game_status": gameState.gamePhase,
    "data": {},
    "framecount": p.frameCount,
    "timestamp": Date.now()
  });
}

export function resumeGame(p) {
  gameState.gamePhase = "PLAYING";
  
  p.logs.game_info.push({
    "game_status": gameState.gamePhase,
    "data": {},
    "framecount": p.frameCount,
    "timestamp": Date.now()
  });
}

export function resetToStart(p) {
  resetToLevel1(); // Sets currentLevel to 1, calls resetGame(), sets player=null
  gameState.gamePhase = "START"; // Explicitly sets phase to START
  gameState.autoRestartTimer = null; // Ensure auto-restart timer is cleared on manual reset
  
  p.logs.game_info.push({
    "game_status": gameState.gamePhase,
    "data": {},
    "framecount": p.frameCount,
    "timestamp": Date.now()
  });
}