// Replay Controller - Loads and manages replay data from inputs.json and logs.json
import { KEY, gameState } from './globals.js';

let replayData = null;
let isReplayActive = false;
let currentFrame = 0;
let replaySpeed = 1.0;
let isPaused = false;
let frameInputs = {}; // Map of frame number to active inputs
let gameInfoEvents = []; // Game state transitions
let playerInfoLog = []; // Logged player positions for validation
let startTimestamp = null;
let originalStartTimestamp = null;

// Map keyboard event keys to game KEY constants
const KEY_MAP = {
  "ArrowLeft": KEY.LEFT,
  "ArrowRight": KEY.RIGHT,
  "ArrowUp": KEY.UP,
  "ArrowDown": KEY.DOWN,
  "z": KEY.Z,
  "Z": KEY.Z,
  " ": KEY.SPACE,
  "Shift": KEY.SHIFT,
  "Enter": KEY.ENTER,
  "Escape": KEY.ESC,
  "Esc": KEY.ESC,
  "r": KEY.R,
  "R": KEY.R,
  "1": KEY.KEY_1,
  "2": KEY.KEY_2,
  "3": KEY.KEY_3,
  "4": KEY.KEY_4,
};

// Map keyboard event codes to game KEY constants
const CODE_MAP = {
  "ArrowLeft": KEY.LEFT,
  "ArrowRight": KEY.RIGHT,
  "ArrowUp": KEY.UP,
  "ArrowDown": KEY.DOWN,
  "KeyZ": KEY.Z,
  "Space": KEY.SPACE,
  "ShiftLeft": KEY.SHIFT,
  "ShiftRight": KEY.SHIFT,
  "Enter": KEY.ENTER,
  "Escape": KEY.ESC,
  "KeyR": KEY.R,
  "Digit1": KEY.KEY_1,
  "Digit2": KEY.KEY_2,
  "Digit3": KEY.KEY_3,
  "Digit4": KEY.KEY_4,
};

/**
 * Initialize replay mode with inputs.json and logs.json data
 * @param {Object} inputsData - Parsed inputs.json data
 * @param {Object} logsData - Parsed logs.json data
 */
export function initReplay(inputsData, logsData) {
  replayData = {
    inputs: inputsData,
    logs: logsData
  };
  
  isReplayActive = true;
  currentFrame = 0;
  replaySpeed = 1.0;
  isPaused = false;
  frameInputs = {};
  gameInfoEvents = [];
  playerInfoLog = [];
  
  // Parse logs.json FIRST (has framecounts, more accurate)
  if (logsData) {
    if (logsData.game_info) {
      gameInfoEvents = logsData.game_info;
    }
    if (logsData.player_info) {
      playerInfoLog = logsData.player_info;
    }
    if (logsData.inputs && logsData.inputs.length > 0) {
      // Prioritize inputs from logs.json - they have framecounts!
      console.log('[Replay] Loading inputs from logs.json (with framecounts)');
      parseInputEvents(logsData.inputs);
    }
  }
  
  // Parse inputs.json events (fallback or supplement)
  if (inputsData && inputsData.events && inputsData.events.length > 0) {
    // Use inputs.json if logs.json didn't have inputs, or if we want to supplement
    const hasLogsInputs = logsData && logsData.inputs && logsData.inputs.length > 0;
    
    if (!hasLogsInputs) {
      console.log('[Replay] Loading inputs from inputs.json (timestamp-based, using frame anchors)');
      // Use improved frame estimation with anchors from logs
      parseInputEventsWithAnchors(inputsData.events, gameInfoEvents, playerInfoLog);
    } else {
      console.log('[Replay] Using logs.json inputs (inputs.json available but not needed)');
    }
  }
  
  // Find the first timestamp to use as baseline
  if (inputsData && inputsData.events && inputsData.events.length > 0) {
    originalStartTimestamp = inputsData.events[0].timestamp;
    startTimestamp = Date.now();
  } else if (gameInfoEvents.length > 0) {
    originalStartTimestamp = gameInfoEvents[0].timestamp;
    startTimestamp = Date.now();
  }
  
  console.log('[Replay] Initialized replay:', {
    inputEventsFromInputsJson: inputsData?.events?.length || 0,
    inputEventsFromLogsJson: logsData?.inputs?.length || 0,
    gameInfoEvents: gameInfoEvents.length,
    playerInfoFrames: playerInfoLog.length,
    totalFrameInputs: Object.keys(frameInputs).length,
    firstFrame: gameInfoEvents.length > 0 ? gameInfoEvents[0].framecount : 0,
    lastFrame: playerInfoLog.length > 0 ? playerInfoLog[playerInfoLog.length - 1].framecount : 0
  });
}

/**
 * Parse input events and build frame-based input map
 * @param {Array} events - Array of input events
 */
function parseInputEvents(events) {
  // Group events by framecount if available, otherwise estimate from timestamps
  let frameEstimate = 0;
  let lastTimestamp = null;
  let firstTimestamp = null;
  const frameRate = 60; // 60 FPS
  
  for (const event of events) {
    let eventFrame = frameEstimate;
    
    // If event has framecount, use it (most accurate)
    if (event.framecount !== undefined && event.framecount !== null) {
      eventFrame = event.framecount;
      // Update frame estimate to match
      frameEstimate = eventFrame;
    } else if (event.timestamp) {
      // Estimate frame from timestamp
      if (firstTimestamp === null) {
        firstTimestamp = event.timestamp;
      }
      
      if (lastTimestamp !== null) {
        // Estimate frame from timestamp difference
        const timeDiff = event.timestamp - lastTimestamp;
        const framesDiff = Math.round((timeDiff / 1000) * frameRate);
        eventFrame = frameEstimate + Math.max(0, framesDiff); // Don't go backwards
      } else {
        // First event - estimate from first timestamp
        const timeFromStart = event.timestamp - firstTimestamp;
        eventFrame = Math.round((timeFromStart / 1000) * frameRate);
      }
    }
    
    // Map event to key code
    let keyCode = null;
    const eventType = event.type || event.input_type;
    
    // Check data object first (for logs.json format with nested data)
    const eventData = event.data || event;
    
    // Try to get keyCode from various sources (data object first, then direct)
    if (eventData.keyCode !== undefined) {
      keyCode = eventData.keyCode;
    } else if (event.keyCode !== undefined) {
      keyCode = event.keyCode;
    } else if (eventData.key) {
      keyCode = KEY_MAP[eventData.key];
    } else if (event.key) {
      keyCode = KEY_MAP[event.key];
    } else if (eventData.code) {
      keyCode = CODE_MAP[eventData.code];
    } else if (event.code) {
      keyCode = CODE_MAP[event.code];
    }
    
    // If we found a keyCode, add it to the frame inputs
    if (keyCode !== null && keyCode !== undefined) {
      if (!frameInputs[eventFrame]) {
        frameInputs[eventFrame] = { pressed: new Set(), released: new Set() };
      }
      
      if (eventType === "keydown" || eventType === "keyPressed" || eventType === "press") {
        frameInputs[eventFrame].pressed.add(keyCode);
      } else if (eventType === "keyup" || eventType === "keyReleased" || eventType === "release") {
        frameInputs[eventFrame].released.add(keyCode);
      }
    }
    
    if (event.timestamp) {
      lastTimestamp = event.timestamp;
    }
    frameEstimate = eventFrame;
  }
}

/**
 * Check if replay mode is active
 * @returns {boolean} True if replay is active
 */
export function isReplayMode() {
  return isReplayActive;
}

/**
 * Get inputs for the current frame
 * @param {number} frame - Current frame number
 * @returns {Object} Object with pressed and released key sets
 */
export function getInputsForFrame(frame) {
  if (!isReplayActive) {
    return { pressed: new Set(), released: new Set() };
  }
  
  return frameInputs[frame] || { pressed: new Set(), released: new Set() };
}

/**
 * Check if game phase should transition at this frame
 * @param {number} frame - Current frame number
 * @returns {Object|null} Game info event if transition should occur, null otherwise
 */
export function shouldTransitionPhase(frame) {
  if (!isReplayActive || gameInfoEvents.length === 0) {
    return null;
  }
  
  // Find the game_info event that matches this frame
  for (const event of gameInfoEvents) {
    if (event.framecount === frame) {
      return event;
    }
  }
  
  return null;
}

/**
 * Get logged player position for validation
 * @param {number} frame - Current frame number
 * @returns {Object|null} Player info if available
 */
export function getLoggedPlayerInfo(frame) {
  if (!isReplayActive || playerInfoLog.length === 0) {
    return null;
  }
  
  // Find the closest player_info entry for this frame
  for (const info of playerInfoLog) {
    if (info.framecount === frame) {
      return info;
    }
  }
  
  // If exact match not found, find closest
  let closest = null;
  let minDiff = Infinity;
  for (const info of playerInfoLog) {
    const diff = Math.abs(info.framecount - frame);
    if (diff < minDiff) {
      minDiff = diff;
      closest = info;
    }
  }
  
  return closest;
}

/**
 * Update current frame (called from game loop)
 * @param {number} frame - New frame number
 */
export function setCurrentFrame(frame) {
  currentFrame = frame;
}

/**
 * Get current frame
 * @returns {number} Current frame number
 */
export function getCurrentFrame() {
  return currentFrame;
}

/**
 * Set replay speed multiplier
 * @param {number} speed - Speed multiplier (1.0 = normal, 2.0 = 2x, 0.5 = 0.5x)
 */
export function setReplaySpeed(speed) {
  replaySpeed = Math.max(0.1, Math.min(5.0, speed));
}

/**
 * Get replay speed
 * @returns {number} Current replay speed
 */
export function getReplaySpeed() {
  return replaySpeed;
}

/**
 * Pause replay
 */
export function pauseReplay() {
  isPaused = true;
}

/**
 * Resume replay
 */
export function resumeReplay() {
  isPaused = false;
}

/**
 * Check if replay is paused
 * @returns {boolean} True if paused
 */
export function isReplayPaused() {
  return isPaused;
}

/**
 * Stop replay mode
 */
export function stopReplay() {
  isReplayActive = false;
  replayData = null;
  frameInputs = {};
  gameInfoEvents = [];
  playerInfoLog = [];
  currentFrame = 0;
  
  // Reset key state
  if (typeof resetReplayKeyState === 'function') {
    // Import dynamically to avoid circular dependency
    import('./input.js').then(module => {
      if (module.resetReplayKeyState) {
        module.resetReplayKeyState();
      }
    }).catch(() => {
      // Ignore if import fails
    });
  }
}

/**
 * Get replay metadata
 * @returns {Object} Replay metadata
 */
export function getReplayMetadata() {
  if (!replayData) {
    return null;
  }
  
  return {
    totalFrames: playerInfoLog.length > 0 ? playerInfoLog[playerInfoLog.length - 1].framecount : 0,
    gameInfoEvents: gameInfoEvents.length,
    playerInfoFrames: playerInfoLog.length,
    inputEvents: replayData.inputs?.events?.length || 0
  };
}
