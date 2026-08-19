// Replay UI - Controls and display for replay mode
import { isReplayMode, pauseReplay, resumeReplay, isReplayPaused, getCurrentFrame, getReplayMetadata } from './replay_controller.js';
import { gameState } from './globals.js';

let uiContainer = null;
let isUIVisible = true;

/**
 * Initialize replay UI
 */
export function initReplayUI() {
  if (uiContainer) {
    return; // Already initialized
  }
  
  // Create UI container
  uiContainer = document.createElement('div');
  uiContainer.id = 'replay-ui';
  uiContainer.style.cssText = `
    position: fixed;
    top: 10px;
    right: 10px;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 15px;
    border-radius: 8px;
    font-family: monospace;
    font-size: 12px;
    z-index: 10000;
    min-width: 250px;
    display: none;
  `;
  
  // Create UI content
  uiContainer.innerHTML = `
    <div style="margin-bottom: 10px; font-weight: bold; border-bottom: 1px solid #555; padding-bottom: 5px;">
      REPLAY CONTROLS
    </div>
    <div id="replay-status" style="margin-bottom: 10px;">
      Status: <span id="replay-status-text">Loading...</span>
    </div>
    <div id="replay-frame-info" style="margin-bottom: 10px;">
      Frame: <span id="replay-frame">0</span> / <span id="replay-total-frames">0</span>
    </div>
    <div style="margin-bottom: 10px;">
      <button id="replay-play-pause" style="padding: 5px 10px; cursor: pointer;">Pause</button>
    </div>
    <div id="replay-validation" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #555; display: none;">
      <div style="font-size: 10px; color: #ffaa00;">
        Validation: <span id="replay-validation-count">0</span> discrepancies
      </div>
    </div>
  `;
  
  document.body.appendChild(uiContainer);
  
  // Wire up controls
  setupControls();
  
  // Start update loop
  updateUI();
}

/**
 * Setup control event handlers
 */
function setupControls() {
  const playPauseBtn = document.getElementById('replay-play-pause');
  
  if (playPauseBtn) {
    playPauseBtn.addEventListener('click', () => {
      if (isReplayPaused()) {
        resumeReplay();
        playPauseBtn.textContent = 'Pause';
      } else {
        pauseReplay();
        playPauseBtn.textContent = 'Play';
      }
    });
  }
}

/**
 * Update UI display
 */
function updateUI() {
  if (!uiContainer || !isReplayMode()) {
    return;
  }
  
  // Update frame counter
  const frameEl = document.getElementById('replay-frame');
  const totalFramesEl = document.getElementById('replay-total-frames');
  const statusText = document.getElementById('replay-status-text');
  const playPauseBtn = document.getElementById('replay-play-pause');
  
  if (frameEl) {
    frameEl.textContent = getCurrentFrame();
  }
  
  const metadata = getReplayMetadata();
  if (totalFramesEl && metadata) {
    totalFramesEl.textContent = metadata.totalFrames;
  }
  
  if (statusText) {
    if (isReplayPaused()) {
      statusText.textContent = 'Paused';
      statusText.style.color = '#ffaa00';
    } else {
      statusText.textContent = 'Playing';
      statusText.style.color = '#00ff00';
    }
  }
  
  if (playPauseBtn) {
    playPauseBtn.textContent = isReplayPaused() ? 'Play' : 'Pause';
  }
  
  // Update validation display
  if (gameState.replayValidation.enabled) {
    const validationDiv = document.getElementById('replay-validation');
    const validationCount = document.getElementById('replay-validation-count');
    if (validationDiv) {
      validationDiv.style.display = 'block';
    }
    if (validationCount) {
      validationCount.textContent = gameState.replayValidation.discrepancies.length;
      if (gameState.replayValidation.discrepancies.length > 0) {
        validationCount.style.color = '#ff4444';
      } else {
        validationCount.style.color = '#00ff00';
      }
    }
  }
  
  // Schedule next update
  requestAnimationFrame(updateUI);
}

/**
 * Show replay UI
 */
export function showReplayUI() {
  if (uiContainer) {
    uiContainer.style.display = 'block';
    isUIVisible = true;
  }
}

/**
 * Hide replay UI
 */
export function hideReplayUI() {
  if (uiContainer) {
    uiContainer.style.display = 'none';
    isUIVisible = false;
  }
}

/**
 * Toggle replay UI visibility
 */
export function toggleReplayUI() {
  if (isUIVisible) {
    hideReplayUI();
  } else {
    showReplayUI();
  }
}

/**
 * Enable validation mode
 */
export function enableValidation() {
  gameState.replayValidation.enabled = true;
  gameState.replayValidation.discrepancies = [];
}

/**
 * Get validation results
 */
export function getValidationResults() {
  return {
    enabled: gameState.replayValidation.enabled,
    discrepancies: gameState.replayValidation.discrepancies,
    totalDiscrepancies: gameState.replayValidation.discrepancies.length
  };
}
