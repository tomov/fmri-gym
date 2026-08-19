// game.js - Main game file
import { gameState, GAME_PHASES, CANVAS_WIDTH, CANVAS_HEIGHT } from './globals.js';
import { loadLevel } from './levels.js';
import { 
  renderStartScreen, 
  renderGameOverScreen, 
  renderUI,
  renderDarkness,
  renderMessages 
} from './rendering.js';

const p5 = window.p5;

let gameInstance = new p5(p => {
  // Initialize logs
  p.logs = {
    game_info: [],
    inputs: [],
    player_info: []
  };

  let lastPlayerLogFrame = -10;

  p.setup = function() {
    p.createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    p.frameRate(60);
    p.randomSeed(42);
    
    // Log initial state
    p.logs.game_info.push({
      data: { phase: gameState.gamePhase, level: gameState.level },
      framecount: p.frameCount,
      timestamp: Date.now()
    });
  };

  p.draw = function() {
    p.background(40, 35, 45);

    if (gameState.gamePhase === GAME_PHASES.START) {
      renderStartScreen(p);
    } else if (gameState.gamePhase === GAME_PHASES.PLAYING) {
      updateGame(p);
      renderGame(p);
      renderUI(p);
      renderDarkness(p);
      renderMessages(p);
    } else if (gameState.gamePhase === GAME_PHASES.PAUSED) {
      renderGame(p);
      renderUI(p);
      renderDarkness(p);
      renderMessages(p);
    } else if (gameState.gamePhase === GAME_PHASES.GAME_OVER_WIN) {
      renderGameOverScreen(p, true);
    } else if (gameState.gamePhase === GAME_PHASES.GAME_OVER_LOSE) {
      renderGameOverScreen(p, false);
    }

    // Log player info periodically
    if (gameState.player && gameState.gamePhase === GAME_PHASES.PLAYING) {
      if (p.frameCount - lastPlayerLogFrame >= 10) {
        p.logs.player_info.push({
          screen_x: gameState.player.x,
          screen_y: gameState.player.y,
          game_x: gameState.player.x,
          game_y: gameState.player.y,
          framecount: p.frameCount
        });
        lastPlayerLogFrame = p.frameCount;
      }
    }
  };

  function updateGame(p) {
    // Don't update if game is over
    if (gameState.gamePhase === GAME_PHASES.GAME_OVER_LOSE || 
        gameState.gamePhase === GAME_PHASES.GAME_OVER_WIN) {
      return;
    }

    // Update player
    if (gameState.player) {
      gameState.player.update(p);
    }

    // Update monsters
    for (const antagonist of gameState.antagonists) {
      antagonist.update(p);
    }

    // Update items
    for (const item of gameState.items) {
      item.update();
    }

    // Update messages
    updateMessages();

    // Check for level completion
    if (gameState.exitZone && gameState.exitZone.checkPlayer() && !gameState.levelComplete) {
      gameState.levelComplete = true;
      
      // Award completion bonus
      gameState.score += 100;
      addMessage("Level Complete! +100", "success");
      
      // Award undetected bonus
      if (gameState.undetectedBonus) {
        gameState.score += 50;
        addMessage("Undetected Bonus! +50", "success");
      }
      
      // Check if final level (now level 6)
      if (gameState.level >= 6) {
        setTimeout(() => {
          gameState.gamePhase = GAME_PHASES.GAME_OVER_WIN;
          p.logs.game_info.push({
            data: { phase: gameState.gamePhase, won: true, finalScore: gameState.score },
            framecount: p.frameCount,
            timestamp: Date.now()
          });
        }, 500);
      } else {
        // Load next level immediately
        gameState.level++;
        loadLevel(gameState.level);
        addMessage(`Level ${gameState.level}`, "info");
        p.logs.game_info.push({
          data: { phase: "LEVEL_COMPLETE", nextLevel: gameState.level },
          framecount: p.frameCount,
          timestamp: Date.now()
        });
      }
    }
  }

  function renderGame(p) {
    // Background
    p.fill(60, 55, 65);
    p.noStroke();
    p.rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // Walls
    p.fill(80, 70, 75);
    p.stroke(60, 50, 55);
    p.strokeWeight(2);
    for (const wall of gameState.walls) {
      p.rect(wall.x, wall.y, wall.w, wall.h);
    }

    // Doors
    for (const door of gameState.doors) {
      door.render(p);
    }

    // Hiding spots
    for (const spot of gameState.hidingSpots) {
      spot.render(p);
    }

    // Items
    for (const item of gameState.items) {
      item.render(p);
    }

    // Exit zone
    if (gameState.exitZone) {
      gameState.exitZone.render(p);
    }

    // monsters
    for (const antagonist of gameState.antagonists) {
      antagonist.render(p);
    }

    // Player
    if (gameState.player) {
      gameState.player.render(p);
    }
  }

  p.keyPressed = function() {
    // Log input
    p.logs.inputs.push({
      input_type: "keyPressed",
      data: { key: p.key, keyCode: p.keyCode },
      framecount: p.frameCount,
      timestamp: Date.now()
    });

    // Since test modes are removed, this always applies
    // Game phase transitions - Esc = pause, Enter = resume (when paused)
    if (p.keyCode === 13) { // ENTER - start or resume from pause
      if (gameState.gamePhase === GAME_PHASES.START) {
        gameState.gamePhase = GAME_PHASES.PLAYING;
        gameState.level = 1;
        gameState.score = 0;
        loadLevel(1);
        p.logs.game_info.push({
          data: { phase: gameState.gamePhase, level: gameState.level },
          framecount: p.frameCount,
          timestamp: Date.now()
        });
      } else if (gameState.gamePhase === GAME_PHASES.PAUSED) {
        gameState.gamePhase = GAME_PHASES.PLAYING;
        p.logs.game_info.push({
          data: { phase: gameState.gamePhase },
          framecount: p.frameCount,
          timestamp: Date.now()
        });
      }
    } else if (p.keyCode === 27) { // ESC - pause (and resume)
      if (gameState.gamePhase === GAME_PHASES.PLAYING) {
        gameState.gamePhase = GAME_PHASES.PAUSED;
        p.logs.game_info.push({
          data: { phase: gameState.gamePhase },
          framecount: p.frameCount,
          timestamp: Date.now()
        });
      } else if (gameState.gamePhase === GAME_PHASES.PAUSED) {
        gameState.gamePhase = GAME_PHASES.PLAYING;
        p.logs.game_info.push({
          data: { phase: gameState.gamePhase },
          framecount: p.frameCount,
          timestamp: Date.now()
        });
      }
    } else if (p.keyCode === 82) { // R
      gameState.gamePhase = GAME_PHASES.START;
      gameState.level = 1;
      gameState.score = 0;
      gameState.inventory = [];
      gameState.hasFlashlight = false;
      gameState.flashlightOn = false;
      gameState.messages = [];
      gameState.isRunning = false;
      
      p.logs.game_info.push({
        data: { phase: gameState.gamePhase, action: "restart" },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    }

    // Gameplay controls
    if (gameState.gamePhase === GAME_PHASES.PLAYING) {
      if (p.keyCode === 16) { // SHIFT - toggle sprint mode
        gameState.isRunning = !gameState.isRunning;
        addMessage(gameState.isRunning ? "Sprint Mode ON" : "Sprint Mode OFF", "info");
      } else if (p.keyCode === 32) { // SPACE
        handleInteraction();
      } else if (p.keyCode === 90) { // Z
        if (gameState.hasFlashlight) {
          gameState.flashlightOn = !gameState.flashlightOn;
          addMessage(gameState.flashlightOn ? "Flashlight ON" : "Flashlight OFF", "info");
        }
      }
      
      // CONTINUOUS MOVEMENT: Track key pressed state
      if (p.keyCode === 37 || p.keyCode === 38 || p.keyCode === 39 || p.keyCode === 40) {
        gameState.keysPressed[p.keyCode] = true;
      }
    }

    return false;
  };

  p.keyReleased = function() {
    // Log input
    p.logs.inputs.push({
      input_type: "keyReleased",
      data: { key: p.key, keyCode: p.keyCode },
      framecount: p.frameCount,
      timestamp: Date.now()
    });

    // Since test modes are removed, this always applies
    // CONTINUOUS MOVEMENT: Track key released state
    if (p.keyCode === 37 || p.keyCode === 38 || p.keyCode === 39 || p.keyCode === 40) {
      gameState.keysPressed[p.keyCode] = false;
    }

    return false;
  };

  function handleInteraction() {
    if (!gameState.player) return;

    // Check if hiding
    if (gameState.player.hiding) {
      gameState.player.unhide();
      addMessage("Left hiding spot", "info");
      return;
    }

    // Check hiding spots
    for (const spot of gameState.hidingSpots) {
      if (spot.canHide()) {
        gameState.player.hide(spot);
        spot.occupied = true;
        addMessage("Hiding...", "info");
        return;
      }
    }

    // Check doors - increased interaction range to 50 pixels
    for (const door of gameState.doors) {
      const dist = Math.hypot(
        door.x + door.w / 2 - gameState.player.x,
        door.y + door.h / 2 - gameState.player.y
      );
      if (dist < 50) {
        const result = door.interact();
        if (result === "unlocked") {
          addMessage("Door unlocked!", "success");
        } else if (result === "locked") {
          addMessage("Door is locked. Need key!", "warning");
        } else if (result === "opened") {
          addMessage("Door opened", "info");
        }
        return;
      }
    }

    // Check items
    for (const item of gameState.items) {
      if (item.checkPickup()) {
        item.collect();
        return;
      }
    }
  }
});

// Message system
function addMessage(text, type = "info") {
  if (!gameState.messages) {
    gameState.messages = [];
  }
  gameState.messages.push({
    text: text,
    type: type,
    time: 120 // frames to display
  });
}

function updateMessages() {
  if (!gameState.messages) {
    gameState.messages = [];
    return;
  }
  
  for (let i = gameState.messages.length - 1; i >= 0; i--) {
    gameState.messages[i].time--;
    if (gameState.messages[i].time <= 0) {
      gameState.messages.splice(i, 1);
    }
  }
}

// Expose globally
window.addMessage = addMessage;

// Expose game instance globally
window.gameInstance = gameInstance;
// Expose level loading for dev mode
window.loadLevel = function(levelNum) {
  const state = window.getGameState ? window.getGameState() : (window.gameState || (window.gameInstance && window.gameInstance.gameState));
  if (state) {
    state.currentLevel = levelNum;
    if (typeof loadLevel === 'function') {
      loadLevel(levelNum);
    } else if (typeof initializeLevel === 'function') {
      initializeLevel(levelNum);
    }
    if (state.gamePhase !== undefined) {
      state.gamePhase = "PLAYING";
    }
  }
};

// Control mode switching
window.setControlMode = function(mode) {
  // Force control mode to HUMAN as test modes are removed
  gameState.controlMode = "HUMAN"; 
  
  // Update button states - only humanModeBtn should exist and be active
  const allControlButtons = document.querySelectorAll('.control-button');
  allControlButtons.forEach(btn => {
    btn.classList.remove('active');
  });
  
  const humanModeBtn = document.getElementById('humanModeBtn');
  if (humanModeBtn) {
    humanModeBtn.classList.add('active');
  }
};