import { CANVAS_WIDTH, CANVAS_HEIGHT, KEY, gameState, resetGame, getGameState, nextLevel, resetToLevel1 } from './globals.js';
import { Player, Enemy, Bullet, Obstacle, Pickup, ExtractionPoint, WeaponPickup } from './entities.js';
import { generateLevel } from './level.js';
import { drawUI, drawStartScreen, drawGameOverScreen } from './ui.js';
import { setupInputHandlers, keys, startGame, pauseGame, resumeGame, resetToStart, applyReplayInputs } from './input.js';
import { random } from './rng.js';
import { isReplayMode, shouldTransitionPhase, getLoggedPlayerInfo, isReplayPaused, getReplaySpeed, initReplay } from './replay_controller.js';
import { setSeed } from './rng.js';
import { initReplayUI, showReplayUI, enableValidation } from './replay_ui.js';

function checkPointInRect(pointX, pointY, rectX, rectY, rectWidth, rectHeight) {
  return pointX >= rectX && pointX <= rectX + rectWidth && 
         pointY >= rectY && pointY <= rectY + rectHeight;
}

window.getGameState = getGameState;

window.setControlMode = function(mode) {
  gameState.controlMode = mode;
  
  const buttons = document.querySelectorAll('.control-button');
  buttons.forEach(button => {
    button.classList.remove('active');
  });
  
  const activeButton = document.getElementById(mode === "HUMAN" ? "humanModeBtn" : `test_${mode.split('_')[1]}_ModeBtn`);
  if (activeButton) {
    activeButton.classList.add('active');
  }
  
  if (window.gameInstance && window.gameInstance.logs) {
    window.gameInstance.logs.game_info.push({
      "game_status": gameState.gamePhase,
      "data": { "controlMode": mode },
      "framecount": window.gameInstance.frameCount,
      "timestamp": Date.now()
    });
  }
};

const p5 = window.p5;
let gameInstance = new p5(p => {
  p.setup = function() {
    p.createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    p.frameRate(60);
    p.randomSeed(42);
    setSeed(42); // Initialize seeded RNG with same seed
    
    p.logs = {
      "game_info": [],
      "player_info": [],
      "inputs": []
    };
    
    gameState.gamePhase = "START";
    gameState.controlMode = "HUMAN";
    gameState.replayMode = false;
    gameState.autoRestartTimer = null; // Initialize auto-restart timer
    
    setupInputHandlers(p);
    
    // Check if replay mode should be initialized from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const replayInputsUrl = urlParams.get('replay_inputs');
    const replayLogsUrl = urlParams.get('replay_logs');
    
    if (replayInputsUrl && replayLogsUrl) {
      // Load replay data asynchronously
      Promise.all([
        fetch(replayInputsUrl).then(r => r.json()),
        fetch(replayLogsUrl).then(r => r.json())
      ]).then(([inputsData, logsData]) => {
        initReplay(inputsData, logsData);
        gameState.replayMode = true;
        gameState.controlMode = "REPLAY";
        
        // Reset RNG to ensure determinism
        setSeed(42);
        if (window.gameInstance) {
          window.gameInstance.randomSeed(42);
        }
        
        // Initialize replay UI
        initReplayUI();
        showReplayUI();
        
        // Enable validation
        enableValidation();
        
        console.log('[Replay] Replay mode activated from URL parameters');
      }).catch(err => {
        console.error('[Replay] Failed to load replay data:', err);
      });
    }
    
    p.logs.game_info.push({
      "game_status": gameState.gamePhase,
      "data": {},
      "framecount": p.frameCount,
      "timestamp": Date.now()
    });
  };
  
  p.draw = function() {
    // Handle replay mode
    if (isReplayMode()) {
      // Check if replay is paused
      if (isReplayPaused()) {
        // Still render but don't update
        p.background(30);
        // Render current state without updating
        renderCurrentState(p);
        return;
      }
      
      // Apply replay inputs for this frame
      applyReplayInputs(p);
      
      // Check for phase transitions
      const phaseEvent = shouldTransitionPhase(p.frameCount);
      if (phaseEvent) {
        const newPhase = phaseEvent.game_status;
        if (newPhase !== gameState.gamePhase) {
          console.log(`[Replay] Phase transition at frame ${p.frameCount}: ${gameState.gamePhase} -> ${newPhase}`);
          gameState.gamePhase = newPhase;
          
          // Handle level transitions
          if (phaseEvent.data && phaseEvent.data.level) {
            gameState.currentLevel = phaseEvent.data.level;
            resetGame();
            // Reset RNG to ensure determinism after restart
            setSeed(42);
            p.randomSeed(42);
            generateLevel(p);
            gameState.player = new Player(gameState.level.width / 2, gameState.level.height / 2);
          }
          
          // Handle game start/restart
          if (newPhase === "START" || (newPhase === "PLAYING" && !gameState.player)) {
            resetGame();
            // Reset RNG to ensure determinism after restart
            setSeed(42);
            p.randomSeed(42);
            generateLevel(p);
            gameState.player = new Player(gameState.level.width / 2, gameState.level.height / 2);
            console.log(`[Replay] Game reset at frame ${p.frameCount}, RNG reseeded`);
          }
        }
      }
      
      // Validation is performed regardless of pause state, but only if enabled
      if (gameState.replayValidation.enabled) {
        const loggedInfo = getLoggedPlayerInfo(p.frameCount);
        if (loggedInfo && gameState.player) {
          const actualX = gameState.player.x;
          const actualY = gameState.player.y;
          const loggedX = loggedInfo.game_x;
          const loggedY = loggedInfo.game_y;
          const diffX = Math.abs(actualX - loggedX);
          const diffY = Math.abs(actualY - loggedY);
          
          // Use a tolerance of 2 pixels to account for floating point precision
          // and small timing differences
          const tolerance = 2;
          
          if (diffX > tolerance || diffY > tolerance) {
            // Only log if we haven't already logged too many discrepancies
            // to avoid memory issues
            if (gameState.replayValidation.discrepancies.length < 1000) {
              gameState.replayValidation.discrepancies.push({
                frame: p.frameCount,
                actual: { x: actualX, y: actualY },
                logged: { x: loggedX, y: loggedY },
                diff: { x: diffX, y: diffY }
              });
            }
            
            // Log first few discrepancies for debugging
            if (gameState.replayValidation.discrepancies.length <= 5) {
              console.warn(`[Replay Validation] Frame ${p.frameCount}: Position mismatch`, {
                actual: { x: actualX, y: actualY },
                logged: { x: loggedX, y: loggedY },
                diff: { x: diffX, y: diffY }
              });
            }
          }
        }
      }
    }
    
    p.background(30);
    
    switch (gameState.gamePhase) {
      case "START":
        drawStartScreen(p);
        break;
        
      case "PLAYING":
        if (!gameState.player) {
          resetGame();
          generateLevel(p);
          gameState.player = new Player(gameState.level.width / 2, gameState.level.height / 2);
        }
        
        gameState.timeElapsed += 1/60;
        
        // Handle input - RL mode takes precedence over other input methods
        let inputKeys = keys;
        
        if (window.gymAPI && window.gymAPI.isRLMode && window.gymAPI.isRLMode()) {
          // RL mode active - read actions from RL agent
          const rlAction = window.gymAPI.getRLAction();
          
          // Convert action object to keys format (string properties, not keycodes)
          inputKeys = {
            left: rlAction.left || false,
            right: rlAction.right || false,
            up: rlAction.up || false,
            down: rlAction.down || false,
            shoot: rlAction.shoot || false,
            sprint: rlAction.sprint || false,
            crouch: rlAction.crouch || false,
          };
        }
        // Otherwise use normal keyboard input (keys object from input.js)
        
        gameState.player.update(p, inputKeys);
        
        p.logs.player_info.push({
          "screen_x": gameState.player.x - gameState.level.cameraX,
          "screen_y": gameState.player.y - gameState.level.cameraY,
          "game_x": gameState.player.x,
          "game_y": gameState.player.y,
          "framecount": p.frameCount,
          "timestamp": Date.now()
        });
        
        if (gameState.extractionPointObj) {
          gameState.extractionPointObj.update(p);
          
          if (p.dist(gameState.player.x, gameState.player.y, 
                      gameState.extractionPoint.x, gameState.extractionPoint.y) < 40) {
            
            // Check win condition: Reached extraction AND enough kills
            if (gameState.enemiesKilled >= gameState.requiredKills) {
              gameState.gamePhase = "GAME_OVER_WIN";
              gameState.score += 1000;
              
              p.logs.game_info.push({
                "game_status": gameState.gamePhase,
                "data": { "score": gameState.score },
                "framecount": p.frameCount,
                "timestamp": Date.now()
              });
            } else {
              // Show hint that more kills are needed
              p.push();
              p.fill(255, 50, 50);
              p.textSize(20);
              p.textAlign(p.CENTER, p.CENTER);
              p.text("ELIMINATE MORE ENEMIES TO EXTRACT!", CANVAS_WIDTH / 2, CANVAS_HEIGHT / 4);
              p.pop();
            }
          }
        }
        
        for (let i = gameState.pickups.length - 1; i >= 0; i--) {
          const pickup = gameState.pickups[i];
          pickup.update(p);
          
          if (p.dist(gameState.player.x, gameState.player.y, pickup.x, pickup.y) < gameState.player.radius + pickup.radius) {
            if (pickup.type === "health") {
              gameState.player.health = Math.min(gameState.player.maxHealth, gameState.player.health + 30);
            } else if (pickup.type === "ammo") {
              // Add to reserve ammo instead of just filling clip
              const weapon = gameState.player.weapons[gameState.player.currentWeapon];
              weapon.reserveAmmo = Math.min(weapon.maxReserveAmmo, weapon.reserveAmmo + weapon.maxAmmo * 2);
              gameState.player.reserveAmmo = weapon.reserveAmmo;
              
              // Also fill current clip if possible
              if (gameState.player.ammo < gameState.player.maxAmmo) {
                 // But reload logic handles this usually, let's just add to reserve to force reload mechanic usage
                 // Or we can be nice and fill the clip too? Let's stick to reserve to make it "scarce" feeling
              }
            }
            
            gameState.pickups.splice(i, 1);
            gameState.score += 50;
          }
        }
        
        // Handle weapon pickups
        for (let i = gameState.weaponPickups.length - 1; i >= 0; i--) {
          const weaponPickup = gameState.weaponPickups[i];
          weaponPickup.update(p);
          
          if (p.dist(gameState.player.x, gameState.player.y, weaponPickup.x, weaponPickup.y) < gameState.player.radius + weaponPickup.radius) {
            gameState.player.pickupWeapon(weaponPickup.weaponType);
            gameState.weaponPickups.splice(i, 1);
            gameState.score += 100;
          }
        }
        
        // Helper function for explosions
        const createExplosion = (x, y, radius, damage) => {
          // Visual effect
          p.push();
          p.noStroke();
          p.fill(255, 100, 0, 150);
          p.circle(x - gameState.level.cameraX, y - gameState.level.cameraY, radius * 2);
          p.fill(255, 200, 0, 200);
          p.circle(x - gameState.level.cameraX, y - gameState.level.cameraY, radius);
          p.pop();
          
          // Damage enemies
          for (const enemy of gameState.enemies) {
            const dist = p.dist(x, y, enemy.x, enemy.y);
            if (dist < radius + enemy.radius) {
              // Calculate falloff damage
              const damageFactor = 1 - (dist / (radius + enemy.radius));
              const actualDamage = Math.ceil(damage * damageFactor);
              
              const killed = enemy.takeDamage(actualDamage);
              if (killed && !enemy.dead) { // Prevent double counting
                enemy.dead = true;
                // We'll handle removal in the main loop or mark for removal
              }
            }
          }
        };

        for (let i = gameState.enemies.length - 1; i >= 0; i--) {
          const enemy = gameState.enemies[i];
          if (enemy.dead) { // Handle explosion kills
             gameState.enemies.splice(i, 1);
             gameState.enemiesKilled++;
             gameState.score += 100; // Simplified score for explosion kills
             continue;
          }
          
          enemy.update(p, gameState.player);
          
          for (let j = gameState.bullets.length - 1; j >= 0; j--) {
            const bullet = gameState.bullets[j];
            if (p.dist(enemy.x, enemy.y, bullet.x, bullet.y) < enemy.radius + bullet.radius) {
              let killed = false;
              
              if (bullet.type === "rocket") {
                createExplosion(bullet.x, bullet.y, bullet.explosionRadius, bullet.explosionDamage);
                killed = enemy.takeDamage(bullet.damage); // Direct hit damage
              } else {
                killed = enemy.takeDamage(bullet.damage);
              }
              
              if (killed || enemy.dead) {
                if (!enemy.dead) { // If not already dead from explosion
                    gameState.enemies.splice(i, 1);
                    gameState.enemiesKilled++;
                    
                    // Score based on enemy type
                    if (enemy.type === "elite") gameState.score += 200;
                    else if (enemy.type === "heavy") gameState.score += 300;
                    else if (enemy.type === "scout") gameState.score += 150;
                    else if (enemy.type === "sniper") gameState.score += 250;
                    else if (enemy.type === "tank") gameState.score += 400;
                    else gameState.score += 100;
                    
                    if (random() < 0.3) {
                      const pickupType = random() < 0.4 ? "health" : "ammo";
                      gameState.pickups.push(new Pickup(enemy.x, enemy.y, pickupType));
                    }
                }
              }
              
              gameState.bullets.splice(j, 1);
              break;
            }
          }
        }
        
        for (let i = gameState.bullets.length - 1; i >= 0; i--) {
          const bullet = gameState.bullets[i];
          const remove = bullet.update();
          
          let hitObstacle = false;
          for (const obstacle of gameState.obstacles) {
            if (checkPointInRect(bullet.x, bullet.y, obstacle.x, obstacle.y, obstacle.width, obstacle.height)) {
              if (bullet.type === "rocket") {
                createExplosion(bullet.x, bullet.y, bullet.explosionRadius, bullet.explosionDamage);
              }
              gameState.bullets.splice(i, 1);
              hitObstacle = true;
              break;
            }
          }
          
          if (!hitObstacle && remove && i < gameState.bullets.length) {
            gameState.bullets.splice(i, 1);
          }
        }
        
        for (let i = gameState.enemyBullets.length - 1; i >= 0; i--) {
          const bullet = gameState.enemyBullets[i];
          const remove = bullet.update();
          
          if (p.dist(gameState.player.x, gameState.player.y, bullet.x, bullet.y) < gameState.player.radius + bullet.radius) {
            gameState.player.takeDamage(bullet.damage);
            gameState.enemyBullets.splice(i, 1);
            continue;
          }
          
          for (const obstacle of gameState.obstacles) {
            if (checkPointInRect(bullet.x, bullet.y, obstacle.x, obstacle.y, obstacle.width, obstacle.height)) {
              gameState.enemyBullets.splice(i, 1);
              break;
            }
          }
          
          if (remove && i < gameState.enemyBullets.length) {
            gameState.enemyBullets.splice(i, 1);
          }
        }
        
        if (gameState.extractionPointObj) {
          gameState.extractionPointObj.draw(p);
        }
        
        for (const obstacle of gameState.obstacles) {
          obstacle.draw(p);
        }
        
        for (const pickup of gameState.pickups) {
          pickup.draw(p);
        }
        
        for (const weaponPickup of gameState.weaponPickups) {
          weaponPickup.draw(p);
        }
        
        for (const enemy of gameState.enemies) {
          enemy.draw(p);
        }
        
        gameState.player.draw(p);
        
        for (const bullet of gameState.bullets) {
          bullet.draw(p);
        }
        
        for (const bullet of gameState.enemyBullets) {
          bullet.draw(p);
        }
        
        drawUI(p);
        break;
        
      case "PAUSED":
        if (gameState.extractionPointObj) {
          gameState.extractionPointObj.draw(p);
        }
        
        for (const obstacle of gameState.obstacles) {
          obstacle.draw(p);
        }
        
        for (const pickup of gameState.pickups) {
          pickup.draw(p);
        }
        
        for (const weaponPickup of gameState.weaponPickups) {
          weaponPickup.draw(p);
        }
        
        for (const enemy of gameState.enemies) {
          enemy.draw(p);
        }
        
        if (gameState.player) {
          gameState.player.draw(p);
        }
        
        for (const bullet of gameState.bullets) {
          bullet.draw(p);
        }
        
        for (const bullet of gameState.enemyBullets) {
          bullet.draw(p);
        }
        
        drawUI(p);
        // Pause overlay removed as per feedback
        break;
        
      case "GAME_OVER_WIN":
      case "GAME_OVER_LOSE":
        p.push();
        p.tint(255, 100);
        
        if (gameState.extractionPointObj) {
          gameState.extractionPointObj.draw(p);
        }
        
        for (const obstacle of gameState.obstacles) {
          obstacle.draw(p);
        }
        
        for (const pickup of gameState.pickups) {
          pickup.draw(p);
        }
        
        for (const weaponPickup of gameState.weaponPickups) {
          weaponPickup.draw(p);
        }
        
        for (const enemy of gameState.enemies) {
          enemy.draw(p);
        }
        
        if (gameState.player) {
          gameState.player.draw(p);
        }
        
        p.pop();
        
        drawGameOverScreen(p, gameState.gamePhase === "GAME_OVER_WIN");

        // Auto-restart logic
        if (!isReplayMode()) {
            if (gameState.autoRestartTimer === null) { // Initialize timer if not set
                gameState.autoRestartTimer = p.frameCount;
            }
            const framesToWait = p.frameRate(); // 1 second (60 frames at 60fps)
            if (p.frameCount - gameState.autoRestartTimer >= framesToWait) {
                // Clear the auto-restart timer immediately
                gameState.autoRestartTimer = null; 

                // Determine restart action
                if (gameState.gamePhase === "GAME_OVER_WIN") {
                    nextLevel(); // Increments level, calls resetGame(), sets player = null
                    gameState.gamePhase = "PLAYING"; // Immediately transition to PLAYING
                } else { // GAME_OVER_LOSE
                    // Reset game state but keep current level
                    resetGame(); // Calls resetGame(), sets player = null, keeps currentLevel
                    gameState.gamePhase = "PLAYING"; // Transition to PLAYING
                }
                
                // Reset RNG for deterministic levels
                setSeed(42);
                p.randomSeed(42);

                // If phase is PLAYING, immediately generate level and player
                if (gameState.gamePhase === "PLAYING") {
                  generateLevel(p);
                  gameState.player = new Player(gameState.level.width / 2, gameState.level.height / 2);
                }
                
                // Log the auto-restart event
                p.logs.game_info.push({
                    "game_status": gameState.gamePhase,
                    "data": { "autoRestart": true, "level": gameState.currentLevel },
                    "framecount": p.frameCount,
                    "timestamp": Date.now()
                });
            }
        }
        break;
    }
  };
});

window.gameInstance = gameInstance;

// Expose level loading functions for dev mode
window.loadLevel = function(levelNum) {
  const state = getGameState();
  if (state) {
    state.currentLevel = levelNum;
    resetGame();
    if (window.gameInstance) {
      generateLevel(window.gameInstance);
      state.player = new Player(state.level.width / 2, state.level.height / 2);
      state.gamePhase = "PLAYING";
    }
  }
};

// Expose replay UI initialization (will be imported and called directly)

// Expose replay initialization function
window.initReplay = async function(inputsUrl, logsUrl) {
  try {
    const [inputsResponse, logsResponse] = await Promise.all([
      fetch(inputsUrl),
      fetch(logsUrl)
    ]);
    
    const inputsData = await inputsResponse.json();
    const logsData = await logsResponse.json();
    
    initReplay(inputsData, logsData);
    gameState.replayMode = true;
    gameState.controlMode = "REPLAY";
    
    // Reset RNG to ensure determinism
    setSeed(42);
    if (window.gameInstance) {
      window.gameInstance.randomSeed(42);
    }
    
    // Initialize replay UI
    initReplayUI();
    showReplayUI();
    
    // Enable validation
    enableValidation();
    
    // Reset game to start
    resetGame();
    if (window.gameInstance) {
      generateLevel(window.gameInstance);
      gameState.player = new Player(gameState.level.width / 2, gameState.level.height / 2);
      gameState.gamePhase = "START";
    }
    
    console.log('[Replay] Replay initialized successfully');
    return true;
  } catch (error) {
    console.error('[Replay] Failed to initialize replay:', error);
    return false;
  }
};

// Helper function to render current state without updating (for paused replay)
function renderCurrentState(p) {
  if (gameState.extractionPointObj) {
    gameState.extractionPointObj.draw(p);
  }
  
  for (const obstacle of gameState.obstacles) {
    obstacle.draw(p);
  }
  
  for (const pickup of gameState.pickups) {
    pickup.draw(p);
  }
  
  for (const weaponPickup of gameState.weaponPickups) {
    weaponPickup.draw(p);
  }
  
  for (const enemy of gameState.enemies) {
    enemy.draw(p);
  }
  
  if (gameState.player) {
    gameState.player.draw(p);
  }
  
  for (const bullet of gameState.bullets) {
    bullet.draw(p);
  }
  
  for (const bullet of gameState.enemyBullets) {
    bullet.draw(p);
  }
  
  drawUI(p);
}