// physics.js - Physics and collision handling

import { 
  gameState, 
  ENTITY_TYPES,
  CANVAS_HEIGHT,
  GAME_PHASES
} from './globals.js';
import { checkCollisionAABB } from './utils.js';
import { resetToCheckpoint } from './level.js';

export function updatePhysics() {
  const player = gameState.player;
  if (!player) return;
  
  // Apply gravity to player
  player.applyGravity();
  
  // Apply gravity to crates
  const crates = gameState.entities.filter(e => e.type === ENTITY_TYPES.CRATE);
  crates.forEach(crate => {
    crate.update();
  });
  
  // Update player position
  player.x += player.vx;
  player.y += player.vy;
  
  // Reset ground state
  player.onGround = false;
  
  // Check collisions with platforms
  const platforms = gameState.entities.filter(e => 
    e.type === ENTITY_TYPES.PLATFORM || e.type === ENTITY_TYPES.MOVING_PLATFORM
  );
  
  platforms.forEach(platform => {
    if (platform.type === ENTITY_TYPES.MOVING_PLATFORM) {
      platform.update();
    }
    
    if (checkCollisionAABB(player, platform)) {
      // Landing on top
      if (player.vy > 0 && player.y + player.height - player.vy <= platform.y) {
        player.y = platform.y - player.height;
        player.vy = 0;
        player.onGround = true;
        
        // Move with platform
        if (platform.type === ENTITY_TYPES.MOVING_PLATFORM) {
          player.x += platform.vx;
        }
      }
      // Hit from below
      else if (player.vy < 0 && player.y - player.vy >= platform.y + platform.height) {
        player.y = platform.y + platform.height;
        player.vy = 0;
      }
      // Side collision
      else {
        if (player.vx > 0) {
          player.x = platform.x - player.width;
        } else if (player.vx < 0) {
          player.x = platform.x + platform.width;
        }
      }
    }
    
    // Crate collisions with platforms
    crates.forEach(crate => {
      if (checkCollisionAABB(crate, platform)) {
        if (crate.vy > 0 && crate.y + crate.height - crate.vy <= platform.y) {
          crate.y = platform.y - crate.height;
          crate.vy = 0;
          crate.onGround = true;
          
          if (platform.type === ENTITY_TYPES.MOVING_PLATFORM) {
            crate.x += platform.vx;
          }
        } else if (crate.vy < 0 && crate.y - crate.vy >= platform.y + platform.height) {
          crate.y = platform.y + platform.height;
          crate.vy = 0;
        } else {
          if (crate.vx > 0) {
            crate.x = platform.x - crate.width;
          } else if (crate.vx < 0) {
            crate.x = platform.x + platform.width;
          }
        }
      }
    });
  });
  
  // Player can stand on crates
  crates.forEach(crate => {
    if (checkCollisionAABB(player, crate)) {
      if (player.vy > 0 && player.y + player.height - player.vy <= crate.y) {
        player.y = crate.y - player.height;
        player.vy = 0;
        player.onGround = true;
      } else {
        // Push crate
        if (player.interacting && player.onGround) {
          if (player.facingRight && player.x < crate.x) {
            crate.x += 2;
            crate.beingPushed = true;
          } else if (!player.facingRight && player.x > crate.x) {
            crate.x -= 2;
            crate.beingPushed = true;
          }
        }
      }
    }
  });
  
  // Update crate positions
  crates.forEach(crate => {
    crate.y += crate.vy;
    crate.x += crate.vx;
    crate.vx *= 0.9; // Friction
    crate.beingPushed = false;
  });
  
  // Check spike collisions
  const spikes = gameState.entities.filter(e => e.type === ENTITY_TYPES.SPIKE);
  spikes.forEach(spike => {
    if (checkCollisionAABB(player, spike)) {
      resetToCheckpoint();
    }
  });
  
  // Check gate collisions
  const gates = gameState.entities.filter(e => e.type === ENTITY_TYPES.GATE);
  gates.forEach(gate => {
    gate.update();
    if (gate.isBlocking() && checkCollisionAABB(player, gate)) {
      if (player.vx > 0) {
        player.x = gate.x - player.width;
      } else if (player.vx < 0) {
        player.x = gate.x + gate.width;
      }
    }
  });
  
  // Check checkpoint collisions
  const checkpoints = gameState.entities.filter(e => e.type === ENTITY_TYPES.CHECKPOINT);
  checkpoints.forEach(checkpoint => {
    checkpoint.update();
    if (checkCollisionAABB(player, checkpoint)) {
      if (!checkpoint.activated) {
        checkpoint.activate();
        if (checkpoint.index > gameState.currentCheckpoint) {
          gameState.currentCheckpoint = checkpoint.index;
          gameState.score += 100;
        }
        
        // Check for final checkpoint
        if (checkpoint.index === 4) {
          gameState.levelComplete = true;
          gameState.gamePhase = GAME_PHASES.GAME_OVER_WIN;
        }
      }
    }
  });
  
  // Check lever interaction
  if (player.interacting) {
    const levers = gameState.entities.filter(e => e.type === ENTITY_TYPES.LEVER);
    levers.forEach(lever => {
      const interactDistance = 40;
      const dx = (lever.x + lever.width/2) - (player.x + player.width/2);
      const dy = (lever.y + lever.height/2) - (player.y + player.height/2);
      const distance = Math.sqrt(dx*dx + dy*dy);
      
      if (distance < interactDistance) {
        if (lever.activate()) {
          // Open corresponding gate
          gates.forEach(gate => {
            if (gate.id === lever.targetGateId) {
              gate.open = true;
              gameState.score += 50;
            }
          });
        }
      }
    });
  }
  
  // World bounds
  if (player.x < 0) player.x = 0;
  if (player.x > gameState.worldWidth - player.width) {
    player.x = gameState.worldWidth - player.width;
  }
  
  // Fall death
  if (player.y > CANVAS_HEIGHT + 50) {
    resetToCheckpoint();
  }
  
  // Clamp velocities
  player.vx *= 0.8; // Friction
}

export function handlePlayerInput(keys) {
  const player = gameState.player;
  if (!player) return;
  
  let moveDir = 0;
  if (keys.left) moveDir -= 1;
  if (keys.right) moveDir += 1;
  
  player.move(moveDir);
  
  if (keys.jump) {
    player.jump();
  }
  
  player.interacting = keys.interact;
}