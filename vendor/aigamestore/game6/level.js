// level.js - Level creation and management

import { 
  gameState, 
  ENTITY_TYPES,
  CANVAS_HEIGHT
} from './globals.js';
import { 
  Player, 
  Crate, 
  Spike, 
  Lever, 
  Gate, 
  Checkpoint, 
  Platform,
  MovingPlatform
} from './entities.js';

export function createLevel() {
  gameState.entities = [];
  
  const groundY = CANVAS_HEIGHT - 40;
  
  // Player
  const spawnPoint = getCheckpointPosition(gameState.currentCheckpoint);
  gameState.player = new Player(spawnPoint.x, spawnPoint.y);
  gameState.entities.push(gameState.player);
  
  // Ground platforms
  gameState.entities.push(new Platform(0, groundY, 800, 40));
  gameState.entities.push(new Platform(900, groundY, 400, 40));
  gameState.entities.push(new Platform(1400, groundY, 600, 40));
  gameState.entities.push(new Platform(2100, groundY, 900, 40));
  
  // Section 1: Basic movement and first crate puzzle
  gameState.entities.push(new Checkpoint(50, groundY - 60, 0));
  gameState.entities.push(new Crate(150, groundY - 40));
  gameState.entities.push(new Platform(250, groundY - 80, 100, 20));
  gameState.entities.push(new Spike(400, groundY - 15, 120));
  gameState.entities.push(new Crate(560, groundY - 40));
  
  // Section 2: Lever and gate puzzle
  gameState.entities.push(new Checkpoint(650, groundY - 60, 1));
  gameState.entities.push(new Lever(700, groundY - 30, 1));
  gameState.entities.push(new Gate(800, groundY - 100, 100, 1));
  gameState.entities.push(new Platform(820, groundY - 120, 80, 20));
  
  // Section 3: Gap jumping with platforms
  gameState.entities.push(new Platform(950, groundY - 60, 80, 20));
  gameState.entities.push(new Platform(1080, groundY - 100, 80, 20));
  gameState.entities.push(new Platform(1210, groundY - 80, 80, 20));
  gameState.entities.push(new Spike(1000, groundY - 15, 300));
  
  // Section 4: Crate stacking puzzle
  gameState.entities.push(new Checkpoint(1450, groundY - 60, 2));
  gameState.entities.push(new Crate(1500, groundY - 40));
  gameState.entities.push(new Crate(1550, groundY - 40));
  gameState.entities.push(new Platform(1650, groundY - 120, 100, 20));
  gameState.entities.push(new Lever(1700, groundY - 150, 2));
  gameState.entities.push(new Gate(1950, groundY - 100, 100, 2));
  
  // Section 5: Moving platform challenge
  gameState.entities.push(new MovingPlatform(2050, groundY - 100, 80, 15, 2050, 2300, 1.5));
  gameState.entities.push(new Spike(2100, groundY - 15, 300));
  gameState.entities.push(new Platform(2400, groundY - 80, 100, 20));
  
  // Section 6: Complex spike navigation
  gameState.entities.push(new Checkpoint(2500, groundY - 60, 3));
  gameState.entities.push(new Platform(2550, groundY - 50, 80, 20));
  gameState.entities.push(new Spike(2600, groundY - 15, 60));
  gameState.entities.push(new Platform(2660, groundY - 90, 80, 20));
  gameState.entities.push(new Spike(2700, groundY - 15, 60));
  gameState.entities.push(new Platform(2760, groundY - 60, 80, 20));
  
  // Final checkpoint
  gameState.entities.push(new Checkpoint(2850, groundY - 60, 4));
}

export function getCheckpointPosition(index) {
  const groundY = CANVAS_HEIGHT - 40;
  const checkpoints = [
    { x: 50, y: groundY - 30 },
    { x: 650, y: groundY - 30 },
    { x: 1450, y: groundY - 30 },
    { x: 2500, y: groundY - 30 },
    { x: 2850, y: groundY - 30 }
  ];
  
  return checkpoints[Math.min(index, checkpoints.length - 1)];
}

export function resetToCheckpoint() {
  const pos = getCheckpointPosition(gameState.currentCheckpoint);
  gameState.player.x = pos.x;
  gameState.player.y = pos.y;
  gameState.player.vx = 0;
  gameState.player.vy = 0;
  gameState.player.onGround = false;
  gameState.deaths++;
}