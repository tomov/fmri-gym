import { CANVAS_WIDTH, CANVAS_HEIGHT, gameState } from './globals.js';
import { Obstacle, Pickup, Enemy, ExtractionPoint, WeaponPickup } from './entities.js';

function checkCollisionWithRect(circleX, circleY, circleRadius, rectX, rectY, rectWidth, rectHeight) {
  const closestX = Math.max(rectX, Math.min(circleX, rectX + rectWidth));
  const closestY = Math.max(rectY, Math.min(circleY, rectY + rectHeight));
  const distanceX = circleX - closestX;
  const distanceY = circleY - closestY;
  return (distanceX * distanceX + distanceY * distanceY) < (circleRadius * circleRadius);
}

export function generateLevel(p) {
  const { level } = gameState;
  
  gameState.obstacles = [];
  gameState.enemies = [];
  gameState.pickups = [];
  gameState.weaponPickups = [];
  
  // Create outer walls
  gameState.obstacles.push(new Obstacle(-10, -10, level.width + 20, 10));
  gameState.obstacles.push(new Obstacle(-10, level.height, level.width + 20, 10));
  gameState.obstacles.push(new Obstacle(-10, -10, 10, level.height + 20));
  gameState.obstacles.push(new Obstacle(level.width, -10, 10, level.height + 20));
  
  // Generate random obstacles - more obstacles at higher levels
  const numObstacles = 20 + gameState.currentLevel * 2;
  for (let i = 0; i < numObstacles; i++) {
    const obstacleWidth = 40 + Math.floor(p.random(80));
    const obstacleHeight = 40 + Math.floor(p.random(80));
    
    let x, y;
    let validPosition = false;
    
    while (!validPosition) {
      x = p.random(100, level.width - 100 - obstacleWidth);
      y = p.random(100, level.height - 100 - obstacleHeight);
      
      const centerDist = p.dist(x + obstacleWidth/2, y + obstacleHeight/2, level.width/2, level.height/2);
      if (centerDist > 150) {
        validPosition = true;
      }
    }
    
    gameState.obstacles.push(new Obstacle(x, y, obstacleWidth, obstacleHeight));
  }
  
  // Generate enemies - ensure enough to meet kill requirement plus extras
  // Reduced the multiplier for high levels to make it less hard
  const minEnemies = gameState.requiredKills + 2;
  const extraEnemies = Math.floor(gameState.currentLevel * 1.0); // Reduced from 1.5
  const numEnemies = minEnemies + extraEnemies;
  
  for (let i = 0; i < numEnemies; i++) {
    let x, y;
    let validPosition = false;
    
    while (!validPosition) {
      x = p.random(100, level.width - 100);
      y = p.random(100, level.height - 100);
      
      const centerDist = p.dist(x, y, level.width/2, level.height/2);
      if (centerDist > 300) {
        let collides = false;
        for (const obstacle of gameState.obstacles) {
          if (checkCollisionWithRect(x, y, 15, obstacle.x, obstacle.y, obstacle.width, obstacle.height)) {
            collides = true;
            break;
          }
        }
        
        if (!collides) {
          validPosition = true;
        }
      }
    }
    
    // Enemy type distribution based on 6-level progression
    let enemyType = "regular";
    const rand = p.random();
    const currentLevel = gameState.currentLevel;
    
    if (currentLevel === 1) {
      // Easy 1: Mostly regular, rare scout
      if (rand < 0.9) enemyType = "regular";
      else enemyType = "scout"; 
    } else if (currentLevel === 2) {
      // Easy 2: Intro Elite
      if (rand < 0.7) enemyType = "regular";
      else if (rand < 0.9) enemyType = "scout";
      else enemyType = "elite";
    } else if (currentLevel === 3) {
      // Medium 1: More Elites, Intro Heavy
      if (rand < 0.5) enemyType = "regular";
      else if (rand < 0.75) enemyType = "scout";
      else if (rand < 0.9) enemyType = "elite";
      else enemyType = "heavy";
    } else if (currentLevel === 4) {
      // Medium 2: More Heavies
      if (rand < 0.4) enemyType = "regular";
      else if (rand < 0.6) enemyType = "scout";
      else if (rand < 0.8) enemyType = "elite";
      else enemyType = "heavy";
    } else if (currentLevel === 5) {
      // Hard 1: Intro Sniper
      if (rand < 0.3) enemyType = "regular";
      else if (rand < 0.5) enemyType = "scout";
      else if (rand < 0.7) enemyType = "elite";
      else if (rand < 0.85) enemyType = "heavy";
      else enemyType = "sniper";
    } else {
      // Hard 2 (Level 6+): Intro Tank, Chaos
      if (rand < 0.2) enemyType = "regular";
      else if (rand < 0.4) enemyType = "scout";
      else if (rand < 0.6) enemyType = "elite";
      else if (rand < 0.75) enemyType = "heavy";
      else if (rand < 0.9) enemyType = "sniper";
      else enemyType = "tank";
    }
    
    gameState.enemies.push(new Enemy(x, y, enemyType));
  }
  
  // Generate health pickups
  const numHealthPickups = 6 + gameState.currentLevel;
  for (let i = 0; i < numHealthPickups; i++) {
    placePickup(p, "health");
  }
  
  // Generate ammo pickups
  const numAmmoPickups = 4 + Math.floor(gameState.currentLevel);
  for (let i = 0; i < numAmmoPickups; i++) {
    placePickup(p, "ammo");
  }
  
  // Generate weapon pickups - more weapons at higher levels
  const weaponTypes = ["rifle", "shotgun", "sniper", "rocket_launcher"];
  const numWeaponPickups = Math.min(gameState.currentLevel, 4);
  for (let i = 0; i < numWeaponPickups; i++) {
    // Cycle through weapon types
    const weaponType = weaponTypes[i % weaponTypes.length];
    placeWeaponPickup(p, weaponType);
  }
  
  // ALWAYS generate extraction point now
  let x, y;
  let validPosition = false;
  
  while (!validPosition) {
    x = p.random(100, level.width - 100);
    y = p.random(100, level.height - 100);
    
    const centerDist = p.dist(x, y, level.width/2, level.height/2);
    if (centerDist > 500) {
      let collides = false;
      for (const obstacle of gameState.obstacles) {
        if (checkCollisionWithRect(x, y, 20, obstacle.x, obstacle.y, obstacle.width, obstacle.height)) {
          collides = true;
          break;
        }
      }
      
      if (!collides) {
        validPosition = true;
      }
    }
  }
  
  gameState.extractionPoint = { x, y };
  gameState.extractionPointObj = new ExtractionPoint(x, y);
}

function placePickup(p, type) {
  let x, y;
  let validPosition = false;
  
  while (!validPosition) {
    x = p.random(100, gameState.level.width - 100);
    y = p.random(100, gameState.level.height - 100);
    
    let collides = false;
    for (const obstacle of gameState.obstacles) {
      if (checkCollisionWithRect(x, y, 10, obstacle.x, obstacle.y, obstacle.width, obstacle.height)) {
        collides = true;
        break;
      }
    }
    
    let tooClose = false;
    for (const pickup of gameState.pickups) {
      if (p.dist(x, y, pickup.x, pickup.y) < 60) {
        tooClose = true;
        break;
      }
    }
    
    if (!collides && !tooClose) {
      validPosition = true;
    }
  }
  
  gameState.pickups.push(new Pickup(x, y, type));
}

function placeWeaponPickup(p, weaponType) {
  let x, y;
  let validPosition = false;
  
  while (!validPosition) {
    x = p.random(100, gameState.level.width - 100);
    y = p.random(100, gameState.level.height - 100);
    
    let collides = false;
    for (const obstacle of gameState.obstacles) {
      if (checkCollisionWithRect(x, y, 12, obstacle.x, obstacle.y, obstacle.width, obstacle.height)) {
        collides = true;
        break;
      }
    }
    
    let tooClose = false;
    for (const pickup of [...gameState.pickups, ...gameState.weaponPickups]) {
      if (p.dist(x, y, pickup.x, pickup.y) < 80) {
        tooClose = true;
        break;
      }
    }
    
    if (!collides && !tooClose) {
      validPosition = true;
    }
  }
  
  gameState.weaponPickups.push(new WeaponPickup(x, y, weaponType));
}