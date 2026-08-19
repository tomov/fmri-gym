import { gameState, WEAPONS } from './globals.js';
import { Player } from './player.js';

export function updatePhysics(p) {
  // Update projectiles
  for (let i = gameState.projectiles.length - 1; i >= 0; i--) {
    const proj = gameState.projectiles[i];
    proj.x += proj.vx;
    proj.y += proj.vy;
    proj.lifetime--;

    // Safe check for stage width (currentStage may be null during transitions)
    const stageWidth = gameState.currentStage?.width || 1200;
    if (proj.lifetime <= 0 || proj.x < -50 || proj.x > stageWidth + 50 ||
        proj.y < -50 || proj.y > 500) {
      gameState.projectiles.splice(i, 1);
      continue;
    }

    // Check collision with platforms
    for (let plat of gameState.platformBlocks) {
      if (proj.x > plat.x && proj.x < plat.x + plat.width &&
          proj.y > plat.y && proj.y < plat.y + plat.height) {
        gameState.projectiles.splice(i, 1);
        break;
      }
    }
  }

  // Projectile collision with entities
  for (let i = gameState.projectiles.length - 1; i >= 0; i--) {
    const proj = gameState.projectiles[i];
    
    if (proj.isPlayerProjectile) {
      // Hit enemies
      for (let j = gameState.entities.length - 1; j >= 0; j--) {
        const entity = gameState.entities[j];
        if (entity === gameState.player) continue;

        if (proj.x > entity.x && proj.x < entity.x + entity.width &&
            proj.y > entity.y && proj.y < entity.y + entity.height) {
          const died = entity.takeDamage(proj.damage, proj.weapon);
          gameState.projectiles.splice(i, 1);
          
          if (died) {
            gameState.entities.splice(j, 1);
            gameState.score += 100;
            
            // Drop weapon energy
            if (p.random() < 0.3) {
              gameState.drops.push({
                x: entity.x + entity.width / 2,
                y: entity.y + entity.height / 2,
                type: 'weapon_energy',
                amount: 5,
                vy: -2,
                lifetime: 300
              });
            }
            
            // Update boss health
            if (entity.health !== undefined && entity.maxHealth) {
              gameState.bossHealth = entity.health;
            }
          }
          break;
        }
      }
    } else {
      // Hit player
      const player = gameState.player;
      if (player && proj.x > player.x && proj.x < player.x + player.width &&
          proj.y > player.y && proj.y < player.y + player.height) {
        player.takeDamage(p, proj.damage);
        gameState.projectiles.splice(i, 1);
      }
    }
  }

  // Check hazard collision with player
  const player = gameState.player;
  if (player) {
    for (let hazard of gameState.hazards) {
      if (player.x + player.width > hazard.x && player.x < hazard.x + hazard.width &&
          player.y + player.height > hazard.y && player.y < hazard.y + hazard.height) {
        player.takeDamage(p, 10);
      }
    }
  }

  // Update drops
  for (let i = gameState.drops.length - 1; i >= 0; i--) {
    const drop = gameState.drops[i];
    drop.y += drop.vy;
    drop.vy += 0.3;
    drop.lifetime--;

    // Check collision with platforms
    for (let plat of gameState.platformBlocks) {
      if (drop.y + 10 > plat.y && drop.y < plat.y &&
          drop.x + 10 > plat.x && drop.x < plat.x + plat.width) {
        drop.y = plat.y - 10;
        drop.vy = 0;
        break;
      }
    }

    // Check collection by player
    if (player && drop.x + 10 > player.x && drop.x < player.x + player.width &&
        drop.y + 10 > player.y && drop.y < player.y + player.height) {
      if (drop.type === 'weapon_energy') {
        const currentWeapon = gameState.unlockedWeapons[gameState.currentWeapon];
        if (currentWeapon !== 'BUSTER' && gameState.weaponEnergy[currentWeapon] !== undefined) {
          gameState.weaponEnergy[currentWeapon] = Math.min(
            gameState.weaponEnergy[currentWeapon] + drop.amount,
            28
          );
        }
      }
      gameState.drops.splice(i, 1);
      continue;
    }

    if (drop.lifetime <= 0) {
      gameState.drops.splice(i, 1);
    }
  }

  // Decrease invincibility frames
  if (gameState.invincibilityFrames > 0) {
    gameState.invincibilityFrames--;
  }
}

export function updateParticles(p) {
  for (let i = gameState.particles.length - 1; i >= 0; i--) {
    const particle = gameState.particles[i];
    particle.x += particle.vx;
    particle.y += particle.vy;
    particle.vy += 0.2;
    particle.lifetime--;
    particle.alpha = particle.lifetime * 5;

    if (particle.lifetime <= 0) {
      gameState.particles.splice(i, 1);
    }
  }
}

export function createExplosion(x, y, color) {
  for (let i = 0; i < 10; i++) {
    const angle = (i / 10) * Math.PI * 2;
    gameState.particles.push({
      x: x,
      y: y,
      vx: Math.cos(angle) * 3,
      vy: Math.sin(angle) * 3,
      color: color,
      alpha: 255,
      lifetime: 30
    });
  }
}