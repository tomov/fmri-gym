import { gameState, BIRD_TYPES, SCORE_SMALL_PIG, SCORE_LARGE_PIG, SCORE_WOOD_BLOCK, SCORE_STONE_BLOCK, BIRD_AIR_FRICTION } from './globals.js';
import { createParticleEffect } from './physics.js';

const Matter = window.Matter;
const Bodies = Matter.Bodies;
const Body = Matter.Body;
const World = Matter.World;

export class Bird {
  constructor(x, y, birdType) {
    this.type = 'bird';
    this.birdType = birdType;
    
    // Different properties for different bird types
    let radius = 10;
    let density = 0.004;
    
    if (birdType === BIRD_TYPES.BLACK) {
      radius = 12;
      density = 0.008; // Heavy bird
    } else if (birdType === BIRD_TYPES.WHITE) {
      radius = 11;
      density = 0.005;
    }
    
    this.body = Bodies.circle(x, y, radius, {
      density: density,
      restitution: 0.4,
      friction: 0.3,
      frictionAir: BIRD_AIR_FRICTION
    });
    this.active = true;
    this.abilityUsed = false;
    this.trail = [];
    this.maxTrailLength = 15;
    this.size = birdType === BIRD_TYPES.BLACK ? 24 : 20;
    
    World.add(gameState.matterWorld, this.body);
  }

  updateTrail() {
    this.trail.push({ x: this.body.position.x, y: this.body.position.y });
    if (this.trail.length > this.maxTrailLength) {
      this.trail.shift();
    }
  }

  useAbility(gameState, p) {
    if (this.abilityUsed) return null;
    this.abilityUsed = true;

    if (this.birdType === BIRD_TYPES.BLUE) {
      // Split into three birds
      const miniBirds = [];
      for (let i = -1; i <= 1; i++) {
        const miniBird = new Bird(this.body.position.x, this.body.position.y, BIRD_TYPES.BLUE);
        miniBird.size = 12;
        
        // Recreate body with smaller size
        World.remove(gameState.matterWorld, miniBird.body);
        miniBird.body = Bodies.circle(this.body.position.x, this.body.position.y, 6, {
          density: 0.003,
          restitution: 0.4,
          friction: 0.3,
          frictionAir: BIRD_AIR_FRICTION
        });
        World.add(gameState.matterWorld, miniBird.body);
        
        Body.setVelocity(miniBird.body, {
          x: this.body.velocity.x + i * 2,
          y: this.body.velocity.y - 2 + i * 0.5
        });
        miniBird.abilityUsed = true; // Mini birds can't split again
        miniBirds.push(miniBird);
      }
      this.active = false;
      if (this.body && gameState.matterWorld) {
        World.remove(gameState.matterWorld, this.body);
        this.body = null;
      }
      return miniBirds;
    } else if (this.birdType === BIRD_TYPES.YELLOW) {
      // Speed boost
      const speed = Math.sqrt(
        this.body.velocity.x * this.body.velocity.x +
        this.body.velocity.y * this.body.velocity.y
      );
      if (speed > 0) {
        const multiplier = 2.5;
        Body.setVelocity(this.body, {
          x: this.body.velocity.x * multiplier,
          y: this.body.velocity.y * multiplier
        });
      }
    } else if (this.birdType === BIRD_TYPES.GREEN) {
      // Explosion effect - damage nearby entities
      const explosionRadius = 80;
      const explosionForce = 15;
      const pos = this.body.position;
      
      // Create explosion particles
      const particles = createParticleEffect(pos.x, pos.y, [50, 255, 50], 20);
      gameState.particleEffects.push(...particles);
      
      // Damage nearby entities
      gameState.entities.forEach(entity => {
        if (!entity.active || !entity.body || entity === this) return;
        
        const dx = entity.body.position.x - pos.x;
        const dy = entity.body.position.y - pos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < explosionRadius) {
          // Apply damage
          if (entity.type === 'pig') {
            entity.takeDamage(2);
          } else if (entity.type === 'block') {
            entity.takeDamage(1);
          }
          
          // Apply force
          const force = explosionForce / (distance + 1);
          const angle = Math.atan2(dy, dx);
          Body.applyForce(entity.body, entity.body.position, {
            x: Math.cos(angle) * force * 0.01,
            y: Math.sin(angle) * force * 0.01
          });
        }
      });
      
      this.active = false;
      if (this.body && gameState.matterWorld) {
        World.remove(gameState.matterWorld, this.body);
        this.body = null;
      }
    } else if (this.birdType === BIRD_TYPES.WHITE) {
      // Drop egg bomb
      const egg = new EggBomb(this.body.position.x, this.body.position.y + 5);
      Body.setVelocity(egg.body, {
        x: this.body.velocity.x * 0.3,
        y: 8 // Drop downward
      });
      gameState.entities.push(egg);
      
      // White bird bounces upward after dropping egg
      Body.setVelocity(this.body, {
        x: this.body.velocity.x,
        y: this.body.velocity.y - 5
      });
    }
    
    return null;
  }
}

export class EggBomb {
  constructor(x, y) {
    this.type = 'egg';
    this.body = Bodies.circle(x, y, 8, {
      density: 0.006,
      restitution: 0.2,
      friction: 0.3,
      frictionAir: BIRD_AIR_FRICTION
    });
    this.active = true;
    this.size = 16;
    this.hasExploded = false;
    
    World.add(gameState.matterWorld, this.body);
  }
  
  explode() {
    if (this.hasExploded) return;
    this.hasExploded = true;
    
    const explosionRadius = 60;
    const explosionForce = 12;
    const pos = this.body.position;
    
    // Create explosion particles
    const particles = createParticleEffect(pos.x, pos.y, [255, 255, 200], 15);
    gameState.particleEffects.push(...particles);
    
    // Damage nearby entities
    gameState.entities.forEach(entity => {
      if (!entity.active || !entity.body || !entity.body.position || entity === this) return;
      
      const dx = entity.body.position.x - pos.x;
      const dy = entity.body.position.y - pos.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance < explosionRadius) {
        // Apply damage
        if (entity.type === 'pig') {
          entity.takeDamage(2);
        } else if (entity.type === 'block') {
          entity.takeDamage(1);
        }
        
        // Apply force
        const force = explosionForce / (distance + 1);
        const angle = Math.atan2(dy, dx);
        Body.applyForce(entity.body, entity.body.position, {
          x: Math.cos(angle) * force * 0.01,
          y: Math.sin(angle) * force * 0.01
        });
      }
    });
    
    this.active = false;
    if (this.body && gameState.matterWorld) {
      World.remove(gameState.matterWorld, this.body);
      this.body = null;
    }
  }
}

export class Pig {
  constructor(x, y, isLarge = false) {
    this.type = 'pig';
    const size = isLarge ? 15 : 10;
    this.body = Bodies.circle(x, y, size, {
      density: 0.003,
      restitution: 0.3,
      friction: 0.5,
      frictionAir: 0.01
    });
    this.isLarge = isLarge;
    this.health = isLarge ? 2 : 1;
    this.size = isLarge ? 30 : 20;
    this.active = true;
    
    World.add(gameState.matterWorld, this.body);
  }

  takeDamage(amount) {
    this.health -= amount;
    if (this.health <= 0) {
      this.active = false;
      if (this.body && gameState.matterWorld) {
        const pos = { x: this.body.position.x, y: this.body.position.y };
        World.remove(gameState.matterWorld, this.body);
        this.body = null;
        
        // Add score
        const points = this.isLarge ? SCORE_LARGE_PIG : SCORE_SMALL_PIG;
        gameState.score += points;
        gameState.pigsRemaining--;
        
        // Particle effect
        const particles = createParticleEffect(pos.x, pos.y, [100, 200, 100], 10);
        gameState.particleEffects.push(...particles);
      }
      
      return true;
    }
    return false;
  }
}

export class StructureBlock {
  constructor(x, y, width, height, material) {
    this.type = 'block';
    this.material = material; // 'WOOD' or 'STONE'
    this.width = width;
    this.height = height;
    
    if (material === 'WOOD') {
      this.body = Bodies.rectangle(x, y, width, height, {
        density: 0.0015,
        restitution: 0.2,
        friction: 0.8,
        frictionAir: 0.01
      });
      this.durability = 5;
      this.health = 1;
      this.maxHealth = 1;
    } else if (material === 'STONE') {
      this.body = Bodies.rectangle(x, y, width, height, {
        density: 0.003,
        restitution: 0.15,
        friction: 0.9,
        frictionAir: 0.01
      });
      this.durability = 5; // Reduced from 10 to 5
      this.health = 1; // Reduced from 2 to 1
      this.maxHealth = 1;
    }
    
    this.active = true;
    World.add(gameState.matterWorld, this.body);
  }

  takeDamage(amount) {
    this.health -= amount;
    if (this.health <= 0) {
      this.active = false;
      if (this.body && gameState.matterWorld) {
        const pos = { x: this.body.position.x, y: this.body.position.y };
        const color = this.material === 'WOOD' ? [139, 90, 43] : [120, 120, 120];
        World.remove(gameState.matterWorld, this.body);
        this.body = null;
        
        // Add score
        const points = this.material === 'WOOD' ? SCORE_WOOD_BLOCK : SCORE_STONE_BLOCK;
        gameState.score += points;
        
        // Particle effect
        const particles = createParticleEffect(pos.x, pos.y, color, 6);
        gameState.particleEffects.push(...particles);
      }
      
      return true;
    }
    return false;
  }
}
