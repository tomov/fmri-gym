import { gameState, GROUND_Y, CANVAS_WIDTH, CANVAS_HEIGHT, GRAVITY } from './globals.js';

const Matter = window.Matter;
const Engine = Matter.Engine;
const World = Matter.World;
const Bodies = Matter.Bodies;
const Body = Matter.Body;
const Events = Matter.Events;

// Initialize Matter.js physics engine
export function initPhysicsEngine() {
  gameState.matterEngine = Engine.create({
    gravity: { x: 0, y: GRAVITY }
  });
  
  gameState.matterWorld = gameState.matterEngine.world;
  
  createGround();
  
  // Setup collision detection
  Events.on(gameState.matterEngine, 'collisionStart', handleCollisionStart);
}

// Create ground body
function createGround() {
  gameState.groundBody = Bodies.rectangle(CANVAS_WIDTH / 2, GROUND_Y + 25, CANVAS_WIDTH, 50, {
    isStatic: true,
    friction: 0.8,
    restitution: 0.2
  });
  World.add(gameState.matterWorld, gameState.groundBody);
}

// Update physics
export function updatePhysics() {
  if (!gameState.matterEngine) return;
  
  Engine.update(gameState.matterEngine, 1000 / 60);
  
  // Clean up bodies that fell off screen
  gameState.entities.forEach(entity => {
    if (entity.body && entity.body.position.y > CANVAS_HEIGHT + 100) {
      entity.active = false;
      if (entity.body && gameState.matterWorld) {
        World.remove(gameState.matterWorld, entity.body);
        entity.body = null;
      }
    }
  });
  
  // Check egg collisions and trigger explosions
  gameState.entities.forEach(entity => {
    if (entity.type === 'egg' && entity.active && !entity.hasExploded) {
      // Check if egg is moving very slowly or hit something
      const speed = Math.sqrt(
        entity.body.velocity.x * entity.body.velocity.x +
        entity.body.velocity.y * entity.body.velocity.y
      );
      if (speed < 1 || entity.body.position.y > GROUND_Y - 10) {
        entity.explode();
      }
    }
  });
  
  // Update particle effects
  gameState.particleEffects = gameState.particleEffects.filter(particle => {
    particle.x += particle.vx;
    particle.y += particle.vy;
    particle.vy += 0.2; // Gravity
    particle.life--;
    return particle.life > 0;
  });
}

// Collision handling
function handleCollisionStart(event) {
  const pairs = event.pairs;
  
  pairs.forEach(pair => {
    const bodyA = pair.bodyA;
    const bodyB = pair.bodyB;
    
    // Find entities for these bodies
    const entityA = gameState.entities.find(e => e.body === bodyA);
    const entityB = gameState.entities.find(e => e.body === bodyB);
    
    // Check for egg hitting anything - trigger explosion
    if (entityA && entityA.type === 'egg' && !entityA.hasExploded) {
      entityA.explode();
    }
    if (entityB && entityB.type === 'egg' && !entityB.hasExploded) {
      entityB.explode();
    }
    
    // Check for pig hitting ground with high velocity
    if (bodyA === gameState.groundBody && entityB && entityB.type === 'pig') {
      handleGroundImpact(entityB, bodyB);
    } else if (bodyB === gameState.groundBody && entityA && entityA.type === 'pig') {
      handleGroundImpact(entityA, bodyA);
    }
    
    if (!entityA || !entityB) return;
    
    // Calculate impact force
    const speed = Math.sqrt(
      Math.pow(bodyA.velocity.x - bodyB.velocity.x, 2) +
      Math.pow(bodyA.velocity.y - bodyB.velocity.y, 2)
    );
    
    const mass = Math.min(bodyA.mass, bodyB.mass);
    const force = speed * mass;
    
    if (force > 1.5) {
      handleCollisionDamage(entityA, entityB, force);
    }
  });
}

function handleGroundImpact(pig, pigBody) {
  // Calculate impact velocity (only vertical component matters for ground)
  const impactSpeed = Math.abs(pigBody.velocity.y);
  
  // If pig is falling fast, take damage
  if (impactSpeed > 1.5) {
    const damage = Math.ceil(impactSpeed / 1.5);
    pig.takeDamage(damage);
  }
}

function handleCollisionDamage(e1, e2, force) {
  const isBird1 = e1.type === 'bird';
  const isBird2 = e2.type === 'bird';
  const isPig1 = e1.type === 'pig';
  const isPig2 = e2.type === 'pig';
  const isBlock1 = e1.type === 'block';
  const isBlock2 = e2.type === 'block';
  
  // Bird hitting pig
  if (isBird1 && isPig2 && force > 1.5) {
    const damage = Math.max(1, Math.floor(force / 2));
    e2.takeDamage(damage);
  } else if (isBird2 && isPig1 && force > 1.5) {
    const damage = Math.max(1, Math.floor(force / 2));
    e1.takeDamage(damage);
  }
  
  // Bird or block hitting block
  if ((isBird1 || isBlock1) && isBlock2 && force > e2.durability) {
    e2.takeDamage(1);
  }
  if ((isBird2 || isBlock2) && isBlock1 && force > e1.durability) {
    e1.takeDamage(1);
  }
  
  // Block landing on pig - INSTANT KILL if falling from above
  if (isBlock1 && isPig2) {
    const isAbove = e1.body.position.y < e2.body.position.y;
    const isFalling = e1.body.velocity.y > 0.5;
    if (isAbove && isFalling) {
      e2.takeDamage(100);
    } else if (force > 2.5) {
      const damage = Math.max(1, Math.ceil(force / 3));
      e2.takeDamage(damage);
    }
  } else if (isBlock2 && isPig1) {
    const isAbove = e2.body.position.y < e1.body.position.y;
    const isFalling = e2.body.velocity.y > 0.5;
    if (isAbove && isFalling) {
      e1.takeDamage(100);
    } else if (force > 2.5) {
      const damage = Math.max(1, Math.ceil(force / 3));
      e1.takeDamage(damage);
    }
  }
  
  // Pig landing on pig - INSTANT KILL the one below
  if (isPig1 && isPig2) {
    const pig1Above = e1.body.position.y < e2.body.position.y;
    const pig2Above = e2.body.position.y < e1.body.position.y;
    const pig1Falling = e1.body.velocity.y > 0.5;
    const pig2Falling = e2.body.velocity.y > 0.5;
    
    if (pig1Above && pig1Falling) {
      e2.takeDamage(100);
    }
    if (pig2Above && pig2Falling) {
      e1.takeDamage(100);
    }
    
    if (force > 2 && (pig1Falling || pig2Falling) && !pig1Above && !pig2Above) {
      const damage = Math.max(1, Math.floor(force / 3));
      if (pig1Falling) e1.takeDamage(damage);
      if (pig2Falling) e2.takeDamage(damage);
    }
  }
}

// Create particle effect
export function createParticleEffect(x, y, color, count = 8) {
  const particles = [];
  for (let i = 0; i < count; i++) {
    const angle = (Math.PI * 2 * i) / count;
    const speed = 2 + Math.random() * 3;
    particles.push({
      x: x,
      y: y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      life: 30,
      maxLife: 30,
      color: color,
      size: 3 + Math.random() * 3
    });
  }
  return particles;
}

// Clear all bodies from world
export function clearPhysicsWorld() {
  if (!gameState.matterWorld) return;
  
  // Remove all entities
  gameState.entities.forEach(entity => {
    if (entity.body) {
      World.remove(gameState.matterWorld, entity.body);
    }
  });
  
  // Remove ground if it exists
  if (gameState.groundBody) {
    World.remove(gameState.matterWorld, gameState.groundBody);
    gameState.groundBody = null;
  }
  
  // Recreate ground
  createGround();
}