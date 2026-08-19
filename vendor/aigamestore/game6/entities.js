// entities.js - Entity classes

import { 
  ENTITY_TYPES, 
  PLAYER_WIDTH, 
  PLAYER_HEIGHT, 
  PLAYER_SPEED,
  JUMP_FORCE,
  GRAVITY,
  MAX_FALL_SPEED,
  gameState
} from './globals.js';
import { checkCollisionAABB } from './utils.js';

export class Entity {
  constructor(x, y, width, height, type) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
    this.type = type;
    this.active = true;
  }

  update() {}
  render(p) {}
}

export class Player extends Entity {
  constructor(x, y) {
    super(x, y, PLAYER_WIDTH, PLAYER_HEIGHT, ENTITY_TYPES.PLAYER);
    this.vx = 0;
    this.vy = 0;
    this.onGround = false;
    this.facingRight = true;
    this.interacting = false;
    this.holdingObject = null;
    this.animFrame = 0;
  }

  update() {
    this.animFrame += 0.1;
  }

  applyGravity() {
    this.vy += GRAVITY;
    if (this.vy > MAX_FALL_SPEED) {
      this.vy = MAX_FALL_SPEED;
    }
  }

  move(direction) {
    this.vx = direction * PLAYER_SPEED;
    if (direction !== 0) {
      this.facingRight = direction > 0;
    }
  }

  jump() {
    if (this.onGround) {
      this.vy = JUMP_FORCE;
      this.onGround = false;
    }
  }

  render(p) {
    const headRadius = 8;
    const bodyHeight = 15;
    const legLength = 7;
    
    // Body
    p.push();
    p.fill(50, 50, 60);
    p.noStroke();
    
    // Head
    p.circle(this.x + this.width/2, this.y + headRadius, headRadius * 2);
    
    // Body rectangle
    p.rect(this.x + this.width/2 - 6, this.y + headRadius * 2, 12, bodyHeight);
    
    // Simple legs (animated)
    const legOffset = Math.sin(this.animFrame) * 3;
    p.rect(this.x + this.width/2 - 7, this.y + headRadius * 2 + bodyHeight, 3, legLength + legOffset);
    p.rect(this.x + this.width/2 + 4, this.y + headRadius * 2 + bodyHeight, 3, legLength - legOffset);
    
    // Eyes (glowing)
    p.fill(220, 220, 255);
    const eyeX = this.facingRight ? 2 : -2;
    p.circle(this.x + this.width/2 + eyeX, this.y + headRadius - 1, 3);
    
    p.pop();
  }
}

export class Crate extends Entity {
  constructor(x, y) {
    super(x, y, 40, 40, ENTITY_TYPES.CRATE);
    this.vx = 0;
    this.vy = 0;
    this.onGround = false;
    this.beingPushed = false;
  }

  update() {
    // Apply gravity
    this.vy += GRAVITY;
    if (this.vy > MAX_FALL_SPEED) {
      this.vy = MAX_FALL_SPEED;
    }
  }

  render(p) {
    p.push();
    p.fill(100, 70, 50);
    p.stroke(60, 40, 30);
    p.strokeWeight(2);
    p.rect(this.x, this.y, this.width, this.height);
    
    // Wood grain
    p.stroke(80, 55, 40);
    p.strokeWeight(1);
    p.line(this.x + 10, this.y, this.x + 10, this.y + this.height);
    p.line(this.x + 30, this.y, this.x + 30, this.y + this.height);
    
    // Cross pattern
    p.line(this.x + 5, this.y + 5, this.x + this.width - 5, this.y + this.height - 5);
    p.line(this.x + this.width - 5, this.y + 5, this.x + 5, this.y + this.height - 5);
    
    p.pop();
  }
}

export class Spike extends Entity {
  constructor(x, y, width) {
    super(x, y, width, 15, ENTITY_TYPES.SPIKE);
    this.numSpikes = Math.floor(width / 15);
  }

  render(p) {
    p.push();
    p.fill(80, 80, 90);
    p.noStroke();
    
    for (let i = 0; i < this.numSpikes; i++) {
      const spikeX = this.x + i * 15;
      p.triangle(
        spikeX, this.y + this.height,
        spikeX + 7.5, this.y,
        spikeX + 15, this.y + this.height
      );
    }
    p.pop();
  }
}

export class Lever extends Entity {
  constructor(x, y, targetGateId) {
    super(x, y, 20, 30, ENTITY_TYPES.LEVER);
    this.activated = false;
    this.targetGateId = targetGateId;
  }

  activate() {
    if (!this.activated) {
      this.activated = true;
      return true;
    }
    return false;
  }

  render(p) {
    p.push();
    
    // Base
    p.fill(60, 60, 70);
    p.noStroke();
    p.rect(this.x, this.y + 20, this.width, 10);
    
    // Lever arm
    p.stroke(80, 80, 90);
    p.strokeWeight(4);
    if (this.activated) {
      p.line(this.x + 10, this.y + 20, this.x + 18, this.y + 5);
      p.fill(100, 200, 100);
    } else {
      p.line(this.x + 10, this.y + 20, this.x + 2, this.y + 5);
      p.fill(200, 100, 100);
    }
    
    // Handle
    p.noStroke();
    p.circle(this.activated ? this.x + 18 : this.x + 2, this.y + 5, 8);
    
    p.pop();
  }
}

export class Gate extends Entity {
  constructor(x, y, height, id) {
    super(x, y, 20, height, ENTITY_TYPES.GATE);
    this.id = id;
    this.open = false;
    this.openProgress = 0;
  }

  update() {
    if (this.open && this.openProgress < 1) {
      this.openProgress += 0.05;
      if (this.openProgress > 1) this.openProgress = 1;
    }
  }

  render(p) {
    if (this.openProgress >= 1) return;
    
    p.push();
    const currentHeight = this.height * (1 - this.openProgress);
    
    p.fill(70, 70, 80);
    p.stroke(50, 50, 60);
    p.strokeWeight(2);
    p.rect(this.x, this.y + this.height - currentHeight, this.width, currentHeight);
    
    // Horizontal bars
    for (let i = 0; i < 4; i++) {
      const barY = this.y + this.height - currentHeight + (currentHeight / 4) * i;
      p.line(this.x, barY, this.x + this.width, barY);
    }
    
    p.pop();
  }

  isBlocking() {
    return !this.open || this.openProgress < 1;
  }
}

export class Checkpoint extends Entity {
  constructor(x, y, index) {
    super(x, y, 30, 60, ENTITY_TYPES.CHECKPOINT);
    this.index = index;
    this.activated = false;
    this.glowPhase = 0;
  }

  update() {
    this.glowPhase += 0.1;
  }

  activate() {
    this.activated = true;
  }

  render(p) {
    p.push();
    
    // Pole
    p.stroke(100, 100, 110);
    p.strokeWeight(3);
    p.line(this.x + 5, this.y, this.x + 5, this.y + this.height);
    
    // Flag
    if (this.activated) {
      p.fill(100, 200, 255, 150 + Math.sin(this.glowPhase) * 50);
      p.noStroke();
      p.triangle(
        this.x + 5, this.y,
        this.x + this.width, this.y + 15,
        this.x + 5, this.y + 30
      );
      
      // Glow effect
      p.fill(100, 200, 255, 50);
      p.circle(this.x + 15, this.y + 15, 40 + Math.sin(this.glowPhase) * 5);
    } else {
      p.fill(100, 100, 120);
      p.noStroke();
      p.triangle(
        this.x + 5, this.y,
        this.x + this.width, this.y + 15,
        this.x + 5, this.y + 30
      );
    }
    
    p.pop();
  }
}

export class Platform extends Entity {
  constructor(x, y, width, height) {
    super(x, y, width, height, ENTITY_TYPES.PLATFORM);
  }

  render(p) {
    p.push();
    p.fill(60, 60, 70);
    p.stroke(40, 40, 50);
    p.strokeWeight(2);
    p.rect(this.x, this.y, this.width, this.height);
    
    // Texture lines
    p.stroke(50, 50, 60);
    p.strokeWeight(1);
    for (let i = 10; i < this.width; i += 20) {
      p.line(this.x + i, this.y, this.x + i, this.y + this.height);
    }
    p.pop();
  }
}

export class MovingPlatform extends Platform {
  constructor(x, y, width, height, startX, endX, speed) {
    super(x, y, width, height);
    this.type = ENTITY_TYPES.MOVING_PLATFORM;
    this.startX = startX;
    this.endX = endX;
    this.speed = speed;
    this.direction = 1;
    this.vx = speed;
  }

  update() {
    this.x += this.vx;
    
    if (this.x <= this.startX) {
      this.x = this.startX;
      this.vx = this.speed;
    } else if (this.x >= this.endX) {
      this.x = this.endX;
      this.vx = -this.speed;
    }
  }

  render(p) {
    p.push();
    p.fill(80, 80, 100);
    p.stroke(60, 60, 80);
    p.strokeWeight(2);
    p.rect(this.x, this.y, this.width, this.height);
    
    // Direction indicator
    p.fill(120, 120, 140);
    p.noStroke();
    const arrowX = this.x + this.width / 2;
    const arrowY = this.y + this.height / 2;
    if (this.vx > 0) {
      p.triangle(arrowX - 5, arrowY - 5, arrowX + 5, arrowY, arrowX - 5, arrowY + 5);
    } else {
      p.triangle(arrowX + 5, arrowY - 5, arrowX - 5, arrowY, arrowX + 5, arrowY + 5);
    }
    
    p.pop();
  }
}