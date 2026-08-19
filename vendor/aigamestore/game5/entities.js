import { gameState, CANVAS_HEIGHT } from './globals.js';
import { handlePlayerHit } from './collision.js';

export class Player {
  constructor(p, x, y) {
    this.p = p;
    this.x = x;
    this.y = y;
    this.w = 25;
    this.h = 25;
    this.vx = 0;
    this.vy = 0;
    this.onGround = false;
    this.moveSpeed = 3;
    this.jumpPower = -8;
    this.gravity = 0.4;
    this.facingRight = true;
    this.animFrame = 0;
    this.animTimer = 0;
  }

  update() {
    // Apply gravity
    this.vy += this.gravity;
    
    // Apply velocity
    this.x += this.vx;
    this.y += this.vy;
    
    // Check ground collision
    this.onGround = false;
    const platforms = gameState.entities.filter(e => e.type === 'platform');
    for (const platform of platforms) {
      if (this.checkPlatformCollision(platform)) {
        this.onGround = true;
        this.vy = 0;
        this.y = platform.y - this.h / 2;
        break;
      }
    }
    
    // Keep player in horizontal bounds
    this.x = this.p.constrain(this.x, this.w / 2, 600 - this.w / 2);
    
    // Check if player fell off the map
    if (this.y > CANVAS_HEIGHT + 20) {
      handlePlayerHit(this.p);
      return;
    }
    
    // Animation
    if (this.vx !== 0 && this.onGround) {
      this.animTimer++;
      if (this.animTimer > 8) {
        this.animFrame = (this.animFrame + 1) % 2;
        this.animTimer = 0;
      }
    } else {
      this.animFrame = 0;
      this.animTimer = 0;
    }
  }

  checkPlatformCollision(platform) {
    return this.p.collideRectRect(
      this.x - this.w / 2, this.y - this.h / 2, this.w, this.h,
      platform.x, platform.y, platform.w, platform.h
    ) && this.vy >= 0 && this.y - this.h / 2 < platform.y + 5;
  }

  draw() {
    this.p.push();
    this.p.translate(this.x, this.y);
    if (!this.facingRight) {
      this.p.scale(-1, 1);
    }
    
    // Mouse body (grey)
    this.p.fill(160, 160, 160);
    this.p.noStroke();
    this.p.ellipse(0, 0, this.w, this.h);
    
    // Ears
    this.p.fill(140, 140, 140);
    this.p.ellipse(-8, -8, 8, 10);
    this.p.ellipse(8, -8, 8, 10);
    
    // Eyes
    this.p.fill(0);
    this.p.ellipse(-5, -2, 3, 3);
    this.p.ellipse(5, -2, 3, 3);
    
    // Nose
    this.p.fill(255, 180, 200);
    this.p.ellipse(0, 3, 4, 3);
    
    // Whiskers
    this.p.stroke(80);
    this.p.strokeWeight(1);
    this.p.line(-10, 0, -15, -2);
    this.p.line(-10, 2, -15, 4);
    this.p.line(10, 0, 15, -2);
    this.p.line(10, 2, 15, 4);
    
    // Legs animation
    if (this.animFrame === 1 && this.onGround) {
      this.p.noStroke();
      this.p.fill(150, 150, 150);
      this.p.ellipse(-6, 10, 4, 6);
      this.p.ellipse(6, 10, 4, 6);
    }
    
    this.p.pop();
  }

  jump() {
    if (this.onGround) {
      this.vy = this.jumpPower;
      this.onGround = false;
    }
  }

  moveLeft() {
    this.vx = -this.moveSpeed;
    this.facingRight = false;
  }

  moveRight() {
    this.vx = this.moveSpeed;
    this.facingRight = true;
  }

  stopMove() {
    this.vx = 0;
  }
}

export class Cat {
  constructor(p, x, y, patrolPath, speed) {
    this.p = p;
    this.x = x;
    this.y = y;
    this.w = 35;
    this.h = 35;
    this.type = 'cat';
    this.patrolPath = patrolPath;
    this.patrolIndex = 0;
    this.speed = speed;
    this.facingRight = true;
    this.animFrame = 0;
    this.animTimer = 0;
  }

  update() {
    if (this.patrolPath.length > 0) {
      const target = this.patrolPath[this.patrolIndex];
      const dx = target.x - this.x;
      const dy = target.y - this.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < 5) {
        this.patrolIndex = (this.patrolIndex + 1) % this.patrolPath.length;
      } else {
        this.x += (dx / dist) * this.speed;
        this.y += (dy / dist) * this.speed;
        this.facingRight = dx > 0;
      }
      
      // Animation
      this.animTimer++;
      if (this.animTimer > 10) {
        this.animFrame = (this.animFrame + 1) % 2;
        this.animTimer = 0;
      }
    }
  }

  draw() {
    this.p.push();
    this.p.translate(this.x, this.y);
    if (!this.facingRight) {
      this.p.scale(-1, 1);
    }
    
    // Cat body (orange-brown)
    this.p.fill(200, 120, 60);
    this.p.noStroke();
    this.p.ellipse(0, 0, this.w, this.h);
    
    // Ears (triangular)
    this.p.fill(180, 100, 50);
    this.p.triangle(-10, -15, -6, -10, -14, -10);
    this.p.triangle(10, -15, 6, -10, 14, -10);
    
    // Eyes (more menacing)
    this.p.fill(255, 255, 0);
    this.p.ellipse(-7, -3, 6, 8);
    this.p.ellipse(7, -3, 6, 8);
    this.p.fill(0);
    this.p.ellipse(-7, -2, 3, 6);
    this.p.ellipse(7, -2, 3, 6);
    
    // Nose
    this.p.fill(255, 180, 180);
    this.p.triangle(-2, 3, 2, 3, 0, 5);
    
    // Whiskers
    this.p.stroke(60);
    this.p.strokeWeight(1);
    this.p.line(-12, 0, -18, -2);
    this.p.line(-12, 2, -18, 4);
    this.p.line(12, 0, 18, -2);
    this.p.line(12, 2, 18, 4);
    
    // Legs animation
    if (this.animFrame === 1) {
      this.p.noStroke();
      this.p.fill(180, 100, 50);
      this.p.ellipse(-8, 14, 5, 8);
      this.p.ellipse(8, 14, 5, 8);
    }
    
    this.p.pop();
  }
}

export class Cheese {
  constructor(p, x, y) {
    this.p = p;
    this.x = x;
    this.y = y;
    this.w = 20;
    this.h = 20;
    this.type = 'cheese';
    this.collected = false;
    this.pulseTimer = 0;
  }

  update() {
    this.pulseTimer += 0.1;
  }

  draw() {
    if (this.collected) return;
    
    this.p.push();
    this.p.translate(this.x, this.y);
    
    const pulse = Math.sin(this.pulseTimer) * 2;
    const size = this.w + pulse;
    
    // Cheese wedge (yellow)
    this.p.fill(255, 220, 60);
    this.p.noStroke();
    this.p.beginShape();
    this.p.vertex(-size / 2, size / 2);
    this.p.vertex(size / 2, size / 2);
    this.p.vertex(0, -size / 2);
    this.p.endShape(this.p.CLOSE);
    
    // Holes
    this.p.fill(230, 180, 40);
    this.p.ellipse(-3, 3, 4, 4);
    this.p.ellipse(4, 5, 3, 3);
    this.p.ellipse(0, -3, 3, 3);
    
    // Sparkle
    this.p.stroke(255, 255, 200, 150);
    this.p.strokeWeight(2);
    const sparkle = Math.abs(Math.sin(this.pulseTimer * 2));
    this.p.line(-size / 2 - 3, -size / 2 - 3, -size / 2 - 3 - sparkle * 3, -size / 2 - 3 - sparkle * 3);
    this.p.line(size / 2 + 3, -size / 2 - 3, size / 2 + 3 + sparkle * 3, -size / 2 - 3 - sparkle * 3);
    
    this.p.pop();
  }
}

export class Platform {
  constructor(p, x, y, w, h) {
    this.p = p;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.type = 'platform';
  }

  update() {}

  draw() {
    this.p.push();
    this.p.fill(100, 80, 60);
    this.p.stroke(80, 60, 40);
    this.p.strokeWeight(2);
    this.p.rect(this.x, this.y, this.w, this.h);
    
    // Add texture lines
    this.p.stroke(120, 100, 80);
    this.p.strokeWeight(1);
    for (let i = 10; i < this.w; i += 20) {
      this.p.line(this.x + i, this.y, this.x + i, this.y + this.h);
    }
    this.p.pop();
  }
}

export class MouseHole {
  constructor(p, x, y) {
    this.p = p;
    this.x = x;
    this.y = y;
    this.w = 35;
    this.h = 35;
    this.type = 'mousehole';
    this.glowTimer = 0;
  }

  update() {
    if (gameState.mouseHoleActive) {
      this.glowTimer += 0.15;
    }
  }

  draw() {
    this.p.push();
    this.p.translate(this.x, this.y);
    
    // Glow effect when active
    if (gameState.mouseHoleActive) {
      const glow = Math.abs(Math.sin(this.glowTimer)) * 20 + 10;
      this.p.fill(255, 255, 150, glow);
      this.p.noStroke();
      this.p.ellipse(0, 0, this.w + 20, this.h + 20);
      
      this.p.fill(255, 255, 100, glow * 2);
      this.p.ellipse(0, 0, this.w + 10, this.h + 10);
    }
    
    // Hole
    this.p.fill(...(gameState.mouseHoleActive ? [60, 50, 40] : [30, 25, 20]));
    this.p.noStroke();
    this.p.ellipse(0, 0, this.w, this.h);
    
    // Arch detail
    this.p.fill(...(gameState.mouseHoleActive ? [80, 70, 60] : [50, 45, 40]));
    this.p.arc(0, 5, this.w - 5, this.h - 5, this.p.PI, 0, this.p.CHORD);
    
    this.p.pop();
  }
}

export class Particle {
  constructor(p, x, y, vx, vy, color, life) {
    this.p = p;
    this.x = x;
    this.y = y;
    this.vx = vx;
    this.vy = vy;
    this.color = color;
    this.life = life;
    this.maxLife = life;
    this.size = 4;
    this.type = 'particle';
  }

  update() {
    this.x += this.vx;
    this.y += this.vy;
    this.vy += 0.2; // gravity
    this.life--;
  }

  draw() {
    const alpha = (this.life / this.maxLife) * 255;
    this.p.push();
    this.p.fill(...this.color, alpha);
    this.p.noStroke();
    this.p.ellipse(this.x, this.y, this.size, this.size);
    this.p.pop();
  }

  isDead() {
    return this.life <= 0;
  }
}