// shooter.js - Shooter and projectile management

import { gameState, CANVAS_WIDTH, SHOOTER_Y, BUBBLE_RADIUS } from './globals.js';
import { createBubble } from './bubble.js';
import { getRandomColorFromGrid } from './grid.js';

export class Shooter {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.angle = -Math.PI / 2;
    this.rotationSpeed = 0.05;
  }

  rotateLeft() {
    this.angle -= this.rotationSpeed;
    const minAngle = -Math.PI + 0.1;
    if (this.angle < minAngle) this.angle = minAngle;
    gameState.shooterAngle = this.angle;
  }

  rotateRight() {
    this.angle += this.rotationSpeed;
    const maxAngle = -0.1;
    if (this.angle > maxAngle) this.angle = maxAngle;
    gameState.shooterAngle = this.angle;
  }

  draw(p) {
    p.push();
    p.translate(this.x, this.y);
    p.rotate(this.angle + Math.PI / 2);

    // Shooter triangle
    p.fill(150, 150, 150);
    p.stroke(255);
    p.strokeWeight(2);
    p.triangle(-20, 20, 20, 20, 0, -30);

    p.pop();
  }

  getShootDirection() {
    return {
      x: Math.cos(this.angle),
      y: Math.sin(this.angle)
    };
  }
}

export class Projectile {
  constructor(x, y, colorIndex, vx, vy) {
    this.x = x;
    this.y = y;
    this.colorIndex = colorIndex;
    this.color = [255, 255, 255]; // Will be set from bubble
    this.radius = BUBBLE_RADIUS;
    this.vx = vx;
    this.vy = vy;
    this.speed = 8;
    this.active = true;
  }

  update(p) {
    if (!this.active) return;

    this.x += this.vx * this.speed;
    this.y += this.vy * this.speed;

    // Wall bouncing
    if (this.x - this.radius < 0) {
      this.x = this.radius;
      this.vx = Math.abs(this.vx);
    }
    if (this.x + this.radius > CANVAS_WIDTH) {
      this.x = CANVAS_WIDTH - this.radius;
      this.vx = -Math.abs(this.vx);
    }

    // Top boundary
    if (this.y - this.radius < 0) {
      this.y = this.radius;
      this.active = false;
    }
  }

  draw(p) {
    if (!this.active) return;

    p.push();
    p.fill(...this.color);
    p.stroke(255, 255, 255, 200);
    p.strokeWeight(2);
    p.ellipse(this.x, this.y, this.radius * 2);
    p.pop();
  }
}

export function createShooter() {
  return new Shooter(CANVAS_WIDTH / 2, SHOOTER_Y);
}

export function createProjectileFromBubble(shooter, bubble) {
  const dir = shooter.getShootDirection();
  const startX = shooter.x + dir.x * 30;
  const startY = shooter.y + dir.y * 30;

  const projectile = new Projectile(startX, startY, bubble.colorIndex, dir.x, dir.y);
  projectile.color = bubble.color;
  return projectile;
}

export function generateNextBubble(p) {
  const colorIndex = getRandomColorFromGrid(gameState.bubbleGrid, p);
  return createBubble(CANVAS_WIDTH / 2 + 60, SHOOTER_Y, colorIndex);
}

export function drawAimingGuide(shooter, p) {
  const dir = shooter.getShootDirection();
  const startX = shooter.x + dir.x * 30;
  const startY = shooter.y + dir.y * 30;

  let x = startX;
  let y = startY;
  let vx = dir.x;
  let vy = dir.y;

  const points = [[x, y]];
  const maxSteps = 100;

  for (let i = 0; i < maxSteps; i++) {
    x += vx * 5;
    y += vy * 5;

    // Wall bouncing
    if (x < BUBBLE_RADIUS) {
      x = BUBBLE_RADIUS;
      vx = Math.abs(vx);
    }
    if (x > CANVAS_WIDTH - BUBBLE_RADIUS) {
      x = CANVAS_WIDTH - BUBBLE_RADIUS;
      vx = -Math.abs(vx);
    }

    // Top boundary
    if (y < BUBBLE_RADIUS) {
      y = BUBBLE_RADIUS;
      break;
    }

    points.push([x, y]);

    // Stop at grid area
    if (y < 200) break;
  }

  p.push();
  p.stroke(255, 255, 255, 100);
  p.strokeWeight(2);
  p.noFill();
  p.beginShape();
  for (const [px, py] of points) {
    p.vertex(px, py);
  }
  p.endShape();
  p.pop();
}