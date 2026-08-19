// bubble.js - Bubble class and related functions

import { BUBBLE_RADIUS, BUBBLE_COLORS } from './globals.js';

export class Bubble {
  constructor(x, y, colorIndex, gridRow = -1, gridCol = -1) {
    this.x = x;
    this.y = y;
    this.colorIndex = colorIndex;
    this.color = BUBBLE_COLORS[colorIndex];
    this.radius = BUBBLE_RADIUS;
    this.gridRow = gridRow;
    this.gridCol = gridCol;
    this.popping = false;
    this.popProgress = 0;
    this.falling = false;
    this.velocityY = 0;
  }

  update() {
    if (this.popping) {
      this.popProgress += 0.1;
      if (this.popProgress >= 1) {
        return true; // Mark for removal
      }
    }
    if (this.falling) {
      this.velocityY += 0.5; // Gravity
      this.y += this.velocityY;
      if (this.y > 500) {
        return true; // Mark for removal
      }
    }
    return false;
  }

  draw(p) {
    p.push();
    if (this.popping) {
      const scale = 1 - this.popProgress;
      const alpha = 255 * (1 - this.popProgress);
      p.fill(...this.color, alpha);
      p.noStroke();
      p.ellipse(this.x, this.y, this.radius * 2 * scale);
    } else {
      p.fill(...this.color);
      p.stroke(255, 255, 255, 150);
      p.strokeWeight(2);
      p.ellipse(this.x, this.y, this.radius * 2);
    }
    p.pop();
  }

  startPop() {
    this.popping = true;
    this.popProgress = 0;
  }

  startFall() {
    this.falling = true;
    this.velocityY = 0;
  }
}

export function createBubble(x, y, colorIndex, gridRow = -1, gridCol = -1) {
  return new Bubble(x, y, colorIndex, gridRow, gridCol);
}