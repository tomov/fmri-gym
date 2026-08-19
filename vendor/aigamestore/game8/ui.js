// ui.js - UI rendering

import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT, LOSE_LINE_Y } from './globals.js';
import { calculateShotBonus } from './levels.js';

export function drawUI(p) {
  p.push();
  p.textAlign(p.LEFT, p.TOP);
  p.textSize(20);
  p.fill(255);
  p.noStroke();

  // Score
  p.text(`SCORE: ${gameState.score}`, 10, 10);

  // Shots
  p.textAlign(p.CENTER, p.TOP);
  p.text(`SHOTS: ${gameState.shotsRemaining}`, CANVAS_WIDTH / 2, 10);

  // Level
  p.textAlign(p.RIGHT, p.TOP);
  p.text(`LEVEL: ${gameState.currentLevel}`, CANVAS_WIDTH - 10, 10);

  // Lose line indicator
  p.stroke(255, 0, 0, 100);
  p.strokeWeight(2);
  p.line(0, LOSE_LINE_Y, CANVAS_WIDTH, LOSE_LINE_Y);

  p.pop();
}

export function drawStartScreen(p) {
  p.push();
  p.background(20, 30, 50);

  // New main title / start prompt
  p.textAlign(p.CENTER, p.CENTER);
  p.fill(255, 220, 100);
  p.textSize(36); // Make this prominent
  p.noStroke();
  p.text('press enter to begin', CANVAS_WIDTH / 2, 150); // Centered and higher

  // Controls (kept as per request)
  p.fill(255);
  p.textSize(14);
  p.text('ARROW KEYS: Aim', CANVAS_WIDTH / 2, 250); // Adjusted Y position
  p.text('SPACE: Fire Bubble', CANVAS_WIDTH / 2, 270); // Adjusted Y position
  p.text('Z: Swap Bubbles', CANVAS_WIDTH / 2, 290); // Adjusted Y position

  p.pop();
}

export function drawGameOverScreen(p, isWin) {
  p.push();
  p.fill(0, 0, 0, 150);
  p.noStroke();
  p.rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  p.textAlign(p.CENTER, p.CENTER);
  p.fill(isWin ? [100, 255, 100] : [255, 100, 100]);
  p.textSize(48);
  p.noStroke();
  p.text(isWin ? 'YOU WIN!' : 'GAME OVER', CANVAS_WIDTH / 2, 120);

  p.fill(255);
  p.textSize(24);
  p.text(`FINAL SCORE: ${gameState.score}`, CANVAS_WIDTH / 2, 200);

  p.fill(255, 255, 100);
  p.textSize(20);
  p.text('PRESS R TO RESTART', CANVAS_WIDTH / 2, 280);

  p.pop();
}

export function drawLevelTransition(p) {
  p.push();
  p.fill(0, 0, 0, 150);
  p.noStroke();
  p.rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

  p.textAlign(p.CENTER, p.CENTER);
  p.fill(100, 255, 100);
  p.textSize(36);
  p.noStroke();
  p.text(`LEVEL ${gameState.currentLevel - 1} COMPLETE!`, CANVAS_WIDTH / 2, 150);

  p.fill(255);
  p.textSize(24);
  const bonus = calculateShotBonus();
  p.text(`Shot Bonus: +${bonus}`, CANVAS_WIDTH / 2, 200);
  p.text(`Score: ${gameState.score}`, CANVAS_WIDTH / 2, 230);

  p.fill(255, 255, 100);
  p.textSize(20);
  p.text(`GET READY FOR LEVEL ${gameState.currentLevel}!`, CANVAS_WIDTH / 2, 280);

  p.pop();
}
