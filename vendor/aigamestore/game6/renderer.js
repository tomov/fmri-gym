// renderer.js - Rendering functions

import { 
  gameState, 
  GAME_PHASES,
  CANVAS_WIDTH,
  CANVAS_HEIGHT
} from './globals.js';
import { worldToScreen, isOnScreen } from './utils.js';

export function renderGame(p) {
  // Dark atmospheric background
  p.background(20, 15, 25);
  
  // Render atmosphere gradient
  for (let i = 0; i < CANVAS_HEIGHT; i += 10) {
    const alpha = p.map(i, 0, CANVAS_HEIGHT, 30, 60);
    p.fill(30, 25, 35, alpha);
    p.noStroke();
    p.rect(0, i, CANVAS_WIDTH, 10);
  }
  
  // Render entities
  gameState.entities.forEach(entity => {
    if (!entity.active) return;
    
    const screenPos = worldToScreen(entity.x, entity.y);
    
    if (isOnScreen(entity.x, entity.width)) {
      p.push();
      p.translate(screenPos.x - entity.x, 0);
      entity.render(p);
      p.pop();
    }
  });
  
  // UI
  renderUI(p);
}

export function renderUI(p) {
  p.push();
  p.fill(200, 200, 220);
  p.textSize(16);
  p.textAlign(p.LEFT, p.TOP);
  p.text(`Score: ${gameState.score}`, 10, 10);
  p.text(`Deaths: ${gameState.deaths}`, 10, 30);
  p.text(`Checkpoint: ${gameState.currentCheckpoint + 1}/5`, 10, 50);
  
  // Interaction hint
  if (gameState.player && gameState.player.interacting) {
    p.textAlign(p.CENTER, p.TOP);
    p.fill(150, 200, 255);
    p.text("[Z] Interacting", CANVAS_WIDTH / 2, 10);
  }
  
  p.pop();
}

export function renderStartScreen(p) {
  p.background(15, 10, 20);
  
  p.push();
  
  // Replaced title with "press enter to begin"
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(36); // Slightly smaller than old title, but prominent
  p.fill(150, 140, 180);
  p.text("press enter to begin", CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2);
  
  p.pop();
}

export function renderGameOverScreen(p) {
  p.background(15, 10, 20);
  
  p.push();
  p.textAlign(p.CENTER, p.CENTER);
  
  if (gameState.gamePhase === GAME_PHASES.GAME_OVER_WIN) {
    // Victory screen
    p.textSize(48);
    p.fill(100, 255, 150);
    p.text("JOURNEY COMPLETE", CANVAS_WIDTH / 2, 120);
    
    p.textSize(20);
    p.fill(180, 180, 200);
    p.text("You found your way through the shadows.", CANVAS_WIDTH / 2, 180);
  } else {
    // Lose screen (shouldn't normally reach this, but included for completeness)
    p.textSize(48);
    p.fill(255, 100, 100);
    p.text("GAME OVER", CANVAS_WIDTH / 2, 120);
  }
  
  // Stats
  p.textSize(18);
  p.fill(200, 200, 220);
  p.text(`Final Score: ${gameState.score}`, CANVAS_WIDTH / 2, 220);
  p.text(`Deaths: ${gameState.deaths}`, CANVAS_WIDTH / 2, 250);
  p.text(`Checkpoints Reached: ${gameState.currentCheckpoint + 1}/5`, CANVAS_WIDTH / 2, 280);
  
  // Restart prompt
  const pulseAlpha = 150 + Math.sin(p.frameCount * 0.1) * 100;
  p.textSize(20);
  p.fill(100, 200, 255, pulseAlpha);
  p.text("PRESS R TO RESTART", CANVAS_WIDTH / 2, 340);
  
  p.pop();
}