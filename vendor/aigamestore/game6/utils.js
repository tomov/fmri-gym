// utils.js - Utility functions

import { CANVAS_WIDTH, CANVAS_HEIGHT, gameState } from './globals.js';

export function worldToScreen(worldX, worldY) {
  return {
    x: worldX - gameState.cameraOffset,
    y: worldY
  };
}

export function isOnScreen(worldX, width) {
  const screenX = worldX - gameState.cameraOffset;
  return screenX + width > 0 && screenX < CANVAS_WIDTH;
}

export function updateCamera(targetX) {
  const centerX = CANVAS_WIDTH / 2;
  const desiredOffset = targetX - centerX;
  
  // Clamp camera
  const maxOffset = gameState.worldWidth - CANVAS_WIDTH;
  gameState.cameraOffset = Math.max(0, Math.min(maxOffset, desiredOffset));
}

export function checkCollisionAABB(a, b) {
  return a.x < b.x + b.width &&
         a.x + a.width > b.x &&
         a.y < b.y + b.height &&
         a.y + a.height > b.y;
}