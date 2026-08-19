// collision.js - Collision detection and handling

import { gameState, BUBBLE_RADIUS, LOSE_LINE_Y } from './globals.js';
import {
  findNearestGridPosition,
  getGridPosition,
  isPositionOccupied,
  getConnectedBubbles,
  findDetachedBubbles
} from './grid.js';
import { createBubble } from './bubble.js';

export function checkProjectileCollision(projectile, p) {
  if (!projectile.active) return false;

  // Check collision with bubbles in grid
  for (const bubble of gameState.bubbleGrid) {
    if (bubble.popping || bubble.falling) continue;

    const dist = p.dist(projectile.x, projectile.y, bubble.x, bubble.y);
    if (dist < BUBBLE_RADIUS * 2) {
      return handleCollision(projectile, bubble, p);
    }
  }

  // Check if reached top
  if (projectile.y <= BUBBLE_RADIUS) {
    return handleTopCollision(projectile, p);
  }

  return false;
}

function handleCollision(projectile, hitBubble, p) {
  projectile.active = false;

  // Find nearest empty grid position
  const { row, col } = findNearestGridPosition(projectile.x, projectile.y);

  // Try to place at nearest position
  let placed = false;
  const positions = [
    [row, col],
    [row - 1, col],
    [row + 1, col],
    [row, col - 1],
    [row, col + 1],
    [row - 1, col - 1],
    [row - 1, col + 1],
    [row + 1, col - 1],
    [row + 1, col + 1]
  ];

  for (const [r, c] of positions) {
    if (r >= 0 && c >= 0 && !isPositionOccupied(gameState.bubbleGrid, r, c)) {
      const pos = getGridPosition(r, c);
      const newBubble = createBubble(pos.x, pos.y, projectile.colorIndex, r, c);
      gameState.bubbleGrid.push(newBubble);
      placed = true;

      // Check for matches
      processMatches(newBubble, p);
      break;
    }
  }

  return true;
}

function handleTopCollision(projectile, p) {
  projectile.active = false;

  const { row, col } = findNearestGridPosition(projectile.x, BUBBLE_RADIUS);
  if (!isPositionOccupied(gameState.bubbleGrid, row, col)) {
    const pos = getGridPosition(row, col);
    const newBubble = createBubble(pos.x, pos.y, projectile.colorIndex, row, col);
    gameState.bubbleGrid.push(newBubble);
    processMatches(newBubble, p);
  }

  return true;
}

function processMatches(bubble, p) {
  const connected = getConnectedBubbles(gameState.bubbleGrid, bubble);

  if (connected.length >= 3) {
    // Pop matched bubbles
    for (const b of connected) {
      b.startPop();
      gameState.score += 10;
    }

    // Log game info
    p.logs.game_info.push({
      data: { event: 'bubbles_popped', count: connected.length, score: gameState.score },
      framecount: p.frameCount,
      timestamp: Date.now()
    });

    // Check for detached bubbles
    setTimeout(() => {
      const detached = findDetachedBubbles(gameState.bubbleGrid);
      if (detached.length > 0) {
        for (const b of detached) {
          b.startFall();
          gameState.score += 20;
        }

        p.logs.game_info.push({
          data: { event: 'bubbles_detached', count: detached.length, score: gameState.score },
          framecount: p.frameCount,
          timestamp: Date.now()
        });
      }
    }, 100);
  }
}

export function checkLoseCondition() {
  for (const bubble of gameState.bubbleGrid) {
    if (!bubble.popping && !bubble.falling && bubble.y + BUBBLE_RADIUS > LOSE_LINE_Y) {
      return true;
    }
  }
  return false;
}

export function checkWinCondition() {
  return gameState.bubbleGrid.every(b => b.popping || b.falling);
}