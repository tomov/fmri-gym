// grid.js - Grid management and bubble placement

import { BUBBLE_RADIUS, BUBBLE_DIAMETER, CANVAS_WIDTH, BUBBLE_COLORS } from './globals.js';
import { createBubble } from './bubble.js';

export function getGridPosition(row, col) {
  const offsetX = (row % 2 === 1) ? BUBBLE_RADIUS : 0;
  const x = col * BUBBLE_DIAMETER + BUBBLE_RADIUS + offsetX + 60;
  const y = row * (BUBBLE_DIAMETER * 0.866) + BUBBLE_RADIUS + 30;
  return { x, y };
}

export function findNearestGridPosition(x, y) {
  let minDist = Infinity;
  let bestRow = 0;
  let bestCol = 0;

  for (let row = 0; row < 12; row++) {
    const colCount = (row % 2 === 0) ? 12 : 11;
    for (let col = 0; col < colCount; col++) {
      const pos = getGridPosition(row, col);
      const dist = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
      if (dist < minDist) {
        minDist = dist;
        bestRow = row;
        bestCol = col;
      }
    }
  }

  return { row: bestRow, col: bestCol };
}

export function createLevelLayout(level) {
  const layouts = [
    // Level 1: Easy - sparse, simple clusters
    [
      [-1, -1, -1, 0, 0, 1, 1, 0, 0, -1, -1, -1],
      [-1, -1, -1, 0, 1, 1, 0, 0, -1, -1, -1],
      [-1, -1, 2, 2, 0, 0, 1, 1, -1, -1, -1, -1],
      [-1, -1, 2, 0, 0, 1, 1, -1, -1, -1, -1],
      [-1, -1, 3, 3, 3, 0, 0, -1, -1, -1, -1, -1]
    ],
    // Level 2: Easy - similar to level 1, different pattern
    [
      [-1, -1, -1, 1, 1, 0, 0, 1, 1, -1, -1, -1],
      [-1, -1, -1, 1, 0, 0, 1, 1, -1, -1, -1],
      [-1, -1, 0, 0, 2, 2, 1, 1, 0, 0, -1, -1],
      [-1, -1, 0, 2, 2, 1, 1, 0, 0, -1, -1],
      [-1, -1, 2, 2, 1, 1, 3, 3, -1, -1, -1, -1]
    ],
    // Level 3: Easy - simple clusters, easy matching
    [
      [-1, -1, -1, 0, 0, 2, 2, 1, 1, -1, -1, -1],
      [-1, -1, -1, 0, 2, 2, 1, 1, -1, -1, -1],
      [-1, -1, 1, 1, 3, 3, 0, 0, 2, 2, -1, -1],
      [-1, -1, 1, 3, 3, 0, 0, 2, 2, -1, -1],
      [-1, -1, 3, 3, 0, 0, 1, 1, -1, -1, -1, -1]
    ],
    // Level 4: Medium - denser, more colors
    [
      [-1, -1, 0, 1, 2, 3, 0, 1, 2, 3, -1, -1],
      [-1, -1, 1, 0, 3, 2, 1, 0, 3, -1, -1],
      [-1, -1, 2, 3, 0, 1, 2, 3, 0, 1, -1, -1],
      [-1, -1, 0, 1, 2, 3, 0, 1, 2, -1, -1],
      [-1, -1, 1, 0, 4, 4, 1, 0, 3, 2, -1, -1],
      [-1, -1, 2, 4, 4, 1, 0, 3, 2, -1, -1]
    ],
    // Level 5: Medium - mixed patterns, more strategic
    [
      [-1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, -1],
      [-1, 1, 2, 3, 4, 0, 1, 2, 3, 4, -1],
      [-1, 2, 2, 0, 0, 1, 1, 3, 3, 4, 4, -1],
      [-1, 2, 0, 0, 1, 1, 3, 3, 4, 4, -1],
      [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1],
      [-1, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1],
      [-1, -1, 1, 1, 2, 2, 3, 3, 4, -1, -1, -1]
    ],
    // Level 6: Medium - complex patterns, all basic colors
    [
      [-1, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, -1],
      [-1, 1, 0, 3, 2, 5, 4, 1, 0, 3, -1],
      [-1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, -1],
      [-1, 3, 4, 5, 0, 1, 2, 3, 4, 5, -1],
      [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1],
      [-1, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1],
      [-1, -1, 5, 5, 0, 0, 1, 1, 2, -1, -1, -1]
    ],
    // Level 7: Hard - very dense, complex patterns
    [
      [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
      [5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1],
      [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
      [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0],
      [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
      [5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1],
      [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
      [0, 0, 0, 5, 5, 5, 1, 1, 1, 2, 2]
    ],
    // Level 8: Hard - challenging layout, many colors
    [
      [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
      [1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
      [2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1],
      [3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1],
      [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
      [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0],
      [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
      [5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1],
      [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    ],
    // Level 9: Hard - ultimate challenge, full grid
    [
      [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
      [5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1],
      [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
      [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0],
      [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
      [5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1],
      [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
      [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0],
      [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
      [1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    ]
  ];

  const layout = layouts[level - 1];
  const bubbles = [];

  for (let row = 0; row < layout.length; row++) {
    for (let col = 0; col < layout[row].length; col++) {
      const colorIndex = layout[row][col];
      if (colorIndex >= 0) {
        const pos = getGridPosition(row, col);
        const bubble = createBubble(pos.x, pos.y, colorIndex, row, col);
        bubbles.push(bubble);
      }
    }
  }

  return bubbles;
}

export function getAvailableColors(bubbleGrid) {
  const colors = new Set();
  for (const bubble of bubbleGrid) {
    if (!bubble.popping && !bubble.falling) {
      colors.add(bubble.colorIndex);
    }
  }
  return Array.from(colors);
}

export function getRandomColorFromGrid(bubbleGrid, p) {
  const availableColors = getAvailableColors(bubbleGrid);
  if (availableColors.length === 0) {
    return Math.floor(p.random(BUBBLE_COLORS.length));
  }
  return availableColors[Math.floor(p.random(availableColors.length))];
}

export function isPositionOccupied(bubbleGrid, row, col) {
  return bubbleGrid.some(b => b.gridRow === row && b.gridCol === col && !b.popping && !b.falling);
}

export function getConnectedBubbles(bubbleGrid, startBubble) {
  const visited = new Set();
  const connected = [];
  const queue = [startBubble];

  while (queue.length > 0) {
    const bubble = queue.shift();
    const key = `${bubble.gridRow},${bubble.gridCol}`;

    if (visited.has(key)) continue;
    visited.add(key);

    if (bubble.colorIndex === startBubble.colorIndex && !bubble.popping && !bubble.falling) {
      connected.push(bubble);
      const neighbors = getNeighbors(bubbleGrid, bubble.gridRow, bubble.gridCol);
      queue.push(...neighbors);
    }
  }

  return connected;
}

export function getNeighbors(bubbleGrid, row, col) {
  const neighbors = [];
  const isEvenRow = row % 2 === 0;

  const offsets = isEvenRow
    ? [[-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]]
    : [[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]];

  for (const [dr, dc] of offsets) {
    const newRow = row + dr;
    const newCol = col + dc;
    const bubble = bubbleGrid.find(b => b.gridRow === newRow && b.gridCol === newCol && !b.popping && !b.falling);
    if (bubble) {
      neighbors.push(bubble);
    }
  }

  return neighbors;
}

export function findDetachedBubbles(bubbleGrid) {
  const attached = new Set();
  const queue = [];

  // Start with top row
  for (const bubble of bubbleGrid) {
    if (bubble.gridRow === 0 && !bubble.popping && !bubble.falling) {
      queue.push(bubble);
      attached.add(`${bubble.gridRow},${bubble.gridCol}`);
    }
  }

  // BFS to find all attached bubbles
  while (queue.length > 0) {
    const bubble = queue.shift();
    const neighbors = getNeighbors(bubbleGrid, bubble.gridRow, bubble.gridCol);

    for (const neighbor of neighbors) {
      const key = `${neighbor.gridRow},${neighbor.gridCol}`;
      if (!attached.has(key)) {
        attached.add(key);
        queue.push(neighbor);
      }
    }
  }

  // Find detached bubbles
  const detached = [];
  for (const bubble of bubbleGrid) {
    const key = `${bubble.gridRow},${bubble.gridCol}`;
    if (!attached.has(key) && !bubble.popping && !bubble.falling) {
      detached.push(bubble);
    }
  }

  return detached;
}