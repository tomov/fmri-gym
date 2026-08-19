export const levelConfigs = [
  {
    // Level 1: The Pantry Maze (Easy)
    level: 1,
    playerStart: { x: 60, y: 340 },
    platforms: [
      { x: 0, y: 380, w: 600, h: 20 }, // floor
      { x: 150, y: 300, w: 150, h: 15 },
      { x: 350, y: 250, w: 150, h: 15 }
    ],
    cheese: [
      { x: 100, y: 350 },
      { x: 220, y: 270 },
      { x: 420, y: 220 }
    ],
    cats: [
      { x: 400, y: 340, patrolPath: [{ x: 350, y: 340 }, { x: 500, y: 340 }], speed: 0.8 }
    ],
    mouseHole: { x: 550, y: 350 }
  },
  {
    // Level 2: Kitchen Climb (Easy)
    level: 2,
    playerStart: { x: 60, y: 340 },
    platforms: [
      { x: 0, y: 380, w: 200, h: 20 },
      { x: 150, y: 320, w: 100, h: 15 },
      { x: 300, y: 260, w: 120, h: 15 },
      { x: 450, y: 200, w: 150, h: 15 },
      { x: 350, y: 140, w: 100, h: 15 }
    ],
    cheese: [
      { x: 120, y: 350 },
      { x: 200, y: 290 },
      { x: 360, y: 230 },
      { x: 520, y: 170 },
      { x: 400, y: 110 }
    ],
    cats: [
      { x: 200, y: 310, patrolPath: [{ x: 150, y: 310 }, { x: 250, y: 310 }], speed: 1.0 }
    ],
    mouseHole: { x: 550, y: 170 }
  },
  {
    // Level 3: Dining Room (Medium)
    // Redesigned: Clearer safe zones, cats patrol away from cheese
    level: 3,
    playerStart: { x: 50, y: 340 },
    platforms: [
      { x: 0, y: 380, w: 600, h: 20 }, // Floor
      { x: 150, y: 250, w: 300, h: 15 }, // Main Table
      { x: 60, y: 310, w: 60, h: 15 }, // Left Chair
      { x: 480, y: 310, w: 60, h: 15 }, // Right Chair
      { x: 250, y: 150, w: 100, h: 15 }  // High Shelf
    ],
    cheese: [
      { x: 90, y: 290 }, // On Left Chair
      { x: 510, y: 290 }, // On Right Chair
      { x: 170, y: 230 }, // Table Left Edge (Safe from cat)
      { x: 430, y: 230 }, // Table Right Edge (Safe from cat)
      { x: 300, y: 130 }  // Top Shelf
    ],
    cats: [
      // Table Cat: Patrols middle of table (220-380), leaving edges (150-220 & 380-450) safe
      { x: 300, y: 235, patrolPath: [{ x: 220, y: 235 }, { x: 380, y: 235 }], speed: 1.0 },
      // Floor Cat: Patrols center floor (200-400), avoiding chairs
      { x: 300, y: 340, patrolPath: [{ x: 200, y: 340 }, { x: 400, y: 340 }], speed: 1.0 }
    ],
    mouseHole: { x: 300, y: 110 }
  },
  {
    // Level 4: The Staircase (Medium)
    // Redesigned: Logical climb, larger safe areas on top beam
    level: 4,
    playerStart: { x: 50, y: 340 },
    platforms: [
      { x: 0, y: 380, w: 600, h: 20 }, // Floor
      { x: 120, y: 320, w: 80, h: 15 }, // Step 1
      { x: 240, y: 260, w: 80, h: 15 }, // Step 2
      { x: 360, y: 200, w: 80, h: 15 }, // Step 3
      { x: 150, y: 140, w: 350, h: 15 }  // Top Beam
    ],
    cheese: [
      { x: 50, y: 350 }, // Floor Start
      { x: 550, y: 350 }, // Floor End
      { x: 160, y: 300 }, // Step 1
      { x: 280, y: 240 }, // Step 2
      { x: 400, y: 180 }, // Step 3
      { x: 180, y: 120 }, // Top Beam Left (Safe)
      { x: 470, y: 120 }  // Top Beam Right (Safe)
    ],
    cats: [
      // Floor Cat: Patrols right side only, leaving start safe
      { x: 450, y: 340, patrolPath: [{ x: 400, y: 340 }, { x: 500, y: 340 }], speed: 1.1 },
      // Top Beam Cat: Patrols center (230-420), leaving edges safe
      { x: 325, y: 125, patrolPath: [{ x: 230, y: 125 }, { x: 420, y: 125 }], speed: 1.1 }
    ],
    mouseHole: { x: 325, y: 100 }
  },
  {
    // Level 5: Grand Feast (Hard)
    // Dense platforming, multiple threats
    level: 5,
    playerStart: { x: 300, y: 340 },
    platforms: [
      { x: 0, y: 380, w: 100, h: 20 },
      { x: 150, y: 380, w: 100, h: 20 },
      { x: 300, y: 380, w: 100, h: 20 },
      { x: 450, y: 380, w: 150, h: 20 },
      { x: 50, y: 330, w: 80, h: 15 },
      { x: 180, y: 310, w: 90, h: 15 },
      { x: 320, y: 330, w: 80, h: 15 },
      { x: 450, y: 310, w: 90, h: 15 },
      { x: 100, y: 270, w: 80, h: 15 },
      { x: 230, y: 250, w: 90, h: 15 },
      { x: 370, y: 270, w: 80, h: 15 },
      { x: 480, y: 240, w: 100, h: 15 },
      { x: 50, y: 210, w: 100, h: 15 },
      { x: 200, y: 190, w: 80, h: 15 },
      { x: 330, y: 210, w: 90, h: 15 },
      { x: 470, y: 170, w: 100, h: 15 },
      { x: 150, y: 130, w: 100, h: 15 },
      { x: 300, y: 150, w: 80, h: 15 },
      { x: 430, y: 100, w: 120, h: 15 }
    ],
    cheese: [
      { x: 50, y: 350 },
      { x: 200, y: 350 },
      { x: 350, y: 350 },
      { x: 90, y: 300 },
      { x: 225, y: 280 },
      { x: 360, y: 300 },
      { x: 490, y: 280 },
      { x: 140, y: 240 },
      { x: 275, y: 220 },
      { x: 410, y: 240 },
      { x: 100, y: 180 },
      { x: 490, y: 70 }
    ],
    cats: [
      { x: 90, y: 320, patrolPath: [{ x: 50, y: 320 }, { x: 130, y: 320 }], speed: 1.5 },
      { x: 360, y: 320, patrolPath: [{ x: 320, y: 320 }, { x: 400, y: 320 }], speed: 1.6 },
      { x: 140, y: 260, patrolPath: [{ x: 100, y: 260 }, { x: 180, y: 260 }], speed: 1.7 },
      { x: 410, y: 260, patrolPath: [{ x: 370, y: 260 }, { x: 450, y: 260 }], speed: 1.8 }
    ],
    mouseHole: { x: 520, y: 140 }
  },
  {
    // Level 6: The Clock Tower (Hard)
    // Vertical challenge, tight jumps
    level: 6,
    playerStart: { x: 300, y: 350 },
    platforms: [
      { x: 200, y: 380, w: 200, h: 20 }, // Start Base
      { x: 50, y: 300, w: 120, h: 15 },
      { x: 430, y: 300, w: 120, h: 15 },
      { x: 250, y: 230, w: 100, h: 15 }, // Center small
      { x: 80, y: 160, w: 150, h: 15 },
      { x: 370, y: 160, w: 150, h: 15 },
      { x: 250, y: 90, w: 100, h: 15 }   // Top peak
    ],
    cheese: [
      { x: 60, y: 270 },
      { x: 540, y: 270 },
      { x: 300, y: 200 },
      { x: 90, y: 130 },
      { x: 510, y: 130 },
      { x: 300, y: 60 }
    ],
    cats: [
      { x: 110, y: 285, patrolPath: [{ x: 60, y: 285 }, { x: 160, y: 285 }], speed: 1.6 },
      { x: 490, y: 285, patrolPath: [{ x: 440, y: 285 }, { x: 540, y: 285 }], speed: 1.6 },
      { x: 300, y: 145, patrolPath: [{ x: 100, y: 145 }, { x: 500, y: 145 }], speed: 2.0 } // Fast sweeper on top
    ],
    mouseHole: { x: 300, y: 330 }
  }
];