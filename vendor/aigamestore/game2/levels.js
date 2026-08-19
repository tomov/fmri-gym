import { Pig, StructureBlock } from './entities.js';
import { BIRD_TYPES } from './globals.js';

export function getLevelData(levelNumber) {
  const levels = {
    // EASY LEVELS (1-3) - Now with 8-10 birds
    1: {
      birds: [
        BIRD_TYPES.RED, BIRD_TYPES.RED, BIRD_TYPES.BLUE, 
        BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.RED,
        BIRD_TYPES.BLUE, BIRD_TYPES.GREEN
      ],
      pigs: [
        { x: 350, y: 300, large: false },
        { x: 380, y: 300, large: false }
      ],
      structures: [
        // Simple ground platform
        { x: 365, y: 330, w: 50, h: 10, mat: 'WOOD' },
        { x: 340, y: 320, w: 10, h: 20, mat: 'WOOD' },
        { x: 390, y: 320, w: 10, h: 20, mat: 'WOOD' },
        { x: 365, y: 305, w: 60, h: 10, mat: 'WOOD' }
      ]
    },
    2: {
      birds: [
        BIRD_TYPES.RED, BIRD_TYPES.BLUE, BIRD_TYPES.RED,
        BIRD_TYPES.YELLOW, BIRD_TYPES.RED, BIRD_TYPES.BLUE,
        BIRD_TYPES.BLACK, BIRD_TYPES.RED, BIRD_TYPES.YELLOW
      ],
      pigs: [
        { x: 380, y: 300, large: false },
        { x: 420, y: 300, large: false },
        { x: 400, y: 270, large: false }
      ],
      structures: [
        // Simple tower
        { x: 370, y: 335, w: 15, h: 30, mat: 'WOOD' },
        { x: 400, y: 335, w: 15, h: 30, mat: 'WOOD' },
        { x: 430, y: 335, w: 15, h: 30, mat: 'WOOD' },
        // Platform 1
        { x: 400, y: 310, w: 70, h: 10, mat: 'WOOD' },
        // Walls
        { x: 380, y: 290, w: 15, h: 30, mat: 'WOOD' },
        { x: 420, y: 290, w: 15, h: 30, mat: 'WOOD' },
        // Top
        { x: 400, y: 270, w: 50, h: 10, mat: 'WOOD' }
      ]
    },
    3: {
      birds: [
        BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.BLUE,
        BIRD_TYPES.RED, BIRD_TYPES.WHITE, BIRD_TYPES.BLUE,
        BIRD_TYPES.YELLOW, BIRD_TYPES.RED, BIRD_TYPES.RED,
        BIRD_TYPES.GREEN
      ],
      pigs: [
        { x: 360, y: 310, large: false },
        { x: 400, y: 310, large: false },
        { x: 380, y: 280, large: false }
      ],
      structures: [
        // Base
        { x: 360, y: 335, w: 15, h: 30, mat: 'WOOD' },
        { x: 380, y: 335, w: 15, h: 30, mat: 'WOOD' },
        { x: 400, y: 335, w: 15, h: 30, mat: 'WOOD' },
        { x: 420, y: 335, w: 15, h: 30, mat: 'WOOD' },
        // Floor 1
        { x: 390, y: 315, w: 70, h: 8, mat: 'WOOD' },
        { x: 370, y: 295, w: 12, h: 30, mat: 'WOOD' },
        { x: 410, y: 295, w: 12, h: 30, mat: 'WOOD' },
        // Floor 2
        { x: 390, y: 275, w: 50, h: 8, mat: 'WOOD' }
      ]
    },
    
    // MEDIUM LEVELS (4-6) - Now with 12-15 birds
    4: {
      birds: [
        BIRD_TYPES.RED, BIRD_TYPES.BLUE, BIRD_TYPES.YELLOW, BIRD_TYPES.RED,
        BIRD_TYPES.BLUE, BIRD_TYPES.BLACK, BIRD_TYPES.YELLOW, BIRD_TYPES.RED,
        BIRD_TYPES.BLUE, BIRD_TYPES.GREEN, BIRD_TYPES.RED, BIRD_TYPES.YELLOW
      ],
      pigs: [
        { x: 380, y: 300, large: false },
        { x: 420, y: 300, large: false },
        { x: 400, y: 270, large: true }
      ],
      structures: [
        // Base with mixed materials
        { x: 370, y: 335, w: 15, h: 30, mat: 'STONE' },
        { x: 400, y: 335, w: 15, h: 30, mat: 'WOOD' },
        { x: 430, y: 335, w: 15, h: 30, mat: 'STONE' },
        // Platform 1
        { x: 400, y: 310, w: 70, h: 10, mat: 'WOOD' },
        // Walls
        { x: 375, y: 290, w: 15, h: 30, mat: 'WOOD' },
        { x: 425, y: 290, w: 15, h: 30, mat: 'WOOD' },
        // Top
        { x: 400, y: 270, w: 60, h: 10, mat: 'STONE' }
      ]
    },
    5: {
      birds: [
        BIRD_TYPES.YELLOW, BIRD_TYPES.BLUE, BIRD_TYPES.RED, BIRD_TYPES.BLUE,
        BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.WHITE, BIRD_TYPES.BLUE,
        BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.RED, BIRD_TYPES.BLUE,
        BIRD_TYPES.BLACK
      ],
      pigs: [
        { x: 350, y: 310, large: false },
        { x: 400, y: 290, large: false },
        { x: 450, y: 310, large: false },
        { x: 400, y: 260, large: true }
      ],
      structures: [
        // Left tower
        { x: 340, y: 335, w: 15, h: 30, mat: 'WOOD' },
        { x: 360, y: 335, w: 15, h: 30, mat: 'STONE' },
        { x: 350, y: 310, w: 30, h: 10, mat: 'WOOD' },
        // Right tower
        { x: 440, y: 335, w: 15, h: 30, mat: 'STONE' },
        { x: 460, y: 335, w: 15, h: 30, mat: 'WOOD' },
        { x: 450, y: 310, w: 30, h: 10, mat: 'WOOD' },
        // Center bridge
        { x: 400, y: 295, w: 120, h: 8, mat: 'WOOD' },
        { x: 385, y: 275, w: 15, h: 30, mat: 'STONE' },
        { x: 415, y: 275, w: 15, h: 30, mat: 'STONE' },
        { x: 400, y: 255, w: 50, h: 10, mat: 'WOOD' }
      ]
    },
    6: {
      birds: [
        BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.BLUE, BIRD_TYPES.YELLOW, BIRD_TYPES.RED,
        BIRD_TYPES.BLUE, BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.GREEN, BIRD_TYPES.BLUE,
        BIRD_TYPES.BLACK, BIRD_TYPES.YELLOW, BIRD_TYPES.RED, BIRD_TYPES.WHITE, BIRD_TYPES.RED
      ],
      pigs: [
        { x: 370, y: 310, large: false },
        { x: 430, y: 310, large: false },
        { x: 400, y: 280, large: true },
        { x: 380, y: 250, large: false },
        { x: 420, y: 250, large: false }
      ],
      structures: [
        // Base
        { x: 360, y: 335, w: 15, h: 30, mat: 'STONE' },
        { x: 385, y: 335, w: 15, h: 30, mat: 'WOOD' },
        { x: 415, y: 335, w: 15, h: 30, mat: 'WOOD' },
        { x: 440, y: 335, w: 15, h: 30, mat: 'STONE' },
        // Floor 1
        { x: 400, y: 315, w: 90, h: 8, mat: 'WOOD' },
        { x: 365, y: 295, w: 15, h: 30, mat: 'STONE' },
        { x: 400, y: 295, w: 15, h: 30, mat: 'WOOD' },
        { x: 435, y: 295, w: 15, h: 30, mat: 'STONE' },
        // Floor 2
        { x: 400, y: 275, w: 80, h: 8, mat: 'STONE' },
        { x: 375, y: 260, w: 15, h: 20, mat: 'WOOD' },
        { x: 405, y: 260, w: 15, h: 20, mat: 'WOOD' },
        { x: 425, y: 260, w: 15, h: 20, mat: 'WOOD' },
        // Floor 3
        { x: 400, y: 245, w: 60, h: 8, mat: 'WOOD' }
      ]
    },
    
    // HARD LEVELS (7-9) - Now with 15-20 birds
    7: {
      birds: [
        BIRD_TYPES.YELLOW, BIRD_TYPES.BLUE, BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.BLUE,
        BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.BLACK, BIRD_TYPES.BLUE, BIRD_TYPES.RED,
        BIRD_TYPES.YELLOW, BIRD_TYPES.GREEN, BIRD_TYPES.BLUE, BIRD_TYPES.RED, BIRD_TYPES.WHITE
      ],
      pigs: [
        { x: 380, y: 300, large: false },
        { x: 440, y: 300, large: false },
        { x: 410, y: 270, large: true },
        { x: 410, y: 240, large: true }
      ],
      structures: [
        // Base fortress
        { x: 365, y: 330, w: 20, h: 40, mat: 'STONE' },
        { x: 395, y: 330, w: 20, h: 40, mat: 'STONE' },
        { x: 425, y: 330, w: 20, h: 40, mat: 'STONE' },
        { x: 455, y: 330, w: 20, h: 40, mat: 'STONE' },
        // Floor 1
        { x: 410, y: 305, w: 100, h: 10, mat: 'STONE' },
        { x: 380, y: 285, w: 15, h: 30, mat: 'WOOD' },
        { x: 440, y: 285, w: 15, h: 30, mat: 'WOOD' },
        // Floor 2
        { x: 410, y: 265, w: 80, h: 10, mat: 'WOOD' },
        { x: 390, y: 250, w: 15, h: 20, mat: 'STONE' },
        { x: 430, y: 250, w: 15, h: 20, mat: 'STONE' },
        // Roof
        { x: 410, y: 235, w: 70, h: 10, mat: 'STONE' }
      ]
    },
    8: {
      birds: [
        BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.BLUE, BIRD_TYPES.BLUE, BIRD_TYPES.YELLOW, BIRD_TYPES.RED,
        BIRD_TYPES.BLUE, BIRD_TYPES.BLACK, BIRD_TYPES.YELLOW, BIRD_TYPES.RED, BIRD_TYPES.BLUE, BIRD_TYPES.YELLOW,
        BIRD_TYPES.RED, BIRD_TYPES.GREEN, BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.WHITE, BIRD_TYPES.BLUE
      ],
      pigs: [
        { x: 320, y: 310, large: false },
        { x: 350, y: 280, large: false },
        { x: 450, y: 310, large: false },
        { x: 480, y: 280, large: true },
        { x: 400, y: 230, large: true }
      ],
      structures: [
        // Left tower
        { x: 320, y: 330, w: 15, h: 40, mat: 'WOOD' },
        { x: 345, y: 330, w: 15, h: 40, mat: 'STONE' },
        { x: 332, y: 300, w: 40, h: 10, mat: 'WOOD' },
        { x: 332, y: 280, w: 15, h: 30, mat: 'STONE' },
        // Right tower
        { x: 455, y: 330, w: 15, h: 40, mat: 'STONE' },
        { x: 480, y: 330, w: 15, h: 40, mat: 'STONE' },
        { x: 467, y: 300, w: 40, h: 10, mat: 'STONE' },
        { x: 467, y: 280, w: 15, h: 30, mat: 'STONE' },
        // Middle bridge
        { x: 400, y: 270, w: 80, h: 10, mat: 'STONE' },
        { x: 385, y: 250, w: 15, h: 30, mat: 'STONE' },
        { x: 415, y: 250, w: 15, h: 30, mat: 'STONE' },
        { x: 400, y: 230, w: 50, h: 10, mat: 'STONE' }
      ]
    },
    9: {
      birds: [
        BIRD_TYPES.YELLOW, BIRD_TYPES.BLUE, BIRD_TYPES.YELLOW, BIRD_TYPES.RED, BIRD_TYPES.RED, BIRD_TYPES.BLUE, BIRD_TYPES.YELLOW,
        BIRD_TYPES.RED, BIRD_TYPES.BLACK, BIRD_TYPES.YELLOW, BIRD_TYPES.RED, BIRD_TYPES.BLUE, BIRD_TYPES.GREEN, BIRD_TYPES.YELLOW,
        BIRD_TYPES.BLUE, BIRD_TYPES.RED, BIRD_TYPES.YELLOW, BIRD_TYPES.WHITE, BIRD_TYPES.BLUE, BIRD_TYPES.BLACK
      ],
      pigs: [
        { x: 360, y: 320, large: false },
        { x: 390, y: 320, large: false },
        { x: 420, y: 320, large: false },
        { x: 450, y: 320, large: false },
        { x: 380, y: 280, large: true },
        { x: 430, y: 280, large: true },
        { x: 405, y: 240, large: true },
        { x: 405, y: 200, large: true }
      ],
      structures: [
        // Massive base
        { x: 350, y: 335, w: 20, h: 30, mat: 'STONE' },
        { x: 380, y: 335, w: 20, h: 30, mat: 'STONE' },
        { x: 410, y: 335, w: 20, h: 30, mat: 'STONE' },
        { x: 440, y: 335, w: 20, h: 30, mat: 'STONE' },
        { x: 470, y: 335, w: 20, h: 30, mat: 'STONE' },
        // Level 1
        { x: 405, y: 312, w: 130, h: 8, mat: 'STONE' },
        { x: 365, y: 295, w: 15, h: 25, mat: 'WOOD' },
        { x: 395, y: 295, w: 15, h: 25, mat: 'STONE' },
        { x: 425, y: 295, w: 15, h: 25, mat: 'STONE' },
        { x: 455, y: 295, w: 15, h: 25, mat: 'WOOD' },
        // Level 2
        { x: 405, y: 275, w: 110, h: 8, mat: 'STONE' },
        { x: 375, y: 260, w: 15, h: 20, mat: 'STONE' },
        { x: 405, y: 260, w: 15, h: 20, mat: 'STONE' },
        { x: 435, y: 260, w: 15, h: 20, mat: 'STONE' },
        // Level 3
        { x: 405, y: 245, w: 90, h: 8, mat: 'STONE' },
        { x: 385, y: 230, w: 15, h: 20, mat: 'STONE' },
        { x: 425, y: 230, w: 15, h: 20, mat: 'STONE' },
        // Level 4
        { x: 405, y: 215, w: 70, h: 8, mat: 'STONE' },
        { x: 390, y: 200, w: 12, h: 20, mat: 'STONE' },
        { x: 420, y: 200, w: 12, h: 20, mat: 'STONE' },
        // Top
        { x: 405, y: 185, w: 50, h: 8, mat: 'STONE' }
      ]
    }
  };

  return levels[levelNumber];
}

export function createLevel(levelNumber) {
  const levelData = getLevelData(levelNumber);
  if (!levelData) return null;

  const pigs = levelData.pigs.map(p => new Pig(p.x, p.y, p.large));
  const structures = levelData.structures.map(s => 
    new StructureBlock(s.x, s.y, s.w, s.h, s.mat)
  );

  return {
    birds: [...levelData.birds],
    pigs: pigs,
    structures: structures
  };
}