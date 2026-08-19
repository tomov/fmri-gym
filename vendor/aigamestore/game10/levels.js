// levels.js - Level data and management
import { gameState, TILE_SIZE, CANVAS_WIDTH, CANVAS_HEIGHT } from './globals.js';
import { Door, Item, HidingSpot, ExitZone } from './entities.js';
import { Antagonist } from './antagonist.js';
import { Player } from './player.js';

export function createLevel1() {
  // Level 1: Tutorial - Super Easy (No enemies!)
  // Scaled to 640x400 canvas
  
  gameState.walls = [
    // Outer walls
    { x: 0, y: 0, w: CANVAS_WIDTH, h: 16 },
    { x: 0, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: CANVAS_WIDTH - 16, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: 0, y: CANVAS_HEIGHT - 16, w: CANVAS_WIDTH, h: 16 },
    
    // One simple obstacle in the middle
    { x: 304, y: 176, w: 32, h: 48 }
  ];

  gameState.doors = [];

  gameState.items = [
    new Item(160, 250, "objective", "obj1_1", true)
  ];

  gameState.requiredItemIds = ["obj1_1"];

  gameState.hidingSpots = [
    new HidingSpot(80, 70, 32, 24, "closet")
  ];

  gameState.exitZone = new ExitZone(480, 300, 80, 56);

  gameState.player = new Player(64, 64);

  gameState.antagonists = [];
  
  gameState.entities = [gameState.player];
}

export function createLevel2() {
  // Level 2: Introduction to enemies - Easy
  // Scaled to 640x400 canvas
  
  gameState.walls = [
    // Outer walls
    { x: 0, y: 0, w: CANVAS_WIDTH, h: 16 },
    { x: 0, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: CANVAS_WIDTH - 16, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: 0, y: CANVAS_HEIGHT - 16, w: CANVAS_WIDTH, h: 16 },
    
    // Vertical divider
    { x: 304, y: 16, w: 16, h: 160 },
    { x: 304, y: 240, w: 16, h: 144 }
  ];

  gameState.doors = [
    new Door(304, 176, 16, 64, true, "key2_1")
  ];

  gameState.items = [
    new Item(80, 64, "key", "key2_1", false),
    new Item(480, 64, "objective", "obj2_1", true)
  ];

  gameState.requiredItemIds = ["obj2_1"];

  gameState.hidingSpots = [
    new HidingSpot(40, 270, 32, 24, "closet"),
    new HidingSpot(440, 270, 32, 24, "closet")
  ];

  gameState.exitZone = new ExitZone(520, 32, 64, 48);

  gameState.player = new Player(48, 48);

  gameState.antagonists = [
    new Antagonist(480, 300, 0.6, [
      [480, 300],
      [544, 300],
      [544, 250],
      [480, 250]
    ])
  ];
  
  gameState.entities = [gameState.player, ...gameState.antagonists];
}

export function createLevel3() {
  // Level 3: Basic challenge - Medium-Easy
  // Scaled to 640x400 canvas
  
  gameState.walls = [
    // Outer walls
    { x: 0, y: 0, w: CANVAS_WIDTH, h: 16 },
    { x: 0, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: CANVAS_WIDTH - 16, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: 0, y: CANVAS_HEIGHT - 16, w: CANVAS_WIDTH, h: 16 },
    
    // Two vertical dividers
    { x: 216, y: 16, w: 16, h: 160 },
    { x: 216, y: 240, w: 16, h: 144 },
    { x: 432, y: 16, w: 16, h: 160 },
    { x: 432, y: 240, w: 16, h: 144 }
  ];

  gameState.doors = [
    new Door(216, 176, 16, 64, false)
  ];

  gameState.items = [
    new Item(104, 300, "objective", "obj3_1", true),
    new Item(544, 300, "objective", "obj3_2", true)
  ];

  gameState.requiredItemIds = ["obj3_1", "obj3_2"];

  gameState.hidingSpots = [
    new HidingSpot(40, 64, 32, 24, "closet"),
    new HidingSpot(304, 64, 48, 16, "table"),
    new HidingSpot(520, 64, 32, 24, "closet")
  ];

  gameState.exitZone = new ExitZone(288, 32, 64, 48);

  gameState.player = new Player(48, 300);

  gameState.antagonists = [
    new Antagonist(544, 80, 0.6, [
      [520, 80],
      [576, 80],
      [576, 120],
      [520, 120]
    ], 80)
  ];
  
  gameState.entities = [gameState.player, ...gameState.antagonists];
}

export function createLevel4() {
  // Level 4: Simple layout with safer item placement
  // Scaled to 640x400 canvas
  
  gameState.walls = [
    // Outer walls
    { x: 0, y: 0, w: CANVAS_WIDTH, h: 16 },
    { x: 0, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: CANVAS_WIDTH - 16, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: 0, y: CANVAS_HEIGHT - 16, w: CANVAS_WIDTH, h: 16 },
    
    // Central vertical wall
    { x: 304, y: 16, w: 16, h: 120 },
    { x: 304, y: 220, w: 16, h: 164 },
    
    // Horizontal walls
    { x: 16, y: 120, w: 176, h: 16 },
    { x: 432, y: 120, w: 192, h: 16 },
    
    // Small barriers
    { x: 104, y: 240, w: 16, h: 80 },
    { x: 488, y: 240, w: 16, h: 80 }
  ];

  gameState.doors = [
    new Door(304, 136, 16, 84, true, "key4_1")
  ];

  gameState.items = [
    new Item(48, 40, "key", "key4_1", false),
    new Item(560, 48, "objective", "obj4_1", true)
  ];

  gameState.requiredItemIds = ["obj4_1"];

  gameState.hidingSpots = [
    new HidingSpot(40, 280, 32, 24, "closet"),
    new HidingSpot(384, 40, 32, 24, "closet")
  ];

  gameState.exitZone = new ExitZone(520, 320, 64, 48);

  gameState.player = new Player(40, 180);

  gameState.antagonists = [
    new Antagonist(416, 300, 1.0, [
      [416, 300],
      [544, 300],
      [544, 200],
      [416, 200]
    ])
  ];
  
  gameState.entities = [gameState.player, ...gameState.antagonists];
}

export function createLevel5() {
  // Level 5: Two-room layout with SAFE spawn positioning
  // Monster placed FAR from player spawn (top-right vs bottom-left)
  // Scaled to 640x400 canvas
  
  gameState.walls = [
    // Outer walls
    { x: 0, y: 0, w: CANVAS_WIDTH, h: 16 },
    { x: 0, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: CANVAS_WIDTH - 16, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: 0, y: CANVAS_HEIGHT - 16, w: CANVAS_WIDTH, h: 16 },
    
    // Central horizontal wall
    { x: 16, y: 200, w: 272, h: 16 },
    { x: 368, y: 200, w: 256, h: 16 },
    
    // Vertical walls
    { x: 216, y: 16, w: 16, h: 100 },
    { x: 408, y: 280, w: 16, h: 104 },
    
    // Small barriers
    { x: 104, y: 80, w: 16, h: 60 },
    { x: 520, y: 80, w: 16, h: 60 }
  ];

  gameState.doors = [
    new Door(288, 200, 80, 16, false)
  ];

  gameState.items = [
    new Item(48, 48, "objective", "obj5_1", true),
    new Item(560, 340, "objective", "obj5_2", true)
  ];

  gameState.requiredItemIds = ["obj5_1", "obj5_2"];

  gameState.hidingSpots = [
    new HidingSpot(152, 40, 32, 24, "closet"),
    new HidingSpot(328, 320, 48, 16, "table"),
    new HidingSpot(480, 320, 32, 24, "closet")
  ];

  gameState.exitZone = new ExitZone(520, 32, 64, 48);

  // Player spawns BOTTOM-LEFT
  gameState.player = new Player(40, 340);

  // Monster spawns and patrols TOP-RIGHT (far from player!)
  gameState.antagonists = [
    new Antagonist(560, 60, 0.9, [
      [560, 60],
      [560, 120],
      [480, 120],
      [480, 60]
    ])
  ];
  
  gameState.entities = [gameState.player, ...gameState.antagonists];
}

export function createLevel6() {
  // Level 6: Three-room maze with SAFE spawn positioning
  // Monsters positioned FAR from player spawn
  // Scaled to 640x400 canvas
  
  gameState.walls = [
    // Outer walls
    { x: 0, y: 0, w: CANVAS_WIDTH, h: 16 },
    { x: 0, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: CANVAS_WIDTH - 16, y: 0, w: 16, h: CANVAS_HEIGHT },
    { x: 0, y: CANVAS_HEIGHT - 16, w: CANVAS_WIDTH, h: 16 },
    
    // Vertical dividers
    { x: 216, y: 16, w: 16, h: 140 },
    { x: 216, y: 220, w: 16, h: 164 },
    { x: 408, y: 16, w: 16, h: 140 },
    { x: 408, y: 220, w: 16, h: 164 },
    
    // Horizontal barriers
    { x: 16, y: 140, w: 104, h: 16 },
    { x: 328, y: 140, w: 64, h: 16 },
    { x: 496, y: 140, w: 112, h: 16 }
  ];

  gameState.doors = [
    new Door(216, 156, 16, 64, true, "key6a"),
    new Door(408, 156, 16, 64, true, "key6b")
  ];

  gameState.items = [
    new Item(190, 135, "key", "key6a", false),
    new Item(328, 280, "key", "key6b", false),
    new Item(304, 280, "objective", "obj6_1", true),
    // Moved obj6_2 closer to middle wall and farther from monster
    new Item(440, 100, "objective", "obj6_2", true)
  ];

  gameState.requiredItemIds = ["obj6_1", "obj6_2"];

  gameState.hidingSpots = [
    new HidingSpot(40, 280, 32, 24, "closet"),
    new HidingSpot(272, 40, 48, 16, "table"),
    new HidingSpot(544, 180, 32, 24, "closet")
  ];

  gameState.exitZone = new ExitZone(288, 320, 64, 48);

  // Player spawns BOTTOM-LEFT
  gameState.player = new Player(40, 340);

  // Monsters patrol TOP areas (far from player spawn!)
  gameState.antagonists = [
    new Antagonist(104, 80, 1.2, [
      [64, 60],
      [180, 60],
      [180, 100],
      [64, 100]
    ]),
    // Right monster - larger patrol path to avoid camping objective
    new Antagonist(550, 60, 1.2, [
      [600, 60],
      [600, 340],
      [440, 340],
      [440, 60]
    ])
  ];
  
  gameState.entities = [gameState.player, ...gameState.antagonists];
}

export function loadLevel(levelNum) {
  // Reset level-specific state
  gameState.walls = [];
  gameState.doors = [];
  gameState.items = [];
  gameState.hidingSpots = [];
  gameState.antagonists = [];
  gameState.entities = [];
  gameState.exitZone = null;
  gameState.inventory = [];
  gameState.inChase = false;
  gameState.levelComplete = false;
  gameState.undetectedBonus = true;
  gameState.requiredItemIds = [];
  gameState.collectedRequiredItems = new Set();

  switch (levelNum) {
    case 1:
      createLevel1();
      break;
    case 2:
      createLevel2();
      break;
    case 3:
      createLevel3();
      break;
    case 4:
      createLevel4();
      break;
    case 5:
      createLevel5();
      break;
    case 6:
      createLevel6();
      break;
  }
}