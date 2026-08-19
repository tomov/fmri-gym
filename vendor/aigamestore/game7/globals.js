// Game state and constants
export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 400;
export const PLAYER_SPEED = 2.5;
export const PLAYER_SPRINT_SPEED = 4;
export const PLAYER_COVER_SPEED = 1.5;
export const BULLET_SPEED = 6;
export const ENEMY_SPEED = 1.2;
export const ENEMY_BULLET_SPEED = 3;

// Weapon definitions
export const WEAPONS = {
  pistol: {
    name: "Pistol",
    fireRate: 300,
    damage: 1,
    ammo: 7,
    maxAmmo: 7,
    reserveAmmo: 21,
    maxReserveAmmo: 42,
    accuracy: 0.05,
    sprintAccuracy: 0.2,
    crouchAccuracy: 0.01,
    bulletSpeed: 6,
    reloadTime: 2500, // Slower reload
    type: "bullet"
  },
  rifle: {
    name: "Rifle",
    fireRate: 150,
    damage: 1,
    ammo: 14,
    maxAmmo: 14,
    reserveAmmo: 42,
    maxReserveAmmo: 84,
    accuracy: 0.03,
    sprintAccuracy: 0.15,
    crouchAccuracy: 0.005,
    bulletSpeed: 8,
    reloadTime: 3000, // Slower reload
    type: "bullet"
  },
  shotgun: {
    name: "Shotgun",
    fireRate: 800,
    damage: 2,
    ammo: 7,
    maxAmmo: 7,
    reserveAmmo: 14,
    maxReserveAmmo: 28,
    accuracy: 0.3,
    sprintAccuracy: 0.5,
    crouchAccuracy: 0.2,
    bulletSpeed: 5,
    reloadTime: 4000, // Very slow reload
    pellets: 5,
    type: "bullet"
  },
  sniper: {
    name: "Sniper",
    fireRate: 1500,
    damage: 5, // Increased damage
    ammo: 5,
    maxAmmo: 5,
    reserveAmmo: 10,
    maxReserveAmmo: 20,
    accuracy: 0.01,
    sprintAccuracy: 0.3,
    crouchAccuracy: 0.001,
    bulletSpeed: 12,
    reloadTime: 4500, // Very slow reload
    type: "bullet"
  },
  rocket_launcher: {
    name: "RPG",
    fireRate: 2000,
    damage: 10, // Direct hit damage
    explosionDamage: 8, // Splash damage
    explosionRadius: 80,
    ammo: 1,
    maxAmmo: 1,
    reserveAmmo: 3,
    maxReserveAmmo: 5,
    accuracy: 0.1,
    sprintAccuracy: 0.4,
    crouchAccuracy: 0.05,
    bulletSpeed: 5,
    reloadTime: 5000, // Extremely slow reload
    type: "rocket"
  }
};

// Game state object to track all game data
export const gameState = {
  player: null,
  bullets: [],
  enemyBullets: [],
  enemies: [],
  obstacles: [],
  pickups: [],
  weaponPickups: [],
  score: 0,
  gamePhase: "START",
  controlMode: "HUMAN",
  currentLevel: 1,
  level: {
    width: 1200,
    height: 800,
    cameraX: 0,
    cameraY: 0
  },
  extractionPoint: { x: 0, y: 0 },
  extractionPointObj: null,
  requiredKills: 10,
  enemiesKilled: 0,
  mission: "combined", // Always combined now
  timeElapsed: 0,
  // Auto-restart timer
  autoRestartTimer: null,
  // Replay state
  replayMode: false,
  replayValidation: {
    enabled: false,
    discrepancies: []
  }
};

// Keyboard codes
export const KEY = {
  LEFT: 37,
  UP: 38,
  RIGHT: 39,
  DOWN: 40,
  SPACE: 32,
  SHIFT: 16,
  Z: 90,
  ENTER: 13,
  ESC: 27,
  R: 82,
  KEY_1: 49,
  KEY_2: 50,
  KEY_3: 51,
  KEY_4: 52,
  KEY_5: 53
};

// Get game state function (exposed globally)
export function getGameState() {
  return gameState;
}

if (typeof window !== 'undefined') {
  window.getGameState = getGameState;
  window.gameState = gameState;
}

// Helper functions
export function resetGame() {
  gameState.bullets = [];
  gameState.enemyBullets = [];
  gameState.enemies = [];
  gameState.obstacles = [];
  gameState.pickups = [];
  gameState.weaponPickups = [];
  // Score carries over between levels; resetToLevel1 sets it back to 0
  gameState.enemiesKilled = 0;
  gameState.timeElapsed = 0;
  gameState.player = null;
  gameState.level.cameraX = 0;
  gameState.level.cameraY = 0;
  
  // Mission is always combined: Kill enemies AND extract
  gameState.mission = "combined";
  
  // Define difficulty progression: 2 Easy, 2 Medium, then Hard but slower scaling
  if (gameState.currentLevel <= 2) {
    // Easy Levels (1-2): Low kill count
    gameState.requiredKills = 5 + (gameState.currentLevel - 1) * 3; // L1: 5, L2: 8
  } else if (gameState.currentLevel <= 4) {
    // Medium Levels (3-4): Medium kill count
    gameState.requiredKills = 12 + (gameState.currentLevel - 3) * 4; // L3: 12, L4: 16
  } else {
    // Hard Levels (5+): Slower scaling
    // Previous: 20 + (L-5)*5 -> 20, 25, 30
    // New: 18 + (L-5)*3 -> 18, 21, 24
    gameState.requiredKills = 18 + (gameState.currentLevel - 5) * 3;
  }
}

export function nextLevel() {
  gameState.currentLevel++;
  resetGame();
}

export function resetToLevel1() {
  gameState.currentLevel = 1;
  gameState.score = 0;
  resetGame();
}