export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 400;

export const GAME_PHASES = {
  START: "START",
  PLAYING: "PLAYING",
  PAUSED: "PAUSED",
  GAME_OVER_WIN: "GAME_OVER_WIN",
  GAME_OVER_LOSE: "GAME_OVER_LOSE"
};

export const CONTROL_MODES = {
  HUMAN: "HUMAN",
  TEST_1: "TEST_1",
  TEST_2: "TEST_2"
};

export const gameState = {
  player: null,
  entities: [],
  gamePhase: GAME_PHASES.START,
  controlMode: CONTROL_MODES.HUMAN,
  camera: { x: 0, y: 0 },
  currentStage: null,
  currentLevel: 0,
  totalLevels: 9,
  robotMastersDefeated: {},
  unlockedWeapons: [],
  currentWeapon: 0,
  weaponEnergy: {},
  bossGauntletIndex: 0,
  wilyStagePhase: 0,
  score: 0,
  lives: 5,
  maxLives: 5,
  playerHealth: 28,
  maxPlayerHealth: 28,
  bossHealth: 0,
  maxBossHealth: 28,
  particles: [],
  projectiles: [],
  platformBlocks: [],
  hazards: [],
  enemySpawners: [],
  drops: [],
  showBossHealthBar: false,
  stageComplete: false,
  transitionTimer: 0,
  invincibilityFrames: 0,
  yokublockTimer: 0,
  yokublockPattern: [],
  healingAvailable: false,
  autoRestartTimer: null, // Added for automatic restart functionality
  frameRate: 60 // Added to store the game's frame rate for timing
};

export const ROBOT_MASTERS = [
  { name: "CUT", color: [200, 50, 50], weakness: "BOMB", weapon: "METAL_BLADE" },
  { name: "ELEC", color: [255, 255, 100], weakness: "METAL_BLADE", weapon: "THUNDER_BEAM" },
  { name: "ICE", color: [100, 200, 255], weakness: "THUNDER_BEAM", weapon: "ICE_SLASHER" },
  { name: "BOMB", color: [150, 150, 150], weakness: "ICE_SLASHER", weapon: "HYPER_BOMB" },
  { name: "FIRE", color: [255, 100, 50], weakness: "ICE_SLASHER", weapon: "FIRE_STORM" },
  { name: "TIME", color: [150, 100, 255], weakness: "FIRE_STORM", weapon: "TIME_STOPPER" }
];

export const WEAPONS = {
  BUSTER: { name: "MEGA BUSTER", energy: Infinity, damage: 1, color: [100, 200, 255] },
  METAL_BLADE: { name: "METAL BLADE", energy: 28, damage: 2, color: [200, 200, 200] },
  THUNDER_BEAM: { name: "THUNDER BEAM", energy: 28, damage: 4, color: [255, 255, 100] },
  ICE_SLASHER: { name: "ICE SLASHER", energy: 28, damage: 3, color: [100, 200, 255] },
  HYPER_BOMB: { name: "HYPER BOMB", energy: 28, damage: 5, color: [80, 80, 80] },
  FIRE_STORM: { name: "FIRE STORM", energy: 28, damage: 3, color: [255, 100, 50] },
  TIME_STOPPER: { name: "TIME STOPPER", energy: 28, damage: 1, color: [200, 150, 255] },
  MAGNET_BEAM: { name: "MAGNET BEAM", energy: 28, damage: 0, color: [255, 50, 150] }
};

export function getGameState() {
  return gameState;
}

// Attach to window
if (typeof window !== 'undefined') {
  window.getGameState = getGameState;
}