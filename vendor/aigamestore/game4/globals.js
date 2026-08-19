/**
 * Global constants and game state management.
 * Defines the core configuration for the physics engine, game loop, and state containers.
 */

// Canvas Dimensions
export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 400;

// Physics Constants
export const GRAVITY = 0.6;
export const THRUST_FORCE = -1.5;
export const TERMINAL_VELOCITY = 10;
export const GAME_INITIAL_SPEED = 5;
export const SPEED_INCREMENT = 0.001; // Speed increase per frame

// Game Configuration
export const GROUND_HEIGHT = 50;
export const ROOF_HEIGHT = 50;
export const SPAWN_BUFFER = 100; // Distance beyond screen to spawn entities

// Scoring
export const COIN_VALUE = 1;
export const DISTANCE_MULTIPLIER = 0.1; // Points per pixel traveled

// Game State Object
// Centralized state container for the entire game lifecycle
export const gameState = {
    // Core Phase State
    gamePhase: "START", // START, PLAYING, PAUSED, GAME_OVER_WIN, GAME_OVER_LOSE
    controlMode: "HUMAN", // HUMAN

    // Performance & Time
    frameCount: 0,
    lastFrameTime: 0,
    deltaTime: 0,
    gameSpeed: GAME_INITIAL_SPEED,
    distanceTraveled: 0,

    // Entities containers
    player: null,
    entities: [],       // General list for physics/updates
    obstacles: [],      // Specific list for zappers/missiles
    collectibles: [],   // Coins/Tokens
    particles: [],      // Visual effects
    projectiles: [],    // Bullets (visual only mostly, or interactive)
    backgroundElements: [], // Decorations

    // Player State Snapshot
    score: 0,       // combined total (distance + coins) — used by score capture
    coinScore: 0,   // coin points only
    coinsCollected: 0,
    highScore: 0,

    // Camera/View
    cameraShake: 0,

    // Debug/Logging
    debugMode: false
};

/**
 * Resets the game state to initial values for a new run.
 * Called when starting or restarting the game.
 */
export function resetGameState() {
    gameState.gameSpeed = GAME_INITIAL_SPEED;
    gameState.distanceTraveled = 0;
    gameState.score = 0;
    gameState.coinScore = 0;
    gameState.coinsCollected = 0;
    gameState.cameraShake = 0;
    
    // Clear entity arrays
    gameState.entities = [];
    gameState.obstacles = [];
    gameState.collectibles = [];
    gameState.particles = [];
    gameState.projectiles = [];
    gameState.backgroundElements = [];
    
    // Player is re-initialized externally
    gameState.player = null;
}

/**
 * Expose gameState globally for debugging and external access
 */
export function getGameState() {
    return gameState;
}

// Attach to window object
if (typeof window !== 'undefined') {
    window.getGameState = getGameState;
}