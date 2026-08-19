import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 400;

// Game Configuration
export const CONFIG = {
    PLAYER_SPEED: 0.15,
    PLAYER_DASH_SPEED: 0.5, // Slightly faster dash
    PLAYER_DASH_DURATION: 0.2, // seconds
    PLAYER_DASH_COOLDOWN: 0.3, // Reduced from 1.0 for "dash more"
    PLAYER_MAX_HEALTH: 100,
    CAMERA_HEIGHT: 20,
    CAMERA_OFFSET_Z: 10, // Slight tilt
    GRAVITY: -0.5,
    TILE_SIZE: 4,
    COLORS: {
        PLAYER: 0x00FFFF, // Cyan
        ENEMY: 0xFF0000,  // Red
        WALL: 0x222222,   // Dark Grey
        FLOOR: 0x111111,  // Almost Black
        PROJECTILE: 0xFFD700, // Gold
        BLOOD: 0x880000,  // Dark Red
        AMMO: 0x00FF00,   // Green
        HEALTH: 0xFF00FF  // Magenta
    }
};

// Global Game State
export const gameState = {
    // Phase
    gamePhase: "START", // START, PLAYING, PAUSED, LEVEL_COMPLETE, GAME_OVER_WIN, GAME_OVER_LOSE
    controlMode: "HUMAN", // HUMAN
    
    // Core Three.js
    scene: null,
    camera: null,
    renderer: null,
    gameContainer: null,
    
    // Systems
    clock: new THREE.Clock(),
    frameCount: 0,
    deltaTime: 0,
    timestamp: 0,
    
    // Level
    currentLevel: 1,
    score: 0,
    
    // Entity Lists
    entities: [],
    enemies: [],
    projectiles: [],
    particles: [],
    pickups: [],
    walls: [], // Static physics geometry
    
    // References
    player: null,
    cameraTarget: new THREE.Vector3(),
    mousePos: new THREE.Vector3(), // Not used for control, but maybe for debugging
    
    // Lighting
    lights: [],
    
    // Input State (Keyboard only)
    input: {
        up: false,
        down: false,
        left: false,
        right: false,
        shootUp: false,
        shootDown: false,
        shootLeft: false,
        shootRight: false,
        dash: false,
        reload: false,
        swap: false,
        walk: false,
        enter: false,
        esc: false,
        restart: false
    }
};

// Expose gameState globally
window.getGameState = () => gameState;

// Logging
export const logs = {
    game_info: [],
    player_info: [],
    inputs: []
};
window.logs = logs;

export function logGameEvent(type, data) {
    logs.game_info.push({
        type: type,
        data: data,
        frame: gameState.frameCount,
        time: Date.now()
    });
}