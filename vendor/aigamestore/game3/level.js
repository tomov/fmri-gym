import { Wall, Enemy, Player } from './entities.js';
import { gameState, CONFIG } from './globals.js';
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';

// Simple Grid Based Level Generation
// W = Wall, . = Floor, P = Player, E = Enemy(Grunt), S = Shooter
const LEVEL_1 = [
    "WWWWWWWWWWWWWWWW",
    "W..............W",
    "W.P............W",
    "W.......W......W",
    "W.......W...E..W",
    "WWWW..WWWW.....W",
    "W..............W",
    "W.....S........W",
    "W..............W",
    "W.......E......W",
    "WWWWWWWWWWWWWWWW"
];

const LEVEL_2 = [
    "WWWWWWWWWWWWWWWWWWWW",
    "W..................W",
    "W.P.....WWWW.......W",
    "W.......W..E.......W",
    "W...E...W..........W",
    "W.......WWWW...S...W",
    "W..................W",
    "W...WWWW....WWWW...W",
    "W...W..........W...W",
    "W...W...S..E...W...W",
    "W...W..........W...W",
    "W..................W",
    "WWWWWWWWWWWWWWWWWWWW"
];

const LEVEL_3 = [
    "WWWWWWWWWWWWWWWWWWWW",
    "W..................W",
    "W.P................W",
    "W...W..W....W..W...W",
    "W...E..........E...W",
    "W..................W",
    "W...W..........W...W",
    "W...S...WWWW...S...W",
    "W...W..........W...W",
    "W..................W",
    "W...E..........E...W",
    "W..................W",
    "WWWWWWWWWWWWWWWWWWWW"
];

const LEVEL_4 = [
    "WWWWWWWWWWWWWWWWWWWWWWWW",
    "W......................W",
    "W.P..WWWWWW..WWWWWW....W",
    "W.......W......W.......W",
    "W.......W..E...W...E...W",
    "W....WWWW......WWWW....W",
    "W......................W",
    "W....WWWW......WWWW....W",
    "W.......W..S...W.......W",
    "W.......W......W...S...W",
    "W....WWWWWW..WWWWWW....W",
    "W......................W",
    "WWWWWWWWWWWWWWWWWWWWWWWW"
];

const LEVEL_5 = [
    "WWWWWWWWWWWWWWWWWWWW",
    "W..................W",
    "W.P................W",
    "W.......E..E.......W",
    "W...WW........WW...W",
    "W...W..........W...W",
    "W...W....S.....W...W",
    "W...W..........W...W",
    "W...WW........WW...W",
    "W.......E..E.......W",
    "W..................W",
    "W.......S..S.......W",
    "WWWWWWWWWWWWWWWWWWWW"
];

const LEVEL_6 = [
    "WWWWWWWWWWWWWWWWWWWWWWWWWW",
    "W........................W",
    "W.P..E...................W",
    "W.......WWWW..WWWWWW.....W",
    "W.......W..........W.....W",
    "W.......W...S..S...W.....W",
    "W.......W..........W.....W",
    "W.......W...E..E...W.....W",
    "W.......W..........W.....W",
    "W.......WWWWWWWWWWWW.....W",
    "W........................W",
    "W...S..E........E..S.....W",
    "WWWWWWWWWWWWWWWWWWWWWWWWWW"
];

const LEVEL_7 = [
    "WWWWWWWWWWWWWWWWWWWWWWWWWW",
    "W........................W",
    "W.P......................W",
    "W......E..E..E..E........W",
    "W........................W",
    "W...WWWW........WWWW.....W",
    "W...W..............W.....W",
    "W...W...S..S..S....W.....W",
    "W...W..............W.....W",
    "W...WWWW........WWWW.....W",
    "W........................W",
    "W......E..E..E..E........W",
    "W........................W",
    "W.........S..S...........W",
    "WWWWWWWWWWWWWWWWWWWWWWWWWW"
];

const LEVELS = [LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4, LEVEL_5, LEVEL_6, LEVEL_7];

const LEVEL_COLORS = [
    { bg: 0x050505, floor: 0x111111 }, // 1: Dark (Default)
    { bg: 0x000022, floor: 0x111133 }, // 2: Midnight Blue
    { bg: 0x001100, floor: 0x112211 }, // 3: Toxic Green
    { bg: 0x220000, floor: 0x331111 }, // 4: Red Alert
    { bg: 0x110022, floor: 0x221133 }, // 5: Purple Void
    { bg: 0x222200, floor: 0x333311 }, // 6: Golden Halls
    { bg: 0x333333, floor: 0x555555 }  // 7: White Lab
];

export function loadLevel(levelIndex) {
    clearLevel();
    
    // Safety clamp
    const idx = Math.max(1, Math.min(levelIndex, LEVELS.length)) - 1;
    const layout = LEVELS[idx];
    const colors = LEVEL_COLORS[idx] || LEVEL_COLORS[0];
    
    // Set Environment Colors
    if (gameState.scene) {
        gameState.scene.background = new THREE.Color(colors.bg);
        // Also update fog if we had it, but we don't.
    }

    const scale = CONFIG.TILE_SIZE;
    const offsetX = (layout[0].length * scale) / 2;
    const offsetZ = (layout.length * scale) / 2;
    
    for (let row = 0; row < layout.length; row++) {
        for (let col = 0; col < layout[row].length; col++) {
            const char = layout[row][col];
            const x = col * scale - offsetX;
            const z = row * scale - offsetZ;
            
            if (char === 'W') {
                new Wall(x, z, scale, scale);
            } else if (char === 'P') {
                gameState.player = new Player(x, z);
                gameState.entities.push(gameState.player);
            } else if (char === 'E') {
                const e = new Enemy(x, z, 'GRUNT');
                gameState.enemies.push(e);
                gameState.entities.push(e);
            } else if (char === 'S') {
                const e = new Enemy(x, z, 'SHOOTER');
                gameState.enemies.push(e);
                gameState.entities.push(e);
            }
        }
    }
    
    // Create Floor with specific color
    createFloor(layout[0].length * scale, layout.length * scale, colors.floor);
}

function clearLevel() {
    // Remove all entities
    gameState.entities.forEach(e => gameState.scene.remove(e.mesh));
    gameState.walls.forEach(w => gameState.scene.remove(w.mesh));
    
    gameState.entities = [];
    gameState.enemies = [];
    gameState.projectiles = [];
    gameState.walls = [];
    
    if (gameState.floorMesh) gameState.scene.remove(gameState.floorMesh);
}

function createFloor(width, height, colorHex) {
    const geometry = new THREE.PlaneGeometry(width + 20, height + 20); // Make it larger than level
    const material = new THREE.MeshStandardMaterial({ 
        color: colorHex,
        roughness: 0.9 
    });
    const floor = new THREE.Mesh(geometry, material);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    gameState.scene.add(floor);
    gameState.floorMesh = floor;
}