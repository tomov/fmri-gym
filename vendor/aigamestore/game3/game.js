import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { gameState, CONFIG, CANVAS_WIDTH, CANVAS_HEIGHT, logs, logGameEvent } from './globals.js';
import { setupInput } from './input.js';
import { setupLighting, updateLighting } from './lighting.js';
import { loadLevel } from './level.js';
import { setupUI, renderUI } from './ui.js';
import { initParticles, updateParticles } from './particles.js';
import { setSeed } from './utils.js';

function init() {
    // 1. Setup Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    gameState.scene = scene;
    
    // 2. Setup Camera
    const camera = new THREE.PerspectiveCamera(45, CANVAS_WIDTH / CANVAS_HEIGHT, 1, 100);
    camera.position.set(0, CONFIG.CAMERA_HEIGHT, CONFIG.CAMERA_OFFSET_Z);
    camera.lookAt(0, 0, 0);
    gameState.camera = camera;
    
    // 3. Setup Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(CANVAS_WIDTH, CANVAS_HEIGHT);
    renderer.shadowMap.enabled = true;
    gameState.renderer = renderer;
    // Use same id as p5.js main canvas so frame.locator("#defaultCanvas0").first works for all games
    renderer.domElement.id = 'defaultCanvas0';
    
    // Container
    const container = document.getElementById('game-container') || document.body;
    if (container.id !== 'game-container') {
        const newContainer = document.createElement('div');
        newContainer.id = 'game-container';
        newContainer.style.position = 'relative';
        newContainer.style.width = `${CANVAS_WIDTH}px`;
        newContainer.style.height = `${CANVAS_HEIGHT}px`;
        newContainer.style.overflow = 'hidden';
        document.body.appendChild(newContainer);
        newContainer.appendChild(renderer.domElement);
        gameState.gameContainer = newContainer;
    } else {
        gameState.gameContainer = container;
        container.appendChild(renderer.domElement);
    }
    
    // 4. Subsystems
    setupInput();
    setupLighting();
    setupUI();
    initParticles();
    
    // 5. Initial State
    setSeed('42');
    
    // Start Loop
    gameState.clock.start();
    requestAnimationFrame(gameLoop);
}

function gameLoop() {
    requestAnimationFrame(gameLoop);
    
    const dt = gameState.clock.getDelta();
    gameState.deltaTime = dt;
    gameState.frameCount++;
    
    update(dt);
    render();
}

function update(dt) {
    if (gameState.input.restart) {
        restartGame();
        gameState.input.restart = false;
    }

    if (gameState.gamePhase === "PLAYING") {
        // --- GAMEPLAY LOGIC ---
        
        // Entities
        if (gameState.player) gameState.player.update(dt);
        gameState.enemies.forEach(e => e.update(dt));
        gameState.projectiles.forEach(p => p.update(dt));
        
        // Particles
        updateParticles(dt);
        
        // Lighting follows player
        updateLighting();
        
        // Camera Follow
        if (gameState.player) {
            const targetPos = new THREE.Vector3(
                gameState.player.mesh.position.x,
                CONFIG.CAMERA_HEIGHT,
                gameState.player.mesh.position.z + CONFIG.CAMERA_OFFSET_Z
            );
            gameState.camera.position.lerp(targetPos, 0.1); // Smooth follow
        }
        
        // Win Condition - Updated for 7 levels
        if (gameState.enemies.length === 0 && gameState.currentLevel <= 7) {
            // Level Clear
            logGameEvent('LEVEL_CLEAR', { level: gameState.currentLevel });
            gameState.currentLevel++;
            if (gameState.currentLevel > 7) {
                gameState.gamePhase = "GAME_OVER_WIN";
            } else {
                loadLevel(gameState.currentLevel);
            }
        }
    }
    
    // Start game trigger from logic if needed (handled in input usually)
    if (gameState.gamePhase === "PLAYING" && gameState.entities.length === 0) {
        // First load
        loadLevel(1);
    }
    
    renderUI();
}

function render() {
    gameState.renderer.render(gameState.scene, gameState.camera);
}

function restartGame() {
    gameState.score = 0;
    gameState.currentLevel = 1;
    gameState.gamePhase = "PLAYING";
    loadLevel(1);
    logGameEvent('RESTART', {});
}

// Global hook for controls
window.setControlMode = (mode) => {
    gameState.controlMode = mode;
    console.log(`Control Mode set to: ${mode}`);
};

// Start
init();