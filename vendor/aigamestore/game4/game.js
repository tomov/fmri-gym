/**
 * Main Game Controller.
 * Initializes p5 instance, runs the game loop, and manages level generation.
 */

import { gameState, resetGameState, getGameState, CANVAS_WIDTH, CANVAS_HEIGHT, SPEED_INCREMENT, SPAWN_BUFFER } from './globals.js';
import { Player, Zapper, Missile, Coin, VehicleToken } from './entities.js';
import { BackgroundManager } from './background.js';
import { setupInput, handleInput } from './input.js';
import { updatePhysics, checkRectCollision, checkCircleCollision, checkCircleRectCollision } from './physics.js';
import { renderStartScreen, renderHUD, renderGameOverScreen } from './ui.js';
import { spawnExplosion, spawnCoinSparkle } from './particles.js';

// Access p5 from global scope
const p5 = window.p5;

// Global Reset Function for easy access
window.resetGameInstance = function() {
    resetGameState();
    // Re-init player
    // gameState.player = new Player(); // Done in setup/restart logic within p5
};

const gameInstance = new p5(p => {
    
    let backgroundManager;
    
    // Spawning Timers
    let zapperTimer = 0;
    let missileTimer = 0;
    let coinTimer = 0;
    
    p.setup = function() {
        p.createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
        p.frameRate(60);
        p.randomSeed(42);
        
        resetGameState();
        
        backgroundManager = new BackgroundManager();
        setupInput(p);
        
        gameState.gamePhase = 'START';
        
    };

    p.draw = function() {
        // --- Logic Update ---
        const currentTime = p.millis();
        gameState.deltaTime = (currentTime - gameState.lastFrameTime) / 1000;
        gameState.lastFrameTime = currentTime;
        gameState.frameCount = p.frameCount;

        // Camera Shake decay
        if (gameState.cameraShake > 0) {
            p.translate(p.random(-gameState.cameraShake, gameState.cameraShake), p.random(-gameState.cameraShake, gameState.cameraShake));
            gameState.cameraShake *= 0.9;
        }

        // Phase Handling
        switch(gameState.gamePhase) {
            case 'START':
                backgroundManager.render(p); // Render BG for ambiance
                renderStartScreen(p);
                break;
                
            case 'PLAYING':
                updateGameLogic(p);
                renderGame(p);
                renderHUD(p);
                break;
                
            case 'PAUSED':
                renderGame(p);
                renderHUD(p);
                break;
                
            case 'GAME_OVER_WIN':
            case 'GAME_OVER_LOSE':
                renderGame(p);
                renderGameOverScreen(p);
                break;
        }
    };
    
    function startPlaying() {
        resetGameState();
        gameState.player = new Player();
        gameState.entities.push(gameState.player);
        gameState.gamePhase = 'PLAYING';
    }
    
    // Check if phase changed from START to PLAYING via Input (ENTER)
    // We need to initialize player once when entering PLAYING
    // The input handler sets the string 'PLAYING'. We detect the transition here?
    // Actually input handler sets the variable directly. 
    // We should check if player is null in PLAYING to init.
    
    function updateGameLogic(p) {
        // 1. Init Player if needed
        if (!gameState.player) {
            gameState.player = new Player();
            gameState.entities.push(gameState.player);
        }

        // 2. Input Handling
        let inputActive = handleInput(p);

        // 3. Update Player
        gameState.player.update(p, inputActive);

        // 4. Update Physics & Entities
        gameState.gameSpeed += SPEED_INCREMENT;
        gameState.distanceTraveled += gameState.gameSpeed * 0.1;
        gameState.score = Math.floor(gameState.distanceTraveled) + gameState.coinScore;

        // Background
        backgroundManager.update();

        // Update Obstacles
        gameState.obstacles.forEach((obs, index) => {
            obs.update();
            if (!obs.active) gameState.obstacles.splice(index, 1);
        });

        // Update Collectibles
        gameState.collectibles.forEach((col, index) => {
            col.update();
            if (!col.active) gameState.collectibles.splice(index, 1);
        });

        // Update Particles
        for (let i = gameState.particles.length - 1; i >= 0; i--) {
            gameState.particles[i].update();
            if (gameState.particles[i].markedForDeletion) {
                gameState.particles.splice(i, 1);
            }
        }

        // 5. Spawning Logic (Procedural Generation)
        handleSpawning(p);

        // 6. Collision Detection
        handleCollisions(p);
        
        // 7. Check Death
        if (gameState.player.isDead) {
            gameState.gamePhase = 'GAME_OVER_LOSE';
            gameState.cameraShake = 20;
        }
    }

    function handleSpawning(p) {
        // Zappers
        zapperTimer++;
        if (zapperTimer > 200 - Math.min(100, gameState.gameSpeed * 10)) {
            const y = p.random(100, CANVAS_HEIGHT - 100);
            const type = p.random() > 0.7 ? 'ROTATING' : 'STATIC';
            const len = p.random(100, 150);
            const zapper = new Zapper(CANVAS_WIDTH + SPAWN_BUFFER, y, len, p.random(p.TWO_PI), type);
            gameState.obstacles.push(zapper);
            zapperTimer = 0;
        }

        // Missiles
        missileTimer++;
        if (missileTimer > 300) {
            const y = p.random(50, CANVAS_HEIGHT - 50);
            gameState.obstacles.push(new Missile(y));
            missileTimer = 0;
        }

        // Coins (Pattern spawning)
        coinTimer++;
        if (coinTimer > 150) {
            spawnCoinPattern(p);
            coinTimer = 0;
        }
        
        // Vehicle Tokens (Rare)
        if (p.frameCount % 1200 === 0 && !gameState.player.vehicle) {
            gameState.collectibles.push(new VehicleToken(CANVAS_WIDTH + SPAWN_BUFFER, p.random(100, 300)));
        }
    }

    function spawnCoinPattern(p) {
        const pattern = Math.floor(p.random(3));
        const startX = CANVAS_WIDTH + SPAWN_BUFFER;
        const startY = p.random(100, CANVAS_HEIGHT - 100);

        if (pattern === 0) { // Line
            for (let i = 0; i < 5; i++) {
                gameState.collectibles.push(new Coin(startX + i * 30, startY));
            }
        } else if (pattern === 1) { // Sine wave
            for (let i = 0; i < 10; i++) {
                gameState.collectibles.push(new Coin(startX + i * 30, startY + Math.sin(i * 0.5) * 50));
            }
        } else if (pattern === 2) { // Block
            for (let x = 0; x < 3; x++) {
                for (let y = 0; y < 3; y++) {
                    gameState.collectibles.push(new Coin(startX + x * 30, startY + y * 30));
                }
            }
        }
    }

    function handleCollisions(p) {
        const player = gameState.player;
        if (!player) return;

        // Player Bounds (Approximated as rect)
        const playerRect = {
            x: player.x,
            y: player.y,
            width: player.width,
            height: player.height
        };

        // Obstacles
        for (let obs of gameState.obstacles) {
            if (player.invincibilityTimer > 0) continue;

            let collision = false;
            
            if (obs instanceof Zapper) {
                // Simplified Circle-Line or Circle-Rect collision
                // For simplicity in this constraints, treat zapper as a rect rotated
                // OR just use multiple circles along the line
                
                // Using a simplified rect check for center
                const obsRect = {
                    x: obs.x - obs.width/2,
                    y: obs.y - obs.length/2,
                    width: obs.width,
                    height: obs.length
                };
                
                // If rotating, bounds are larger.
                // Simple AABB first
                if (Math.abs(player.x - obs.x) < 100 && Math.abs(player.y - obs.y) < 100) {
                     // Check collision based on type
                     // For high fidelity, use p5.collide2d logic manually
                     // Check player center distance to line segment of zapper
                     // Zapper line: (obs.x + sin(a)*len/2, obs.y + cos(a)*len/2) to (obs.x - ..., ...)
                     
                     // Better: checkRectCollision with the unrotated AABB is bad.
                     // Use point to line distance?
                     if (p.collideLineRect) {
                         const x1 = obs.x - Math.sin(obs.angle) * obs.length/2;
                         const y1 = obs.y + Math.cos(obs.angle) * obs.length/2;
                         const x2 = obs.x + Math.sin(obs.angle) * obs.length/2;
                         const y2 = obs.y - Math.cos(obs.angle) * obs.length/2;
                         collision = p.collideLineRect(x1, y1, x2, y2, player.x, player.y, player.width, player.height);
                     }
                }
            } else if (obs instanceof Missile) {
                if (obs.state === 'LAUNCHED') {
                    const missileRect = {x: obs.x - 20, y: obs.y - 7, width: 40, height: 15};
                    collision = checkRectCollision(playerRect, missileRect);
                }
            }

            if (collision && !gameState.debugMode) {
                if (player.vehicle) {
                    player.vehicle = null;
                    player.invincibilityTimer = 60; // 1 second invincibility
                    spawnExplosion(player.x, player.y, 20);
                    gameState.cameraShake = 10;
                } else {
                    player.isDead = true;
                    spawnExplosion(player.x, player.y);
                }
            }
        }

        // Collectibles
        for (let col of gameState.collectibles) {
            if (!col.active) continue;
            
            let collected = false;
            if (col instanceof Coin) {
                collected = checkCircleRectCollision({x: col.x, y: col.y, radius: col.hitRadius}, playerRect);
                if (collected) {
                    col.active = false;
                    gameState.coinScore += 10;
                    gameState.coinsCollected++;
                    spawnCoinSparkle(col.x, col.y);
                }
            } else if (col instanceof VehicleToken) {
                // Rect Rect
                collected = checkRectCollision(
                    {x: col.x - 20, y: col.y - 20, width: 40, height: 40}, 
                    playerRect
                );
                if (collected) {
                    col.active = false;
                    player.vehicle = 'DRAGON';
                    gameState.cameraShake = 10;
                    spawnExplosion(player.x, player.y, 50); // Transform effect
                }
            }
        }
    }

    function renderGame(p) {
        p.background(50); // Fallback
        
        // 1. Background
        backgroundManager.render(p);
        
        // 2. Game World Entities
        // Collectibles (behind player)
        gameState.collectibles.forEach(col => col.render(p));
        
        // Obstacles
        gameState.obstacles.forEach(obs => obs.render(p));
        
        // Particles (behind player mostly)
        gameState.particles.forEach(part => part.render(p));
        
        // Player
        if (gameState.player && !gameState.player.isDead) {
            gameState.player.render(p);
        }
    }
});

// Expose instance
window.gameInstance = gameInstance;
