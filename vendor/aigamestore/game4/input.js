/**
 * Input handling for Keyboard.
 */

import { gameState } from './globals.js';

const keys = {};

export function handleInput(p) {
    // Check for control keys
    const up = isKeyDown(p.UP_ARROW) || isKeyDown(32); // 32 is SPACE
    
    return up; // Return active state of 'thrust'
}

export function setupInput(p) {
    p.keyPressed = function() {
        keys[p.keyCode] = true;
        
        // Phase transitions - Esc = pause, Enter = resume (when paused)
        if (p.keyCode === 13) { // ENTER - start or resume from pause
            if (gameState.gamePhase === 'START') {
                gameState.gamePhase = 'PLAYING';
            } else if (gameState.gamePhase === 'PAUSED') {
                gameState.gamePhase = 'PLAYING';
            }
        }
        
        if (p.keyCode === 27) { // ESC - pause (and unpause)
            if (gameState.gamePhase === 'PLAYING') {
                gameState.gamePhase = 'PAUSED';
            } else if (gameState.gamePhase === 'PAUSED') {
                gameState.gamePhase = 'PLAYING';
            }
        }
        
        if (p.keyCode === 82) { // R
            if (gameState.gamePhase === 'GAME_OVER_WIN' || gameState.gamePhase === 'GAME_OVER_LOSE') {
                // Restart handled in game.js loop check or via reload, 
                // but better to signal a reset.
                // We'll set a flag or call reset directly if imported (circular dependency risk).
                // Let's rely on game loop to check this key or update phase to START
                gameState.gamePhase = 'START';
                // Reset logic needs to be called. We can do it here if we expose resetGameState globally.
                if (window.resetGameInstance) window.resetGameInstance();
            }
        }
        
        // Vehicle Special ability
        if (p.keyCode === 90) { // Z
            // Trigger special
        }
    };

    p.keyReleased = function() {
        keys[p.keyCode] = false;
    };
}

function isKeyDown(code) {
    return !!keys[code];
}