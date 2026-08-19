/**
 * User Interface rendering.
 * Start Screen, HUD, Game Over, Paused.
 */

import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT } from './globals.js';

export function renderStartScreen(p) {
    p.background(20, 30, 40);
    p.textAlign(p.CENTER, p.CENTER);
    
    // Replace main game title with "press enter to begin"
    p.fill(255); // White color for the main message
    p.textSize(36); // Prominent size for the new title
    p.text("press enter to begin", CANVAS_WIDTH/2, CANVAS_HEIGHT/2 - 20); // Centered, slightly above mid-point

    // Preserve existing controls section, adjusted for new title's position
    p.fill(200); // Gray color for instructions
    p.textSize(16); // Slightly larger for readability
    p.text("SPACE / UP ARROW to Fly", CANVAS_WIDTH/2, CANVAS_HEIGHT/2 + 40);
    p.text("Avoid Zappers & Missiles!", CANVAS_WIDTH/2, CANVAS_HEIGHT/2 + 60);
}

export function renderHUD(p) {
    p.textAlign(p.LEFT, p.TOP);
    p.textSize(20);
    
    // Calculate Total Score (Distance + Coin Bonuses)
    const totalScore = gameState.score;
    
    // Distance as Score
    p.fill(255);
    p.text(`Score: ${totalScore}`, 20, 20);
    
    // Coins
    p.fill(255, 215, 0);
    p.text(`Coins: ${gameState.coinsCollected}`, 20, 50);
    
}

export function renderGameOverScreen(p) {
    p.push();
    p.fill(0, 0, 0, 200);
    p.rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    p.textAlign(p.CENTER, p.CENTER);
    
    if (gameState.gamePhase === 'GAME_OVER_LOSE') {
        p.fill(255, 50, 50);
        p.textSize(40);
        p.text("YOU DIED", CANVAS_WIDTH/2, CANVAS_HEIGHT/2 - 40);
    } else {
        p.fill(50, 255, 50);
        p.textSize(40);
        p.text("MISSION COMPLETE", CANVAS_WIDTH/2, CANVAS_HEIGHT/2 - 40);
    }
    
    const totalScore = gameState.score;
    
    p.fill(255);
    p.textSize(24);
    p.text(`Final Score: ${totalScore}`, CANVAS_WIDTH/2, CANVAS_HEIGHT/2 + 10);
    p.text(`Coins Collected: ${gameState.coinsCollected}`, CANVAS_WIDTH/2, CANVAS_HEIGHT/2 + 40);
    
    p.fill(200);
    p.textSize(16);
    p.text("Press R to Restart", CANVAS_WIDTH/2, CANVAS_HEIGHT/2 + 80);
    p.pop();
}