import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT } from './globals.js';

let uiCanvas, uiContext;

export function setupUI() {
    uiCanvas = document.createElement('canvas');
    uiCanvas.width = CANVAS_WIDTH;
    uiCanvas.height = CANVAS_HEIGHT;
    uiCanvas.style.position = 'absolute';
    uiCanvas.style.top = '0';
    uiCanvas.style.left = '0';
    uiCanvas.style.pointerEvents = 'none';
    uiCanvas.style.zIndex = '1000';
    
    if (gameState.gameContainer) {
        gameState.gameContainer.appendChild(uiCanvas);
    }
    
    uiContext = uiCanvas.getContext('2d');
}

export function renderUI() {
    if (!uiContext) return;
    
    uiContext.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    const phase = gameState.gamePhase;
    
    if (phase === "START") {
        drawScreen("", "press enter to begin", null);
    } else if (phase === "PLAYING") {
        drawHUD();
    } else if (phase === "PAUSED") {
        drawHUD();
        // Pause overlay and text removed per request
    } else if (phase === "GAME_OVER_WIN") {
        drawScreen("MISSION ACCOMPLISHED", "Press R to Restart", `Score: ${gameState.score}`);
    } else if (phase === "GAME_OVER_LOSE") {
        drawScreen("KIA - GAME OVER", "Press R to Retry", `Score: ${gameState.score}`);
    }
}

function drawHUD() {
    // Health Bar
    const hp = gameState.player ? gameState.player.health : 0;
    const maxHp = 100;
    const barW = 200;
    const barH = 20;
    const x = 20;
    const y = 20;
    
    // Bg
    uiContext.fillStyle = 'rgba(0, 0, 0, 0.5)';
    uiContext.fillRect(x, y, barW, barH);
    
    // Fill
    const ratio = Math.max(0, hp / maxHp);
    uiContext.fillStyle = ratio > 0.3 ? '#00ff00' : '#ff0000';
    uiContext.fillRect(x, y, barW * ratio, barH);
    
    // Text
    uiContext.fillStyle = 'white';
    uiContext.font = '14px Arial';
    uiContext.fillText(`HP: ${Math.ceil(hp)}`, x + 5, y + 15);
    
    // Score
    uiContext.textAlign = 'right';
    uiContext.fillText(`SCORE: ${gameState.score}`, CANVAS_WIDTH - 20, 35);
    
    // Level
    uiContext.textAlign = 'center';
    uiContext.fillText(`FLOOR ${gameState.currentLevel}`, CANVAS_WIDTH / 2, 35);
    
    // Weapon Info
    if (gameState.player) {
        uiContext.textAlign = 'right';
        uiContext.fillText(`WEAPON: ${gameState.player.currentWeapon}`, CANVAS_WIDTH - 20, 60);
    }
}

function drawScreen(title, subtitle, info) {
    uiContext.fillStyle = 'rgba(0, 0, 0, 0.8)';
    uiContext.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Only draw title if it's not empty
    if (title) {
        uiContext.fillStyle = '#ff0000';
        uiContext.font = 'bold 40px Impact, Arial';
        uiContext.textAlign = 'center';
        uiContext.fillText(title, CANVAS_WIDTH / 2, 150);
    }
    
    uiContext.fillStyle = 'white';
    uiContext.font = 'bold 24px Arial';
    // Adjust Y position if title is empty
    const subtitleY = title ? 220 : CANVAS_HEIGHT / 2;
    uiContext.fillText(subtitle, CANVAS_WIDTH / 2, subtitleY);
    
    if (info) {
        uiContext.font = '16px Arial';
        uiContext.fillStyle = '#cccccc';
        uiContext.fillText(info, CANVAS_WIDTH / 2, 280);
    }
}