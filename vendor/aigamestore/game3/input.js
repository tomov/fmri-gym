import { gameState, logs } from './globals.js';

export function setupInput() {
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
}

function handleKeyDown(e) {
    const code = e.keyCode;
    updateKey(code, true);
    
    // Logs
    logs.inputs.push({
        type: 'keydown',
        key: e.key,
        code: code,
        frame: gameState.frameCount,
        time: Date.now()
    });

    // Global Phase Handling
    if (code === 13) { // ENTER - start game or resume from pause
        if (gameState.gamePhase === "START") {
            gameState.gamePhase = "PLAYING";
        } else if (gameState.gamePhase === "LEVEL_COMPLETE") {
            gameState.gamePhase = "PLAYING";
        } else if (gameState.gamePhase === "PAUSED") {
            gameState.gamePhase = "PLAYING";
        }
    }
    
    if (code === 27) { // ESC - pause (and optionally resume)
        if (gameState.gamePhase === "PLAYING") gameState.gamePhase = "PAUSED";
        else if (gameState.gamePhase === "PAUSED") gameState.gamePhase = "PLAYING";
    }

    if (code === 82) { // R
        if (gameState.gamePhase === "GAME_OVER_WIN" || gameState.gamePhase === "GAME_OVER_LOSE") {
            gameState.input.restart = true; // Handled in game loop
        }
    }
}

function handleKeyUp(e) {
    const code = e.keyCode;
    updateKey(code, false);
}

function updateKey(code, isPressed) {
    const i = gameState.input;
    
    // Movement
    if (code === 87) i.up = isPressed;    // W
    if (code === 83) i.down = isPressed;  // S
    if (code === 65) i.left = isPressed;  // A
    if (code === 68) i.right = isPressed; // D
    
    // Shooting
    if (code === 38) i.shootUp = isPressed;    // Arrow Up
    if (code === 40) i.shootDown = isPressed;  // Arrow Down
    if (code === 37) i.shootLeft = isPressed;  // Arrow Left
    if (code === 39) i.shootRight = isPressed; // Arrow Right
    
    // Actions
    if (code === 32) i.dash = isPressed;       // Space
    if (code === 90) i.swap = isPressed;       // Z
    if (code === 16) i.walk = isPressed;       // Shift
    if (code === 13) i.enter = isPressed;      // Enter
    if (code === 27) i.esc = isPressed;        // Esc
}

export function getInputVector() {
    // Movement vector
    const x = (gameState.input.right ? 1 : 0) - (gameState.input.left ? 1 : 0);
    const z = (gameState.input.down ? 1 : 0) - (gameState.input.up ? 1 : 0);
    return { x, z };
}

export function getShootVector() {
    // Shooting vector (8-way)
    const x = (gameState.input.shootRight ? 1 : 0) - (gameState.input.shootLeft ? 1 : 0);
    const z = (gameState.input.shootDown ? 1 : 0) - (gameState.input.shootUp ? 1 : 0);
    return { x, z };
}