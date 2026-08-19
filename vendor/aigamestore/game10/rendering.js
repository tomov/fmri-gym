// rendering.js - Rendering functions
import { gameState, GAME_PHASES, CANVAS_WIDTH, CANVAS_HEIGHT } from './globals.js';

export function renderStartScreen(p) {
  p.background(20, 15, 30);
  
  // Main title replaced with "press enter to begin"
  p.fill(255, 255, 100);
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(32);
  const blink = Math.floor(p.frameCount / 30) % 2;
  if (blink) {
    p.text("press enter to begin", CANVAS_WIDTH / 2, 60);
  }
  
  // Subtitle removed as per instructions
  
  // Instructions
  p.fill(220, 220, 240);
  p.textSize(10);
  p.textAlign(p.LEFT, p.TOP);
  const instructions = [
    "OBJECTIVE: Escape the haunted mansion",
    "- Navigate through 6 challenging levels",
    "- Collect required objectives (★) to unlock exit",
    "- Find keys to unlock doors",
    "- Avoid the blue monster", // Modified to remove "the monster"
    "- Hide when spotted to evade capture",
    "",
    "CONTROLS:",
    "Arrow Keys - Move",
    "Shift - Sprint",
    "Space - Interact (doors, items, hide)",
    "Z - Toggle flashlight (in dark levels)",
    "ESC - Pause game",
    "R - Restart to title"
  ];
  
  let y = 110; // Adjusted starting Y position for instructions
  for (const line of instructions) {
    p.text(line, 70, y);
    y += 14;
  }
  
  // Original "Press Enter prompt" removed as its functionality is now handled by the main title
}

export function renderGameOverScreen(p, won) {
  p.background(20, 15, 30);
  
  if (won) {
    p.fill(100, 255, 100);
    p.textAlign(p.CENTER, p.CENTER);
    p.textSize(32);
    p.text("YOU ESCAPED!", CANVAS_WIDTH / 2, 100);
    
    p.fill(200, 255, 200);
    p.textSize(16);
    p.text("Congratulations on escaping the mansion!", CANVAS_WIDTH / 2, 140);
  } else {
    p.fill(255, 100, 100);
    p.textAlign(p.CENTER, p.CENTER);
    p.textSize(32);
    p.text("GAME OVER", CANVAS_WIDTH / 2, 100);
    
    p.fill(255, 200, 200);
    p.textSize(16);
    p.text("The monster caught you...", CANVAS_WIDTH / 2, 140);
    
    // Add dramatic effect
    p.fill(255, 0, 0, 50);
    p.noStroke();
    p.rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  }
  
  // Final score
  p.fill(255, 255, 255);
  p.textSize(20);
  p.text(`Final Score: ${gameState.score}`, CANVAS_WIDTH / 2, 190);
  
  // Instructions
  p.fill(220, 220, 240);
  p.textSize(14);
  p.text("Press R to restart", CANVAS_WIDTH / 2, 250);
  
  const blink = Math.floor(p.frameCount / 30) % 2;
  if (blink) {
    p.text("▼", CANVAS_WIDTH / 2, 280);
  }
}

export function renderUI(p) {
  // Top UI bar - darker background
  p.push();
  p.fill(0, 0, 0, 180);
  p.noStroke();
  p.rect(0, 0, CANVAS_WIDTH, 28);
  p.pop();
  
  // Score - top right
  p.push();
  p.fill(255, 255, 100);
  p.textAlign(p.RIGHT, p.TOP);
  p.textSize(14);
  p.text(`Score: ${gameState.score}`, CANVAS_WIDTH - 8, 7);
  p.pop();
  
  // Level - top left
  p.push();
  p.fill(100, 200, 255);
  p.textAlign(p.LEFT, p.TOP);
  p.textSize(14);
  p.text(`Level: ${gameState.level} / 6`, 8, 7);
  p.pop();
  
  // Objectives collected indicator - top center
  if (gameState.requiredItemIds.length > 0) {
    p.push();
    const collected = gameState.collectedRequiredItems.size;
    const total = gameState.requiredItemIds.length;
    const isComplete = collected >= total;
    
    p.fill(isComplete ? 100 : 255, isComplete ? 255 : 150, isComplete ? 100 : 150);
    p.textAlign(p.CENTER, p.TOP);
    p.textSize(12);
    p.text(`Objectives: ${collected}/${total} ${isComplete ? '✓' : ''}`, CANVAS_WIDTH / 2, 8);
    p.pop();
  }
  
  // Bottom UI bar
  p.push();
  p.fill(0, 0, 0, 180);
  p.noStroke();
  p.rect(0, CANVAS_HEIGHT - 45, CANVAS_WIDTH, 45);
  p.pop();
  
  // Inventory header
  p.push();
  p.fill(255, 255, 255);
  p.textAlign(p.LEFT, p.TOP);
  p.textSize(12);
  p.text("Inventory:", 8, CANVAS_HEIGHT - 38);
  p.pop();
  
  // Inventory items
  p.push();
  let x = 8;
  const y = CANVAS_HEIGHT - 20;
  
  if (gameState.inventory && gameState.inventory.length > 0) {
    for (const item of gameState.inventory) {
      // Item box
      p.fill(50, 50, 60);
      p.stroke(200, 200, 220);
      p.strokeWeight(2);
      p.rect(x, y - 16, 20, 20, 2);
      
      // Item icon
      p.fill(255, 215, 0);
      p.noStroke();
      p.textAlign(p.CENTER, p.CENTER);
      p.textSize(12);
      if (item.type === "key") {
        p.text("🔑", x + 10, y - 6);
      } else if (item.type === "flashlight") {
        p.text("🔦", x + 10, y - 6);
      } else {
        p.text("📦", x + 10, y - 6);
      }
      
      x += 26;
    }
  } else {
    p.fill(150, 150, 150);
    p.noStroke();
    p.textAlign(p.LEFT, p.CENTER);
    p.textSize(10);
    p.text("(empty)", x, y - 6);
  }
  p.pop();
  
  // Sprint indicator - bottom right
  if (gameState.isRunning) {
    p.push();
    p.fill(100, 255, 100);
    p.textAlign(p.RIGHT, p.TOP);
    p.textSize(11);
    p.text("SPRINTING", CANVAS_WIDTH - 8, CANVAS_HEIGHT - 38);
    p.pop();
  }
  
  // Chase indicator - flashing border effect
  if (gameState.inChase) {
    p.push();
    const alpha = 50 + Math.sin(p.frameCount * 0.2) * 30;
    p.fill(255, 0, 0, alpha);
    p.noStroke();
    p.rect(0, 28, CANVAS_WIDTH, 8);
    p.rect(0, 28, 8, CANVAS_HEIGHT - 73);
    p.rect(CANVAS_WIDTH - 8, 28, 8, CANVAS_HEIGHT - 73);
    p.rect(0, CANVAS_HEIGHT - 45 - 8, CANVAS_WIDTH, 8);
    
    p.fill(255, 100, 100);
    p.textAlign(p.CENTER, p.TOP);
    p.textSize(12);
    p.text("! CHASE !", CANVAS_WIDTH / 2, 32);
    p.pop();
  }
}

export function renderMessages(p) {
  if (!gameState.messages || gameState.messages.length === 0) return;
  
  p.push();
  let y = 50;
  
  for (const msg of gameState.messages) {
    const alpha = Math.min(255, msg.time * 2);
    
    // Message background
    p.fill(0, 0, 0, alpha * 0.7);
    p.noStroke();
    const textW = p.textWidth(msg.text);
    p.rect(CANVAS_WIDTH / 2 - textW / 2 - 10, y - 10, textW + 20, 22, 5);
    
    // Message text
    let color;
    if (msg.type === "success") {
      color = [100, 255, 100, alpha];
    } else if (msg.type === "warning") {
      color = [255, 200, 100, alpha];
    } else if (msg.type === "error") {
      color = [255, 100, 100, alpha];
    } else {
      color = [255, 255, 255, alpha];
    }
    
    p.fill(...color);
    p.textAlign(p.CENTER, p.TOP);
    p.textSize(12);
    p.text(msg.text, CANVAS_WIDTH / 2, y - 5);
    
    y += 26;
  }
  
  p.pop();
}

export function renderDarkness(p) {
  if (gameState.level >= 7 && gameState.player) {
    p.push();
    
    // Create darkness overlay
    p.fill(0, 0, 0, 200);
    p.noStroke();
    p.rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Player light
    const lightRadius = gameState.flashlightOn ? 100 : 65;
    const gradient = p.drawingContext.createRadialGradient(
      gameState.player.x, gameState.player.y, 0,
      gameState.player.x, gameState.player.y, lightRadius
    );
    gradient.addColorStop(0, 'rgba(255, 255, 200, 0.8)');
    gradient.addColorStop(0.5, 'rgba(255, 255, 200, 0.3)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    
    p.drawingContext.globalCompositeOperation = 'destination-out';
    p.drawingContext.fillStyle = gradient;
    p.drawingContext.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    p.drawingContext.globalCompositeOperation = 'source-over';
    
    p.pop();
  }
}