import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT, WEAPONS, ROBOT_MASTERS, GAME_PHASES } from './globals.js';

export function drawGame(p) {
  // Draw stage
  if (gameState.currentStage) {
    gameState.currentStage.draw(p);
  }

  // Draw entities
  for (let entity of gameState.entities) {
    entity.draw(p);
  }

  // Draw projectiles
  for (let proj of gameState.projectiles) {
    const screenX = proj.x - gameState.camera.x;
    const screenY = proj.y - gameState.camera.y;
    
    p.push();
    p.fill(...proj.color);
    p.noStroke();
    
    if (proj.weapon === 'HYPER_BOMB') {
      p.ellipse(screenX, screenY, 8, 8);
    } else if (proj.weapon === 'FIRE_STORM') {
      p.fill(255, 150, 50);
      p.ellipse(screenX, screenY, 10, 10);
      p.fill(255, 50, 50);
      p.ellipse(screenX, screenY, 6, 6);
    } else if (proj.weapon === 'ICE_SLASHER') {
      p.rect(screenX - 4, screenY - 2, 8, 4);
    } else {
      p.ellipse(screenX, screenY, 6, 6);
    }
    p.pop();
  }

  // Draw drops
  for (let drop of gameState.drops) {
    const screenX = drop.x - gameState.camera.x;
    const screenY = drop.y - gameState.camera.y;
    
    p.push();
    if (drop.type === 'weapon_energy') {
      p.fill(255, 200, 100);
      p.rect(screenX, screenY, 10, 10);
      p.fill(255, 255, 150);
      p.rect(screenX + 2, screenY + 2, 6, 6);
    }
    p.pop();
  }

  // Draw particles
  for (let particle of gameState.particles) {
    const screenX = particle.x - gameState.camera.x;
    const screenY = particle.y - gameState.camera.y;
    
    p.push();
    p.fill(...particle.color, particle.alpha);
    p.noStroke();
    p.ellipse(screenX, screenY, 4, 4);
    p.pop();
  }

  // Draw UI
  drawUI(p);
}

export function drawUI(p) {
  const uiColor = [255, 255, 255];
  
  // Level indicator
  p.push();
  p.fill(0, 0, 0, 180);
  p.rect(CANVAS_WIDTH / 2 - 60, 10, 120, 25);
  
  p.fill(100, 200, 255);
  p.textSize(14);
  p.textAlign(p.CENTER, p.TOP);
  p.text(`LEVEL ${gameState.currentLevel}/${gameState.totalLevels}`, CANVAS_WIDTH / 2, 15);
  p.pop();
  
  // Health bar
  p.push();
  p.fill(0, 0, 0, 180);
  p.rect(10, 45, 150, 35);
  
  p.fill(...uiColor);
  p.textSize(12);
  p.textAlign(p.LEFT, p.TOP);
  p.text('ENERGY', 15, 47);
  
  // Health blocks
  const healthBlocks = 28;
  const blockWidth = 4;
  const blockHeight = 12;
  for (let i = 0; i < healthBlocks; i++) {
    if (i < gameState.playerHealth) {
      p.fill(255, 200, 50);
    } else {
      p.fill(50, 50, 50);
    }
    p.rect(15 + i * blockWidth, 63, blockWidth - 1, blockHeight);
  }
  p.pop();

  // Weapon display
  p.push();
  p.fill(0, 0, 0, 180);
  p.rect(10, 90, 150, 60);
  
  const weaponKey = gameState.unlockedWeapons[gameState.currentWeapon];
  const weapon = WEAPONS[weaponKey];
  
  p.fill(...weapon.color);
  p.textSize(10);
  p.textAlign(p.LEFT, p.TOP);
  p.text(weapon.name, 15, 93);
  
  // Weapon energy
  if (weaponKey !== 'BUSTER') {
    const energy = gameState.weaponEnergy[weaponKey] || 0;
    for (let i = 0; i < 28; i++) {
      if (i < energy) {
        p.fill(...weapon.color);
      } else {
        p.fill(50, 50, 50);
      }
      p.rect(15 + i * 4, 110, 3, 8);
    }
    
    p.fill(255, 255, 255);
    p.textSize(10);
    p.text(`${energy}/28`, 15, 125);
  } else {
    p.fill(255, 255, 255);
    p.textSize(10);
    p.text('UNLIMITED', 15, 125);
  }
  
  p.fill(200, 200, 200);
  p.textSize(8);
  p.text('SHIFT: Switch Weapon', 15, 138);
  p.pop();

  // Boss health bar
  if (gameState.showBossHealthBar && gameState.currentStage && gameState.currentStage.boss) {
    p.push();
    p.fill(0, 0, 0, 180);
    p.rect(CANVAS_WIDTH - 160, 45, 150, 35);
    
    p.fill(255, 255, 255);
    p.textSize(10);
    p.textAlign(p.LEFT, p.TOP);
    p.text('BOSS', CANVAS_WIDTH - 155, 47);
    
    const boss = gameState.currentStage.boss;
    const maxBlocks = 28;
    for (let i = 0; i < maxBlocks; i++) {
      if (i < boss.health) {
        p.fill(255, 50, 50);
      } else {
        p.fill(50, 50, 50);
      }
      p.rect(CANVAS_WIDTH - 155 + i * 4, 63, 3, 12);
    }
    p.pop();
  }

  // Score and Lives
  p.push();
  p.fill(0, 0, 0, 180);
  p.rect(10, CANVAS_HEIGHT - 45, 120, 35);
  
  p.fill(255, 255, 255);
  p.textSize(12);
  p.textAlign(p.LEFT, p.TOP);
  p.text(`SCORE: ${gameState.score}`, 15, CANVAS_HEIGHT - 42);
  p.text(`LIVES: ${gameState.lives}`, 15, CANVAS_HEIGHT - 25);
  p.pop();
}

export function drawStartScreen(p) {
  p.background(20, 30, 50);
  
  // Animated background
  for (let i = 0; i < 10; i++) {
    p.fill(50, 80, 120, 100);
    const offset = (p.frameCount * 0.5) % 400;
    p.rect(i * 100 - offset, 0, 80, CANVAS_HEIGHT);
  }

  // Replaced game title and description with "press enter to begin"
  p.push();
  const flashAlpha = (Math.sin(p.frameCount * 0.1) + 1) * 127.5;
  p.fill(100, 255, 100, flashAlpha);
  p.textSize(36); // Made text larger to act as a title
  p.textAlign(p.CENTER, p.CENTER);
  p.text('press enter to begin', CANVAS_WIDTH / 2, 100); // Centered at former title position
  p.pop();

  // Controls - Kept and repositioned slightly higher
  p.push();
  p.fill(0, 0, 0, 180);
  p.rect(CANVAS_WIDTH / 2 - 140, 200, 280, 110); // Moved up from y=230
  
  p.fill(255, 255, 100);
  p.textSize(14);
  p.textAlign(p.LEFT, p.TOP);
  p.text('CONTROLS:', CANVAS_WIDTH / 2 - 130, 210); // Adjusted Y
  
  p.fill(200, 200, 200);
  p.textSize(11);
  p.text('Arrow Keys: Move & Aim', CANVAS_WIDTH / 2 - 130, 230); // Adjusted Y
  p.text('Z: Jump', CANVAS_WIDTH / 2 - 130, 245); // Adjusted Y
  p.text('Space: Shoot', CANVAS_WIDTH / 2 - 130, 260); // Adjusted Y
  p.text('Shift: Cycle Weapons', CANVAS_WIDTH / 2 - 130, 275); // Adjusted Y
  p.text('ESC: Pause    R: Restart', CANVAS_WIDTH / 2 - 130, 290); // Adjusted Y
  p.pop();
}

export function drawGameOver(p, won) {
  p.background(won ? [30, 60, 30] : [60, 30, 30]);
  
  // Title
  p.push();
  p.fill(won ? [100, 255, 100] : [255, 100, 100]);
  p.textSize(48);
  p.textAlign(p.CENTER, p.CENTER);
  p.text(won ? 'VICTORY!' : 'GAME OVER', CANVAS_WIDTH / 2, 120);
  
  // Message
  p.fill(255, 255, 255);
  p.textSize(16);
  if (won) {
    p.text('All Levels Completed!', CANVAS_WIDTH / 2, 180);
    p.text('You are the ultimate champion!', CANVAS_WIDTH / 2, 205);
  } else {
    p.text('The battle is lost...', CANVAS_WIDTH / 2, 180);
    p.text('But the fight continues!', CANVAS_WIDTH / 2, 205);
  }
  
  // Score
  p.fill(255, 255, 100);
  p.textSize(20);
  p.text(`FINAL SCORE: ${gameState.score}`, CANVAS_WIDTH / 2, 250);
  p.text(`Levels Completed: ${gameState.currentLevel - 1}/${gameState.totalLevels}`, CANVAS_WIDTH / 2, 280);
  
  // Restart prompt
  const flashAlpha = (Math.sin(p.frameCount * 0.1) + 1) * 127.5;
  p.fill(200, 200, 200, flashAlpha);
  p.textSize(16);
  p.text('PRESS R TO RESTART', CANVAS_WIDTH / 2, 330);
  p.pop();
}