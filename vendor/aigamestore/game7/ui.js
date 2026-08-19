import { CANVAS_WIDTH, CANVAS_HEIGHT, gameState } from './globals.js';

export function drawUI(p) {
  p.push();
  
  if (gameState.player) {
    const healthPercentage = gameState.player.health / gameState.player.maxHealth;
    p.noStroke();
    p.fill(50);
    p.rect(20, 20, 150, 15);
    p.fill(255, 0, 0);
    p.rect(20, 20, 150 * healthPercentage, 15);
    p.fill(255);
    p.textSize(12);
    p.textAlign(p.CENTER, p.CENTER);
    p.text(`HEALTH: ${gameState.player.health}/${gameState.player.maxHealth}`, 95, 27);
    
    p.fill(50);
    p.rect(20, 45, 150, 15);
    p.fill(255, 200, 0);
    const ammoPercentage = gameState.player.ammo / gameState.player.maxAmmo;
    p.rect(20, 45, 150 * ammoPercentage, 15);
    p.fill(255);
    p.textAlign(p.CENTER, p.CENTER);
    // Display clip ammo and reserve ammo
    p.text(`AMMO: ${gameState.player.ammo} / ${gameState.player.reserveAmmo}`, 95, 52);
    
    if (gameState.player.reloading) {
      p.fill(255, 200, 0);
      p.textAlign(p.LEFT, p.CENTER);
      p.text("RELOADING...", 180, 52);
    } else if (gameState.player.ammo === 0 && gameState.player.reserveAmmo === 0) {
      p.fill(255, 50, 50);
      p.textAlign(p.LEFT, p.CENTER);
      p.text("NO AMMO!", 180, 52);
    }
    
    // Current weapon display
    p.fill(255);
    p.textAlign(p.LEFT, p.TOP);
    p.textSize(14);
    const weaponName = gameState.player.weapons[gameState.player.currentWeapon].name;
    p.text(`WEAPON: ${weaponName}`, 20, 70);
    
    // Available weapons (1-5 keys)
    p.textSize(10);
    p.textAlign(p.LEFT, p.TOP);
    let yPos = 90;
    if (gameState.player.weapons.pistol) {
      p.fill(gameState.player.currentWeapon === "pistol" ? 255 : 150);
      p.text("1: Pistol", 20, yPos);
      yPos += 15;
    }
    if (gameState.player.weapons.rifle) {
      p.fill(gameState.player.currentWeapon === "rifle" ? 255 : 150);
      p.text("2: Rifle", 20, yPos);
      yPos += 15;
    }
    if (gameState.player.weapons.shotgun) {
      p.fill(gameState.player.currentWeapon === "shotgun" ? 255 : 150);
      p.text("3: Shotgun", 20, yPos);
      yPos += 15;
    }
    if (gameState.player.weapons.sniper) {
      p.fill(gameState.player.currentWeapon === "sniper" ? 255 : 150);
      p.text("4: Sniper", 20, yPos);
      yPos += 15;
    }
    if (gameState.player.weapons.rocket_launcher) {
      p.fill(gameState.player.currentWeapon === "rocket_launcher" ? 255 : 150);
      p.text("5: RPG", 20, yPos);
    }
  }
  
  // Level indicator
  p.fill(255);
  p.textAlign(p.LEFT, p.TOP);
  p.textSize(16);
  p.text(`LEVEL ${gameState.currentLevel}`, 20, CANVAS_HEIGHT - 50);
  
  // Score/Kill counter
  p.fill(255);
  p.textAlign(p.RIGHT, p.TOP);
  p.textSize(14);
  p.text(`SCORE: ${gameState.score}`, CANVAS_WIDTH - 20, 20);
  
  // Mission objective - Updated to show both requirements
  p.textAlign(p.CENTER, p.TOP);
  p.text(`MISSION: ELIMINATE ${gameState.enemiesKilled}/${gameState.requiredKills} AND EXTRACT`, CANVAS_WIDTH / 2, 20);
  
  // Time elapsed
  const minutes = Math.floor(gameState.timeElapsed / 60);
  const seconds = Math.floor(gameState.timeElapsed % 60);
  p.textAlign(p.RIGHT, p.TOP);
  p.text(`TIME: ${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`, CANVAS_WIDTH - 20, 40);
  
  // Status indicators
  if (gameState.player) {
    p.textAlign(p.LEFT, p.BOTTOM);
    p.textSize(12);
    if (gameState.player.isSprinting) {
      p.fill(0, 200, 255);
      p.text("SPRINTING", 20, CANVAS_HEIGHT - 20);
    } else if (gameState.player.isCrouching) {
      p.fill(0, 255, 200);
      p.text("TAKING COVER", 20, CANVAS_HEIGHT - 20);
    }
  }
  
  p.pop();
}

export function drawStartScreen(p) {
  p.push();
  p.background(20);
  
  p.fill(255);
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(40);
  p.text("press enter to begin", CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 - 20);
  
  p.textSize(14);
  p.text("CONTROLS:", CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 50);
  p.text("Arrow Keys: Move | Z: Shoot | SPACE: Sprint | SHIFT: Take Cover", CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 70);
  p.text("1-5: Switch Weapons | ESC: Pause | R: Restart", CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 90);
  
  p.pop();
}

export function drawGameOverScreen(p, won) {
  p.push();
  p.fill(0, 150);
  p.rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  
  p.fill(255);
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(40);
  
  if (won) {
    p.fill(0, 255, 100);
    p.text("MISSION COMPLETE", CANVAS_WIDTH / 2, CANVAS_HEIGHT / 3);
    
    p.fill(255);
    p.textSize(24);
    p.text(`LEVEL ${gameState.currentLevel} CLEARED`, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 3 + 40);
  } else {
    p.fill(255, 50, 50);
    p.text("MISSION FAILED", CANVAS_WIDTH / 2, CANVAS_HEIGHT / 3);
  }
  
  p.fill(255);
  p.textSize(20);
  p.text(`SCORE: ${gameState.score}`, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2);
  
  const minutes = Math.floor(gameState.timeElapsed / 60);
  const seconds = Math.floor(gameState.timeElapsed % 60);
  p.text(`TIME: ${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 30);
  
  p.text(`ENEMIES ELIMINATED: ${gameState.enemiesKilled}`, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 60);
  
  p.textSize(20);
  if (Math.floor(p.frameCount / 30) % 2 === 0) {
    if (won) {
      p.text("PRESS ENTER FOR NEXT LEVEL", CANVAS_WIDTH / 2, CANVAS_HEIGHT * 3 / 4 - 20);
    }
    p.text("PRESS R TO RESTART", CANVAS_WIDTH / 2, CANVAS_HEIGHT * 3 / 4 + 10);
  }
  
  p.pop();
}