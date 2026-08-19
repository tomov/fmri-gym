import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT, GROUND_Y, SLINGSHOT_X, SLINGSHOT_Y, BIRD_TYPES, LAUNCH_POWER_MULTIPLIER, GRAVITY, BIRD_AIR_FRICTION, MAX_PULL_DISTANCE } from './globals.js';

export function renderGame(p) {
  // Background
  renderBackground(p);

  if (gameState.gamePhase === "START") {
    renderStartScreen(p);
  } else if (gameState.gamePhase === "PLAYING" || gameState.gamePhase === "PAUSED") {
    renderGameplay(p);
  } else if (gameState.gamePhase === "LEVEL_COMPLETE") {
    renderGameplay(p);
    renderLevelCompleteScreen(p);
  } else if (gameState.gamePhase === "GAME_OVER_WIN" || gameState.gamePhase === "GAME_OVER_LOSE") {
    renderGameplay(p);
    renderGameOverScreen(p);
  }
}

function renderBackground(p) {
  // Sky gradient
  for (let y = 0; y < GROUND_Y; y++) {
    const inter = y / GROUND_Y;
    const c = p.lerpColor(p.color(135, 206, 235), p.color(100, 150, 200), inter);
    p.stroke(c);
    p.line(0, y, CANVAS_WIDTH, y);
  }
  
  // Ground
  p.fill(100, 200, 100);
  p.noStroke();
  p.rect(0, GROUND_Y, CANVAS_WIDTH, CANVAS_HEIGHT - GROUND_Y);
}

function renderStartScreen(p) {
  p.fill(0, 0, 0, 180);
  p.rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  
  p.fill(100, 255, 100);
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(24); // Slightly smaller font for the new title
  p.text("press enter to begin", CANVAS_WIDTH / 2, 60); // Replaced and moved to top
  
  // Adjusted Y positions for controls and bird abilities
  p.textSize(16);
  p.fill(255, 255, 150);
  p.text("CONTROLS:", CANVAS_WIDTH / 2, 120); // Moved up
  p.fill(255);
  p.textSize(12);
  p.text("ARROW KEYS: Hold to adjust slingshot aim", CANVAS_WIDTH / 2, 140); // Moved up
  p.text("SPACE: Launch the bird", CANVAS_WIDTH / 2, 160); // Moved up
  p.text("Z: Activate bird's special ability", CANVAS_WIDTH / 2, 180); // Moved up
  p.text("R: Restart", CANVAS_WIDTH / 2, 200); // Moved up
  
  p.textSize(14);
  p.fill(255, 200, 100);
  p.text("BIRD ABILITIES:", CANVAS_WIDTH / 2, 225); // Moved up
  p.fill(255);
  p.textSize(10);
  p.text("Red: Basic • Blue: Split x3 • Yellow: Speed boost", CANVAS_WIDTH / 2, 245); // Moved up
  p.text("Black: Heavy • Green: Explosion • White: Egg bomb", CANVAS_WIDTH / 2, 260); // Moved up
  
  if (gameState.highScore > 0) {
    p.fill(255, 215, 0);
    p.textSize(14);
    p.text(`HIGH SCORE: ${gameState.highScore}`, CANVAS_WIDTH / 2, 360); // Kept at original Y
  }
  
  // Removed "PRESS ENTER TO START" as it was replaced by the new title message
}

function renderGameplay(p) {
  // Slingshot
  renderSlingshot(p);
  
  // Entities
  gameState.entities.forEach(entity => {
    if (entity.type === 'bird') {
      renderBird(p, entity);
    } else if (entity.type === 'pig') {
      renderPig(p, entity);
    } else if (entity.type === 'block') {
      renderStructureBlock(p, entity);
    } else if (entity.type === 'egg') {
      renderEgg(p, entity);
    }
  });
  
  // Particle effects
  renderParticles(p);
  
  // UI
  renderUI(p);
  
  // Trajectory preview when aiming
  if (gameState.isAiming) {
    renderTrajectoryPreview(p);
    renderAimingInfo(p);
  }
}

function renderSlingshot(p) {
  const baseX = SLINGSHOT_X;
  const baseY = SLINGSHOT_Y;
  
  // Base
  p.fill(80, 50, 30);
  p.noStroke();
  p.rect(baseX - 8, baseY, 16, 30);
  
  // Rubber bands
  if (gameState.isAiming) {
    const pullX = baseX + gameState.slingshotPullPos.x;
    const pullY = baseY + gameState.slingshotPullPos.y;
    
    p.stroke(60, 40, 20);
    p.strokeWeight(3);
    p.line(baseX - 6, baseY, pullX, pullY);
    p.line(baseX + 6, baseY, pullX, pullY);
    
    // Bird at pull position
    const birdType = gameState.birdsRemaining[0];
    drawBirdIcon(p, pullX, pullY, birdType, 20);
  } else if (gameState.birdsRemaining.length > 0 && gameState.activeBirds.length === 0) {
    // Bird ready at slingshot
    const birdType = gameState.birdsRemaining[0];
    drawBirdIcon(p, baseX, baseY, birdType, 20);
  }
}

function renderBird(p, bird) {
  if (!bird.active || !bird.body) return;
  
  drawBirdIcon(p, bird.body.position.x, bird.body.position.y, bird.birdType, bird.size);
  
  // Trail for yellow bird during boost
  if (bird.birdType === BIRD_TYPES.YELLOW && bird.abilityUsed) {
    p.stroke(255, 255, 0, 100);
    p.strokeWeight(bird.size * 0.5);
    for (let i = 0; i < bird.trail.length - 1; i++) {
      const alpha = (i / bird.trail.length) * 100;
      p.stroke(255, 255, 0, alpha);
      p.line(bird.trail[i].x, bird.trail[i].y, bird.trail[i + 1].x, bird.trail[i + 1].y);
    }
  }
}

function drawBirdIcon(p, x, y, birdType, size) {
  p.noStroke();
  
  if (birdType === BIRD_TYPES.RED) {
    p.fill(220, 20, 20);
  } else if (birdType === BIRD_TYPES.BLUE) {
    p.fill(20, 100, 220);
  } else if (birdType === BIRD_TYPES.YELLOW) {
    p.fill(255, 220, 0);
  } else if (birdType === BIRD_TYPES.BLACK) {
    p.fill(30, 30, 30);
  } else if (birdType === BIRD_TYPES.GREEN) {
    p.fill(50, 220, 50);
  } else if (birdType === BIRD_TYPES.WHITE) {
    p.fill(255, 255, 255);
  }
  
  p.ellipse(x, y, size, size);
  
  // Eyes
  p.fill(0);
  p.ellipse(x + size * 0.15, y - size * 0.1, size * 0.2, size * 0.2);
  p.ellipse(x - size * 0.15, y - size * 0.1, size * 0.2, size * 0.2);
  
  // White part of eyes for white bird visibility
  if (birdType === BIRD_TYPES.WHITE) {
    p.fill(255);
    p.ellipse(x + size * 0.15, y - size * 0.1, size * 0.15, size * 0.15);
    p.ellipse(x - size * 0.15, y - size * 0.1, size * 0.15, size * 0.15);
    p.fill(0);
    p.ellipse(x + size * 0.15, y - size * 0.1, size * 0.08, size * 0.08);
    p.ellipse(x - size * 0.15, y - size * 0.1, size * 0.08, size * 0.08);
  }
  
  // Eyebrows for angry look
  p.stroke(0);
  p.strokeWeight(1.5);
  p.line(x - size * 0.25, y - size * 0.25, x - size * 0.1, y - size * 0.15);
  p.line(x + size * 0.25, y - size * 0.25, x + size * 0.1, y - size * 0.15);
}

function renderEgg(p, egg) {
  if (!egg.active || !egg.body) return;
  
  p.noStroke();
  p.fill(255, 255, 220);
  p.ellipse(egg.body.position.x, egg.body.position.y, egg.size, egg.size * 1.2);
  
  // Spots
  p.fill(220, 200, 150);
  p.ellipse(egg.body.position.x - 3, egg.body.position.y - 2, 4, 4);
  p.ellipse(egg.body.position.x + 2, egg.body.position.y + 3, 3, 3);
}

function renderPig(p, pig) {
  if (!pig.active || !pig.body) return;
  
  // Body
  p.fill(100, 200, 100);
  p.noStroke();
  p.ellipse(pig.body.position.x, pig.body.position.y, pig.size, pig.size);
  
  // Eyes
  p.fill(255);
  const eyeOffset = pig.size * 0.2;
  p.ellipse(pig.body.position.x - eyeOffset, pig.body.position.y - eyeOffset * 0.5, pig.size * 0.25, pig.size * 0.25);
  p.ellipse(pig.body.position.x + eyeOffset, pig.body.position.y - eyeOffset * 0.5, pig.size * 0.25, pig.size * 0.25);
  
  p.fill(0);
  p.ellipse(pig.body.position.x - eyeOffset, pig.body.position.y - eyeOffset * 0.5, pig.size * 0.15, pig.size * 0.15);
  p.ellipse(pig.body.position.x + eyeOffset, pig.body.position.y - eyeOffset * 0.5, pig.size * 0.15, pig.size * 0.15);
  
  // Snout
  p.fill(80, 160, 80);
  p.ellipse(pig.body.position.x, pig.body.position.y + eyeOffset, pig.size * 0.4, pig.size * 0.3);
  p.fill(60, 140, 60);
  p.ellipse(pig.body.position.x - pig.size * 0.1, pig.body.position.y + eyeOffset, pig.size * 0.1, pig.size * 0.1);
  p.ellipse(pig.body.position.x + pig.size * 0.1, pig.body.position.y + eyeOffset, pig.size * 0.1, pig.size * 0.1);
  
  // Health bar
  renderPigHealthBar(p, pig);
}

function renderPigHealthBar(p, pig) {
  const barWidth = pig.size * 1.2;
  const barHeight = 4;
  const barX = pig.body.position.x - barWidth / 2;
  const barY = pig.body.position.y - pig.size / 2 - 8;
  
  const maxHealth = pig.isLarge ? 2 : 1;
  const healthPercent = pig.health / maxHealth;
  
  // Background (red)
  p.fill(200, 50, 50);
  p.noStroke();
  p.rect(barX, barY, barWidth, barHeight);
  
  // Health (green)
  p.fill(50, 200, 50);
  p.rect(barX, barY, barWidth * healthPercent, barHeight);
  
  // Border
  p.noFill();
  p.stroke(0);
  p.strokeWeight(1);
  p.rect(barX, barY, barWidth, barHeight);
}

function renderStructureBlock(p, block) {
  if (!block.active || !block.body) return;
  
  p.push();
  p.translate(block.body.position.x, block.body.position.y);
  p.rotate(block.body.angle);
  
  if (block.material === 'WOOD') {
    p.fill(139, 90, 43);
    p.stroke(100, 60, 30);
  } else if (block.material === 'STONE') {
    p.fill(120, 120, 120);
    p.stroke(80, 80, 80);
  }
  
  p.strokeWeight(2);
  p.rectMode(p.CENTER);
  p.rect(0, 0, block.width, block.height);
  
  // Texture lines
  p.stroke(block.material === 'WOOD' ? 100 : 90, block.material === 'WOOD' ? 60 : 90, block.material === 'WOOD' ? 30 : 90, 100);
  p.strokeWeight(1);
  if (block.material === 'WOOD') {
    for (let i = -block.width / 2; i < block.width / 2; i += 5) {
      p.line(i, -block.height / 2, i, block.height / 2);
    }
  }
  
  p.pop();
  
  // Health bar (if damaged)
  if (block.health < block.maxHealth) {
    renderBlockHealthBar(p, block);
  }
}

function renderBlockHealthBar(p, block) {
  const barWidth = Math.max(block.width, 30);
  const barHeight = 3;
  const barX = block.body.position.x - barWidth / 2;
  const barY = block.body.position.y - Math.max(block.height, block.width) / 2 - 6;
  
  const healthPercent = block.health / block.maxHealth;
  
  // Background (red)
  p.fill(200, 50, 50);
  p.noStroke();
  p.rect(barX, barY, barWidth, barHeight);
  
  // Health (yellow/orange for blocks)
  p.fill(255, 180, 0);
  p.rect(barX, barY, barWidth * healthPercent, barHeight);
  
  // Border
  p.noFill();
  p.stroke(0);
  p.strokeWeight(1);
  p.rect(barX, barY, barWidth, barHeight);
}

function renderParticles(p) {
  p.noStroke();
  gameState.particleEffects.forEach(particle => {
    const alpha = (particle.life / particle.maxLife) * 200;
    p.fill(...particle.color, alpha);
    p.ellipse(particle.x, particle.y, particle.size, particle.size);
  });
}

function renderUI(p) {
  p.fill(255);
  p.textAlign(p.LEFT, p.TOP);
  p.textSize(16);
  p.text(`LEVEL: ${gameState.currentLevel}`, 10, 10);
  p.text(`SCORE: ${gameState.score}`, 10, 30);
  
  // Birds remaining
  p.textSize(14);
  p.text(`BIRDS: ${gameState.birdsRemaining.length}`, 10, 55);
  
  // Bird icons
  let iconX = 80;
  for (let i = 0; i < Math.min(3, gameState.birdsRemaining.length); i++) {
    drawBirdIcon(p, iconX, 62, gameState.birdsRemaining[i], 12);
    iconX += 18;
  }
  if (gameState.birdsRemaining.length > 3) {
    p.fill(255);
    p.textSize(12);
    p.text(`+${gameState.birdsRemaining.length - 3}`, iconX, 57);
  }
  
  // Pigs remaining
  p.fill(255);
  p.textSize(14);
  p.text(`PIGS: ${gameState.pigsRemaining}`, 10, 80);
}

function renderAimingInfo(p) {
  if (!gameState.slingshotPullPos) return;
  
  const pullX = gameState.slingshotPullPos.x;
  const pullY = gameState.slingshotPullPos.y;
  
  // Calculate launch velocity (opposite direction of pull)
  const vx = -pullX * LAUNCH_POWER_MULTIPLIER;
  const vy = -pullY * LAUNCH_POWER_MULTIPLIER;
  
  // Calculate angle in degrees (0 degrees = right, 90 degrees = up)
  const angleRad = Math.atan2(-vy, vx); // negative vy because y-axis is inverted in screen coords
  let angleDeg = angleRad * (180 / Math.PI);
  // Normalize to 0-360 range
  if (angleDeg < 0) angleDeg += 360;
  
  // Calculate power as percentage of max pull distance
  const pullDistance = Math.sqrt(pullX * pullX + pullY * pullY);
  const powerPercent = Math.round((pullDistance / MAX_PULL_DISTANCE) * 100);
  
  // Display info in top-right corner to avoid obstructing trajectory
  const boxWidth = 110;
  const boxHeight = 40;
  const displayX = CANVAS_WIDTH - boxWidth - 10;
  const displayY = 10;
  
  // Background box
  p.fill(0, 0, 0, 180);
  p.noStroke();
  p.rect(displayX, displayY, boxWidth, boxHeight, 5);
  
  // Text
  p.fill(255, 255, 100);
  p.textAlign(p.LEFT, p.TOP);
  p.textSize(12);
  p.text(`Angle: ${Math.round(angleDeg)}°`, displayX + 5, displayY + 5);
  p.text(`Power: ${powerPercent}%`, displayX + 5, displayY + 23);
}

function renderTrajectoryPreview(p) {
  if (!gameState.slingshotPullPos) return;
  
  const Matter = window.Matter;
  if (!Matter) return; // Matter.js not loaded yet
  
  const Engine = Matter.Engine;
  const World = Matter.World;
  const Bodies = Matter.Bodies;
  const Body = Matter.Body;
  
  const startX = SLINGSHOT_X;
  const startY = SLINGSHOT_Y;
  
  // Initial velocity using the same multiplier as actual launch
  const vx = -gameState.slingshotPullPos.x * LAUNCH_POWER_MULTIPLIER;
  const vy = -gameState.slingshotPullPos.y * LAUNCH_POWER_MULTIPLIER;
  
  // Create a temporary Matter.js engine for trajectory simulation
  const tempEngine = Engine.create({
    gravity: { x: 0, y: GRAVITY }
  });
  
  // Create a temporary bird body (small circle, same properties as actual bird)
  const tempBird = Bodies.circle(startX, startY, 10, {
    frictionAir: BIRD_AIR_FRICTION,
    density: 0.001
  });
  
  // Set initial velocity
  Body.setVelocity(tempBird, { x: vx, y: vy });
  
  // Create temporary ground
  const tempGround = Bodies.rectangle(CANVAS_WIDTH / 2, GROUND_Y + 25, CANVAS_WIDTH, 50, {
    isStatic: true
  });
  
  // Add to temporary world
  World.add(tempEngine.world, [tempBird, tempGround]);
  
  // Simulate trajectory by running physics updates
  const trajectoryPoints = [];
  const dotSpacing = 4;
  
  // Show only 1/2 of full trajectory (40 frames instead of 80)
  for (let frame = 0; frame < 40; frame++) {
    // Update physics (60fps = 1000/60 ms per frame)
    Engine.update(tempEngine, 1000 / 60);
    
    // Record position at intervals
    if (frame % dotSpacing === 0) {
      trajectoryPoints.push({
        x: tempBird.position.x,
        y: tempBird.position.y
      });
    }
    
    // Stop if bird hits ground or goes off screen
    if (tempBird.position.y > GROUND_Y || 
        tempBird.position.x < 0 || 
        tempBird.position.x > CANVAS_WIDTH || 
        tempBird.position.y < 0) {
      break;
    }
  }
  
  // Clean up temporary engine
  World.clear(tempEngine.world, false);
  Engine.clear(tempEngine);
  
  // Draw trajectory points
  p.noStroke();
  p.fill(255, 255, 255, 180);
  
  trajectoryPoints.forEach(point => {
    p.ellipse(point.x, point.y, 4, 4);
  });
}

function renderLevelCompleteScreen(p) {
  p.fill(0, 0, 0, 200);
  p.rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  
  p.fill(100, 255, 100);
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(40);
  p.text(`LEVEL ${gameState.currentLevel} COMPLETE!`, CANVAS_WIDTH / 2, 120);
  
  p.fill(255);
  p.textSize(20);
  p.text(`Level Score: ${gameState.score - gameState.levelStartScore}`, CANVAS_WIDTH / 2, 180);
  p.text(`Total Score: ${gameState.score}`, CANVAS_WIDTH / 2, 210);
  
  if (gameState.currentLevel < gameState.totalLevels) {
    p.fill(255, 255, 100);
    p.textSize(18);
    p.text("PRESS ENTER FOR NEXT LEVEL", CANVAS_WIDTH / 2, 280);
  } else {
    p.fill(255, 215, 0);
    p.textSize(32);
    p.text("YOU WIN!", CANVAS_WIDTH / 2, 260);
    p.fill(255, 255, 100);
    p.textSize(18);
    p.text("PRESS R TO RESTART", CANVAS_WIDTH / 2, 310);
  }
}

function renderGameOverScreen(p) {
  p.fill(0, 0, 0, 200);
  p.rect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  
  if (gameState.gamePhase === "GAME_OVER_WIN") {
    p.fill(100, 255, 100);
    p.textAlign(p.CENTER, p.CENTER);
    p.textSize(48);
    p.text("YOU WIN!", CANVAS_WIDTH / 2, 120);
  } else {
    p.fill(255, 100, 100);
    p.textAlign(p.CENTER, p.CENTER);
    p.textSize(48);
    p.text("GAME OVER", CANVAS_WIDTH / 2, 120);
  }
  
  p.fill(255);
  p.textSize(20);
  p.text(`Final Score: ${gameState.score}`, CANVAS_WIDTH / 2, 180);
  p.text(`High Score: ${gameState.highScore}`, CANVAS_WIDTH / 2, 210);
  
  p.fill(255, 255, 100);
  p.textSize(18);
  p.text("PRESS R TO RESTART", CANVAS_WIDTH / 2, 280);
}
