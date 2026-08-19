import { gameState, GAME_PHASES, CANVAS_WIDTH, CANVAS_HEIGHT } from './globals.js';

export function renderGame(p) {
  p.background(30, 25, 35);

  // Check level transition first, before game phase
  if (gameState.levelTransitionTimer > 0) {
    renderLevelTransition(p);
  } else if (gameState.gamePhase === GAME_PHASES.START) {
    renderStartScreen(p);
  } else if (gameState.gamePhase === GAME_PHASES.PLAYING || gameState.gamePhase === GAME_PHASES.PAUSED) {
    renderPlaying(p);
  } else if (gameState.gamePhase === GAME_PHASES.GAME_OVER_WIN) {
    renderGameOverWin(p);
  } else if (gameState.gamePhase === GAME_PHASES.GAME_OVER_LOSE) {
    renderGameOverLose(p);
  }
}

function renderStartScreen(p) {
  p.push();
  
  // Replaced Title with "press enter to begin" message
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(36); // Increased size for main title
  p.fill(150, 255, 150); // Brighter green for prominence
  const blink = Math.floor(p.frameCount / 30) % 2 === 0;
  if (blink) {
    p.text('press enter to begin', CANVAS_WIDTH / 2, 100); // New position for title message
  }
  
  // Description (preserved)
  p.textSize(14);
  p.fill(200, 200, 200);
  p.text('Collect all the cheese to activate the mouse hole,', CANVAS_WIDTH / 2, 170);
  p.text('then escape before the cats catch you!', CANVAS_WIDTH / 2, 190);
  
  // Instructions (preserved)
  p.textSize(16);
  p.fill(255, 255, 200);
  p.text('HOW TO PLAY', CANVAS_WIDTH / 2, 230);
  
  p.textSize(14);
  p.fill(220, 220, 220);
  p.textAlign(p.LEFT, p.CENTER);
  const instrX = 150;
  p.text('← →  Move Left/Right', instrX, 260);
  p.text('↑      Jump', instrX, 285);
  p.text('ESC   Pause Game', instrX, 310);
  p.text('R       Restart to Menu', instrX, 335);
  
  // High Score (preserved)
  p.textAlign(p.CENTER, p.CENTER);
  p.textSize(16);
  p.fill(255, 215, 0);
  p.text(`HIGH SCORE: ${gameState.highScore.toString().padStart(6, '0')}`, CANVAS_WIDTH / 2, 370);
  
  // Original "PRESS ENTER TO START" prompt removed as it's now the main title
  
  p.pop();
}

function renderPlaying(p) {
  // Render all entities
  for (const entity of gameState.entities) {
    entity.draw();
  }
  
  // Render invulnerability effect
  if (gameState.invulnerable && gameState.player) {
    const alpha = Math.floor((gameState.invulnerableTimer % 20) / 10) * 100;
    p.push();
    p.noFill();
    p.stroke(100, 200, 255, alpha);
    p.strokeWeight(3);
    p.ellipse(gameState.player.x, gameState.player.y, gameState.player.w + 10, gameState.player.h + 10);
    p.pop();
  }
  
  // Render UI
  renderUI(p);
}

function renderUI(p) {
  p.push();
  
  // Score - top left
  p.textAlign(p.LEFT, p.TOP);
  p.textSize(18);
  p.fill(255, 255, 255);
  p.text(`SCORE: ${gameState.score.toString().padStart(6, '0')}`, 10, 10);
  
  // Level - top right
  p.textAlign(p.RIGHT, p.TOP);
  p.text(`LEVEL: ${gameState.level}`, CANVAS_WIDTH - 10, 10);
  
  // Lives - bottom left
  p.textAlign(p.LEFT, p.TOP);
  p.text('LIVES: ', 10, CANVAS_HEIGHT - 30);
  for (let i = 0; i < gameState.lives; i++) {
    p.fill(255, 50, 50);
    p.noStroke();
    p.beginShape();
    const x = 70 + i * 25;
    const y = CANVAS_HEIGHT - 20;
    p.vertex(x, y - 5);
    p.bezierVertex(x - 5, y - 10, x - 10, y - 5, x, y + 5);
    p.bezierVertex(x + 10, y - 5, x + 5, y - 10, x, y - 5);
    p.endShape(p.CLOSE);
  }
  
  // Cheese counter
  p.textAlign(p.RIGHT, p.TOP);
  p.fill(255, 255, 255);
  p.text(`CHEESE: ${gameState.cheeseCollected}/${gameState.totalCheese}`, CANVAS_WIDTH - 10, CANVAS_HEIGHT - 30);
  
  p.pop();
}

function renderGameOverWin(p) {
  p.push();
  
  p.textAlign(p.CENTER, p.CENTER);
  p.fill(100, 255, 100);
  p.textSize(48);
  p.text('YOU WIN!', CANVAS_WIDTH / 2, 100);
  
  p.textSize(24);
  p.fill(255, 255, 100);
  p.text('All Levels Complete!', CANVAS_WIDTH / 2, 150);
  
  p.textSize(20);
  p.fill(255, 255, 255);
  p.text(`Final Score: ${gameState.score.toString().padStart(6, '0')}`, CANVAS_WIDTH / 2, 210);
  
  if (gameState.score === gameState.highScore) {
    p.fill(255, 215, 0);
    p.textSize(18);
    p.text('NEW HIGH SCORE!', CANVAS_WIDTH / 2, 250);
  }
  
  p.textSize(18);
  p.fill(200, 200, 200);
  p.text('PRESS R TO RESTART', CANVAS_WIDTH / 2, 320);
  
  p.pop();
}

function renderGameOverLose(p) {
  p.push();
  
  p.textAlign(p.CENTER, p.CENTER);
  p.fill(255, 100, 100);
  p.textSize(48);
  p.text('GAME OVER', CANVAS_WIDTH / 2, 120);
  
  p.textSize(20);
  p.fill(255, 255, 255);
  p.text(`Final Score: ${gameState.score.toString().padStart(6, '0')}`, CANVAS_WIDTH / 2, 200);
  p.text(`Reached Level: ${gameState.level}`, CANVAS_WIDTH / 2, 230);
  
  p.textSize(18);
  p.fill(200, 200, 200);
  p.text('PRESS R TO RESTART', CANVAS_WIDTH / 2, 300);
  
  p.pop();
}

function renderLevelTransition(p) {
  p.push();
  
  const prevLevel = gameState.level - 1;
  
  p.background(30, 25, 35);
  p.textAlign(p.CENTER, p.CENTER);
  
  if (gameState.levelTransitionTimer > 120) {
    // First 1 second - show completion
    p.fill(100, 255, 100);
    p.textSize(36);
    p.text(`LEVEL ${prevLevel} COMPLETE!`, CANVAS_WIDTH / 2, 140);
    
    p.fill(255, 255, 255);
    p.textSize(20);
    p.text(`Level Bonus: +1000`, CANVAS_WIDTH / 2, 200);
    p.text(`Lives Bonus: +${(gameState.lives * 500)}`, CANVAS_WIDTH / 2, 230);
    p.text(`Total Score: ${gameState.score.toString().padStart(6, '0')}`, CANVAS_WIDTH / 2, 270);
  } else {
    // Next 2 seconds - get ready
    p.fill(255, 255, 100);
    p.textSize(36);
    p.text(`GET READY FOR`, CANVAS_WIDTH / 2, 160);
    p.text(`LEVEL ${gameState.level}`, CANVAS_WIDTH / 2, 210);
  }
  
  p.pop();
}