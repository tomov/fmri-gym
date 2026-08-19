import { gameState, GAME_PHASES } from './globals.js';
import { Particle } from './entities.js';

export function checkCollisions(p) {
  if (gameState.gamePhase !== GAME_PHASES.PLAYING) return;
  if (!gameState.player) return;

  const player = gameState.player;
  
  // Check cheese collection
  const cheeses = gameState.entities.filter(e => e.type === 'cheese' && !e.collected);
  for (const cheese of cheeses) {
    const dist = Math.sqrt(
      Math.pow(player.x - cheese.x, 2) + 
      Math.pow(player.y - cheese.y, 2)
    );
    if (dist < (player.w / 2 + cheese.w / 2)) {
      cheese.collected = true;
      gameState.cheeseCollected++;
      gameState.score += 50;
      
      // Create particles
      for (let i = 0; i < 10; i++) {
        const angle = (Math.PI * 2 * i) / 10;
        const speed = 2 + Math.random() * 2;
        gameState.entities.push(
          new Particle(
            p,
            cheese.x,
            cheese.y,
            Math.cos(angle) * speed,
            Math.sin(angle) * speed,
            [255, 220, 60],
            30
          )
        );
      }
      
      // Check if all cheese collected
      if (gameState.cheeseCollected >= gameState.totalCheese) {
        gameState.mouseHoleActive = true;
      }
    }
  }
  
  // Check cat collision
  if (!gameState.invulnerable) {
    const cats = gameState.entities.filter(e => e.type === 'cat');
    for (const cat of cats) {
      const dist = Math.sqrt(
        Math.pow(player.x - cat.x, 2) + 
        Math.pow(player.y - cat.y, 2)
      );
      if (dist < (player.w / 2 + cat.w / 2)) {
        handlePlayerHit(p);
        break;
      }
    }
  }
  
  // Check mouse hole collision
  if (gameState.mouseHoleActive) {
    const mouseHole = gameState.entities.find(e => e.type === 'mousehole');
    if (mouseHole) {
      const dist = Math.sqrt(
        Math.pow(player.x - mouseHole.x, 2) + 
        Math.pow(player.y - mouseHole.y, 2)
      );
      if (dist < (player.w / 2 + mouseHole.w / 2)) {
        handleLevelComplete(p);
      }
    }
  }
}

export function handlePlayerHit(p) {
  gameState.lives--;
  
  // Log player death
  p.logs.player_info.push({
    event: 'player_died',
    screen_x: gameState.player.x,
    screen_y: gameState.player.y,
    game_x: gameState.player.x,
    game_y: gameState.player.y,
    lives_remaining: gameState.lives,
    framecount: p.frameCount,
    timestamp: Date.now()
  });
  
  if (gameState.lives <= 0) {
    gameState.gamePhase = GAME_PHASES.GAME_OVER_LOSE;
    p.logs.game_info.push({
      event: 'game_over_lose',
      data: { finalScore: gameState.score, level: gameState.level },
      framecount: p.frameCount,
      timestamp: Date.now()
    });
  } else {
    // Respawn player
    respawnPlayer(p);
    gameState.invulnerable = true;
    gameState.invulnerableTimer = 120; // 2 seconds at 60 FPS
  }
}

function respawnPlayer(p) {
  const levelConfig = gameState.currentLevelConfig;
  if (levelConfig && gameState.player) {
    gameState.player.x = levelConfig.playerStart.x;
    gameState.player.y = levelConfig.playerStart.y;
    gameState.player.vx = 0;
    gameState.player.vy = 0;
    gameState.player.onGround = false;
  }
}

function handleLevelComplete(p) {
  // Calculate level completion bonus
  const levelBonus = 1000;
  const livesBonus = gameState.lives * 500;
  gameState.score += levelBonus + livesBonus;
  
  p.logs.game_info.push({
    event: 'level_complete',
    data: { 
      level: gameState.level, 
      score: gameState.score,
      level_bonus: levelBonus,
      lives_bonus: livesBonus
    },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
  
  if (gameState.level >= 6) {
    // Game complete!
    gameState.gamePhase = GAME_PHASES.GAME_OVER_WIN;
    
    // Update high score
    if (gameState.score > gameState.highScore) {
      gameState.highScore = gameState.score;
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem('game5HighScore', gameState.highScore.toString());
      }
    }
    
    p.logs.game_info.push({
      event: 'game_complete',
      data: { finalScore: gameState.score },
      framecount: p.frameCount,
      timestamp: Date.now()
    });
  } else {
    // Next level
    gameState.level++;
    gameState.levelTransitionTimer = 180; // 3 seconds
  }
}
