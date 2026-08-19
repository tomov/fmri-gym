import { gameState, GAME_PHASES } from './globals.js';
import { levelConfigs } from './levelData.js';
import { Player, Cat, Cheese, Platform, MouseHole } from './entities.js';

export function loadLevel(p, levelNum) {
  const config = levelConfigs[levelNum - 1];
  if (!config) return;

  gameState.currentLevelConfig = config;
  gameState.entities = [];
  gameState.cheeseCollected = 0;
  gameState.totalCheese = config.cheese.length;
  gameState.mouseHoleActive = false;
  gameState.invulnerable = false;
  gameState.invulnerableTimer = 0;

  // Create player
  gameState.player = new Player(p, config.playerStart.x, config.playerStart.y);
  gameState.entities.push(gameState.player);

  // Create platforms
  for (const platData of config.platforms) {
    gameState.entities.push(new Platform(p, platData.x, platData.y, platData.w, platData.h));
  }

  // Create cheese
  for (const cheeseData of config.cheese) {
    gameState.entities.push(new Cheese(p, cheeseData.x, cheeseData.y));
  }

  // Create cats
  for (const catData of config.cats) {
    gameState.entities.push(
      new Cat(p, catData.x, catData.y, catData.patrolPath, catData.speed)
    );
  }

  // Create mouse hole
  gameState.entities.push(new MouseHole(p, config.mouseHole.x, config.mouseHole.y));

  p.logs.game_info.push({
    event: 'level_loaded',
    data: { level: levelNum },
    framecount: p.frameCount,
    timestamp: Date.now()
  });
}

export function updateLevelTransition(p) {
  if (gameState.levelTransitionTimer > 0) {
    gameState.levelTransitionTimer--;
    if (gameState.levelTransitionTimer === 0) {
      loadLevel(p, gameState.level);
      gameState.gamePhase = GAME_PHASES.PLAYING;
    }
  }
}