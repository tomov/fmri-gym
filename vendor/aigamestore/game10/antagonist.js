// antagonist.js - the monster enemy class
import { gameState, ANTAGONIST_SIZE, ANTAGONIST_BASE_SPEED, GAME_PHASES } from './globals.js';

export class Antagonist {
  constructor(x, y, speedMultiplier = 1, patrolPoints = [], sightRange = 120) {
    this.x = x;
    this.y = y;
    this.width = ANTAGONIST_SIZE;
    this.height = ANTAGONIST_SIZE;
    this.baseSpeed = ANTAGONIST_BASE_SPEED * speedMultiplier;
    this.speed = this.baseSpeed;
    this.sightRange = sightRange;
    this.state = "PATROL"; // PATROL, CHASE, IDLE
    this.patrolPoints = patrolPoints.length > 0 ? patrolPoints : [[x, y]];
    this.currentPatrolIndex = 0;
    this.idleTimer = 0;
    this.chaseTimer = 0;
    this.animFrame = 0;
    this.lastSeenPlayerX = null;
    this.lastSeenPlayerY = null;
  }

  update(p) {
    this.animFrame += 0.15;

    if (this.state === "IDLE") {
      this.idleTimer--;
      if (this.idleTimer <= 0) {
        this.state = "PATROL";
      }
      return;
    }

    // Check if can see player
    if (gameState.player && !gameState.player.hiding) {
      const dist = p.dist(this.x, this.y, gameState.player.x, gameState.player.y);
      
      if (dist < this.sightRange && this.hasLineOfSight(p)) {
        if (this.state !== "CHASE") {
          this.state = "CHASE";
          gameState.inChase = true;
          gameState.undetectedBonus = false;
        }
        this.lastSeenPlayerX = gameState.player.x;
        this.lastSeenPlayerY = gameState.player.y;
        this.chaseTimer = 180; // 3 seconds at 60fps
      }
    }

    if (this.state === "CHASE") {
      this.speed = this.baseSpeed * 1.8;
      this.chaseTimer--;
      
      if (this.chaseTimer <= 0 || (gameState.player && gameState.player.hiding)) {
        this.state = "PATROL";
        this.speed = this.baseSpeed;
        
        // Award evasion points
        if (gameState.inChase && Date.now() - gameState.lastEvadeTime > 5000) {
          gameState.score += 10;
          gameState.lastEvadeTime = Date.now();
          if (typeof window.addMessage === 'function') {
            window.addMessage("Evaded! +10", "success");
          }
        }
        gameState.inChase = false;
      } else if (this.lastSeenPlayerX !== null) {
        this.moveTowards(this.lastSeenPlayerX, this.lastSeenPlayerY);
      }
    } else if (this.state === "PATROL") {
      this.speed = this.baseSpeed;
      const target = this.patrolPoints[this.currentPatrolIndex];
      const dist = p.dist(this.x, this.y, target[0], target[1]);
      
      if (dist < 5) {
        this.currentPatrolIndex = (this.currentPatrolIndex + 1) % this.patrolPoints.length;
        this.state = "IDLE";
        this.idleTimer = 30;
      } else {
        this.moveTowards(target[0], target[1]);
      }
    }

    // Check collision with player - trigger game over immediately
    if (gameState.player && !gameState.player.hiding && gameState.gamePhase === GAME_PHASES.PLAYING) {
      const dist = p.dist(this.x, this.y, gameState.player.x, gameState.player.y);
      if (dist < (this.width + gameState.player.width) / 2) {
        gameState.gamePhase = GAME_PHASES.GAME_OVER_LOSE;
        
        // Log game over
        if (p && p.logs) {
          p.logs.game_info.push({
            data: { phase: GAME_PHASES.GAME_OVER_LOSE, reason: "caught", finalScore: gameState.score },
            framecount: p.frameCount,
            timestamp: Date.now()
          });
        }
      }
    }
  }

  moveTowards(targetX, targetY) {
    const angle = Math.atan2(targetY - this.y, targetX - this.x);
    let dx = Math.cos(angle) * this.speed;
    let dy = Math.sin(angle) * this.speed;

    const newX = this.x + dx;
    const newY = this.y + dy;

    // Check wall collisions
    let canMoveX = true;
    let canMoveY = true;

    for (const wall of gameState.walls) {
      if (this.checkCollision(newX, this.y, wall)) {
        canMoveX = false;
      }
      if (this.checkCollision(this.x, newY, wall)) {
        canMoveY = false;
      }
    }

    if (canMoveX) this.x = newX;
    if (canMoveY) this.y = newY;
  }

  checkCollision(x, y, wall) {
    return x - this.width / 2 < wall.x + wall.w &&
           x + this.width / 2 > wall.x &&
           y - this.height / 2 < wall.y + wall.h &&
           y + this.height / 2 > wall.y;
  }

  hasLineOfSight(p) {
    if (!gameState.player) return false;
    
    // Line of sight check - check for walls AND closed doors blocking view
    const steps = 20;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const checkX = p.lerp(this.x, gameState.player.x, t);
      const checkY = p.lerp(this.y, gameState.player.y, t);
      
      // Check walls
      for (const wall of gameState.walls) {
        if (checkX > wall.x && checkX < wall.x + wall.w &&
            checkY > wall.y && checkY < wall.y + wall.h) {
          return false;
        }
      }
      
      // Check closed doors - they should block line of sight too
      for (const door of gameState.doors) {
        if (!door.open) {
          if (checkX > door.x && checkX < door.x + door.w &&
              checkY > door.y && checkY < door.y + door.h) {
            return false;
          }
        }
      }
    }
    return true;
  }

  render(p) {
    p.push();
    p.translate(this.x, this.y);
    
    // Body
    p.fill(50, 80, 180);
    p.stroke(30, 50, 120);
    p.strokeWeight(2);
    p.rect(-this.width / 2, -this.height / 2, this.width, this.height, 4);
    
    // Head bump
    p.fill(40, 70, 160);
    p.ellipse(0, -this.height / 3, this.width * 0.6, this.height * 0.4);
    
    // Eyes (glowing)
    const eyeGlow = this.state === "CHASE" ? 255 : 200;
    p.fill(eyeGlow, 0, 0);
    p.noStroke();
    p.ellipse(-6, -5, 7, 8);
    p.ellipse(6, -5, 7, 8);
    
    // Pupils
    p.fill(0);
    p.ellipse(-6, -4, 3, 4);
    p.ellipse(6, -4, 3, 4);
    
    // Mouth
    if (this.state === "CHASE") {
      p.stroke(0);
      p.strokeWeight(2);
      p.noFill();
      p.arc(0, 3, 12, 8, 0, p.PI);
    }
    
    p.pop();
  }
}