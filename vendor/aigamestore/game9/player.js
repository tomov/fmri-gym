import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT, WEAPONS } from './globals.js';

export class Player {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.vx = 0;
    this.vy = 0;
    this.width = 14;
    this.height = 22;
    this.onGround = false;
    this.facing = 1;
    this.shooting = false;
    this.aimUp = false;
    this.aimDown = false;
    this.shootCooldown = 0;
    this.animFrame = 0;
    this.animTimer = 0;
    this.magnetPlatforms = [];
    this.magnetBeamActive = false;
    this.jumpReleased = true; // Track if jump key has been released since last jump
  }

  update(p, keys) {
    const GRAVITY = 0.6;
    const JUMP_POWER = -10;
    const MOVE_SPEED = 2.0;       // Continuous movement speed
    const AIR_CONTROL = 0.8;       // Air control multiplier
    const GROUND_FRICTION = 0.85;  // Ground friction
    const AIR_FRICTION = 0.98;     // Air friction (less friction in air)
    const MAX_FALL_SPEED = 12;

    // HOLD-BASED MOVEMENT - Apply velocity continuously while key is held
    const moveControl = this.onGround ? 1.0 : AIR_CONTROL;
    
    if (keys.left) {
      this.vx = -MOVE_SPEED * moveControl;
      this.facing = -1;
    } else if (keys.right) {
      this.vx = MOVE_SPEED * moveControl;
      this.facing = 1;
    } else {
      // Apply friction when no movement key is held
      if (this.onGround) {
        this.vx *= GROUND_FRICTION;
        if (Math.abs(this.vx) < 0.1) {
          this.vx = 0;
        }
      } else {
        this.vx *= AIR_FRICTION;
      }
    }

    // Aim - hold to set aim direction
    this.aimUp = keys.up;
    this.aimDown = keys.down && !this.onGround;

    // HOLD-BASED JUMP with single-jump-per-press prevention
    // Player can hold jump button, but only jumps once per key press
    if (!keys.jump) {
      // Jump key released - allow next jump
      this.jumpReleased = true;
    }

    if (keys.jump && this.onGround && this.jumpReleased) {
      this.vy = JUMP_POWER;
      this.onGround = false;
      this.jumpReleased = false; // Prevent additional jumps until key is released
    }

    // Gravity
    if (!this.onGround) {
      this.vy += GRAVITY;
      if (this.vy > MAX_FALL_SPEED) this.vy = MAX_FALL_SPEED;
    }

    // Apply velocity
    this.x += this.vx;
    this.y += this.vy;

    // Collision with ground and platforms
    this.onGround = false;
    const platforms = gameState.platformBlocks.concat(this.magnetPlatforms);
    
    for (let plat of platforms) {
      // Horizontal collision
      if (this.vx > 0 && this.x + this.width > plat.x && this.x < plat.x &&
          this.y + this.height > plat.y && this.y < plat.y + plat.height) {
        this.x = plat.x - this.width;
        this.vx = 0;
      }
      if (this.vx < 0 && this.x < plat.x + plat.width && this.x + this.width > plat.x + plat.width &&
          this.y + this.height > plat.y && this.y < plat.y + plat.height) {
        this.x = plat.x + plat.width;
        this.vx = 0;
      }
      // Vertical collision
      if (this.vy > 0 && this.y + this.height > plat.y && this.y < plat.y &&
          this.x + this.width > plat.x && this.x < plat.x + plat.width) {
        this.y = plat.y - this.height;
        this.vy = 0;
        this.onGround = true;
      }
      if (this.vy < 0 && this.y < plat.y + plat.height && this.y + this.height > plat.y + plat.height &&
          this.x + this.width > plat.x && this.x < plat.x + plat.width) {
        this.y = plat.y + plat.height;
        this.vy = 0;
      }
    }

    // Stage boundaries
    if (this.x < 0) {
      this.x = 0;
      this.vx = 0;
    }
    if (gameState.currentStage && this.x + this.width > gameState.currentStage.width) {
      this.x = gameState.currentStage.width - this.width;
      this.vx = 0;
    }

    // Death pit
    if (this.y > CANVAS_HEIGHT + 50) {
      this.takeDamage(p, 999);
    }

    // Animation
    this.animTimer++;
    if (this.animTimer > 8) {
      this.animTimer = 0;
      this.animFrame = (this.animFrame + 1) % 3;
    }

    // HOLD-BASED SHOOTING - Hold to continuously shoot (with cooldown)
    if (keys.shoot && this.shootCooldown === 0) {
      this.shoot(p);
      this.shootCooldown = 15;
    }
    if (this.shootCooldown > 0) this.shootCooldown--;

    // Update magnet platforms
    this.updateMagnetPlatforms();
  }

  shoot(p) {
    const weaponKey = gameState.unlockedWeapons[gameState.currentWeapon];
    const weapon = WEAPONS[weaponKey];
    
    if (weaponKey === 'MAGNET_BEAM') {
      this.activateMagnetBeam(p);
      return;
    }

    if (gameState.weaponEnergy[weaponKey] <= 0 && weaponKey !== 'BUSTER') {
      return;
    }

    let angle = 0;
    if (this.aimUp) angle = -Math.PI / 2;
    else if (this.aimDown) angle = Math.PI / 2;
    else angle = this.facing > 0 ? 0 : Math.PI;

    const projectile = {
      x: this.x + this.width / 2,
      y: this.y + this.height / 2,
      vx: Math.cos(angle) * 5,
      vy: Math.sin(angle) * 5,
      weapon: weaponKey,
      damage: weapon.damage,
      color: weapon.color,
      lifetime: 120,
      isPlayerProjectile: true
    };

    gameState.projectiles.push(projectile);

    if (weaponKey !== 'BUSTER') {
      gameState.weaponEnergy[weaponKey] -= 1;
      if (gameState.weaponEnergy[weaponKey] < 0) gameState.weaponEnergy[weaponKey] = 0;
    }
  }

  activateMagnetBeam(p) {
    const weaponKey = 'MAGNET_BEAM';
    if (gameState.weaponEnergy[weaponKey] <= 0) return;

    const platformLength = 80;
    const platformY = this.y + this.height - 5;
    const platformX = this.facing > 0 ? this.x + this.width : this.x - platformLength;

    // Check if platform can be placed
    let canPlace = true;
    for (let plat of gameState.platformBlocks) {
      if (platformX + platformLength > plat.x && platformX < plat.x + plat.width &&
          platformY + 10 > plat.y && platformY < plat.y + plat.height) {
        canPlace = false;
        break;
      }
    }

    if (canPlace && this.magnetPlatforms.length < 3) {
      this.magnetPlatforms.push({
        x: platformX,
        y: platformY,
        width: platformLength,
        height: 10,
        lifetime: 180
      });
      gameState.weaponEnergy[weaponKey] -= 3;
      if (gameState.weaponEnergy[weaponKey] < 0) gameState.weaponEnergy[weaponKey] = 0;
    }
  }

  updateMagnetPlatforms() {
    for (let i = this.magnetPlatforms.length - 1; i >= 0; i--) {
      this.magnetPlatforms[i].lifetime--;
      if (this.magnetPlatforms[i].lifetime <= 0) {
        this.magnetPlatforms.splice(i, 1);
      }
    }
  }

  takeDamage(p, damage) {
    if (gameState.invincibilityFrames > 0) return;

    gameState.playerHealth -= damage;
    gameState.invincibilityFrames = 90;

    // Knockback
    this.vx = -this.facing * 3;
    this.vy = -4;

    if (gameState.playerHealth <= 0) {
      gameState.playerHealth = 0;
      this.die(p);
    }
  }

  die(p) {
    gameState.lives--;
    if (gameState.lives <= 0) {
      gameState.gamePhase = "GAME_OVER_LOSE";
      p.logs.game_info.push({
        data: { phase: "GAME_OVER_LOSE", reason: "player_died" },
        framecount: p.frameCount,
        timestamp: Date.now()
      });
    } else {
      // Respawn
      this.x = 50;
      this.y = 100;
      this.vx = 0;
      this.vy = 0;
      gameState.playerHealth = gameState.maxPlayerHealth;
      gameState.invincibilityFrames = 120;
      this.jumpReleased = true; // Reset jump state on respawn
    }
  }

  draw(p) {
    const screenX = this.x - gameState.camera.x;
    const screenY = this.y - gameState.camera.y;

    p.push();
    
    // Invincibility flicker
    if (gameState.invincibilityFrames > 0 && Math.floor(gameState.invincibilityFrames / 4) % 2 === 0) {
      p.pop();
      return;
    }

    // Body
    p.fill(100, 200, 255);
    p.rect(screenX + 3, screenY + 6, 8, 12);
    
    // Head
    p.fill(100, 200, 255);
    p.rect(screenX + 4, screenY, 6, 6);
    
    // Helmet
    p.fill(50, 150, 255);
    p.rect(screenX + 2, screenY - 2, 10, 4);
    
    // Arms
    p.fill(100, 200, 255);
    p.rect(screenX + (this.facing > 0 ? 11 : 0), screenY + 8, 3, 6);
    
    // Legs
    if (this.onGround && Math.abs(this.vx) > 0) {
      p.rect(screenX + 3, screenY + 18, 3, 4);
      p.rect(screenX + 8, screenY + 18, 3, 4);
    } else {
      p.rect(screenX + 4, screenY + 18, 6, 4);
    }

    // Buster
    p.fill(80, 100, 150);
    const busterX = this.facing > 0 ? screenX + 11 : screenX;
    p.rect(busterX, screenY + 10, 3, 3);

    // Magnet platforms
    for (let plat of this.magnetPlatforms) {
      const platScreenX = plat.x - gameState.camera.x;
      const platScreenY = plat.y - gameState.camera.y;
      const alpha = Math.min(255, plat.lifetime * 2);
      p.fill(255, 50, 150, alpha);
      p.rect(platScreenX, platScreenY, plat.width, plat.height);
      p.fill(255, 150, 200, alpha);
      for (let i = 0; i < plat.width; i += 10) {
        p.rect(platScreenX + i, platScreenY + 2, 4, 2);
      }
    }

    p.pop();
  }
}