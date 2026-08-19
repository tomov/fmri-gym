/**
 * Game Entities.
 * Contains the Player, Enemies (Zappers, Missiles), and Collectibles.
 */

import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT, GRAVITY, THRUST_FORCE, TERMINAL_VELOCITY, GROUND_HEIGHT, ROOF_HEIGHT } from './globals.js';
import { Particle } from './particles.js';

/* ================= PLAYER CLASS ================= */
export class Player {
    constructor() {
        this.x = 100;
        this.y = CANVAS_HEIGHT - 100;
        this.width = 30;
        this.height = 40;
        
        // Physics
        this.vx = 0;
        this.vy = 0;
        this.ax = 0;
        this.ay = 0;
        
        this.onGround = false;
        this.isDead = false;
        
        // Booster / Mechanics
        this.isThrusting = false;
        this.vehicle = null; // 'DRAGON', 'STOMPER' or null
        this.invincibilityTimer = 0;
        
        // Animation
        this.frameTimer = 0;
        this.runFrame = 0;
        this.rotation = 0;
    }

    update(p, inputActive) {
        this.isThrusting = inputActive;

        // Invincibility Timer
        if (this.invincibilityTimer > 0) {
            this.invincibilityTimer--;
        }

        // --- Physics Logic based on Vehicle ---
        if (this.vehicle === 'DRAGON') {
            // Flappy bird style physics
            // If input is pressed THIS FRAME (needs logic in input to detect discrete press, 
            // but for smooth gameplay we might use hold for flap strength or repeated press)
            // Let's implement: Holding space makes dragon fly up against gravity
            if (this.isThrusting) {
                this.vy += THRUST_FORCE * 0.8; // Dragon heavy
            }
        } else {
            // Normal Booster Physics
            if (this.isThrusting) {
                this.vy += THRUST_FORCE;
                
                // Spawn particles
                if (gameState.frameCount % 2 === 0) {
                    gameState.particles.push(new Particle(this.x - 5, this.y + this.height - 10, 'FIRE'));
                    gameState.particles.push(new Particle(this.x - 5, this.y + this.height - 10, 'SMOKE'));
                }
                
                // Spawn bullets
                if (gameState.frameCount % 4 === 0) {
                    const cx = this.x + this.width / 2;
                    const cy = this.y + this.height / 2;
                    const cos = Math.cos(this.rotation);
                    const sin = Math.sin(this.rotation);
                    
                    // Nozzle offsets relative to center (-12, 10) and (-16, 10)
                    // x' = x*cos - y*sin
                    // y' = x*sin + y*cos
                    
                    const ox1 = -12; const oy1 = 10;
                    const ox2 = -16; const oy2 = 10;
                    
                    const px1 = cx + (ox1 * cos - oy1 * sin);
                    const py1 = cy + (ox1 * sin + oy1 * cos);
                    
                    const px2 = cx + (ox2 * cos - oy2 * sin);
                    const py2 = cy + (ox2 * sin + oy2 * cos);
                    
                    gameState.particles.push(new Particle(px1, py1, 'BULLET'));
                    gameState.particles.push(new Particle(px2, py2, 'BULLET'));
                }

                // Spawn bullet casings (machine gun booster)
                if (gameState.frameCount % 5 === 0) {
                     gameState.particles.push(new Particle(this.x, this.y + 20, 'BULLET_CASING'));
                }
            }
        }

        // Apply Gravity
        this.vy += GRAVITY;

        // Terminal Velocity
        this.vy = Math.max(Math.min(this.vy, TERMINAL_VELOCITY), -TERMINAL_VELOCITY);

        // Update Position
        this.y += this.vy;

        // Floor/Roof Collision
        const floorY = CANVAS_HEIGHT - GROUND_HEIGHT;
        const roofY = ROOF_HEIGHT;

        if (this.y < roofY) {
            this.y = roofY;
            this.vy = 0;
        }

        if (this.y + this.height > floorY) {
            this.y = floorY - this.height;
            this.vy = 0;
            this.onGround = true;
            this.rotation = 0; // Reset rotation on ground
        } else {
            this.onGround = false;
            // Calculate rotation based on velocity for visual flair
            this.rotation = p.map(this.vy, -10, 10, -0.2, 0.2);
        }

        // Animation update
        if (this.onGround) {
            this.frameTimer++;
            if (this.frameTimer > 5) {
                this.runFrame = (this.runFrame + 1) % 4;
                this.frameTimer = 0;
            }
        } else {
            this.runFrame = 1; // Jumping pose
        }
    }

    render(p) {
        // Flash if invincible
        if (this.invincibilityTimer > 0 && Math.floor(this.invincibilityTimer / 4) % 2 === 0) {
            return;
        }

        p.push();
        p.translate(this.x + this.width / 2, this.y + this.height / 2);
        p.rotate(this.rotation);

        if (this.vehicle === 'DRAGON') {
            this.renderDragon(p);
        } else {
            this.renderPlayer(p);
        }

        p.pop();
    }

    renderPlayer(p) {
        // --- Draw Player ---
        p.rectMode(p.CENTER);

        // Legs (animated)
        p.fill(50, 50, 150); // Blue pants
        if (this.onGround) {
            const legOffset = (this.runFrame % 2 === 0) ? 5 : -5;
            p.rect(-5 + legOffset, 15, 6, 10);
            p.rect(5 - legOffset, 15, 6, 10);
        } else {
            // Flying legs
            p.rect(-5, 18, 6, 10); // Back leg hanging
            p.rect(5, 15, 6, 8);   // Front leg bent
        }

        // Body
        p.fill(200, 200, 200); // Suit
        p.rect(0, 0, 20, 25);
        
        // Head
        p.fill(255, 200, 180); // Skin
        p.rect(0, -18, 16, 16);
        p.fill(100, 50, 0); // Hair
        p.rect(0, -24, 18, 6);

        // Arm
        p.fill(200, 200, 200); // Sleeve
        p.rect(5, 0, 8, 20); // Holding controls

        // Booster
        p.fill(100);
        p.rect(-12, -2, 10, 20); // Backpack
        p.fill(50);
        p.rect(-12, 10, 4, 10); // Nozzle
        p.rect(-16, 10, 4, 10); // Nozzle
    }

    renderDragon(p) {
        // --- Mechanical Dragon Visual ---
        p.rectMode(p.CENTER);
        
        // Body segments
        p.fill(200, 0, 0); // Red metal
        p.stroke(100, 0, 0);
        p.strokeWeight(2);
        
        // Tail
        p.rect(-30, 0, 20, 10);
        
        // Main Body
        p.rect(0, 0, 40, 25);
        
        // Head
        p.rect(30, -10, 25, 20);
        
        // Jaw
        p.rect(30, 5, 20, 8);
        
        // Eye
        p.fill(255, 255, 0);
        p.noStroke();
        p.circle(35, -12, 8);
        
        // Wing
        p.fill(150, 150, 150);
        p.stroke(0);
        p.triangle(-10, -10, 10, -10, 0, -40);
        
        // Rider (the player visible on top)
        p.fill(255, 200, 180);
        p.circle(0, -15, 10);
    }
}

/* ================= OBSTACLE CLASSES ================= */

export class Zapper {
    constructor(x, y, length, angle, type) {
        this.x = x;
        this.y = y;
        this.length = length;
        this.angle = angle;
        this.type = type; // 'STATIC', 'ROTATING'
        this.rotationSpeed = type === 'ROTATING' ? 0.02 : 0;
        
        // For collision
        this.width = 10; // Thickness
        this.height = length; // Treated as height when upright
        
        this.active = true;
        this.oscillation = 0;
    }

    update() {
        if (this.type === 'ROTATING') {
            this.angle += this.rotationSpeed;
        }
        this.oscillation += 0.5;
        
        // Move with world
        this.x -= gameState.gameSpeed;
        
        // Remove if off screen
        if (this.x < -200) {
            this.active = false;
        }
    }

    render(p) {
        p.push();
        p.translate(this.x, this.y);
        p.rotate(this.angle);
        
        // Draw End Caps
        p.fill(50);
        p.stroke(0);
        p.circle(0, -this.length/2, 15);
        p.circle(0, this.length/2, 15);
        
        // Draw Electric Field
        p.stroke(100, 200, 255);
        p.strokeWeight(3 + Math.sin(this.oscillation) * 2);
        p.line(0, -this.length/2, 0, this.length/2);
        
        // Inner white core
        p.stroke(255);
        p.strokeWeight(1);
        p.line(0, -this.length/2, 0, this.length/2);
        
        p.pop();
    }
    
    // Get Collision Polygon (approximated as a rotated rectangle)
    getPolygon(p) {
        // Calculate corner points based on center x,y, length, and angle
        const halfL = this.length / 2;
        const halfW = 5; // half thickness
        
        // We need 4 points relative to center, then rotated and translated
        const points = [
            p.createVector(-halfW, -halfL),
            p.createVector(halfW, -halfL),
            p.createVector(halfW, halfL),
            p.createVector(-halfW, halfL)
        ];
        
        const poly = [];
        for(let pt of points) {
            // Rotate
            const x = pt.x * Math.cos(this.angle) - pt.y * Math.sin(this.angle);
            const y = pt.x * Math.sin(this.angle) + pt.y * Math.cos(this.angle);
            // Translate
            poly.push(p.createVector(x + this.x, y + this.y));
        }
        return poly;
    }
}

export class Missile {
    constructor(y) {
        this.x = CANVAS_WIDTH + 50;
        this.y = y;
        this.width = 40;
        this.height = 15;
        this.state = 'WARNING'; // WARNING -> LAUNCHED
        this.timer = 0;
        this.warningTime = 60; // 1 second warning
        this.active = true;
        this.speed = 0;
    }

    update() {
        this.timer++;
        
        if (this.state === 'WARNING') {
            // Keep position relative to screen right edge for warning
            this.x = CANVAS_WIDTH - 50;
            
            // Track player Y slightly (homing warning)
            if (gameState.player) {
                const dy = gameState.player.y - this.y;
                this.y += dy * 0.05;
            }
            
            if (this.timer > this.warningTime) {
                this.state = 'LAUNCHED';
                this.x = CANVAS_WIDTH + 50; // Reset to offscreen right to fly in
                this.speed = gameState.gameSpeed * 2 + 5;
            }
        } else {
            // Fly left
            this.x -= this.speed;
            
            // Smoke trail
            if (this.timer % 3 === 0) {
                gameState.particles.push(new Particle(this.x + 40, this.y, 'SMOKE'));
            }
            
            if (this.x < -100) {
                this.active = false;
            }
        }
    }

    render(p) {
        if (this.state === 'WARNING') {
            // Draw Warning Sign
            if (this.timer % 10 < 5) { // Flash
                p.fill(255, 0, 0);
                p.noStroke();
                p.triangle(this.x, this.y - 20, this.x + 20, this.y, this.x, this.y + 20);
                p.fill(255);
                p.textSize(20);
                p.textAlign(p.CENTER, p.CENTER);
                p.text('!', this.x + 5, this.y);
            }
        } else {
            // Draw Missile
            p.push();
            p.translate(this.x, this.y);
            
            p.fill(200, 50, 50); // Red Body
            p.rectMode(p.CENTER);
            p.rect(0, 0, 40, 15, 5);
            
            p.fill(50); // Fins
            p.rect(15, 0, 5, 25);
            
            p.fill(255); // Tip
            p.arc(-20, 0, 15, 15, p.HALF_PI, p.PI + p.HALF_PI);
            
            p.pop();
        }
    }
}

/* ================= COLLECTIBLES ================= */

export class Coin {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.radius = 12; // visual size
        this.hitRadius = 15; // collision tolerance
        this.active = true;
        this.oscillationOffset = Math.random() * Math.PI * 2;
    }

    update() {
        this.x -= gameState.gameSpeed;
        
        // Remove if off screen
        if (this.x < -50) this.active = false;
    }

    render(p) {
        p.push();
        p.translate(this.x, this.y);
        
        // Spin effect using scale
        const scaleX = Math.sin(gameState.frameCount * 0.1 + this.oscillationOffset);
        
        p.fill(255, 215, 0); // Gold
        p.stroke(200, 150, 0);
        p.strokeWeight(2);
        p.ellipse(0, 0, this.radius * 2 * scaleX, this.radius * 2);
        
        // Inner detail
        p.fill(255, 255, 200);
        p.noStroke();
        p.ellipse(0, 0, this.radius * 1.5 * scaleX, this.radius * 1.5);
        
        p.pop();
    }
}

export class VehicleToken {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.width = 40;
        this.height = 40;
        this.active = true;
        this.floatY = 0;
    }
    
    update() {
        this.x -= gameState.gameSpeed;
        this.floatY = Math.sin(gameState.frameCount * 0.1) * 5;
        
        if (this.x < -50) this.active = false;
    }
    
    render(p) {
        p.push();
        p.translate(this.x, this.y + this.floatY);
        
        // Box
        p.fill(0, 255, 255, 150);
        p.stroke(255);
        p.rectMode(p.CENTER);
        p.rect(0, 0, 40, 40, 5);
        
        // Icon (Gear)
        p.fill(255);
        p.circle(0, 0, 15);
        p.stroke(255);
        p.strokeWeight(4);
        p.line(-10, 0, 10, 0);
        p.line(0, -10, 0, 10);
        
        p.pop();
    }
}