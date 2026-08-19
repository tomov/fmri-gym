/**
 * Particle systems for visual effects.
 * Includes Booster smoke, Coin sparkles, Explosions, and Bullet casings.
 */

import { gameState, CANVAS_HEIGHT } from './globals.js';

export class Particle {
    constructor(x, y, type) {
        this.x = x;
        this.y = y;
        this.type = type;
        this.age = 0;
        this.markedForDeletion = false;
        
        // Default properties, overridden by type
        this.vx = (Math.random() - 0.5) * 2;
        this.vy = (Math.random() - 0.5) * 2;
        this.size = 5;
        this.lifetime = 60;
        this.color = [255, 255, 255];
        this.alpha = 255;
        this.gravity = 0;
        this.drag = 0.98;

        this.initType(type);
    }

    initType(type) {
        switch (type) {
            case 'SMOKE':
                this.vx = -gameState.gameSpeed * 0.5 + (Math.random() - 0.5);
                this.vy = 2 + Math.random() * 2; // Move down
                this.lifetime = 40;
                this.size = Math.random() * 10 + 5;
                this.color = [100, 100, 100];
                this.startAlpha = 150;
                break;
            case 'FIRE':
                this.vx = -gameState.gameSpeed + (Math.random() - 0.5) * 2;
                this.vy = 3 + Math.random() * 3; // Blast down
                this.lifetime = 20;
                this.size = Math.random() * 8 + 4;
                this.color = [255, 100, 0]; // Orange
                this.startAlpha = 255;
                break;
            case 'BULLET':
                this.vx = (Math.random() - 0.5) * 1; 
                this.vy = 15; // Fast down
                this.lifetime = 10;
                this.size = 3;
                this.color = [255, 255, 100];
                this.startAlpha = 255;
                this.drag = 1; // No drag
                break;
            case 'BULLET_CASING':
                this.vx = (Math.random() * -3) - 1; // Eject left/back
                this.vy = (Math.random() * -4) - 2; // Up and out
                this.gravity = 0.5;
                this.lifetime = 100;
                this.size = 3;
                this.color = [255, 215, 0]; // Gold
                this.startAlpha = 255;
                break;
            case 'SPARKLE':
                this.vx = (Math.random() - 0.5) * 4;
                this.vy = (Math.random() - 0.5) * 4;
                this.lifetime = 30;
                this.size = Math.random() * 4 + 2;
                this.color = [255, 255, 0];
                this.startAlpha = 200;
                break;
            case 'EXPLOSION':
                this.vx = (Math.random() - 0.5) * 10;
                this.vy = (Math.random() - 0.5) * 10;
                this.lifetime = 45;
                this.size = Math.random() * 15 + 5;
                this.color = Math.random() > 0.5 ? [255, 50, 0] : [255, 200, 0];
                this.startAlpha = 255;
                this.drag = 0.9;
                break;
            case 'WARNING_GLOW':
                this.vx = 0;
                this.vy = 0;
                this.lifetime = 10;
                this.size = 40;
                this.color = [255, 0, 0];
                this.startAlpha = 100;
                break;
        }
        this.alpha = this.startAlpha;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vy += this.gravity;
        
        // Apply drag
        this.vx *= this.drag;
        this.vy *= this.drag;

        // Apply game speed relative motion for world objects (except casings which fall relative to world?)
        // Actually, smoke should drift back with the world
        if (this.type === 'SMOKE' || this.type === 'FIRE' || this.type === 'EXPLOSION') {
            this.x -= gameState.gameSpeed * 0.2; // Parallax effect on smoke
        }

        // Floor collision for casings
        if (this.type === 'BULLET_CASING' && this.y > CANVAS_HEIGHT - 50) {
            this.y = CANVAS_HEIGHT - 50;
            this.vy *= -0.5; // Bounce
            this.vx *= 0.8;  // Friction
        }

        this.age++;
        
        // Fade out
        this.alpha = this.startAlpha * (1 - this.age / this.lifetime);

        if (this.age >= this.lifetime) {
            this.markedForDeletion = true;
        }
    }

    render(p) {
        p.push();
        
        if (this.type === 'BULLET') {
            p.stroke(this.color[0], this.color[1], this.color[2], this.alpha);
            p.strokeWeight(2);
            p.line(this.x, this.y, this.x, this.y - 12); // Speed streak
        } else {
            p.noStroke();
            p.fill(this.color[0], this.color[1], this.color[2], this.alpha);
            
            if (this.type === 'BULLET_CASING') {
                p.translate(this.x, this.y);
                p.rotate(this.age * 0.2);
                p.rectMode(p.CENTER);
                p.rect(0, 0, this.size, this.size * 2);
            } else {
                p.circle(this.x, this.y, this.size);
            }
        }
        p.pop();
    }
}

export function spawnExplosion(x, y, count = 20) {
    for (let i = 0; i < count; i++) {
        gameState.particles.push(new Particle(x, y, 'EXPLOSION'));
    }
}

export function spawnCoinSparkle(x, y) {
    for (let i = 0; i < 5; i++) {
        gameState.particles.push(new Particle(x, y, 'SPARKLE'));
    }
}