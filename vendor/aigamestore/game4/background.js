/**
 * Handles the background rendering and parallax scrolling effects.
 */

import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT } from './globals.js';

class BackgroundLayer {
    constructor(speedMult, renderFn) {
        this.speedMult = speedMult;
        this.renderFn = renderFn;
        this.offsetX = 0;
    }

    update() {
        this.offsetX -= gameState.gameSpeed * this.speedMult;
        if (this.offsetX <= -CANVAS_WIDTH) {
            this.offsetX += CANVAS_WIDTH;
        }
    }

    render(p) {
        p.push();
        p.translate(this.offsetX, 0);
        // Render twice for seamless looping
        this.renderFn(p, 0);
        this.renderFn(p, CANVAS_WIDTH);
        p.pop();
    }
}

export class BackgroundManager {
    constructor() {
        this.layers = [];
        this.initLayers();
    }

    initLayers() {
        // Wall Layer (Slowest)
        this.layers.push(new BackgroundLayer(0.2, (p, xOffset) => {
            p.fill(40, 40, 50);
            p.rect(xOffset, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
            
            // Wall panels
            p.stroke(60, 60, 70);
            p.strokeWeight(2);
            for(let i=0; i<CANVAS_WIDTH; i+=100) {
                p.line(xOffset + i, 50, xOffset + i, CANVAS_HEIGHT - 50);
            }
        }));

        // Decor Layer (Pipes, Lights)
        this.layers.push(new BackgroundLayer(0.5, (p, xOffset) => {
            // Pipes
            p.noStroke();
            p.fill(30, 30, 40);
            p.rect(xOffset, 100, CANVAS_WIDTH, 20);
            p.rect(xOffset, 300, CANVAS_WIDTH, 20);
            
            // Lights
            for(let i=50; i<CANVAS_WIDTH; i+=200) {
                p.fill(200, 255, 200, 100);
                p.circle(xOffset + i, 110, 10);
                p.circle(xOffset + i, 310, 10);
            }
        }));
    }

    update() {
        this.layers.forEach(layer => layer.update());
    }

    render(p) {
        this.layers.forEach(layer => layer.render(p));
        
        // Draw Floor and Roof fixed relative to screen Y but scrolling texture?
        // Just static gray bars for floor/roof
        p.fill(20);
        p.rect(0, 0, CANVAS_WIDTH, 50); // Roof
        p.rect(0, CANVAS_HEIGHT - 50, CANVAS_WIDTH, 50); // Floor
        
        // Floor details (stripes to show speed)
        p.fill(255, 255, 0);
        const stripeOffset = (gameState.distanceTraveled * 1.5) % 100;
        for (let i = -100; i < CANVAS_WIDTH + 100; i += 100) {
            p.rect(i - stripeOffset, CANVAS_HEIGHT - 45, 50, 10);
        }
    }
}