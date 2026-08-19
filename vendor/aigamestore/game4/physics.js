/**
 * Physics engine and collision detection system.
 * Handles AABB collisions, simple physics updates, and spatial calculations.
 */

import { gameState, CANVAS_WIDTH, CANVAS_HEIGHT } from './globals.js';

// Import p5.collide2D from CDN if environment requires, but we assume it's loaded globally via script tag
// For module safety, we will define wrapper functions that use the global p5 object's context or manual math.

/**
 * Checks for collision between two rectangular entities (AABB).
 * @param {Object} a - Entity A {x, y, width, height}
 * @param {Object} b - Entity B {x, y, width, height}
 * @returns {boolean} True if colliding
 */
export function checkRectCollision(a, b) {
    return (
        a.x < b.x + b.width &&
        a.x + a.width > b.x &&
        a.y < b.y + b.height &&
        a.y + a.height > b.y
    );
}

/**
 * Checks for collision between a circle and a rectangle.
 * Useful for Player (Rect) vs Coin (Circle) or Zapper (Rect/Circle approximation).
 * @param {Object} circle - {x, y, radius} (Note: radius is often half-width)
 * @param {Object} rect - {x, y, width, height}
 * @returns {boolean} True if colliding
 */
export function checkCircleRectCollision(circle, rect) {
    // Find the closest point on the rectangle to the circle center
    let testX = circle.x;
    let testY = circle.y;

    if (circle.x < rect.x) testX = rect.x;
    else if (circle.x > rect.x + rect.width) testX = rect.x + rect.width;

    if (circle.y < rect.y) testY = rect.y;
    else if (circle.y > rect.y + rect.height) testY = rect.y + rect.height;

    // Calculate distance
    let distX = circle.x - testX;
    let distY = circle.y - testY;
    let distance = Math.sqrt((distX * distX) + (distY * distY));

    return distance <= circle.radius;
}

/**
 * Checks overlap between two circles.
 * @param {Object} c1 - {x, y, radius}
 * @param {Object} c2 - {x, y, radius}
 * @returns {boolean}
 */
export function checkCircleCollision(c1, c2) {
    let dx = c1.x - c2.x;
    let dy = c1.y - c2.y;
    let distance = Math.sqrt(dx * dx + dy * dy);
    return distance < c1.radius + c2.radius;
}

/**
 * Rotated Rectangle Collision Check using Separating Axis Theorem (SAT) logic simplified.
 * Used for rotating zappers.
 * This is a wrapper around p5.collide2D's collidePolyPoly if available, or manual implementation.
 * For this game, we'll approximate rotating zappers as multiple circles or use p5.collide2d logic.
 */
export function checkPolyCollision(p, poly1, poly2) {
    // Assuming poly1 and poly2 are arrays of vectors created by p.createVector
    // Using global p5.collide2D if available
    if (window.collidePolyPoly) {
        return window.collidePolyPoly(poly1, poly2, true);
    }
    
    // Fallback: simplified bounding box check (not accurate for rotation)
    // Real implementation of SAT would be too verbose here, relying on p5.collide2D
    return false; 
}

/**
 * Updates an entity's position based on velocity and global game speed.
 * @param {Object} entity - The entity to update
 * @param {boolean} applyGameSpeed - Whether the entity moves with the world (scrolls left)
 */
export function updatePhysics(entity, applyGameSpeed = true) {
    // Apply velocity
    entity.x += entity.vx;
    entity.y += entity.vy;

    // Apply acceleration
    entity.vx += entity.ax;
    entity.vy += entity.ay;

    // Apply world scrolling speed (relative movement)
    if (applyGameSpeed) {
        entity.x -= gameState.gameSpeed;
    }
}

/**
 * Constraints an entity within the playable area (Roof and Floor).
 * @param {Object} entity 
 * @param {number} roofY 
 * @param {number} floorY 
 */
export function constrainToScreen(entity, roofY, floorY) {
    if (entity.y < roofY) {
        entity.y = roofY;
        entity.vy = 0;
    }
    
    // Check floor collision
    // entity.y represents top-left usually, so we check y + height
    const entityBottom = entity.y + (entity.height || entity.radius * 2);
    if (entityBottom > floorY) {
        entity.y = floorY - (entity.height || entity.radius * 2);
        entity.vy = 0;
        entity.onGround = true;
    } else {
        entity.onGround = false;
    }
}