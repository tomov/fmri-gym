import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { gameState } from './globals.js';

// Random Number Generator
let rng = new Math.seedrandom('42'); // Default seed

export function setSeed(seed) {
    rng = new Math.seedrandom(seed);
}

export function random() {
    return rng();
}

export function randomRange(min, max) {
    return min + rng() * (max - min);
}

export function randomInt(min, max) {
    return Math.floor(min + rng() * (max - min + 1));
}

export function randomPointInCircle(radius) {
    const angle = rng() * Math.PI * 2;
    const r = Math.sqrt(rng()) * radius;
    return {
        x: Math.cos(angle) * r,
        z: Math.sin(angle) * r
    };
}

// Math Helpers
export function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

// Physics Helpers
export function boxIntersectsBox(box1, box2) {
    // box: { pos: Vector3, size: Vector3 } (center position)
    return (
        Math.abs(box1.pos.x - box2.pos.x) * 2 < (box1.size.x + box2.size.x) &&
        Math.abs(box1.pos.z - box2.pos.z) * 2 < (box1.size.z + box2.size.z) // Top down, ignore Y for collision mostly
    );
}

export function getDistance(v1, v2) {
    const dx = v1.x - v2.x;
    const dz = v1.z - v2.z;
    return Math.sqrt(dx * dx + dz * dz);
}

export function normalizeVector(v) {
    const len = Math.sqrt(v.x * v.x + v.z * v.z);
    if (len === 0) return { x: 0, z: 0 };
    return { x: v.x / len, z: v.z / len };
}

// Raycasting for Logic (not rendering)
export function raycastScene(origin, direction, maxDistance, ignoreList = []) {
    // Simple raycast against WALLS only for LoS
    // Returns { hit: boolean, distance: number, point: Vector3, object: object }
    
    let closestHit = null;
    let minDist = maxDistance;

    // Check Walls
    for (const wall of gameState.walls) {
        // AABB Ray intersection
        const result = intersectRayAABB(origin, direction, wall.boundingBox);
        if (result && result.distance < minDist) {
            minDist = result.distance;
            closestHit = {
                hit: true,
                distance: result.distance,
                point: result.point,
                object: wall,
                type: 'WALL'
            };
        }
    }
    
    // Check Enemies (if we want to shoot them)
    for (const ent of gameState.enemies) {
        if (ignoreList.includes(ent)) continue;
        
        const result = intersectRaySphere(origin, direction, ent.mesh.position, ent.radius || 0.5);
        if (result && result < minDist) {
            minDist = result;
            closestHit = {
                hit: true,
                distance: result,
                point: new THREE.Vector3().copy(origin).add(direction.clone().multiplyScalar(result)),
                object: ent,
                type: 'ENEMY'
            };
        }
    }

    // Check Player
    if (gameState.player && !ignoreList.includes(gameState.player)) {
        const result = intersectRaySphere(origin, direction, gameState.player.mesh.position, 0.5);
        if (result && result < minDist) {
            minDist = result;
            closestHit = {
                hit: true,
                distance: result,
                point: new THREE.Vector3().copy(origin).add(direction.clone().multiplyScalar(result)),
                object: gameState.player,
                type: 'PLAYER'
            };
        }
    }

    return closestHit;
}

// Math for Ray-AABB
function intersectRayAABB(origin, dir, box) {
    // box = { min: Vector3, max: Vector3 }
    let tmin = (box.min.x - origin.x) / dir.x;
    let tmax = (box.max.x - origin.x) / dir.x;
    if (tmin > tmax) [tmin, tmax] = [tmax, tmin];

    let tymin = (box.min.y - origin.y) / dir.y;
    let tymax = (box.max.y - origin.y) / dir.y;
    if (tymin > tymax) [tymin, tymax] = [tymax, tymin];

    if ((tmin > tymax) || (tymin > tmax)) return null;
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    let tzmin = (box.min.z - origin.z) / dir.z;
    let tzmax = (box.max.z - origin.z) / dir.z;
    if (tzmin > tzmax) [tzmin, tzmax] = [tzmax, tzmin];

    if ((tmin > tzmax) || (tzmin > tmax)) return null;
    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    if (tmax < 0) return null; // Behind ray
    
    const dist = tmin > 0 ? tmin : tmax;
    return {
        distance: dist,
        point: new THREE.Vector3().copy(origin).add(dir.clone().multiplyScalar(dist))
    };
}

function intersectRaySphere(origin, dir, sphereCenter, radius) {
    const L = new THREE.Vector3().subVectors(sphereCenter, origin);
    const tca = L.dot(dir);
    if (tca < 0) return null;
    const d2 = L.dot(L) - tca * tca;
    if (d2 > radius * radius) return null;
    const thc = Math.sqrt(radius * radius - d2);
    const t0 = tca - thc;
    const t1 = tca + thc;
    if (t0 < 0 && t1 < 0) return null;
    return t0 < 0 ? t1 : t0;
}