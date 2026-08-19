import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { gameState, CONFIG } from './globals.js';
import { randomRange } from './utils.js';

class Particle {
    constructor() {
        this.mesh = new THREE.Mesh(
            new THREE.BoxGeometry(0.2, 0.2, 0.2),
            new THREE.MeshBasicMaterial({ color: 0xFFFFFF })
        );
        this.mesh.castShadow = false;
        this.velocity = new THREE.Vector3();
        this.life = 0;
        this.maxLife = 1;
        this.active = false;
        this.rotSpeed = new THREE.Vector3();
        gameState.scene.add(this.mesh);
        this.mesh.visible = false;
    }

    activate(pos, color, speed, size, life) {
        this.mesh.position.copy(pos);
        this.mesh.material.color.setHex(color);
        this.mesh.scale.set(size, size, size);
        this.velocity.set(
            randomRange(-1, 1),
            randomRange(0.5, 2), // Pop up slightly
            randomRange(-1, 1)
        ).normalize().multiplyScalar(speed);
        
        this.rotSpeed.set(randomRange(-0.2, 0.2), randomRange(-0.2, 0.2), randomRange(-0.2, 0.2));
        
        this.life = life;
        this.maxLife = life;
        this.active = true;
        this.mesh.visible = true;
    }

    update(dt) {
        if (!this.active) return;

        this.life -= dt;
        if (this.life <= 0) {
            this.active = false;
            this.mesh.visible = false;
            return;
        }

        // Gravity
        this.velocity.y += CONFIG.GRAVITY * 0.1;

        this.mesh.position.add(this.velocity.clone().multiplyScalar(dt * 10)); // Scale up speed slightly
        this.mesh.rotation.x += this.rotSpeed.x;
        this.mesh.rotation.y += this.rotSpeed.y;
        this.mesh.rotation.z += this.rotSpeed.z;

        // Ground collision simple
        if (this.mesh.position.y < 0.1) {
            this.mesh.position.y = 0.1;
            this.velocity.x *= 0.8;
            this.velocity.z *= 0.8;
            this.velocity.y = 0;
        }

        // Fade out scale
        const s = (this.life / this.maxLife);
        this.mesh.scale.set(s, s, s);
    }
}

const PARTICLE_POOL_SIZE = 200;
const pool = [];

export function initParticles() {
    for (let i = 0; i < PARTICLE_POOL_SIZE; i++) {
        pool.push(new Particle());
    }
    gameState.particles = pool;
}

export function spawnParticles(pos, type, count = 5) {
    let color = 0xFFFFFF;
    let speed = 0.5;
    let size = 1;
    let life = 0.5;

    if (type === 'BLOOD') {
        color = CONFIG.COLORS.BLOOD;
        speed = 0.8;
        size = 0.5; // Smaller chunks
        life = 1.0;
    } else if (type === 'EXPLOSION') {
        color = 0xFFA500; // Orange
        speed = 1.5;
        size = 0.8;
        life = 0.4;
    } else if (type === 'WALL_HIT') {
        color = 0x888888;
        speed = 0.5;
        size = 0.3;
        life = 0.3;
    }

    let spawned = 0;
    for (const p of pool) {
        if (!p.active) {
            p.activate(pos, color, speed * randomRange(0.5, 1.5), size, life);
            spawned++;
            if (spawned >= count) break;
        }
    }
}

export function updateParticles(dt) {
    pool.forEach(p => p.update(dt));
}